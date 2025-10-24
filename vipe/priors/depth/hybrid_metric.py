# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hybrid Metric Depth Model combining MoGe and GeometryCrafter.

This model uses:
- MoGe for metric depth reference on keyframes
- GeometryCrafter for temporally consistent relative depth on all frames
- Scale optimization to combine both advantages
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .moge import MogeModel
from .scale_optimizer import ScaleOptimizer
from vipe.utils.misc import unpack_optional

logger = logging.getLogger(__name__)


class HybridMetricDepthModel(DepthEstimationModel, nn.Module):
    """
    Hybrid depth model combining MoGe's metric accuracy with GeometryCrafter's temporal consistency.
    
    Pipeline:
    1. Run GeometryCrafter on all frames → relative depth
    2. Run MoGe on keyframes → metric depth
    3. Compute scale factors at keyframes
    4. Interpolate and optimize scales across video
    5. Apply scales to get metric depth for all frames
    """
    
    def __init__(
        self,
        moge_model_name: str = "Ruicheng/moge-2-vitl",
        keyframe_interval: int = 10,
        optimize_scales: bool = True,
        lambda_metric: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_accel: float = 0.01,
        cache_dir: str | None = None,
    ):
        """
        Initialize the hybrid depth model.
        
        Args:
            moge_model_name: HuggingFace model name for MoGe
            keyframe_interval: Interval between keyframes for MoGe inference
            optimize_scales: Whether to optimize scales for temporal smoothness
            lambda_metric: Weight for metric accuracy loss in optimization
            lambda_smooth: Weight for temporal smoothness loss
            lambda_accel: Weight for acceleration penalty
            cache_dir: Cache directory for model weights
        """
        super().__init__()
        
        self.keyframe_interval = keyframe_interval
        self.optimize_scales = optimize_scales
        
        # Initialize MoGe model
        logger.info(f"Initializing MoGe model: {moge_model_name}")
        self.moge = MogeModel(cache_dir=cache_dir)
        
        # Initialize scale optimizer
        self.scale_optimizer = ScaleOptimizer(
            lambda_metric=lambda_metric,
            lambda_smooth=lambda_smooth,
            lambda_accel=lambda_accel,
        )
        
        # Cache for computed scales
        self.scale_cache: Dict[str, torch.Tensor] = {}
        
        logger.info(
            f"HybridMetricDepthModel initialized: "
            f"keyframe_interval={keyframe_interval}, "
            f"optimize_scales={optimize_scales}"
        )
    
    @property
    def depth_type(self) -> DepthType:
        """Return depth type as metric depth."""
        return DepthType.MODEL_METRIC_DEPTH
    
    def select_keyframes(self, n_frames: int) -> List[int]:
        """
        Select keyframe indices for MoGe inference.
        
        Args:
            n_frames: Total number of frames
        
        Returns:
            List of keyframe indices
        """
        # Simple fixed-interval strategy
        keyframes = list(range(0, n_frames, self.keyframe_interval))
        
        # Always include the last frame
        if (n_frames - 1) not in keyframes:
            keyframes.append(n_frames - 1)
        
        logger.info(f"Selected {len(keyframes)} keyframes from {n_frames} total frames")
        return keyframes
    
    def estimate_video(
        self,
        frames: List[DepthEstimationInput],
        gc_depths: torch.Tensor,
    ) -> List[DepthEstimationResult]:
        """
        Estimate metric depth for an entire video using hybrid approach.
        
        Args:
            frames: List of video frames as DepthEstimationInput
            gc_depths: GeometryCrafter relative depths (n_frames, H, W)
        
        Returns:
            List of DepthEstimationResult with metric depth
        """
        n_frames = len(frames)
        logger.info(f"Processing video with {n_frames} frames using hybrid depth")
        
        # Step 1: Select keyframes
        keyframe_indices = self.select_keyframes(n_frames)
        
        # Step 2: Run MoGe on keyframes
        logger.info(f"Running MoGe on {len(keyframe_indices)} keyframes...")
        moge_depths = {}
        for kf_idx in keyframe_indices:
            logger.debug(f"Processing keyframe {kf_idx}/{n_frames}")
            moge_result = self.moge.estimate(frames[kf_idx])
            moge_depths[kf_idx] = moge_result.metric_depth
        
        # Step 3: Compute scale factors at keyframes
        logger.info("Computing scale factors at keyframes...")
        keyframe_scales = {}
        for kf_idx, moge_depth in moge_depths.items():
            gc_depth = gc_depths[kf_idx]
            scale = self.scale_optimizer.compute_scale_at_frame(
                moge_depth, gc_depth
            )
            keyframe_scales[kf_idx] = scale
            logger.debug(f"Keyframe {kf_idx}: scale={scale:.3f}")
        
        # Step 4: Interpolate scales across all frames
        logger.info("Interpolating scales across video...")
        scales = self.scale_optimizer.interpolate_scales(
            keyframe_scales, n_frames
        )
        
        # Step 5: Optimize scales for temporal smoothness (optional)
        if self.optimize_scales:
            logger.info("Optimizing scales for temporal smoothness...")
            scales = self.scale_optimizer.optimize_temporal_smoothness(
                scales, moge_depths, gc_depths
            )
        
        # Log scale statistics
        scale_stats = self.scale_optimizer.compute_scale_statistics(scales)
        logger.info(f"Scale statistics: {scale_stats}")
        
        # Step 6: Apply scales to get metric depths
        logger.info("Applying scales to produce metric depths...")
        results = []
        for frame_idx in range(n_frames):
            metric_depth = gc_depths[frame_idx] * scales[frame_idx]
            results.append(
                DepthEstimationResult(metric_depth=metric_depth)
            )
        
        logger.info(f"Hybrid depth estimation complete for {n_frames} frames")
        return results
    
    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """
        Estimate depth for a single frame.
        
        For single frame estimation, we fall back to MoGe only.
        For video processing, use estimate_video() instead.
        
        Args:
            src: Input depth estimation data
        
        Returns:
            DepthEstimationResult with metric depth
        """
        logger.warning(
            "HybridMetricDepthModel.estimate() called for single frame. "
            "Falling back to MoGe only. For optimal results, use estimate_video()."
        )
        return self.moge.estimate(src)
    
    def estimate_batch(
        self,
        srcs: List[DepthEstimationInput],
        gc_depths: torch.Tensor | None = None,
    ) -> List[DepthEstimationResult]:
        """
        Estimate depth for a batch of frames.
        
        Args:
            srcs: List of input depth estimation data
            gc_depths: Optional pre-computed GeometryCrafter depths (n_frames, H, W)
        
        Returns:
            List of DepthEstimationResult with metric depth
        """
        if gc_depths is None:
            logger.warning(
                "No GeometryCrafter depths provided to estimate_batch(). "
                "This defeats the purpose of the hybrid model. "
                "Consider providing gc_depths for better results."
            )
            # Fall back to MoGe for all frames
            return [self.moge.estimate(src) for src in srcs]
        
        return self.estimate_video(srcs, gc_depths)
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for compatibility with nn.Module.
        
        Args:
            images: Input images (B, C, H, W)
            **kwargs: Additional arguments
        
        Returns:
            Metric depth maps (B, H, W)
        """
        # For single image or batch without temporal context, use MoGe
        return self.moge.forward_image(images, **kwargs)[0][:, :, :, 2]  # Extract Z component
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the hybrid model configuration."""
        return {
            "model_type": "hybrid_metric",
            "moge_model": "Ruicheng/moge-2-vitl",
            "keyframe_interval": self.keyframe_interval,
            "optimize_scales": self.optimize_scales,
            "lambda_metric": self.scale_optimizer.lambda_metric,
            "lambda_smooth": self.scale_optimizer.lambda_smooth,
            "lambda_accel": self.scale_optimizer.lambda_accel,
        }

