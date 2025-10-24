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
Scale optimizer for hybrid metric depth estimation.

This module provides utilities to compute and optimize scale factors that convert
relative depth from GeometryCrafter to metric depth using MoGe as a reference.
"""

import logging
from typing import Dict

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ScaleOptimizer:
    """
    Compute and optimize scale factors for hybrid metric depth estimation.
    
    This class handles:
    1. Computing scale factors between MoGe and GeometryCrafter depths
    2. Interpolating scales between keyframes
    3. Optimizing scales for temporal smoothness
    """
    
    def __init__(
        self,
        lambda_metric: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_accel: float = 0.01,
        min_valid_ratio: float = 0.1,
    ):
        """
        Initialize the scale optimizer.
        
        Args:
            lambda_metric: Weight for metric accuracy loss
            lambda_smooth: Weight for temporal smoothness loss
            lambda_accel: Weight for acceleration penalty
            min_valid_ratio: Minimum ratio of valid pixels required
        """
        self.lambda_metric = lambda_metric
        self.lambda_smooth = lambda_smooth
        self.lambda_accel = lambda_accel
        self.min_valid_ratio = min_valid_ratio
    
    def compute_scale_at_frame(
        self,
        moge_depth: torch.Tensor,
        gc_depth: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> float:
        """
        Compute scale factor for a single frame using robust median.
        
        Args:
            moge_depth: Metric depth from MoGe (H, W)
            gc_depth: Relative depth from GeometryCrafter (H, W)
            valid_mask: Optional mask for valid pixels (H, W)
        
        Returns:
            Scale factor as float
        """
        # Create validity mask
        if valid_mask is None:
            valid_mask = torch.ones_like(moge_depth, dtype=torch.bool)
        
        # Filter out invalid depths (too small or too large)
        depth_valid = (
            (gc_depth > 0.01) & 
            (gc_depth < 100.0) &
            (moge_depth > 0.01) & 
            (moge_depth < 100.0) &
            valid_mask
        )
        
        # Check if we have enough valid pixels
        valid_ratio = depth_valid.float().mean().item()
        if valid_ratio < self.min_valid_ratio:
            logger.warning(
                f"Low valid pixel ratio: {valid_ratio:.2%}, "
                f"using fallback scale of 1.0"
            )
            return 1.0
        
        # Compute per-pixel scales
        per_pixel_scales = moge_depth[depth_valid] / gc_depth[depth_valid]
        
        # Use robust median to avoid outliers
        scale = torch.median(per_pixel_scales).item()
        
        # Sanity check: scale should be reasonable
        if scale < 0.1 or scale > 10.0:
            logger.warning(
                f"Computed scale {scale:.3f} is outside reasonable range [0.1, 10.0], "
                f"clamping to safe range"
            )
            scale = max(0.1, min(10.0, scale))
        
        logger.debug(
            f"Computed scale: {scale:.3f}, "
            f"valid pixels: {valid_ratio:.2%}, "
            f"scale std: {per_pixel_scales.std().item():.3f}"
        )
        
        return scale
    
    def interpolate_scales(
        self,
        keyframe_scales: Dict[int, float],
        n_frames: int,
    ) -> torch.Tensor:
        """
        Interpolate scale factors between keyframes.
        
        Args:
            keyframe_scales: Dictionary mapping keyframe indices to scale factors
            n_frames: Total number of frames in the video
        
        Returns:
            Tensor of scale factors for all frames (n_frames,)
        """
        scales = torch.ones(n_frames)
        kf_indices = sorted(keyframe_scales.keys())
        
        if len(kf_indices) == 0:
            logger.warning("No keyframes provided, using scale 1.0 for all frames")
            return scales
        
        # Handle frames before first keyframe
        first_kf = kf_indices[0]
        scales[:first_kf + 1] = keyframe_scales[first_kf]
        
        # Handle frames after last keyframe
        last_kf = kf_indices[-1]
        scales[last_kf:] = keyframe_scales[last_kf]
        
        # Interpolate between keyframes
        for i in range(len(kf_indices) - 1):
            kf_start = kf_indices[i]
            kf_end = kf_indices[i + 1]
            
            scale_start = keyframe_scales[kf_start]
            scale_end = keyframe_scales[kf_end]
            
            # Linear interpolation
            for frame_idx in range(kf_start + 1, kf_end):
                alpha = (frame_idx - kf_start) / (kf_end - kf_start)
                scales[frame_idx] = (1 - alpha) * scale_start + alpha * scale_end
        
        logger.info(
            f"Interpolated scales for {n_frames} frames "
            f"using {len(kf_indices)} keyframes"
        )
        
        return scales
    
    def optimize_temporal_smoothness(
        self,
        initial_scales: torch.Tensor,
        moge_depths: Dict[int, torch.Tensor],
        gc_depths: torch.Tensor,
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> torch.Tensor:
        """
        Optimize scale factors for temporal smoothness while maintaining metric accuracy.
        
        Args:
            initial_scales: Initial scale factors (n_frames,)
            moge_depths: Dictionary mapping keyframe indices to MoGe depths
            gc_depths: GeometryCrafter depths for all frames (n_frames, H, W)
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
        
        Returns:
            Optimized scale factors (n_frames,)
        """
        n_frames = len(initial_scales)
        
        # Create optimizable scales
        scales = initial_scales.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([scales], lr=learning_rate)
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            loss = 0
            
            # Metric accuracy loss (at keyframes)
            metric_loss = 0
            for kf_idx, moge_depth in moge_depths.items():
                if kf_idx >= n_frames:
                    continue
                    
                gc_depth = gc_depths[kf_idx]
                scaled_depth = gc_depth * scales[kf_idx]
                
                # Valid mask
                valid = (gc_depth > 0.01) & (moge_depth > 0.01)
                if valid.sum() > 0:
                    metric_loss += F.mse_loss(
                        scaled_depth[valid], 
                        moge_depth[valid]
                    )
            
            if len(moge_depths) > 0:
                metric_loss = metric_loss / len(moge_depths)
                loss += self.lambda_metric * metric_loss
            
            # Temporal smoothness loss (first derivative)
            smooth_loss = 0
            for i in range(n_frames - 1):
                smooth_loss += (scales[i + 1] - scales[i]).pow(2)
            smooth_loss = smooth_loss / (n_frames - 1)
            loss += self.lambda_smooth * smooth_loss
            
            # Acceleration penalty (second derivative)
            accel_loss = 0
            for i in range(n_frames - 2):
                accel = scales[i + 2] - 2 * scales[i + 1] + scales[i]
                accel_loss += accel.pow(2)
            if n_frames > 2:
                accel_loss = accel_loss / (n_frames - 2)
                loss += self.lambda_accel * accel_loss
            
            loss.backward()
            optimizer.step()
            
            # Clamp scales to reasonable range
            with torch.no_grad():
                scales.clamp_(0.1, 10.0)
            
            if iteration % 20 == 0:
                logger.debug(
                    f"Iteration {iteration}: "
                    f"loss={loss.item():.4f}, "
                    f"metric={metric_loss.item():.4f}, "
                    f"smooth={smooth_loss.item():.4f}, "
                    f"accel={accel_loss.item():.4f}"
                )
        
        optimized_scales = scales.detach()
        
        logger.info(
            f"Optimization complete: "
            f"scale range [{optimized_scales.min():.3f}, {optimized_scales.max():.3f}], "
            f"mean={optimized_scales.mean():.3f}"
        )
        
        return optimized_scales
    
    def compute_scale_statistics(
        self,
        scales: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute statistics about the scale factors.
        
        Args:
            scales: Scale factors (n_frames,)
        
        Returns:
            Dictionary of statistics
        """
        return {
            "mean": scales.mean().item(),
            "std": scales.std().item(),
            "min": scales.min().item(),
            "max": scales.max().item(),
            "range": (scales.max() - scales.min()).item(),
        }

