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

"""Helper functions for Modal optimization."""

import logging

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_geometrycrafter_models_cpu(cache_dir: str = "/home/afridi/Depth/GeometryCrafter/workspace/cache"):
    """Load GeometryCrafter models on CPU for Modal snapshot optimization.
    
    This loads the heavy models on CPU (captured in Modal snapshot), then they
    can be quickly transferred to GPU after snapshot restore.
    
    Args:
        cache_dir: Cache directory for model weights
        
    Returns:
        dict containing the loaded models on CPU
    """
    from vipe.priors.depth.geometrycrafter import (
        GeometryCrafterDiffPipeline,
        PMapAutoencoderKLTemporalDecoder,
        UNetSpatioTemporalConditionModelVid2vid,
    )
    from vipe.priors.depth.moge import MogeModel
    
    logger.info("Loading GeometryCrafter models on CPU for Modal snapshot...")
    
    model_type = "diff"  # TODO: make configurable
    
    # Load UNet on CPU (use float32 to avoid CPU float16 issues)
    logger.info("  Loading UNet...")
    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=cache_dir
    ).requires_grad_(False).to("cpu", dtype=torch.float32)
    
    # Load VAE on CPU
    logger.info("  Loading VAE...")
    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='point_map_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=cache_dir
    ).requires_grad_(False).to("cpu", dtype=torch.float32)
    
    # Load MoGe prior model on CPU
    logger.info("  Loading MoGe prior...")
    prior_model = MogeModel(
        cache_dir=cache_dir,
        device="cpu",  # Keep on CPU for Modal snapshot
    ).requires_grad_(False).to('cpu', dtype=torch.float32)
    
    # Load full pipeline on CPU (use float32 to avoid CPU float16 issues)
    logger.info("  Loading GeometryCrafter pipeline...")
    video_depth_model = GeometryCrafterDiffPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        torch_dtype=torch.float32,
        cache_dir=cache_dir
    ).to("cpu")
    
    logger.info("✓ GeometryCrafter models loaded on CPU")
    
    return {
        "point_map_vae": point_map_vae,
        "prior_model": prior_model,
        "video_depth_model": video_depth_model,
    }


def move_geometrycrafter_models_to_gpu(gc_models: dict):
    """Move GeometryCrafter models from CPU to GPU.
    
    Args:
        gc_models: Dict of models from load_geometrycrafter_models_cpu()
        
    Returns:
        dict containing the models on GPU
    """
    logger.info("⚡ Moving GeometryCrafter models from CPU to GPU...")
    
    # Move each model to GPU (convert to float16 for efficiency)
    gc_models["point_map_vae"] = gc_models["point_map_vae"].to("cuda", dtype=torch.float32)
    gc_models["prior_model"] = gc_models["prior_model"].to("cuda", dtype=torch.float32)
    # Convert pipeline to float16 on GPU for better performance
    gc_models["video_depth_model"] = gc_models["video_depth_model"].to("cuda", dtype=torch.float16)
    
    logger.info("✓ GeometryCrafter models now on GPU")
    
    return gc_models


def prebuild_slam_components_cpu(slam_config: DictConfig, camera_type_str: str = "pinhole"):
    """Pre-build SLAM components on CPU for Modal snapshot optimization.
    
    Builds SLAM models with default dimensions so they're included in the snapshot.
    The models will be transferred to GPU after snapshot restore.
    
    Args:
        slam_config: SLAM configuration from Hydra
        camera_type_str: Camera type string (default: "pinhole")
        
    Returns:
        SLAMSystem instance with components built on CPU
    """
    from vipe.slam.system import SLAMSystem
    from vipe.utils.cameras import CameraType
    from vipe.ext.lietorch import SE3
    
    logger.info("Pre-building SLAM components on CPU for Modal snapshot...")
    
    # Create SLAM system on CPU
    slam_system = SLAMSystem(device=torch.device("cpu"), config=slam_config)
    
    # Set default video dimensions and camera type to build components
    # These are typical values - actual dimensions will override during run()
    # Note: Both dimensions must be divisible by 8 for GraphBuffer
    slam_system.config.update({
        "height": 480,  # 480 % 8 == 0 ✓
        "width": 848,   # 848 % 8 == 0 ✓ (closest to 854)
        "n_views": 1,
        "has_init_pose": False,  # Default: no initial pose (will be determined during run())
        "camera_type": CameraType(camera_type_str),
    })
    
    # Initialize default rig (SE3 identity for single-view)
    # This will be overridden during actual run() with real rig
    slam_system.rig = SE3.Identity(1)
    
    # Build all components on CPU
    slam_system._build_components_cpu()
    
    logger.info("✓ SLAM components pre-built on CPU")
    
    return slam_system


def preload_slam_models_cpu():
    """Pre-load SLAM models (DroidNet) on CPU for Modal snapshot optimization.
    
    Only loads the heavy neural network models, not video-specific components
    like GraphBuffer which depend on video dimensions and must be created fresh
    for each video.
    
    Returns:
        dict: Dictionary containing pre-loaded models
            - 'droid_net': DroidNet model on CPU
    """
    from vipe.slam.networks.droid_net import DroidNet
    
    logger.info("Pre-loading SLAM models (DroidNet) on CPU for Modal snapshot...")
    
    # Load DroidNet on CPU - this is the heavy model (~200MB)
    droid_net = DroidNet().to("cpu")
    
    models = {
        'droid_net': droid_net,
    }
    
    logger.info("✓ SLAM models pre-loaded on CPU")
    
    return models


def move_slam_models_to_gpu(slam_models: dict):
    """Transfer pre-loaded SLAM models from CPU to GPU.
    
    Args:
        slam_models: Dictionary from preload_slam_models_cpu()
        
    Returns:
        dict: Dictionary with models moved to GPU
    """
    logger.info("⚡ Moving SLAM models from CPU to GPU...")
    
    moved_models = {}
    if 'droid_net' in slam_models:
        moved_models['droid_net'] = slam_models['droid_net'].to("cuda")
    
    logger.info("✓ SLAM models moved to GPU")
    
    return moved_models


