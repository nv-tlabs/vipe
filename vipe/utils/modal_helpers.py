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


def load_unidepth_model_cpu(model_type: str = "l"):
    """Load UniDepthV2 on CPU for Modal snapshot.
    
    This loads the UniDepthV2 model on CPU (captured in Modal snapshot), then it
    can be quickly transferred to GPU after snapshot restore. This model is shared
    between SLAM (keyframe depth) and AdaptiveDepthProcessor (metric depth) to
    avoid loading it twice.
    
    Args:
        model_type: Model size - "s" (small), "b" (base), or "l" (large)
        
    Returns:
        UniDepth2Model instance on CPU
    """
    from vipe.priors.depth.unidepth import UniDepth2Model
    
    logger.info(f"Loading UniDepthV2-{model_type} on CPU for Modal snapshot...")
    model = UniDepth2Model(type=model_type, device="cpu")
    logger.info("✓ UniDepthV2 loaded on CPU")
    
    return model


def move_unidepth_to_gpu(model):
    """Move UniDepthV2 from CPU to GPU.
    
    Args:
        model: UniDepth2Model instance from load_unidepth_model_cpu()
        
    Returns:
        UniDepth2Model instance on GPU
    """
    logger.info("⚡ Moving UniDepthV2 from CPU to GPU...")
    model.model = model.model.to("cuda")
    model.device = "cuda"
    logger.info("✓ UniDepthV2 moved to GPU")
    
    return model


def load_trackanything_models_cpu():
    """Load SAM and AOT models on CPU for Modal snapshot.
    
    This loads the TrackAnything models (SAM for segmentation, AOT for tracking)
    on CPU, so they're captured in the Modal snapshot and can be quickly moved
    to GPU after snapshot restore.
    
    Returns:
        dict: Dictionary containing pre-loaded models
            - 'sam': SAM model on CPU
            - 'aot': AOT model on CPU
    """
    import torch
    from pathlib import Path
    from vipe.priors.track_anything.sam import sam_model_registry
    from vipe.priors.track_anything.aot import config as engine_config
    from vipe.priors.track_anything.aot.networks.models import build_vos_model
    
    logger.info("Loading TrackAnything models on CPU for Modal snapshot...")
    
    # Load SAM on CPU
    logger.info("  Loading SAM...")
    sam_ckpt_path = Path(torch.hub.get_dir()) / "sam" / "sam_vit_b_01ec64.pth"
    sam_model = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt_path))
    sam_model = sam_model.to("cpu")
    logger.info("  ✓ SAM loaded")
    
    # Load AOT on CPU
    logger.info("  Loading AOT...")
    aot_ckpt_path = Path(torch.hub.get_dir()) / "aot" / "R50_DeAOTL_PRE_YTB_DAV.pth"
    cfg = engine_config.EngineConfig("PRE_YTB_DAV")
    cfg.TEST_CKPT_PATH = str(aot_ckpt_path)
    aot_model = build_vos_model(cfg.MODEL_VOS, cfg)
    # Load checkpoint weights on CPU
    # Note: weights_only=False is needed for older checkpoints with numpy types
    checkpoint = torch.load(str(aot_ckpt_path), map_location="cpu", weights_only=False)
    # Training checkpoints have nested structure - extract just the model state dict
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    aot_model.load_state_dict(checkpoint)
    logger.info("  ✓ AOT loaded")
    
    logger.info("✓ TrackAnything models loaded on CPU")
    
    return {"sam": sam_model, "aot": aot_model}


def move_trackanything_to_gpu(models_dict):
    """Move TrackAnything models from CPU to GPU.
    
    Args:
        models_dict: Dictionary from load_trackanything_models_cpu()
        
    Returns:
        dict: Dictionary with models moved to GPU
    """
    logger.info("⚡ Moving TrackAnything models from CPU to GPU...")
    models_dict["sam"] = models_dict["sam"].to("cuda")
    models_dict["aot"] = models_dict["aot"].to("cuda")
    logger.info("✓ TrackAnything models moved to GPU")
    
    return models_dict


