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
import json
import os
import time
from pathlib import Path
import torch

import torch.nn as nn
from flashpack import FlashPackMixin
from huggingface_hub import hf_hub_download
try:
    from moge.model.v1 import MoGeModel as _MoGeModel
except ModuleNotFoundError:
   _MoGeModel = None

from vipe.utils.misc import unpack_optional
from vipe.utils.cameras import CameraType

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


class FlashPackMoGeModel(nn.Module, FlashPackMixin):
    """FlashPack-enabled wrapper for MoGe."""

    # Tell FlashPack how to initialize this model
    flashpack_init_method = "from_model_config"

    def __init__(self, model_config=None, **kwargs):
        super().__init__()
        # model_config will be passed from from_flashpack via config parameter
        if model_config is None:
            # Extract from kwargs if passed as 'config'
            model_config = kwargs.get('config', None)
        if model_config is None:
            raise ValueError("model_config is required to initialize FlashPackMoGeModel")
        # ALWAYS create the MoGe model structure (required for flashpack to assign weights)
        self.moge_model = _MoGeModel(**model_config)

    def forward(self, *args, **kwargs):
        return self.moge_model(*args, **kwargs)

    def infer(self, *args, **kwargs):
        return self.moge_model.infer(*args, **kwargs)
def focal_length_to_fov_degrees(focal_length: float, image_width: float) -> float:
    """Compute horizontal field of view from focal length."""
    fov_rad = 2 * torch.atan(torch.tensor(image_width / (2 * focal_length)))
    fov_deg = torch.rad2deg(fov_rad)
    return fov_deg.item()


class MogeModel(DepthEstimationModel, nn.Module):
    """https://github.com/microsoft/MoGe with FlashPack optimization."""

    def __init__(self, cache_dir: str | None = None, flashpack_cache_dir: str = None, use_flashpack: bool = True,  device: str = "cuda") -> None:
        super().__init__()
        if _MoGeModel is None:
            raise RuntimeError(
                "moge is not found in the environment. You can install it via pip install `git+https://github.com/microsoft/MoGe.git`"
            )
        model_name = "Ruicheng/moge-vitl"

        if use_flashpack:
            # Setup flashpack cache
            if flashpack_cache_dir is None:
                flashpack_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vipe_moge_flashpack")
            else:
                os.makedirs(flashpack_cache_dir, exist_ok=True)
                flashpack_cache_dir = Path(flashpack_cache_dir)

            save_dir = os.path.join(flashpack_cache_dir, "moge-vitl")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "model.flashpack")
            config_path = os.path.join(save_dir, "model_config.json")

            # Download the original checkpoint to get config (HF caches this)
            cached_checkpoint_path = hf_hub_download(
                repo_id=model_name,
                repo_type="model",
                filename="model.pt",
                cache_dir=cache_dir
            )

            # If flashpack doesn't exist, create it
            if not os.path.exists(model_path):
                print(f"Creating flashpack for {model_name}...")
                start = time.time()

                # Load checkpoint to get config
                checkpoint = torch.load(cached_checkpoint_path, map_location='cpu', weights_only=True)
                model_config = checkpoint['model_config']

                # Save config for later use
                with open(config_path, 'w') as f:
                    json.dump(model_config, f)

                # Create model and load state dict
                base_model = _MoGeModel(**model_config)
                base_model.load_state_dict(checkpoint['model'])

                # Wrap and save as flashpack
                wrapped_model = FlashPackMoGeModel(model_config)
                wrapped_model.moge_model = base_model
                wrapped_model.save_flashpack(model_path, target_dtype=torch.float32)

                print(f"Flashpack creation took {time.time() - start:.2f}s")
                del wrapped_model, base_model
                torch.cuda.empty_cache()

            # Load from flashpack
            print(f"Loading {model_name} from flashpack...")
            start = time.time()

            # Load the saved config
            with open(config_path, 'r') as f:
                model_config = json.load(f)

            # Load weights from flashpack (will call __init__ with model_config)
            wrapped_model = FlashPackMoGeModel.from_flashpack(
                model_path,
                config=model_config,  # This gets passed to __init__ as 'config' kwarg
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            self.model = wrapped_model.moge_model
            print(f"Flashpack loading took {time.time() - start:.2f}s")
        else:
            # Original loading method
            self.model = _MoGeModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.device = device
        self.model = self.model.to(device).eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.camera_type == CameraType.PINHOLE, "MoGe only supports pinhole cameras"

        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        w = rgb.shape[2]
        input_image_for_depth = rgb.moveaxis(-1, 1).to(self.device)

        # MoGe inference
        moge_input_dict = {"fov_x": focal_length_to_fov_degrees(focal_length, w)}

        with torch.no_grad():
            moge_output_full = self.model.infer(input_image_for_depth, **moge_input_dict)

        moge_depth_hw_full = moge_output_full["depth"]
        moge_mask_hw_full = moge_output_full["mask"]

        # Process depth
        moge_depth_tensor = torch.nan_to_num(moge_depth_hw_full, nan=1e4)
        moge_depth_tensor = torch.clamp(moge_depth_tensor, min=0, max=1e4)

        moge_depth_tensor = moge_depth_tensor * moge_mask_hw_full.float()

        if not batch_dim:
            moge_depth_tensor = moge_depth_tensor.squeeze(0)
            moge_mask_hw_full = moge_mask_hw_full.squeeze(0)

        return DepthEstimationResult(metric_depth=moge_depth_tensor)
    @torch.no_grad()
    def forward_image(self, image: torch.Tensor, **kwargs):
        # image: b, 3, h, w 0,1
        output = self.model.infer(image, resolution_level=9, apply_mask=False, **kwargs)
        points = output['points'] # b,h,w,3
        masks = output['mask'] # b,h,w
        return points, masks