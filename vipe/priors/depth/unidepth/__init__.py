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

from typing import Literal
import os
from pathlib import Path
import time
import torch
from flashpack import FlashPackMixin

from vipe.utils.misc import unpack_optional
from vipe.utils.cameras import CameraType

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .models.unidepthv2.unidepthv2 import Pinhole, UniDepthV2

class FlashPackUniDepthV2(UniDepthV2, FlashPackMixin):
    """FlashPack-enabled version of UniDepthV2 for faster model loading."""
    pass
class UniDepth2Model(DepthEstimationModel):
    def __init__(self, type: Literal["s", "b", "l"] = "l", flashpack_cache_dir: Path = None, use_flashpack: bool = True, device: str = "cuda") -> None:
        super().__init__()
        model_name = f"lpiccinelli/unidepth-v2-vit{type}14"
        start_time = time.time()
        if use_flashpack:
            # Setup flashpack cache directory
            if flashpack_cache_dir is None:
                flashpack_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vipe_unidepth_flashpack")
            os.makedirs(flashpack_cache_dir, exist_ok=True)

            save_dir = os.path.join(flashpack_cache_dir, f"unidepth-v2-vit{type}14")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "model.flashpack")
            config_path = os.path.join(save_dir, "config.json")

            # If flashpack doesn't exist, create it
            if not os.path.exists(model_path):
                print(f"Creating flashpack for {model_name}...")
                initial_model = FlashPackUniDepthV2.from_pretrained(model_name)

                # Save config for later loading
                initial_model._save_pretrained(Path(save_dir))

                initial_model.save_flashpack(model_path, target_dtype=torch.float16)
                del initial_model
                torch.cuda.empty_cache()

            # Load config and create model from flashpack
            print(f"Loading {model_name} from flashpack...")
            # Load the config from the saved directory
            from huggingface_hub import hf_hub_download
            import json

            if not os.path.exists(config_path):
                config_path = hf_hub_download(repo_id=model_name, filename="config.json")

            with open(config_path, 'r') as f:
                config = json.load(f)

            self.model = FlashPackUniDepthV2.from_flashpack(
                model_path,
                config=config,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            # Original loading method
            self.model = UniDepthV2.from_pretrained(model_name)
        end_time = time.time()
        print(f"Time taken to load unidepthv2 model: {end_time - start_time} seconds")
        self.model.interpolation_mode = "bilinear"
        self.device = device
        self.model = self.model.to(device).eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"

        assert src.camera_type == CameraType.PINHOLE, "UniDepth only supports pinhole cameras"
        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb = torch.clamp(rgb.moveaxis(-1, 1) * 255.0, max=255.0).byte()
        K = torch.tensor(
            [
                [focal_length, 0, rgb.shape[-1] / 2],
                [0, focal_length, rgb.shape[-2] / 2],
                [0, 0, 1],
            ],
            device=rgb.device,
        ).float()
        camera = Pinhole(K=K[None].repeat(rgb.shape[0], 1, 1))

        predictions = self.model.infer(rgb, camera)
        pred_depth = predictions["depth"].squeeze(1)
        confidence = predictions["confidence"].squeeze(1)

        if not batch_dim:
            pred_depth, confidence = pred_depth[0], confidence[0]

        return DepthEstimationResult(
            metric_depth=pred_depth,
            confidence=confidence,
        )
