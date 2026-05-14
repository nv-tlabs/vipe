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

from dataclasses import dataclass

import torch

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .model_fn import (
    metric3d_convnext_large,
    metric3d_convnext_tiny,
    metric3d_vit_giant2,
    metric3d_vit_large,
    metric3d_vit_small,
)


class Metric3DDepthModel(DepthEstimationModel):
    """
    https://github.com/YvanYin/Metric3D
    """

    @dataclass
    class PadInfo:
        pad_up: int
        pad_down: int
        pad_left: int
        pad_right: int
        size_origin: tuple[int, int]
        focal_length: float

    def __init__(self, version: int = 2, model: str = "giant2") -> None:
        super().__init__()

        if version == 1:
            assert model in ["tiny", "large"]
            if model == "tiny":
                self.model = metric3d_convnext_tiny(pretrain=True, skip_validation=True)
            else:
                self.model = metric3d_convnext_large(pretrain=True, skip_validation=True)

            self.input_size = (544, 1216)

        elif version == 2:
            assert model in ["small", "large", "giant2"]
            if model == "small":
                self.model = metric3d_vit_small(pretrain=True, skip_validation=True)
            elif model == "large":
                self.model = metric3d_vit_large(pretrain=True, skip_validation=True)
            else:
                self.model = metric3d_vit_giant2(pretrain=True, skip_validation=True)

            self.input_size = (616, 1064)

        else:
            raise ValueError("Invalid version number.")

        self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    @torch.no_grad()
    def _prepare_input(self, img: torch.Tensor, focal_length: float) -> tuple[torch.Tensor, PadInfo]:
        """
        Convert a 0-1 RGB image to normalized and padded image required by the model.
        input: (B, H, W, 3) RGB image
        return: (B, 3, H', W') tensor and pad_info
        """

        h_origin, w_origin = img.shape[1:3]
        scale = min(self.input_size[0] / h_origin, self.input_size[1] / w_origin)
        h_scaled, w_scaled = int(h_origin * scale), int(w_origin * scale)

        img = img.permute(0, 3, 1, 2)
        img = torch.nn.functional.interpolate(img, (h_scaled, w_scaled), mode="bilinear")

        # normalize
        mean = (
            torch.tensor([123.675, 116.28, 103.53], device=img.device, dtype=torch.float32)[None, :, None, None] / 255
        )
        std = torch.tensor([58.395, 57.12, 57.375], device=img.device, dtype=torch.float32)[None, :, None, None] / 255
        img = torch.div((img - mean), std)

        # padding to input_size
        pad_h = self.input_size[0] - h_scaled
        pad_w = self.input_size[1] - w_scaled

        pad_up, pad_left = pad_h // 2, pad_w // 2
        pad_down, pad_right = pad_h - pad_up, pad_w - pad_left

        img = torch.nn.functional.pad(
            img,
            (pad_left, pad_right, pad_up, pad_down),
            value=0.0,
        )

        return img, self.PadInfo(
            pad_up,
            pad_down,
            pad_left,
            pad_right,
            (h_origin, w_origin),
            focal_length * scale,
        )

    def _post_process(self, feature: torch.Tensor, pad_info: PadInfo, is_depth: bool) -> torch.Tensor:
        """
        Post-process the feature map to remove padding, and de-canonicalize the depth.
        input: (B, C, H', W') tensor
        return: (B, C, H, W) tensor
        """

        feature = feature[
            ...,
            pad_info.pad_up : feature.size(-2) - pad_info.pad_down,
            pad_info.pad_left : feature.size(-1) - pad_info.pad_right,
        ]
        feature = torch.nn.functional.interpolate(feature, pad_info.size_origin, mode="bilinear")

        if is_depth:
            canonical_to_real_scale = pad_info.focal_length / 1000.0
            feature = feature * canonical_to_real_scale

        return feature

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"

        assert src.camera_type == CameraType.PINHOLE, "Metric3D only supports pinhole cameras"
        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb, pad_info = self._prepare_input(rgb, focal_length)
        pred_depth, confidence, output_dict = self.model.inference({"input": rgb})

        pred_depth = self._post_process(pred_depth, pad_info, is_depth=True)
        confidence = self._post_process(confidence, pad_info, is_depth=False)

        if not batch_dim:
            pred_depth, confidence = pred_depth[0], confidence[0]

        return DepthEstimationResult(
            metric_depth=pred_depth[0],
            confidence=confidence[0],
        )
