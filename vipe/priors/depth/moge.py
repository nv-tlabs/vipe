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

import torch


try:
    from moge.model.v1 import MoGeModel
except ModuleNotFoundError:
    MoGeModel = None

from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


def focal_length_to_fov_degrees(focal_length: float, image_width: float) -> float:
    """Compute horizontal field of view from focal length."""
    fov_rad = 2 * torch.atan(torch.tensor(image_width / (2 * focal_length)))
    fov_deg = torch.rad2deg(fov_rad)
    return fov_deg.item()


class MogeModel(DepthEstimationModel):
    """https://github.com/microsoft/MoGe"""

    def __init__(self) -> None:
        super().__init__()
        if MoGeModel is None:
            raise RuntimeError(
                "moge is not found in the environment. You can install it via pip install `git+https://github.com/microsoft/MoGe.git`"
            )
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"

        focal_length: float = unpack_optional(src.focal_length)

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        w = rgb.shape[2]
        input_image_for_depth = rgb.moveaxis(-1, 1)

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
