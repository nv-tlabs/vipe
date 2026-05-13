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

"""Vendored minimal inference adapter for Depth Anything 3.

The upstream project is Apache-2.0 licensed:
https://github.com/ByteDance-Seed/Depth-Anything-3
"""

import numpy as np
import torch
import torch.nn.functional as F

from vipe.priors.depth.dav3.api import DepthAnything3
from vipe.priors.depth.dav3.utils import logger as dav3_logger
from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


class DepthAnything3Model(DepthEstimationModel):
    """Single-view metric-depth adapter for DAv3."""

    HF_REPO = "depth-anything/DA3METRIC-LARGE"
    MODEL_NAME = "da3metric-large"

    def __init__(self, weights_path: str | None = None) -> None:
        super().__init__()
        dav3_logger.level = 0
        self.model = DepthAnything3.from_pretrained(
            self.HF_REPO,
            model_name=self.MODEL_NAME,
            weights_path=weights_path,
        )
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        """DAv3 metric depth is proportional to focal length."""
        return DepthType.METRIC_DEPTH

    @property
    def supported_camera_types(self) -> list[CameraType]:
        return [CameraType.PINHOLE]

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.camera_type == CameraType.PINHOLE, "DAv3 only supports pinhole cameras"

        intrinsics = unpack_optional(src.intrinsics)
        focal_length: float = intrinsics[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb_images = [(rgb[idx].cpu().numpy() * 255).astype(np.uint8) for idx in range(rgb.shape[0])]

        with torch.inference_mode():
            dav3_result = self.model.inference(
                rgb_images,
                process_res_method="upper_bound_resize",
                process_res=504,
            )

        # DAv3 predicts metric depth at a canonical focal of 300 after internal
        # resizing. Rescale to the caller's focal length and original resolution.
        dav3_camera_focal = focal_length / max(rgb.shape[2], rgb.shape[1]) * 504
        dav3_metric_depth = dav3_result.depth * dav3_camera_focal / 300.0

        if dav3_result.sky is not None:
            dav3_metric_depth = dav3_metric_depth * (~dav3_result.sky).astype(dav3_metric_depth.dtype)

        metric_depth = torch.from_numpy(dav3_metric_depth).to(device=rgb.device, dtype=torch.float32)[:, None]
        metric_depth = F.interpolate(metric_depth, rgb.shape[1:3], mode="nearest")[:, 0]

        if not batch_dim:
            metric_depth = metric_depth.squeeze(0)

        return DepthEstimationResult(metric_depth=metric_depth)


__all__ = ["DepthAnything3", "DepthAnything3Model"]
