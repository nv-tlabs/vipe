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

from vipe.utils.cameras import BaseCameraModel, CameraType
from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


class PinholeDepthAdapter(DepthEstimationModel):
    """
    Adapter for depth estimation models that support only pinhole cameras.
    Works by rectifying the input image to pinhole, perform depth estimation, and then unrectify the depth map.
    """

    def __init__(self, model: DepthEstimationModel):
        super().__init__()
        self.model = model
        assert CameraType.PINHOLE in self.model.supported_camera_types, "Model must support pinhole cameras"

        self.last_intrinsics: torch.Tensor | None = None
        self.last_pinhole_intrinsics: torch.Tensor | None = None
        self.last_rectification_map: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def supported_camera_types(self) -> list[CameraType]:
        return [CameraType.PINHOLE, CameraType.MEI]

    @property
    def depth_type(self) -> DepthType:
        # Model needs to be re-run if intrinsics are updated
        return DepthType.MODEL_METRIC_DEPTH

    def _compute_rectification_map(self, h: int, w: int, src_model: BaseCameraModel, dst_model: BaseCameraModel):
        # Compute rectification map that should be performed on the src img to obtain the dst model img
        device = src_model.intrinsics.device
        y, x = torch.meshgrid(
            torch.arange(h, device=device).float(), torch.arange(w, device=device).float(), indexing="ij"
        )
        pts, _, _ = dst_model.iproj_disp(torch.ones_like(x), x, y)
        coords, _, _ = src_model.proj_points(pts)
        coords_norm = 2.0 * coords / torch.tensor([w, h], device=coords.device) - 1.0
        coords_norm = coords_norm.reshape(1, h, w, 2)
        return coords_norm

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        # Short-cut for supported camera types
        if src.camera_type in self.model.supported_camera_types:
            return self.model.estimate(src)

        rgb: torch.Tensor = unpack_optional(src.rgb)
        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True
        img_h, img_w = rgb.shape[1:3]

        if self.last_intrinsics is None or not torch.allclose(self.last_intrinsics, src.intrinsics):
            tgt_intr = src.camera_type.build_camera_model(src.intrinsics)
            pinhole_intr = tgt_intr.pinhole()
            to_pinhole_map = self._compute_rectification_map(img_h, img_w, tgt_intr, pinhole_intr)
            to_tgt_map = self._compute_rectification_map(img_h, img_w, pinhole_intr, tgt_intr)
            self.last_intrinsics = src.intrinsics.clone()
            self.last_pinhole_intrinsics = pinhole_intr.intrinsics.clone()
            self.last_rectification_map = (to_pinhole_map, to_tgt_map)

        assert self.last_rectification_map is not None
        to_pinhole_map, to_tgt_map = self.last_rectification_map

        rgb = rgb.permute(0, 3, 1, 2)
        rgb = torch.nn.functional.grid_sample(
            rgb,
            to_pinhole_map,
            mode="bilinear",
            align_corners=False,
        )
        rgb = rgb.permute(0, 2, 3, 1)

        est = self.model.estimate(
            DepthEstimationInput(rgb=rgb, intrinsics=self.last_pinhole_intrinsics, camera_type=CameraType.PINHOLE)
        )
        metric_depth = torch.nn.functional.grid_sample(
            est.metric_depth.unsqueeze(1),
            to_tgt_map,
            mode="nearest",
            align_corners=False,
        )
        metric_depth = metric_depth.squeeze(1)

        if not batch_dim:
            metric_depth = metric_depth[0]

        return DepthEstimationResult(metric_depth=metric_depth)
