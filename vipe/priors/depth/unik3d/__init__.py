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

import numpy as np
import torch

from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .unik3d import UniK3D
from .utils.camera import Spherical


class Unik3DModel(DepthEstimationModel):
    def __init__(self, type: Literal["s", "b", "l"] = "l") -> None:
        super().__init__()
        self.model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-vit{type}")
        self.model.resolution_level = 9
        self.model.interpolation_mode = "bilinear"
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DISTANCE

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.focal_length is None, "This is only intended for 360 panoramas"

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb = rgb.moveaxis(-1, 1) * 255.0
        H, W = rgb.shape[-2:]
        hfov2 = np.pi
        camera = Spherical(params=torch.tensor([0, 0, 0, 0, W, H, hfov2, H / W * hfov2]).float().cuda())
        outputs = self.model.infer(rgb, camera=camera, normalize=True)

        pred_distance = outputs["distance"][0]
        confidence = outputs["confidence"][0]

        if not batch_dim:
            pred_distance, confidence = pred_distance[0], confidence[0]

        return DepthEstimationResult(
            metric_depth=pred_distance,
            confidence=confidence,
        )
