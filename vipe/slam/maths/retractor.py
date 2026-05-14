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

from typing import Any

import torch

from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType


class BaseRetractor:
    def oplus(self, x: Any, inds: torch.Tensor, dx: torch.Tensor) -> None:
        x[inds] += dx


class PoseRetractor(BaseRetractor):
    def oplus(self, x: SE3, inds: torch.Tensor, dx: torch.Tensor) -> None:
        x.data[inds] = SE3(x.data[inds]).retr(dx).data


class RigRotationOnlyRetractor(BaseRetractor):
    def oplus(self, x: SE3, inds: torch.Tensor, dx: torch.Tensor) -> None:
        dx = dx.clone()
        dx[:, :3] = 0  # zero out translation part
        x.data[inds] = SE3(x.data[inds]).retr(dx).data


class DenseDispRetractor(BaseRetractor):
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor) -> None:
        dx = torch.where(dx > 10, torch.zeros_like(dx), dx)
        super().oplus(x, inds, dx)


class TracksDispRetractor(BaseRetractor):
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor) -> None:
        super().oplus(x, inds, dx)
        x.clamp_(min=1e-3, max=10)


class IntrinsicsRetractor(BaseRetractor):
    def __init__(self, camera_type: CameraType):
        self.camera_type = camera_type

    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor) -> None:
        if len(dx) == 1:
            # Broadcast dx to all intrinsics
            inds = torch.where(x[:, 0] > 0)[0]
            dx = dx.repeat(len(inds), 1)
        x[inds, :2] += dx[..., :1]
        # Use smaller learning rate for the distortion parameters
        x[inds, 4:] += dx[..., 1:] * 0.01
