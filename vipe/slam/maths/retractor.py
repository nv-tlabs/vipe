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

from .intrinsics import IntrinsicsParameterization, validate_intrinsics_parameterization


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
    def __init__(
        self,
        camera_type: CameraType,
        distortion_update_scale: float = 0.01,
        parameterization: IntrinsicsParameterization = "additive",
    ):
        self.camera_type = camera_type
        self.distortion_update_scale = distortion_update_scale
        self.parameterization = parameterization
        validate_intrinsics_parameterization(parameterization, camera_type)

    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor) -> None:
        if len(dx) == 1:
            # Broadcast dx to all intrinsics
            inds = torch.where(x[:, 0] > 0)[0]
            dx = dx.repeat(len(inds), 1)

        if self.parameterization == "mei_center_log":
            # eta applies the same logarithmic scale to fx/fy, preserving their
            # ratio. beta changes both focal axes together with s=1+xi.
            current = x[inds]
            finfo = torch.finfo(current.dtype)
            one_plus_xi = (1.0 + current[..., 4]).clamp_min(finfo.eps)
            log_one_plus_xi = torch.log(one_plus_xi)
            log_eps = torch.log(current.new_tensor(finfo.eps))
            log_safe_max = torch.log(current.new_tensor(finfo.max / 4.0))
            beta_min = log_eps - log_one_plus_xi
            beta_max = log_safe_max - log_one_plus_xi
            applied_beta_update = dx[..., 1].clamp(min=beta_min, max=beta_max)
            beta_in_domain = ((dx[..., 1] >= beta_min) & (dx[..., 1] <= beta_max)) | (dx[..., 1] == 0)
            beta_multiplication_safe = (
                (dx[..., 1] >= torch.log(current.new_tensor(finfo.tiny))) & (dx[..., 1] <= log_safe_max)
            ) | (dx[..., 1] == 0)
            new_one_plus_xi = torch.where(
                beta_in_domain & beta_multiplication_safe,
                one_plus_xi * torch.exp(dx[..., 1]),
                torch.exp(log_one_plus_xi + applied_beta_update),
            )

            focal = current[..., :2].clamp_min(finfo.tiny)
            focal_update = dx[..., 0] + applied_beta_update
            log_focal = torch.log(focal)
            focal_min_update = (torch.log(current.new_tensor(finfo.tiny)) - log_focal).amax(dim=-1)
            focal_max_update = (log_safe_max - log_focal).amin(dim=-1)
            applied_focal_update = focal_update.clamp(
                min=focal_min_update,
                max=focal_max_update,
            )
            focal_in_domain = ((focal_update >= focal_min_update) & (focal_update <= focal_max_update)) | (
                focal_update == 0
            )
            focal_multiplication_safe = (
                (focal_update >= torch.log(current.new_tensor(finfo.tiny))) & (focal_update <= log_safe_max)
            ) | (focal_update == 0)
            x[inds, :2] = torch.where(
                (focal_in_domain & focal_multiplication_safe)[..., None],
                focal * torch.exp(focal_update[..., None]),
                torch.exp(log_focal + applied_focal_update[..., None]),
            )
            x[inds, 4] = new_one_plus_xi - 1.0
            return

        x[inds, :2] += dx[..., :1]
        # Use smaller learning rate for the distortion parameters
        x[inds, 4:] += dx[..., 1:] * self.distortion_update_scale
