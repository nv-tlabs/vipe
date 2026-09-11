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

from enum import Enum

import torch


class CameraType(Enum):
    PINHOLE = "pinhole"
    PANORAMA = "panorama"
    SIMPLE_DIVISIONAL = "simple_divisional"
    MEI = "mei"

    def build_camera_model(self, intrinsics: torch.Tensor):
        cls = self.camera_model_cls()
        return cls(intrinsics)

    def intrinsics_dim(self) -> int:
        return self.camera_model_cls().intrinsics_dim()

    def camera_model_cls(self) -> type["BaseCameraModel"]:
        if self == CameraType.PINHOLE:
            return PinholeCameraModel
        elif self == CameraType.MEI:
            return MeiCameraModel
        elif self == CameraType.PANORAMA:
            return PanoramaCameraModel
        else:
            raise ValueError(f"Un-implemented camera type: {self}")


class BaseCameraModel:
    """Represent a batch of camera models of the same type."""

    MIN_DEPTH: float = 0.1

    @classmethod
    def intrinsics_dim(cls) -> int:
        raise NotImplementedError

    def __init__(self, intrinsics: torch.Tensor):
        self.intrinsics = intrinsics
        assert self.intrinsics.shape[-1] == self.intrinsics_dim(), (
            f"Intrinsics should have shape (..., {self.intrinsics_dim()})"
        )

    def iproj_disp(
        self,
        disps: torch.Tensor,
        disps_u: torch.Tensor,
        disps_v: torch.Tensor,
        compute_jz: bool = False,
        compute_jf: bool = False,
    ):
        """
        Args:
            disps: (N, ...) tensor of disparities
            disps_u: (N, ...) tensor of u coordinates
            disps_v: (N, ...) tensor of v coordinates
            compute_jz: bool to compute jacobian of the disps
            compute_jf: bool to compute jacobian of the intrinsics

        Returns:
            pts: (N, ..., 4) tensor of homogeneous points
            Jz: (N, ..., 4) tensor of jacobian of the disps
            Jf: (N, ..., 4, 1+D) tensor of jacobian of the intrinsics (focal + distortion)
        """
        raise NotImplementedError

    def proj_points(
        self,
        ps: torch.Tensor,
        compute_jp: bool = False,
        compute_jf: bool = False,
        limit_min_depth: bool = True,
    ):
        """
        Args:
            ps: (N, ..., 4) tensor of homogeneous points
            compute_jp: bool to compute jacobian of the homogeneous points
            compute_jf: bool to compute jacobian of the intrinsics
            limit_min_depth: bool to limit the minimum depth to self.MIN_DEPTH

        Returns:
            coords: (N, ..., 2) tensor of coordinates
            Jp: (N, ..., 2, 4) tensor of jacobian of the homogeneous points
            Jf: (N, ..., 2, 1+D) tensor of jacobian of the focal + distortion
        """
        raise NotImplementedError

    def pinhole(self) -> "PinholeCameraModel":
        """
        Returns:
            PinholeCameraModel.
        """
        raise NotImplementedError

    def scaled(self, scale: float) -> "BaseCameraModel":
        """
        Args:
            scale: scale factor to apply to the camera model.
        """
        raise NotImplementedError

    @classmethod
    def J_scale(cls, scale: float, J: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PinholeCameraModel(BaseCameraModel):
    def __init__(self, intrinsics: torch.Tensor):
        super().__init__(intrinsics)

    @classmethod
    def intrinsics_dim(self) -> int:
        return 4

    def iproj_disp(
        self,
        disps: torch.Tensor,
        disps_u: torch.Tensor,
        disps_v: torch.Tensor,
        compute_jz: bool = False,
        compute_jf: bool = False,
    ):
        # Expand intrinsics.
        intrinsics = self.intrinsics.view((-1,) + (1,) * (disps.dim() - 1) + (4,))
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)

        i = torch.ones_like(disps)
        X = (disps_u - cx) / fx
        Y = (disps_v - cy) / fy
        pts = torch.stack([X, Y, i, disps], dim=-1)

        Jz = None
        if compute_jz:
            Jz = torch.zeros_like(pts)
            Jz[..., -1] = 1.0

        Jf = None
        if compute_jf:
            Jf = torch.zeros_like(pts)[..., None]
            Jf[..., 0, 0] = -X / fx
            Jf[..., 1, 0] = -Y / fy

        return pts, Jz, Jf

    def proj_points(
        self,
        ps: torch.Tensor,
        compute_jp: bool = False,
        compute_jf: bool = False,
        limit_min_depth: bool = True,
    ):
        extra_dim_shapes = ps.shape[1:-1]
        n_extra_dim = len(extra_dim_shapes)  # Dim of "..."

        fx, fy, cx, cy = self.intrinsics.view((-1,) + (1,) * n_extra_dim + (4,)).unbind(dim=-1)
        # Ignore the last component since it will be cancelled out
        X, Y, Z, _ = ps.unbind(dim=-1)

        if limit_min_depth:
            Z = torch.where(Z < self.MIN_DEPTH, torch.ones_like(Z), Z)
        d = Z.reciprocal()

        x = fx * (X * d) + cx
        y = fy * (Y * d) + cy
        coords = torch.stack([x, y], dim=-1)

        Jp = None
        if compute_jp:
            N = d.shape[0]
            o = torch.zeros_like(d)
            Jp = torch.stack(
                [
                    fx * d,
                    o,
                    -fx * X * d * d,
                    o,
                    o,
                    fy * d,
                    -fy * Y * d * d,
                    o,
                ],
                dim=-1,
            ).view(N, *extra_dim_shapes, 2, 4)

        Jf = None
        if compute_jf:
            Jf = torch.zeros_like(coords)[..., None]
            Jf[..., 0, 0] = X * d
            Jf[..., 1, 0] = Y * d

        return coords, Jp, Jf

    def pinhole(self):
        return self

    def scaled(self, scale: float) -> "PinholeCameraModel":
        return PinholeCameraModel(self.intrinsics * scale)

    @classmethod
    def J_scale(cls, scale: float, J: torch.Tensor) -> torch.Tensor:
        return J * scale


class MeiCameraModel(BaseCameraModel):
    def __init__(self, intrinsics: torch.Tensor):
        super().__init__(intrinsics)

    @classmethod
    def intrinsics_dim(self) -> int:
        return 5

    def iproj_disp(
        self,
        disps: torch.Tensor,
        disps_u: torch.Tensor,
        disps_v: torch.Tensor,
        compute_jz: bool = False,
        compute_jf: bool = False,
    ):
        # Expand intrinsics.
        intrinsics = self.intrinsics.view((-1,) + (1,) * (disps.dim() - 1) + (5,))
        fx, fy, cx, cy, k1 = intrinsics.unbind(dim=-1)

        u_bar = (disps_u - cx) / fx
        v_bar = (disps_v - cy) / fy
        r2 = u_bar**2 + v_bar**2
        q = torch.sqrt(1 + (1 - k1**2) * r2)
        factor = (k1 + q) / (1 + r2)

        X = u_bar * factor / (factor - k1)
        Y = v_bar * factor / (factor - k1)
        i = torch.ones_like(disps)

        pts = torch.stack([X, Y, i, disps], dim=-1)

        Jz = None
        if compute_jz:
            Jz = torch.zeros_like(pts)
            Jz[..., -1] = 1.0

        Jf = None
        if compute_jf:
            # (..., 4, 1 [focal] + 1 [distortion])
            Jf = pts.new_zeros(pts.shape + (1 + 1,))
            # Note that fx and fy are assumed to be equivalent for the derivation.
            f_num = (
                -(k1**3) * r2**2
                - k1**3 * r2
                - k1**2 * q * r2
                - k1 * q**2 * r2
                - k1 * q**2
                + k1 * r2**2
                + k1 * r2
                - q**3
            )
            f_denom = fx * q * (k1**2 * r2**2 - 2 * k1 * q * r2 + q**2)
            Jf[..., 0, 0] = u_bar * f_num / f_denom
            Jf[..., 1, 0] = v_bar * f_num / f_denom

            k_num = (k1 + q) * (k1 * r2 + q * (r2 + 1) - q) - (k1 * r2 - q) * (-k1 * (r2 + 1) + k1 + q)
            k_denom = q * (-k1 * (r2 + 1) + k1 + q) ** 2
            Jf[..., 0, 1] = u_bar * k_num / k_denom
            Jf[..., 1, 1] = v_bar * k_num / k_denom

        return pts, Jz, Jf

    def proj_points(
        self,
        ps: torch.Tensor,
        compute_jp: bool = False,
        compute_jf: bool = False,
        limit_min_depth: bool = True,
    ):
        extra_dim_shapes = ps.shape[1:-1]
        n_extra_dim = len(extra_dim_shapes)  # Dim of "..."

        fx, fy, cx, cy, k1 = self.intrinsics.view((-1,) + (1,) * n_extra_dim + (5,)).unbind(dim=-1)
        # Ignore the last component since it will be cancelled out
        X, Y, Z, _ = ps.unbind(dim=-1)

        if limit_min_depth:
            Z = torch.where(Z < self.MIN_DEPTH, torch.ones_like(Z), Z)

        r = torch.sqrt(X**2 + Y**2 + Z**2)
        rbase = Z + k1 * r
        d = rbase.reciprocal()

        x = fx * (X * d) + cx
        y = fy * (Y * d) + cy
        coords = torch.stack([x, y], dim=-1)

        Jp = None
        if compute_jp:
            N = d.shape[0]
            o = torch.zeros_like(d)
            r_denom = rbase**2 * r
            Jp = torch.stack(
                [
                    fx * (-k1 * X**2 + rbase * r) / r_denom,
                    -fx * k1 * X * Y / r_denom,
                    -fx * X * (k1 * Z + r) / r_denom,
                    o,
                    -fy * k1 * X * Y / r_denom,
                    fy * (-k1 * Y**2 + rbase * r) / r_denom,
                    -fy * Y * (k1 * Z + r) / r_denom,
                    o,
                ],
                dim=-1,
            ).view(N, *extra_dim_shapes, 2, 4)

        Jf = None
        if compute_jf:
            # (..., 2, 1 [focal] + 1 [distortion])
            Jf = coords.new_zeros(coords.shape + (1 + 1,))
            Jf[..., 0, 0] = X * d
            Jf[..., 1, 0] = Y * d
            Jf[..., 0, 1] = -fx * r * X * d**2
            Jf[..., 1, 1] = -fy * r * Y * d**2

        return coords, Jp, Jf

    def pinhole(self) -> PinholeCameraModel:
        # Make sure at least at the center point the scale slope is 1.
        k1 = self.intrinsics[..., -1:]
        pinhole_intrinsics = self.intrinsics[..., :-1].clone()
        pinhole_intrinsics[..., 0:2] *= (1 + k1).reciprocal()
        return PinholeCameraModel(pinhole_intrinsics)

    def scaled(self, scale: float) -> "MeiCameraModel":
        scaled_intrinsics = self.intrinsics.clone()
        scaled_intrinsics[..., :-1] *= scale
        return MeiCameraModel(scaled_intrinsics)

    @classmethod
    def J_scale(cls, scale: float, J: torch.Tensor) -> torch.Tensor:
        # The shared focal parameter scales with image resolution, while MEI xi is dimensionless.
        scale_factor = J.new_ones(J.shape[-1])
        scale_factor[0] = scale
        return J * scale_factor


class PanoramaCameraModel(BaseCameraModel):
    def __init__(self, intrinsics: torch.Tensor):
        super().__init__(intrinsics)
        assert torch.all(intrinsics == 0)

    @classmethod
    def intrinsics_dim(cls) -> int:
        return 4

    def iproj_disp(
        self,
        disps: torch.Tensor,
        disps_u: torch.Tensor,
        disps_v: torch.Tensor,
        compute_jz: bool = False,
        compute_jf: bool = False,
    ):
        assert not compute_jf and not compute_jz

        # Note that due to the special nature of this camera model, we require disps_u and v to be
        # within range of [0, 1]
        theta = (disps_u - 0.5) * 2 * torch.pi  # [-pi, pi]
        phi = disps_v * torch.pi  # [0, pi]

        sin_phi = torch.sin(phi)
        x = sin_phi * torch.sin(theta)
        y = -torch.cos(phi)
        z = sin_phi * torch.cos(theta)

        xyz = torch.stack((x, y, z, disps), dim=-1)
        return xyz, None, None

    def proj_points(
        self,
        ps: torch.Tensor,
        compute_jp: bool = False,
        compute_jf: bool = False,
        limit_min_depth: bool = True,
    ):
        raise NotImplementedError

    def pinhole(self):
        # 512x256 camera with 90 horizontal FOV.
        return PinholeCameraModel(torch.tensor([256.0, 256.0, 256.0, 128.0]).to(self.intrinsics.device))

    def scaled(self, scale: float) -> "PanoramaCameraModel":
        return self

    @classmethod
    def J_scale(cls, scale: float, J: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
