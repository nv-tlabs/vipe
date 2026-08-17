# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import torch

from vipe.utils.cameras import CameraType

IntrinsicsParameterization = Literal["additive", "mei_center_log"]


def validate_intrinsics_parameterization(
    parameterization: IntrinsicsParameterization,
    camera_type: CameraType,
) -> None:
    if parameterization not in ("additive", "mei_center_log"):
        raise ValueError(f"Unsupported intrinsics parameterization: {parameterization}")
    if parameterization == "mei_center_log" and camera_type != CameraType.MEI:
        raise ValueError(f"{parameterization} intrinsics parameterization requires the MEI camera model")


def raw_intrinsics_jacobian_to_local(
    jacobian: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_type: CameraType,
    parameterization: IntrinsicsParameterization,
    *,
    source_inverse_projection: bool,
) -> torch.Tensor:
    """Convert raw MEI camera-coordinate derivatives to the axis-log tangent.

    This conversion must happen before a source inverse-projection Jacobian is
    mixed by rotation/projection.  Rows 0 and 1 are respectively the raw X/Y
    (source) or x/y (target) coordinate derivatives, so their focal columns are
    scaled by fx and fy independently.  The resulting columns are already local
    ``(eta, beta)`` derivatives and must not subsequently pass through J_scale.

    The legacy MEI inverse-projection formula normalizes both raw focal rows by
    fx because it was derived for fx == fy.  In axis-log mode only, compensate
    its Y row to a physical-fy derivative before applying the fy chain factor.
    """
    validate_intrinsics_parameterization(parameterization, camera_type)
    if parameterization != "mei_center_log":
        return jacobian

    if jacobian.shape[-1] != 2 or jacobian.shape[-2] < 2:
        raise ValueError(f"Raw MEI intrinsics Jacobian must end in (coordinate>=2, 2), got {jacobian.shape}")
    if intrinsics.ndim != 2 or intrinsics.shape[-1] != 5:
        raise ValueError(f"MEI intrinsics must have shape (n, 5), got {intrinsics.shape}")
    if jacobian.shape[0] != intrinsics.shape[0]:
        raise ValueError(
            "Raw Jacobian and intrinsics must have matching leading dimensions, "
            f"got {jacobian.shape[0]} and {intrinsics.shape[0]}"
        )
    jacobian_f = jacobian[..., 0]
    jacobian_xi = jacobian[..., 1]
    scalar_shape = (intrinsics.shape[0],) + (1,) * (jacobian_f.ndim - 2)
    fx = intrinsics[:, 0].reshape(scalar_shape)
    fy = intrinsics[:, 1].reshape(scalar_shape)
    one_plus_xi = intrinsics[:, 4].add(1.0).reshape((intrinsics.shape[0],) + (1,) * (jacobian_xi.ndim - 1))

    jacobian_eta = jacobian_f * fx[..., None]
    jacobian_f_y = jacobian_f[..., 1]
    if source_inverse_projection:
        # Restore the raw Y derivative's fy denominator before applying the
        # axis-specific logarithmic chain rule. This algebraically cancels to
        # the legacy row times fx, but keeps the physical-fy chain explicit.
        jacobian_f_y = jacobian_f_y * (fx / fy)
    jacobian_eta[..., 1] = jacobian_f_y * fy
    jacobian_beta = jacobian_eta + one_plus_xi * jacobian_xi
    return torch.stack([jacobian_eta, jacobian_beta], dim=-1)
