# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from vipe.utils.cameras import MeiCameraModel


def test_mei_j_scale_only_scales_focal_column():
    jacobian = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])

    scaled = MeiCameraModel.J_scale(0.125, jacobian)

    torch.testing.assert_close(scaled, torch.tensor([[[0.25, 3.0], [0.625, 7.0]]]))


def test_mei_j_scale_matches_autograd_through_scaled_camera():
    scale = 0.125
    base_intrinsics = torch.tensor([[800.0, 800.0, 320.0, 240.0, 0.7]], dtype=torch.float64)
    point = torch.tensor([[[1.2, -0.7, 2.5, 1.0]]], dtype=torch.float64)

    def project(delta: torch.Tensor) -> torch.Tensor:
        intrinsics = base_intrinsics.clone()
        intrinsics[:, :2] += delta[0]
        intrinsics[:, 4] += delta[1]
        return MeiCameraModel(intrinsics).scaled(scale).proj_points(point)[0]

    autodiff_jacobian = torch.autograd.functional.jacobian(project, torch.zeros(2, dtype=torch.float64))
    scaled_camera = MeiCameraModel(base_intrinsics).scaled(scale)
    _, _, analytic_jacobian = scaled_camera.proj_points(point, compute_jf=True)
    assert analytic_jacobian is not None
    analytic_jacobian = MeiCameraModel.J_scale(scale, analytic_jacobian)

    torch.testing.assert_close(analytic_jacobian, autodiff_jacobian, rtol=1e-10, atol=1e-10)
