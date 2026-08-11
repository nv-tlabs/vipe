# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vipe.slam.maths.retractor import IntrinsicsRetractor
from vipe.utils.cameras import CameraType


def test_intrinsics_retractor_uses_configured_distortion_update_scale():
    intrinsics = torch.tensor([[100.0, 100.0, 50.0, 25.0, 0.5]])
    update = torch.tensor([[2.0, 3.0]])

    IntrinsicsRetractor(CameraType.MEI, distortion_update_scale=0.001).oplus(
        intrinsics,
        torch.tensor([0]),
        update,
    )

    torch.testing.assert_close(intrinsics, torch.tensor([[102.0, 102.0, 50.0, 25.0, 0.503]]))


def test_intrinsics_retractor_keeps_legacy_default_scale():
    intrinsics = torch.tensor([[100.0, 100.0, 50.0, 25.0, 0.5]])
    update = torch.tensor([[2.0, 3.0]])

    IntrinsicsRetractor(CameraType.MEI).oplus(intrinsics, torch.tensor([0]), update)

    torch.testing.assert_close(intrinsics, torch.tensor([[102.0, 102.0, 50.0, 25.0, 0.53]]))


def test_mei_center_log_retractor_matches_definition_and_ignores_legacy_scale():
    intrinsics = torch.tensor(
        [
            [120.0, 100.0, 50.0, 25.0, 0.5],
            [240.0, 200.0, 50.0, 25.0, 1.0],
        ],
        dtype=torch.float64,
    )
    original = intrinsics.clone()
    update = torch.tensor(
        [
            [torch.log(torch.tensor(1.2)), torch.log(torch.tensor(1.5))],
            [torch.log(torch.tensor(0.8)), torch.log(torch.tensor(0.75))],
        ],
        dtype=torch.float64,
    )

    IntrinsicsRetractor(
        CameraType.MEI,
        distortion_update_scale=1.0e6,
        parameterization="mei_center_log",
    ).oplus(intrinsics, torch.tensor([0, 1]), update)

    original_s = 1.0 + original[:, 4:5]
    expected_s = original_s * torch.exp(update[:, 1:2])
    expected_center_focal = original[:, :2] / original_s * torch.exp(update[:, :1])
    torch.testing.assert_close(1.0 + intrinsics[:, 4:5], expected_s)
    torch.testing.assert_close(intrinsics[:, :2] / expected_s, expected_center_focal)
    torch.testing.assert_close(intrinsics[:, 0] / intrinsics[:, 1], original[:, 0] / original[:, 1])
    torch.testing.assert_close(intrinsics[:, 2:4], original[:, 2:4])


def test_mei_center_log_beta_update_preserves_center_focal_and_broadcasts():
    intrinsics = torch.tensor(
        [
            [120.0, 100.0, 50.0, 25.0, 0.5],
            [240.0, 200.0, 50.0, 25.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    original = intrinsics.clone()
    update = torch.tensor([[0.0, torch.log(torch.tensor(1.25))]], dtype=torch.float64)

    IntrinsicsRetractor(CameraType.MEI, parameterization="mei_center_log").oplus(
        intrinsics,
        torch.tensor([0]),
        update,
    )

    torch.testing.assert_close(
        intrinsics[:2, :2] / (1.0 + intrinsics[:2, 4:5]),
        original[:2, :2] / (1.0 + original[:2, 4:5]),
    )
    torch.testing.assert_close(
        intrinsics[:2, 0] / intrinsics[:2, 1],
        original[:2, 0] / original[:2, 1],
    )
    torch.testing.assert_close(1.0 + intrinsics[:2, 4], (1.0 + original[:2, 4]) * 1.25)
    torch.testing.assert_close(intrinsics[2], original[2])


def test_mei_center_log_retractor_enforces_numeric_domain_without_arbitrary_xi_cap():
    intrinsics = torch.tensor(
        [
            [100.0, 90.0, 50.0, 25.0, 0.5],
            [100.0, 90.0, 50.0, 25.0, 0.5],
        ]
    )
    update = torch.tensor([[-1.0e6, -1.0e6], [1.0e6, 1.0e6]])

    IntrinsicsRetractor(CameraType.MEI, parameterization="mei_center_log").oplus(
        intrinsics,
        torch.tensor([0, 1]),
        update,
    )

    torch.testing.assert_close(intrinsics[0, 4], torch.tensor(torch.finfo(intrinsics.dtype).eps - 1.0))
    assert intrinsics[1, 4] > 1.0e30
    assert torch.isfinite(intrinsics[:, 4]).all()
    assert torch.isfinite(intrinsics[:, :2]).all()
    assert (intrinsics[:, :2] > 0).all()


@pytest.mark.parametrize("update_value", [-1.0e6, 1.0e6])
def test_mei_center_log_retractor_stays_finite_from_near_numeric_limit(update_value):
    finfo = torch.finfo(torch.float32)
    intrinsics = torch.tensor(
        [[finfo.max / 4.0, finfo.max / 8.0, 50.0, 25.0, finfo.max / 8.0]],
        dtype=torch.float32,
    )

    IntrinsicsRetractor(CameraType.MEI, parameterization="mei_center_log").oplus(
        intrinsics,
        torch.tensor([0]),
        torch.full((1, 2), update_value, dtype=torch.float32),
    )

    assert torch.isfinite(intrinsics[:, [0, 1, 4]]).all()
    assert (intrinsics[:, :2] > 0).all()
    assert intrinsics[0, 4] > -1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bound", ["lower", "lower_plus_one", "upper"])
def test_mei_center_log_retractor_avoids_exp_intermediate_overflow_at_exact_bounds(dtype, bound):
    finfo = torch.finfo(dtype)
    if bound.startswith("lower"):
        focal = torch.tensor([finfo.max / 4.0, finfo.max / 8.0], dtype=dtype)
        focal_logs = torch.log(focal)
        update = (torch.log(torch.tensor(finfo.tiny, dtype=dtype)) - focal_logs).amax()
        if bound == "lower_plus_one":
            update = update + 1.0
    else:
        focal = torch.tensor([finfo.tiny * 4.0, finfo.tiny * 8.0], dtype=dtype)
        focal_logs = torch.log(focal)
        update = (torch.log(torch.tensor(finfo.max / 4.0, dtype=dtype)) - focal_logs).amin()
    intrinsics = torch.tensor([[1.0, 1.0, 50.0, 25.0, 0.5]], dtype=dtype)
    intrinsics[0, :2] = focal

    IntrinsicsRetractor(CameraType.MEI, parameterization="mei_center_log").oplus(
        intrinsics,
        torch.tensor([0]),
        torch.tensor([[update, 0.0]], dtype=dtype),
    )

    assert torch.isfinite(intrinsics[:, :2]).all()
    assert (intrinsics[:, :2] > 0).all()


def test_mei_center_log_rejects_non_mei_camera():
    with pytest.raises(ValueError, match="requires the MEI camera model"):
        IntrinsicsRetractor(CameraType.PINHOLE, parameterization="mei_center_log")


@pytest.mark.parametrize("xi", [-0.3, 0.75, 10.0])
def test_mei_center_log_retractor_zero_step_is_bitwise_identity(xi):
    intrinsics = torch.tensor([[800.0, 720.0, 320.0, 240.0, xi]], dtype=torch.float32)
    original = intrinsics.clone()

    IntrinsicsRetractor(CameraType.MEI, parameterization="mei_center_log").oplus(
        intrinsics,
        torch.tensor([0]),
        torch.zeros((1, 2), dtype=torch.float32),
    )

    assert torch.equal(intrinsics, original)
