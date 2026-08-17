# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from vipe.ext.lietorch import SE3
from vipe.slam.ba.terms import DenseDepthFlowTerm, TracksFlowTerm
from vipe.slam.maths import geom
from vipe.slam.maths.retractor import IntrinsicsRetractor
from vipe.utils.cameras import CameraType, MeiCameraModel


def test_mei_center_log_geom_matches_autograd_with_unequal_focal_feature_scale_and_source_rotation():
    scale = 0.125
    intrinsics = torch.tensor(
        [
            [800.0, 720.0, 320.0, 240.0, 0.5],
            [1000.0, 850.0, 320.0, 240.0, 1.1],
        ],
        dtype=torch.float64,
    )
    angle = 0.3
    poses = SE3(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.08, -0.03, 0.02, 0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)],
            ],
            dtype=torch.float64,
        )
    )
    rig = SE3.Identity(2, dtype=torch.float64)
    disps = torch.tensor([[0.8, 1.0, 0.65]], dtype=torch.float64)
    tracks_uv = torch.tensor([[[22.0, 18.0], [45.0, 28.0], [60.0, 42.0]]], dtype=torch.float64)
    pose_i_inds = torch.tensor([0])
    pose_j_inds = torch.tensor([1])
    rig_i_inds = torch.tensor([0])
    rig_j_inds = torch.tensor([1])

    def project(local_delta: torch.Tensor) -> torch.Tensor:
        updated = intrinsics.clone()
        IntrinsicsRetractor(CameraType.MEI, parameterization="mei_center_log").oplus(
            updated,
            torch.tensor([0, 1]),
            local_delta,
        )
        scaled_intrinsics = MeiCameraModel(updated).scaled(scale).intrinsics
        return geom.iproj_i_proj_j_disp(
            poses,
            disps,
            tracks_uv,
            scaled_intrinsics,
            CameraType.MEI,
            rig,
            pose_i_inds,
            pose_j_inds,
            rig_i_inds,
            rig_j_inds,
            None,
            jacobian_p_d=False,
            jacobian_f=False,
            jacobian_r=False,
            intrinsics_parameterization="mei_center_log",
        )[0]

    scaled_intrinsics = MeiCameraModel(intrinsics).scaled(scale).intrinsics
    _, _, _, (source_jacobian, target_jacobian), _ = geom.iproj_i_proj_j_disp(
        poses,
        disps,
        tracks_uv,
        scaled_intrinsics,
        CameraType.MEI,
        rig,
        pose_i_inds,
        pose_j_inds,
        rig_i_inds,
        rig_j_inds,
        None,
        jacobian_p_d=False,
        jacobian_f=True,
        jacobian_r=False,
        intrinsics_parameterization="mei_center_log",
    )
    assert source_jacobian is not None and target_jacobian is not None

    zero_delta = torch.zeros((2, 2), dtype=torch.float64)
    autodiff = torch.autograd.functional.jacobian(project, zero_delta)
    torch.testing.assert_close(source_jacobian, autodiff[..., 0, :], rtol=1.0e-10, atol=1.0e-10)
    torch.testing.assert_close(target_jacobian, autodiff[..., 1, :], rtol=1.0e-10, atol=1.0e-10)

    epsilon = 1.0e-6
    for camera_idx in range(2):
        for tangent_idx in range(2):
            delta = zero_delta.clone()
            delta[camera_idx, tangent_idx] = epsilon
            finite_difference = (project(delta) - project(-delta)) / (2.0 * epsilon)
            torch.testing.assert_close(
                finite_difference,
                autodiff[..., camera_idx, tangent_idx],
                rtol=1.0e-7,
                atol=1.0e-8,
            )


@pytest.mark.parametrize("parameterization", ["additive", "mei_center_log"])
def test_dense_depth_flow_term_handles_source_and_target_local_parameterization(monkeypatch, parameterization):
    n_terms = 2
    source_jacobian = torch.tensor(
        [
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            [[[[5.0, 6.0], [7.0, 8.0]]]],
        ]
    )
    target_jacobian = source_jacobian + 10.0

    received_parameterizations = []

    def fake_geometry(*args, **kwargs):
        del args
        received_parameterizations.append(kwargs["intrinsics_parameterization"])
        coords = torch.zeros((n_terms, 1, 1, 2))
        valid = torch.ones((n_terms, 1, 1, 1))
        return coords, valid, (None, None, None), (source_jacobian, target_jacobian), (None, None)

    monkeypatch.setattr("vipe.slam.ba.terms.geom.iproj_i_proj_j_disp", fake_geometry)
    rig_i_inds = torch.tensor([1, 0])
    rig_j_inds = torch.tensor([0, 1])
    intrinsics = torch.tensor([[800.0, 800.0, 4.0, 4.0, 0.5], [1200.0, 1200.0, 4.0, 4.0, 1.0]])
    term = DenseDepthFlowTerm(
        pose_i_inds=torch.tensor([0, 0]),
        pose_j_inds=torch.tensor([1, 1]),
        rig_i_inds=rig_i_inds,
        rig_j_inds=rig_j_inds,
        dense_disp_i_inds=torch.tensor([0, 1]),
        target=torch.zeros((n_terms, 1, 1, 2)),
        weight=torch.ones((n_terms, 1, 1, 2)),
        intrinsics=None,
        intrinsics_factor=8.0,
        rig=None,
        image_size=(1, 1),
        camera_type=CameraType.MEI,
        intrinsics_parameterization=parameterization,
    )

    result = term.forward(
        {
            "pose": SE3.Identity(2),
            "dense_disp": torch.ones((2, 1)),
            "intrinsics": intrinsics,
            "rig": SE3.Identity(2),
        },
        active_group_names={"intrinsics"},
    )

    expected = torch.cat([source_jacobian, target_jacobian]).reshape(4, 2, 2)
    if parameterization == "additive":
        expected = MeiCameraModel.J_scale(0.125, expected)
    torch.testing.assert_close(result.J["intrinsics"].data, expected)
    torch.testing.assert_close(result.J["intrinsics"].j_inds, torch.cat([rig_i_inds, rig_j_inds]))
    assert received_parameterizations == [parameterization]


@pytest.mark.parametrize("parameterization", ["additive", "mei_center_log"])
def test_tracks_flow_term_handles_source_and_target_local_parameterization(monkeypatch, parameterization):
    source_jacobian = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target_jacobian = source_jacobian + 5.0

    received_parameterizations = []

    def fake_geometry(*args, **kwargs):
        del args
        received_parameterizations.append(kwargs["intrinsics_parameterization"])
        coords = torch.zeros((1, 1, 2))
        valid = torch.ones((1, 1, 1))
        return coords, valid, (None, None, None), (source_jacobian, target_jacobian), (None, None)

    monkeypatch.setattr("vipe.slam.ba.terms.geom.iproj_i_proj_j_disp", fake_geometry)
    intrinsics = torch.tensor([[800.0, 800.0, 4.0, 4.0, 0.5], [1200.0, 1200.0, 4.0, 4.0, 1.0]])
    term = TracksFlowTerm(
        pose_i_inds=torch.tensor([0]),
        pose_j_inds=torch.tensor([1]),
        rig_i_inds=torch.tensor([1]),
        rig_j_inds=torch.tensor([0]),
        tracks_i_inds=torch.tensor([0]),
        target=torch.zeros((1, 1, 2)),
        weight=torch.ones((1, 1, 2)),
        tracks_uv=torch.zeros((1, 1, 2)),
        intrinsics=None,
        rig=SE3.Identity(2),
        camera_type=CameraType.MEI,
        intrinsics_parameterization=parameterization,
    )

    result = term.forward(
        {
            "pose": SE3.Identity(2),
            "tracks_disp": torch.ones((1, 1)),
            "intrinsics": intrinsics,
        },
        active_group_names={"intrinsics"},
    )

    expected = torch.cat([source_jacobian, target_jacobian]).reshape(2, 2, 2)
    torch.testing.assert_close(result.J["intrinsics"].data, expected)
    torch.testing.assert_close(result.J["intrinsics"].j_inds, torch.tensor([1, 0]))
    assert received_parameterizations == [parameterization]
