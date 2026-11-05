# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical contracts for the optional CPU PyPose graph, using synthetic data."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vipe.metrictrail.optional.loop_optimizer import (
    GraphSettings,
    graph_settings,
    optimize_identity_graph,
)


@pytest.fixture
def pypose_cpu():
    torch = pytest.importorskip("torch")
    pypose = pytest.importorskip("pypose")
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield torch, pypose
    finally:
        torch.set_num_threads(before)


def drifting_loop(count=8):
    progress = np.linspace(0.0, 1.0, count)
    poses = np.tile(np.eye(4), (count, 1, 1))
    angles = 2.0 * np.pi * progress + 0.08 * progress
    poses[:, :3, :3] = Rotation.from_rotvec(angles[:, None] * [0, 0, 1]).as_matrix()
    poses[:, 0, 3] = np.sin(2.0 * np.pi * progress) + 0.25 * progress
    poses[:, 1, 3] = 1.0 - np.cos(2.0 * np.pi * progress)
    return poses


def test_original_graph_settings_and_inlier_normalization():
    expected = {
        "loop_scale": 0.01,
        "translation_scale": 1.0,
        "rotation_scale": 1.0,
        "max_iterations": 30,
        "lambda_init": 1.0e-6,
        "direct_inliers": 20,
        "rigid_min_inliers": 24,
        "loop_score_threshold": 0.025,
        "num_threads": 2,
    }
    assert asdict(GraphSettings()) == expected
    assert asdict(graph_settings("kitti")) == expected
    assert asdict(graph_settings("vbr")) == {
        **expected,
        "translation_scale": 2.0,
        "rotation_scale": 1.2,
    }


def test_import_and_no_loop_path_do_not_load_torch_or_pypose():
    # A fresh process also catches accidental module-level imports hidden by
    # the test runner's dependency cache.
    script = """
import importlib.abc
import sys
import numpy as np
class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'pypose', 'faiss', 'cv2'}:
            raise AssertionError('unexpected optional dependency: ' + name)
sys.meta_path.insert(0, Blocked())
from vipe.metrictrail.optional.loop_optimizer import GraphSettings, optimize_identity_graph
poses = np.tile(np.eye(4), (3, 1, 1))
poses[:, 0, 3] = [0.1234567890123, 1.234567890123, 2.34567890123]
before = poses.copy()
output, report = optimize_identity_graph(poses, [], GraphSettings())
np.testing.assert_array_equal(output, before)
np.testing.assert_array_equal(poses, before)
assert not np.shares_memory(output, poses)
assert report['status'] == 'unchanged_no_loops'
assert report['evaluations'] == 0
assert 'torch' not in sys.modules and 'pypose' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_initial_objective_uses_original_inlier_weight_and_mean(pypose_cpu):
    # Identity rotations make this hand-computable: two zero odometry
    # residuals, one loop with translation -2, each with six coordinates.
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [0.0, 1.0, 2.0]
    settings = replace(GraphSettings(), max_iterations=1)
    _, report = optimize_identity_graph(poses, [(2, 0)], settings)
    expected = (2.0 * 0.01) ** 2 * (20.0 / 24.0) / (3.0 * 6.0)
    assert report["initial_cost"] == pytest.approx(expected, rel=1.0e-6, abs=1.0e-12)
    assert report["iterations"] == 1
    assert report["status"] == "completed"
    assert report["termination"] == "iteration_limit"
    assert report["evaluations"] >= 2
    assert np.isfinite(report["final_cost"])
    assert report["final_cost"] <= report["initial_cost"] + 1.0e-12


def test_nonidentity_anchor_uses_float32_se3_round_trip(pypose_cpu):
    torch, pypose = pypose_cpu
    poses = drifting_loop()
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [0.17, -0.31, 0.23]).as_matrix()
    world[:3, 3] = [0.1234567890123, -0.9876543210123, 2.34567890123]
    poses = world[None] @ poses
    before = poses.copy()
    corrected, report = optimize_identity_graph(poses, [(len(poses) - 1, 0)], GraphSettings())

    # The gauge stays fixed in the solver's float32 inverse-pose logarithm.
    # Its matrix need not be bit-identical to the float64 input anchor.
    coordinates = np.concatenate((poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()), axis=1).astype(
        np.float32
    )
    internal = pypose.SE3(torch.from_numpy(coordinates))
    anchor = pypose.Exp(internal.Inv().Log()).Inv().matrix()[0].numpy().astype(np.float64)
    np.testing.assert_allclose(corrected[0], anchor, rtol=0, atol=5.0e-7)
    np.testing.assert_allclose(corrected[0], poses[0], rtol=0, atol=1.0e-6)
    np.testing.assert_array_equal(poses, before)
    assert corrected.dtype == np.float64
    np.testing.assert_array_equal(corrected, corrected.astype(np.float32).astype(np.float64))
    assert corrected.shape == poses.shape and np.isfinite(corrected).all()
    np.testing.assert_array_equal(corrected[:, 3], np.tile([0, 0, 0, 1], (len(poses), 1)))
    rotations = corrected[:, :3, :3]
    np.testing.assert_allclose(
        rotations @ rotations.transpose(0, 2, 1),
        np.tile(np.eye(3), (len(poses), 1, 1)),
        rtol=0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, rtol=0, atol=2.0e-6)
    assert report["final_cost"] < report["initial_cost"]


def test_sparse_float64_system_eliminates_anchor_variables(monkeypatch, pypose_cpu):
    import scipy.sparse
    import scipy.sparse.linalg

    original = scipy.sparse.linalg.spsolve
    captured = []

    def observe(matrix, rhs, *args, **kwargs):
        assert scipy.sparse.issparse(matrix)
        assert matrix.shape == (18, 18)  # Four SE3 poses, first one fixed.
        assert matrix.dtype == np.float64
        assert np.asarray(rhs).shape == (18,)
        assert np.asarray(rhs).dtype == np.float64
        assert matrix.nnz < 18 * 18
        assert np.isfinite(matrix.data).all()
        assert np.isfinite(rhs).all()
        captured.append(matrix.shape)
        return original(matrix, rhs, *args, **kwargs)

    monkeypatch.setattr(scipy.sparse.linalg, "spsolve", observe)
    settings = replace(GraphSettings(), max_iterations=1)
    corrected, report = optimize_identity_graph(drifting_loop(4), [(3, 0)], settings)
    assert captured == [(18, 18)]
    assert np.isfinite(corrected).all()
    assert report["iterations"] == 1


@pytest.mark.parametrize("preset", ["kitti", "vbr"])
def test_both_presets_reduce_drifting_loop_objective(pypose_cpu, preset):
    poses = drifting_loop()
    settings = graph_settings(preset)
    corrected, report = optimize_identity_graph(poses, [(len(poses) - 1, 0)], settings)
    assert 1 <= report["iterations"] <= settings.max_iterations
    assert report["settings"] == asdict(settings)
    assert report["status"] == "completed"
    assert report["termination"] in {"iteration_limit", "small_cost_plateau"}
    assert np.isfinite(report["initial_cost"]) and np.isfinite(report["final_cost"])
    assert report["final_cost"] < report["initial_cost"]
    assert np.linalg.norm(corrected[-1, :3, 3] - corrected[0, :3, 3]) < 0.25
    relative_rotation = corrected[0, :3, :3].T @ corrected[-1, :3, :3]
    assert np.linalg.norm(Rotation.from_matrix(relative_rotation).as_rotvec()) < 0.08


def test_solver_restores_callers_thread_setting(pypose_cpu):
    torch, _ = pypose_cpu
    torch.set_num_threads(1)
    _, report = optimize_identity_graph(drifting_loop(), [(7, 0)], GraphSettings())
    assert report["runtime"]["torch_num_threads"] == 2
    assert torch.get_num_threads() == 1


def test_solver_restores_threads_after_failure(monkeypatch, pypose_cpu):
    from vipe.metrictrail.optional import loop_optimizer

    torch, _ = pypose_cpu
    torch.set_num_threads(1)

    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 2
        raise RuntimeError("test failure")

    monkeypatch.setattr(loop_optimizer, "_fit_graph", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        optimize_identity_graph(drifting_loop(), [(7, 0)], GraphSettings())
    assert torch.get_num_threads() == 1
