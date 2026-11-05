# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU SE(3) graph fitting with the historical MetricTrail numerical settings.

Retrieval is external. This module contains neither SALAD nor a retrieval
runtime. PyPose supplies Lie operations and derivatives; SciPy solves the
sparse normal equations. Dependencies on PyPose and PyTorch are lazy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GraphSettings:
    """Residual multipliers, not calibrated information or measured inliers."""

    loop_scale: float = 0.01
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    direct_inliers: int = 20
    rigid_min_inliers: int = 24
    max_iterations: int = 30
    lambda_init: float = 1.0e-6
    loop_score_threshold: float = 0.025
    num_threads: int = 2

    @property
    def effective_loop_scale(self) -> float:
        # The two legacy constants normalize weak identity factors. They are
        # not counts produced by geometric verification of the supplied pairs.
        return self.loop_scale * math.sqrt(self.direct_inliers / self.rigid_min_inliers)


def graph_settings(dataset: str) -> GraphSettings:
    if dataset == "vbr":
        return GraphSettings(translation_scale=2.0, rotation_scale=1.2)
    if dataset == "kitti":
        return GraphSettings()
    raise ValueError(f"unknown loop-closure preset: {dataset}")


def _dependencies():
    try:
        import pypose
        import torch
    except ImportError as error:
        raise ImportError(
            "Nonempty loop closure requires PyPose in the ViPE environment: python -m pip install 'pypose==0.6.8'"
        ) from error
    if pypose.__version__ != "0.6.8":
        raise RuntimeError("MetricTrail's reference LC settings require pypose==0.6.8")
    return pypose, torch


def _pose_vectors(matrices: np.ndarray) -> np.ndarray:
    """Convert float64 input matrices to translation/xyzw float32 Lie data."""
    vectors = np.empty((len(matrices), 7), dtype=np.float32)
    vectors[:, :3] = matrices[:, :3, 3]
    vectors[:, 3:] = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return vectors


def _normal_step(
    jacobians: tuple[Any, Any],
    endpoints: np.ndarray,
    error: Any,
    damping: float,
    count: int,
):
    """Assemble edge blocks in float64, eliminate the first pose, solve once."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    edges = len(endpoints)
    # (edge, residual coordinate, variable coordinate, endpoint). The paired
    # endpoint order also makes sparse accumulation deterministic.
    blocks = np.stack([block.detach().numpy().astype(np.float64) for block in jacobians], axis=-1)
    row_ids = np.broadcast_to(np.arange(6 * edges).reshape(edges, 6, 1, 1), blocks.shape)
    col_ids = np.broadcast_to(
        6 * endpoints[:, None, None, :] + np.arange(6)[None, None, :, None],
        blocks.shape,
    )
    matrix = coo_matrix(
        (blocks.ravel(), (row_ids.ravel(), col_ids.ravel())),
        shape=(6 * edges, 6 * count),
    ).tocsc()
    rhs = -(matrix.T @ error.detach().numpy().astype(np.float64).ravel())
    normal = matrix.T @ matrix
    normal.setdiag(normal.diagonal() * (1.0 + damping))
    free = np.arange(6, 6 * count, dtype=np.int64)
    step = np.zeros(6 * count, dtype=np.float64)
    step[free] = spsolve(normal[free][:, free].tocsc(), rhs[free])
    if not np.isfinite(step).all():
        raise RuntimeError("PyPose graph linear solve produced a non-finite step")
    return torch.from_numpy(step.astype(np.float32).reshape(count, 6))


def optimize_identity_graph(
    baseline: np.ndarray,
    loop_pairs: Sequence[tuple[int, int]],
    settings: GraphSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit adjacent odometry and weak identity loops; never modify baseline.

    The first inverse-pose tangent is fixed. Nonempty-loop output includes the
    historical float32 Lie conversion, including at that anchor. No-loop input
    is copied exactly without importing PyPose. A finite accepted iterate can
    be returned at the iteration budget; this is not a convergence guarantee.
    """
    poses = np.asarray(baseline, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 1 or not np.isfinite(poses).all():
        raise ValueError("pose graph requires finite camera transforms with shape [N,4,4]")
    if len(loop_pairs) == 0:
        return poses.copy(), {
            "status": "unchanged_no_loops",
            "evaluations": 0,
            "iterations": 0,
            "initial_cost": 0.0,
            "final_cost": 0.0,
            "settings": asdict(settings),
        }
    positive = (
        settings.loop_scale,
        settings.translation_scale,
        settings.rotation_scale,
        settings.direct_inliers,
        settings.rigid_min_inliers,
        settings.lambda_init,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive) or settings.max_iterations < 1:
        raise ValueError("pose graph settings must be positive and finite")
    loops = np.asarray(loop_pairs)
    count = len(poses)
    if (
        loops.ndim != 2
        or loops.shape[1] != 2
        or loops.dtype.kind not in "iu"
        or np.any(loops < 0)
        or np.any(loops >= count)
        or np.any(loops[:, 0] == loops[:, 1])
    ):
        raise ValueError("loop positions must identify distinct poses in the graph")
    if type(settings.num_threads) is not int or settings.num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    pp, torch = _dependencies()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(settings.num_threads)
        return _fit_graph(poses, loops.astype(np.int64), settings, pp, torch)
    finally:
        torch.set_num_threads(previous_threads)


def _fit_graph(poses, loops, settings, pp, torch):
    count = len(poses)
    adjacent = np.column_stack((np.arange(count - 1), np.arange(1, count)))
    endpoints = np.concatenate((adjacent, loops.astype(np.int64)))
    # Measurements are T_j^-1 T_i in the input float64 representation. Do not
    # replace the inverse by a rotation transpose or first cast poses to float32.
    measurements = np.tile(np.eye(4), (len(endpoints), 1, 1))
    for index in range(count - 1):
        measurements[index] = np.linalg.inv(poses[index + 1]) @ poses[index]
    constants = pp.SE3(torch.from_numpy(_pose_vectors(measurements)))
    state = pp.SE3(torch.from_numpy(_pose_vectors(poses))).Inv().Log().detach()
    src = torch.from_numpy(endpoints[:, 0].copy())
    dst = torch.from_numpy(endpoints[:, 1].copy())
    scales = torch.ones((len(endpoints), 1), dtype=torch.float32)
    scales[count - 1 :] = settings.effective_loop_scale

    def error(measurement, left, right):
        value = (measurement @ pp.Exp(left) @ pp.Exp(right).Inv()).Log().tensor().clone()
        value[:, :3] *= settings.translation_scale
        value[:, 3:] *= settings.rotation_scale
        return value

    def summed_error(measurement, left, right):
        return error(measurement, left, right).sum(dim=0)

    def cost(parameters):
        return float((error(constants, parameters[src], parameters[dst]) * scales).square().mean().item())

    history = []
    prior_costs = []
    damping = settings.lambda_init
    initial_cost = cost(state)
    if not math.isfinite(initial_cost):
        raise RuntimeError("PyPose graph has a non-finite initial cost")
    termination = "iteration_limit"
    for iteration in range(settings.max_iterations):
        unscaled = error(constants, state[src], state[dst])
        derivative = torch.autograd.functional.jacobian(
            summed_error,
            (constants, state[src], state[dst]),
            vectorize=True,
        )
        blocks = tuple(part.permute(1, 0, 2).contiguous() * scales.unsqueeze(-1) for part in derivative[1:])
        weighted = unscaled * scales
        previous = float(weighted.square().mean().item())
        prior_costs.append(previous)
        trial = state + _normal_step(blocks, endpoints, weighted, damping, count)
        proposed = cost(trial)
        if not math.isfinite(proposed):
            raise RuntimeError("PyPose graph produced a non-finite trial cost")
        take_step = proposed <= previous + 1.0e-12
        history.append(
            {
                "iteration": iteration + 1,
                "cost": previous,
                "trial_cost": proposed,
                "accepted": take_step,
                "lambda": damping,
            }
        )
        if take_step:
            state = trial
            damping = max(1.0e-12, damping * 0.5)
        else:
            damping = min(1.0e8, damping * 2.0)
        if iteration >= 4 and previous < 1.0e-6 and prior_costs[-5] / max(previous, 1.0e-12) < 1.2:
            termination = "small_cost_plateau"
            break
    final_cost = cost(state)
    corrected = pp.Exp(state).Inv().matrix().detach().numpy().astype(np.float64)
    if not np.isfinite(corrected).all():
        raise RuntimeError("PyPose graph produced non-finite corrected poses")
    return corrected, {
        "status": "completed",
        "termination": termination,
        "iterations": len(history),
        "evaluations": 2 * len(history) + 2,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "cost_definition": "mean_squared_weighted_SE3_log_residual",
        "effective_loop_scale": settings.effective_loop_scale,
        "settings": asdict(settings),
        "iteration_history": history,
        "runtime": {
            "pypose": str(pp.__version__),
            "torch": str(torch.__version__),
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
        },
        "pose_jacobian_dtype": "float32",
        "linear_solve_dtype": "float64",
        "anchor": "first_inverse_pose_tangent_fixed_with_float32_roundtrip",
    }
