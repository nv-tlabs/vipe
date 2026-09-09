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

"""Sim3 pose-graph optimization over all keyframes of a long sequence.

Nodes are keyframes (global index = creation order), edges are relative poses
recorded at retirement time (plus, in phase 2, loop closures). Poses-only —
no dense per-pixel state — so this scales to hundreds of thousands of frames
in megabytes. Solved with gtsam ``Similarity3`` between-factors: each window's
metric scale is anchored independently by the keyframe-depth prior, and the
7th (scale) DoF absorbs any residual inter-window scale drift.

gtsam is an optional dependency: it is only imported when a long-sequence
pipeline actually optimizes, never by the standard pipelines.
"""

import logging

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from vipe.ext.lietorch import SE3

logger = logging.getLogger(__name__)

# Isotropic noise sigmas in the Similarity3 tangent space [rot(3), trans(3), scale(1)].
# Edges recorded within one window epoch are mutually consistent, so the exact
# magnitudes matter little in phase 1; they become load-bearing relative to the
# loop-closure sigmas in phase 2.
_EDGE_SIGMAS = np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.01])
# The first keyframe pins the gauge (world frame and global scale).
_PRIOR_SIGMAS = np.full(7, 1.0e-6)


def _se3_data_to_rotation_translation(pose_data: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Split (7,) lietorch SE3 data [tx ty tz qx qy qz qw] into a 3x3 R and a 3-vector t."""
    data = pose_data.detach().cpu().double().numpy()
    rotation = Rotation.from_quat(data[3:7]).as_matrix()
    return rotation, data[:3]


def _rotation_translation_to_se3_data(rotation: np.ndarray, translation: np.ndarray) -> torch.Tensor:
    quat = Rotation.from_matrix(rotation).as_quat()  # xyzw, matching lietorch
    return torch.tensor([*translation, *quat], dtype=torch.float32)


class Sim3PoseGraph:
    """Thin, testable wrapper building and solving the gtsam Sim3 pose graph.

    Poses are camera-to-world SE3 (as (N, 7) lietorch data); the result is the
    corrected absolute Sim3 pose per node, split into ``(scale, rigid SE3)``
    for consumption by ``TrajectoryLedger.apply_corrections``.
    """

    def __init__(self) -> None:
        self._initial_c2w: torch.Tensor | None = None
        self._edges: list[tuple[int, int, torch.Tensor, float]] = []

    def set_nodes(self, initial_c2w: torch.Tensor) -> None:
        """Set all node initial values at once ((N, 7) c2w SE3 data, node id = row)."""
        self._initial_c2w = initial_c2w.detach().cpu()

    def add_edges(
        self,
        src_inds: torch.Tensor,
        dst_inds: torch.Tensor,
        relative_c2w: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """Add between-edges: ``relative_c2w[k]`` measures ``g_src⁻¹ ∘ g_dst`` (gtsam convention)."""
        for k in range(src_inds.shape[0]):
            self._edges.append(
                (int(src_inds[k].item()), int(dst_inds[k].item()), relative_c2w[k], float(weights[k].item()))
            )

    def optimize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the pose graph; returns per-node ``(scale (N,), rigid c2w SE3 data (N, 7))``.

        With no edges (or a single node) this degenerates gracefully to the
        initial values with unit scales.
        """
        assert self._initial_c2w is not None, "set_nodes must be called before optimize"
        num_nodes = self._initial_c2w.shape[0]
        if num_nodes <= 1 or not self._edges:
            return torch.ones(num_nodes), self._initial_c2w.clone()

        # Deferred: gtsam is only needed by the long-sequence pipeline.
        import gtsam

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        for node in range(num_nodes):
            rotation, translation = _se3_data_to_rotation_translation(self._initial_c2w[node])
            values.insert(node, gtsam.Similarity3(gtsam.Rot3(rotation), gtsam.Point3(translation), 1.0))

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(_PRIOR_SIGMAS)
        rotation, translation = _se3_data_to_rotation_translation(self._initial_c2w[0])
        graph.add(
            gtsam.PriorFactorSimilarity3(
                0, gtsam.Similarity3(gtsam.Rot3(rotation), gtsam.Point3(translation), 1.0), prior_noise
            )
        )

        for src, dst, relative, weight in self._edges:
            rotation, translation = _se3_data_to_rotation_translation(relative)
            noise = gtsam.noiseModel.Diagonal.Sigmas(_EDGE_SIGMAS / max(weight, 1.0e-3))
            graph.add(
                gtsam.BetweenFactorSimilarity3(
                    src, dst, gtsam.Similarity3(gtsam.Rot3(rotation), gtsam.Point3(translation), 1.0), noise
                )
            )

        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        result = optimizer.optimize()
        logger.info(
            f"Pose graph: {num_nodes} nodes, {len(self._edges)} edges, "
            f"error {graph.error(values):.3e} -> {graph.error(result):.3e}"
        )

        scales = torch.ones(num_nodes)
        rigid = torch.zeros(num_nodes, 7)
        for node in range(num_nodes):
            similarity = result.atSimilarity3(node)
            scales[node] = float(similarity.scale())
            rigid[node] = _rotation_translation_to_se3_data(
                similarity.rotation().matrix(), np.asarray(similarity.translation()).reshape(3)
            )
        return scales, rigid

    def relative_from_w2c(self, poses_w2c: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Compute ``g_src⁻¹ ∘ g_dst`` (c2w between-measurements) from w2c poses.

        With ``g = w2c⁻¹``, the between-measurement simplifies to
        ``w2c_src ∘ w2c_dst⁻¹`` — computed directly, no explicit inversion of
        the absolute poses needed.
        """
        return (SE3(poses_w2c[src]) * SE3(poses_w2c[dst]).inv()).data  # type: ignore
