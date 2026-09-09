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

"""Trajectory ledger for the long-sequence SLAM recipe.

The ledger is the CPU-side record of everything that outlives a keyframe's
stay in the GPU window: keyframe poses at retirement, pose-graph edges,
per-frame infilled poses with their anchor snapshots, and (optionally) map
point chunks. It is bytes-per-frame — the dense per-keyframe state never
crosses to the host.

Pose conventions: buffer poses are world-to-camera (w2c) SE3, matching
``GraphBuffer.poses``. Pose-graph work happens in camera-to-world (c2w),
matching gtsam's convention; the ledger converts at its boundaries.

Sim3 corrections are represented as ``(scale, rigid SE3)`` pairs acting on
points as ``x -> scale * (R @ x) + t``.
"""

from dataclasses import dataclass

import torch

from vipe.ext.lietorch import SE3

from ..interface import SLAMMap


def _scaled_translation(pose_data: torch.Tensor, scale: torch.Tensor) -> SE3:
    """Return the SE3 with its translation multiplied by ``scale`` (rotation kept)."""
    data = pose_data.clone()
    data[..., :3] = data[..., :3] * scale.unsqueeze(-1)
    return SE3(data)


def _interpolate_se3(pose_a: SE3, pose_b: SE3, weight: torch.Tensor) -> SE3:
    """Geodesic interpolation from ``pose_a`` (weight 0) to ``pose_b`` (weight 1)."""
    return pose_a * SE3.exp(weight.unsqueeze(-1) * (pose_a.inv() * pose_b).log())  # type: ignore


@dataclass(slots=True)
class Sim3Correction:
    """Per-keyframe similarity corrections ``x -> scale * (R @ x) + t``.

    ``rigid`` holds the (N, 7) SE3 data of the (R, t) part; ``scale`` is (N,).
    Index k corrects everything that was expressed relative to keyframe k's
    snapshot pose: the keyframe itself, frames infilled against it, and its
    map points.
    """

    scale: torch.Tensor
    rigid: torch.Tensor

    @staticmethod
    def from_corrected_poses(
        corrected_scale: torch.Tensor,
        corrected_rigid_c2w: torch.Tensor,
        snapshot_c2w: torch.Tensor,
    ) -> "Sim3Correction":
        """Build the correction mapping ``snapshot_c2w`` onto the corrected Sim3 pose.

        For a corrected Sim3 pose ``S = (s, T)`` (acting as ``T ∘ scale_s``) and a
        rigid snapshot ``g``, the correction is ``Δ = S ∘ g⁻¹``, which factors as
        ``Δ.rigid = T ∘ scale_translation(g⁻¹, s)`` with ``Δ.scale = s``.
        """
        snapshot_inv = SE3(snapshot_c2w).inv()
        rigid = SE3(corrected_rigid_c2w) * _scaled_translation(snapshot_inv.data, corrected_scale)  # type: ignore
        return Sim3Correction(scale=corrected_scale, rigid=rigid.data)

    def interpolate(self, inds_a: torch.Tensor, inds_b: torch.Tensor, weight: torch.Tensor) -> "Sim3Correction":
        """Blend corrections of two anchors along the geodesic (per-element weights)."""
        rigid = _interpolate_se3(SE3(self.rigid[inds_a]), SE3(self.rigid[inds_b]), weight)
        log_scale_a, log_scale_b = self.scale[inds_a].log(), self.scale[inds_b].log()
        scale = torch.exp((1.0 - weight) * log_scale_a + weight * log_scale_b)
        return Sim3Correction(scale=scale, rigid=rigid.data)

    def apply_to_c2w(self, c2w: SE3) -> SE3:
        """Correct camera-to-world poses: rotation composes rigidly, camera centers move like points."""
        return SE3(self.rigid) * _scaled_translation(c2w.data, self.scale)  # type: ignore

    def apply_to_points(self, index: int, points: torch.Tensor) -> torch.Tensor:
        """Apply correction ``index`` to (M, 3) world points."""
        matrix = SE3(self.rigid[index]).matrix()
        rotation, translation = matrix[:3, :3], matrix[:3, 3]
        return self.scale[index] * (points @ rotation.T) + translation


@dataclass(slots=True)
class FilledSpan:
    """Per-frame poses infilled at one retirement, with their deformation anchors.

    Every filled frame is bracketed by two keyframes (``anchor_a`` temporally
    before or at the frame, ``anchor_b`` after; equal at the sequence ends).
    ``snapshot_*`` store the anchors' w2c poses at infill time — the poses the
    filled trajectory is consistent with. The final pose-graph correction of
    each anchor relative to its snapshot is what deforms the span.
    """

    frame_inds: torch.Tensor  # (F,) long, ascending
    poses_w2c: torch.Tensor  # (F, 7)
    anchor_a: torch.Tensor  # (F,) long, global keyframe indices
    anchor_b: torch.Tensor  # (F,) long
    interp_weight: torch.Tensor  # (F,) float in [0, 1]: 0 -> anchor_a, 1 -> anchor_b
    snapshot_kf_inds: torch.Tensor  # (K,) long, global keyframe indices
    snapshot_poses_w2c: torch.Tensor  # (K, 7)


@dataclass(slots=True)
class MapChunk:
    """World-space map points of one retired chunk (CPU), extracted at retirement."""

    slam_map: SLAMMap
    kf_global_inds: torch.Tensor  # (C,) long, one per packinfo row of slam_map


class TrajectoryLedger:
    """Append-only host-side record of the long-sequence trajectory.

    Owns: keyframe poses/tstamps (pose-graph nodes), relative-pose edges,
    infilled per-frame spans with anchor snapshots, and optional map chunks.
    ``apply_corrections`` consumes the pose-graph result and deforms all of it
    consistently.
    """

    def __init__(self) -> None:
        self._kf_poses_w2c: list[torch.Tensor] = []  # chunks of (C, 7)
        self._kf_tstamps: list[torch.Tensor] = []  # chunks of (C,) long
        self._edges: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._spans: list[FilledSpan] = []
        self._map_chunks: list[MapChunk] = []
        self._corrections: Sim3Correction | None = None

    @property
    def num_keyframes(self) -> int:
        return sum(int(t.shape[0]) for t in self._kf_tstamps)

    @property
    def num_filled_frames(self) -> int:
        return sum(int(s.frame_inds.shape[0]) for s in self._spans)

    def record_keyframes(self, poses_w2c: torch.Tensor, tstamps: torch.Tensor) -> None:
        """Append retiring keyframes; global indices continue from the previous count."""
        self._kf_poses_w2c.append(poses_w2c.detach().cpu().clone())
        self._kf_tstamps.append(tstamps.detach().cpu().long().clone())

    def record_edges(
        self,
        src_inds: torch.Tensor,
        dst_inds: torch.Tensor,
        relative_c2w: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """Append relative-pose edges (global indices, gtsam ``between`` convention g_i⁻¹ g_j)."""
        self._edges.append(
            (
                src_inds.detach().cpu().long(),
                dst_inds.detach().cpu().long(),
                relative_c2w.detach().cpu(),
                weights.detach().cpu(),
            )
        )

    def record_filled_span(self, span: FilledSpan) -> None:
        """Append one infilled span; spans must arrive in temporal order."""
        if self._spans:
            last_end = int(self._spans[-1].frame_inds[-1].item())
            assert int(span.frame_inds[0].item()) == last_end + 1, "filled spans must be contiguous"
        self._spans.append(span)

    def record_map_chunk(self, slam_map: SLAMMap, kf_global_inds: torch.Tensor) -> None:
        self._map_chunks.append(MapChunk(slam_map=slam_map, kf_global_inds=kf_global_inds.detach().cpu().long()))

    def pose_graph_inputs(
        self,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Return (initial keyframe c2w poses (N, 7), recorded edge groups) for the pose graph."""
        poses_w2c = torch.cat(self._kf_poses_w2c, dim=0)
        return SE3(poses_w2c).inv().data, self._edges  # type: ignore

    def apply_corrections(self, corrected_scale: torch.Tensor, corrected_rigid_c2w: torch.Tensor) -> None:
        """Deform the whole ledger with the pose-graph result.

        ``corrected_*`` are the absolute Sim3 c2w poses of all keyframes, indexed
        by global keyframe index. Keyframe poses are replaced outright; filled
        frames and map points are moved by their anchors' corrections
        (interpolated between the two brackets for continuity).
        """
        num_kf = self.num_keyframes
        assert corrected_rigid_c2w.shape[0] == num_kf and corrected_scale.shape[0] == num_kf

        # The recorded retirement poses double as the pose-graph node initial
        # values and as the extraction poses of the map chunks; capture them
        # before replacing.
        initial_c2w = SE3(torch.cat(self._kf_poses_w2c, dim=0)).inv().data

        # Keyframe poses: replaced by the (rigid part of the) corrected poses.
        corrected_w2c = SE3(corrected_rigid_c2w).inv().data
        self._kf_poses_w2c = [corrected_w2c.clone()]
        self._kf_tstamps = [torch.cat(self._kf_tstamps, dim=0)]

        # Filled frames: each span's anchors have their own snapshots, so the
        # correction is computed per span against those snapshots.
        for span in self._spans:
            snapshot_c2w = SE3(span.snapshot_poses_w2c).inv().data
            per_snapshot = Sim3Correction.from_corrected_poses(
                corrected_scale=corrected_scale[span.snapshot_kf_inds],
                corrected_rigid_c2w=corrected_rigid_c2w[span.snapshot_kf_inds],
                snapshot_c2w=snapshot_c2w,
            )
            # Map global anchor indices to positions inside the snapshot arrays.
            global_to_local = {int(g.item()): local for local, g in enumerate(span.snapshot_kf_inds)}
            local_a = torch.tensor([global_to_local[int(g.item())] for g in span.anchor_a], dtype=torch.long)
            local_b = torch.tensor([global_to_local[int(g.item())] for g in span.anchor_b], dtype=torch.long)

            frame_correction = per_snapshot.interpolate(local_a, local_b, span.interp_weight)
            corrected_c2w = frame_correction.apply_to_c2w(SE3(span.poses_w2c).inv())
            span.poses_w2c = corrected_c2w.inv().data

        # Map chunks: points of keyframe k move by k's correction. Snapshot ==
        # the pose the points were extracted at == the recorded retirement pose
        # (captured above as the node initial values).
        if self._map_chunks:
            per_kf = Sim3Correction.from_corrected_poses(corrected_scale, corrected_rigid_c2w, initial_c2w)
            for chunk in self._map_chunks:
                packinfo = chunk.slam_map.dense_disp_packinfo  # (C, V, 2)
                for row, kf_global in enumerate(chunk.kf_global_inds.tolist()):
                    for view in range(packinfo.shape[1]):
                        start, count = packinfo[row, view].tolist()
                        if count == 0:
                            continue
                        points = chunk.slam_map.dense_disp_xyz[start : start + count]
                        chunk.slam_map.dense_disp_xyz[start : start + count] = per_kf.apply_to_points(kf_global, points)

        self._corrections = Sim3Correction(scale=corrected_scale, rigid=corrected_rigid_c2w)

    def trajectory_w2c(self) -> SE3:
        """Concatenate all filled spans into the full per-frame w2c trajectory."""
        assert self._spans, "no filled frames recorded"
        poses = torch.cat([span.poses_w2c for span in self._spans], dim=0)
        return SE3(poses)

    def keyframe_tstamps(self) -> torch.Tensor:
        return torch.cat(self._kf_tstamps, dim=0)

    def assemble_map(self, device: torch.device) -> SLAMMap | None:
        """Concatenate map chunks into one SLAMMap on ``device`` (None if none kept)."""
        if not self._map_chunks:
            return None
        xyz, rgb, packinfos, frame_inds = [], [], [], []
        point_offset = 0
        for chunk in self._map_chunks:
            slam_map = chunk.slam_map
            xyz.append(slam_map.dense_disp_xyz)
            rgb.append(slam_map.dense_disp_rgb)
            packinfo = slam_map.dense_disp_packinfo.clone()
            packinfo[..., 0] += point_offset
            packinfos.append(packinfo)
            frame_inds.extend(slam_map.dense_disp_frame_inds)
            point_offset += slam_map.dense_disp_xyz.shape[0]
        return SLAMMap(
            dense_disp_xyz=torch.cat(xyz, dim=0).to(device),
            dense_disp_rgb=torch.cat(rgb, dim=0).to(device),
            dense_disp_packinfo=torch.cat(packinfos, dim=0).to(device),
            dense_disp_frame_inds=frame_inds,
        )
