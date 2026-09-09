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

# mypy: ignore-errors

"""Unit tests for the long-sequence sliding-window SLAM machinery.

Covers the shared-component primitives (GraphBuffer.retire_head,
FactorGraph.shift_indices), the trajectory ledger with its Sim3 deformation,
the gtsam pose graph, and LongSequenceSLAMSystem construction/validation.
CPU-only except where noted.
"""

from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from vipe.ext.lietorch import SE3
from vipe.slam.components.buffer import GraphBuffer
from vipe.slam.components.sparse_tracks import DummySparseTracks
from vipe.slam.longseq.frame_cache import RollingFrameCache
from vipe.slam.longseq.ledger import FilledSpan, Sim3Correction, TrajectoryLedger
from vipe.slam.longseq.pose_graph import Sim3PoseGraph
from vipe.slam.longseq.system import LongSequenceSLAMSystem
from vipe.utils.cameras import CameraType

HEIGHT, WIDTH = 32, 40


def _make_buffer(buffer_size: int = 12, n_frames: int = 10) -> GraphBuffer:
    buffer = GraphBuffer(
        height=HEIGHT,
        width=WIDTH,
        n_views=1,
        buffer_size=buffer_size,
        init_disp=1.0,
        cross_view_idx=None,
        ba_config=OmegaConf.create({"dense_disp_alpha": 0.001, "fused": False}),
        sparse_tracks=DummySparseTracks(1),
        camera_type=CameraType.PINHOLE,
        device=torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(20260721)
    buffer.n_frames = n_frames
    buffer.tstamp[:n_frames] = torch.arange(n_frames) * 3  # non-trivial tstamps
    buffer.images[:n_frames] = torch.rand(buffer.images[:n_frames].shape, generator=generator).half()
    buffer.poses[:n_frames] = torch.randn(n_frames, 7, generator=generator)
    buffer.poses[:n_frames, 3:] /= buffer.poses[:n_frames, 3:].norm(dim=-1, keepdim=True)
    buffer.disps[:n_frames] = torch.rand(buffer.disps[:n_frames].shape, generator=generator)
    buffer.fmaps[:n_frames] = torch.randn(buffer.fmaps[:n_frames].shape, generator=generator).half()
    return buffer


def _random_se3(n: int, seed: int, translation_scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    data = torch.randn(n, 7, generator=generator)
    data[:, :3] *= translation_scale
    data[:, 3:] /= data[:, 3:].norm(dim=-1, keepdim=True)
    return data


class RetireHeadTest(unittest.TestCase):
    def test_contents_shift_and_count_drops(self):
        buffer = _make_buffer(n_frames=10)
        expected_tstamp = buffer.tstamp[3:10].clone()
        expected_images = buffer.images[3:10].clone()
        expected_poses = buffer.poses[3:10].clone()
        expected_fmaps = buffer.fmaps[3:10].clone()

        buffer.retire_head(3)

        self.assertEqual(buffer.n_frames, 7)
        torch.testing.assert_close(buffer.tstamp[:7], expected_tstamp, atol=0, rtol=0)
        torch.testing.assert_close(buffer.images[:7], expected_images, atol=0, rtol=0)
        torch.testing.assert_close(buffer.poses[:7], expected_poses, atol=0, rtol=0)
        torch.testing.assert_close(buffer.fmaps[:7], expected_fmaps, atol=0, rtol=0)
        # Temporal order (and hence sorted tstamps) must survive compaction.
        self.assertTrue(torch.all(buffer.tstamp[1:7] > buffer.tstamp[:6]))
        # Cross-view self-reference invariant: slot k points at time index k.
        torch.testing.assert_close(buffer.cross_view_idx[:7, :, 0], torch.arange(7).view(-1, 1), atol=0, rtol=0)

    def test_rejects_retiring_everything(self):
        buffer = _make_buffer(n_frames=5)
        with self.assertRaises(AssertionError):
            buffer.retire_head(5)


class ShiftIndicesTest(unittest.TestCase):
    def _make_graph(self, buffer: GraphBuffer):
        # FactorGraph needs no network for index bookkeeping tests.
        from vipe.slam.components.factor_graph import FactorGraph

        graph = FactorGraph.__new__(FactorGraph)
        graph.buffer = buffer
        device = buffer.device
        ht, wd = HEIGHT // 8, WIDTH // 8
        buffer_size = buffer.tstamp.shape[0]
        graph.ii = torch.tensor([5, 6, 7], dtype=torch.long, device=device)
        graph.jj = torch.tensor([6, 7, 9], dtype=torch.long, device=device)
        graph.ii_inac = torch.tensor([1, 2, 5], dtype=torch.long, device=device)
        graph.jj_inac = torch.tensor([2, 6, 8], dtype=torch.long, device=device)
        graph.target_inac = torch.arange(3, dtype=torch.float).view(1, 3, 1, 1, 1).expand(1, 3, ht, wd, 2).clone()
        graph.weight_inac = graph.target_inac.clone()
        # damping is indexed by buffer slot; give every slot a distinct value.
        graph.damping = torch.arange(buffer_size, dtype=torch.float).view(-1, 1, 1).expand(-1, ht, wd).clone()
        return graph

    def test_active_edges_shift_and_dead_inactive_dropped(self):
        buffer = _make_buffer()
        graph = self._make_graph(buffer)
        graph.shift_indices(3)

        torch.testing.assert_close(graph.ii, torch.tensor([2, 3, 4]), atol=0, rtol=0)
        torch.testing.assert_close(graph.jj, torch.tensor([3, 4, 6]), atol=0, rtol=0)
        # Inactive edges (1,2) and (2,6) touch retired slots -> dropped; (5,8) shifts.
        torch.testing.assert_close(graph.ii_inac, torch.tensor([2]), atol=0, rtol=0)
        torch.testing.assert_close(graph.jj_inac, torch.tensor([5]), atol=0, rtol=0)
        self.assertEqual(graph.target_inac.shape[1], 1)
        self.assertEqual(float(graph.target_inac[0, 0, 0, 0, 0]), 2.0)  # the surviving edge's payload
        # Slot-indexed damping must follow the compaction: row k <- old row k+3.
        self.assertEqual(float(graph.damping[0, 0, 0]), 3.0)
        self.assertEqual(float(graph.damping[5, 0, 0]), 8.0)

    def test_active_edge_into_retired_slot_asserts(self):
        buffer = _make_buffer()
        graph = self._make_graph(buffer)
        with self.assertRaises(AssertionError):
            graph.shift_indices(6)  # active edge (5, 6) would go negative


class TrajectoryLedgerTest(unittest.TestCase):
    def _identity_corrections(self, ledger: TrajectoryLedger):
        initial_c2w, _ = ledger.pose_graph_inputs()
        return torch.ones(initial_c2w.shape[0]), initial_c2w

    def _make_simple_ledger(self) -> tuple[TrajectoryLedger, torch.Tensor, torch.Tensor]:
        """Two keyframes at t=0 and t=4, three filled frames anchored between them."""
        ledger = TrajectoryLedger()
        kf_poses_w2c = _random_se3(2, seed=1)
        ledger.record_keyframes(kf_poses_w2c, torch.tensor([0, 4]))

        filled_w2c = _random_se3(5, seed=2)
        span = FilledSpan(
            frame_inds=torch.arange(5),
            poses_w2c=filled_w2c.clone(),
            anchor_a=torch.tensor([0, 0, 0, 0, 1]),
            anchor_b=torch.tensor([1, 1, 1, 1, 1]),
            interp_weight=torch.tensor([0.0, 0.25, 0.5, 0.75, 0.0]),
            snapshot_kf_inds=torch.tensor([0, 1]),
            snapshot_poses_w2c=kf_poses_w2c.clone(),
        )
        ledger.record_filled_span(span)
        return ledger, kf_poses_w2c, filled_w2c

    def test_identity_corrections_leave_trajectory_unchanged(self):
        ledger, _, filled_w2c = self._make_simple_ledger()
        ledger.apply_corrections(*self._identity_corrections(ledger))
        torch.testing.assert_close(ledger.trajectory_w2c().data, filled_w2c, atol=1e-5, rtol=1e-5)

    def test_rigid_global_move_deforms_frames_rigidly(self):
        # Move every keyframe by the same rigid transform Delta: every filled
        # frame must move by exactly Delta as well (c2w_new = Delta * c2w_old).
        ledger, kf_poses_w2c, filled_w2c = self._make_simple_ledger()
        delta = SE3(_random_se3(1, seed=3))

        snapshot_c2w = SE3(kf_poses_w2c).inv()
        corrected_c2w = (delta * snapshot_c2w).data
        ledger.apply_corrections(torch.ones(2), corrected_c2w)

        expected_c2w = (delta * SE3(filled_w2c).inv()).data
        torch.testing.assert_close(SE3(ledger.trajectory_w2c().data).inv().data, expected_c2w, atol=1e-5, rtol=1e-5)

    def test_interpolation_is_continuous_at_anchors(self):
        # A frame with weight 0 must move exactly like anchor_a; weight 1 like anchor_b.
        ledger, kf_poses_w2c, filled_w2c = self._make_simple_ledger()
        # Perturb only keyframe 1.
        snapshot_c2w = SE3(kf_poses_w2c).inv().data
        corrected_c2w = snapshot_c2w.clone()
        perturbation = SE3(_random_se3(1, seed=4, translation_scale=0.1))
        corrected_c2w[1] = (perturbation * SE3(snapshot_c2w[1:2])).data[0]
        ledger.apply_corrections(torch.ones(2), corrected_c2w)

        result_c2w = SE3(ledger.trajectory_w2c().data).inv().data
        # Frame 0 (weight 0, anchored at kf 0, which did not move): unchanged.
        torch.testing.assert_close(result_c2w[0], SE3(filled_w2c).inv().data[0], atol=1e-5, rtol=1e-5)
        # Frame 4 (weight 0 on anchor pair (1,1)): moves exactly by the kf-1 perturbation.
        expected = (perturbation * SE3(SE3(filled_w2c[4:5]).inv().data))[0].data
        torch.testing.assert_close(result_c2w[4], expected, atol=1e-5, rtol=1e-5)

    def test_scale_correction_scales_camera_centers(self):
        # A pure scale correction s about the origin must scale camera centers by s.
        ledger, kf_poses_w2c, filled_w2c = self._make_simple_ledger()
        scale = 2.0
        snapshot_c2w = SE3(kf_poses_w2c).inv().data
        corrected_c2w = snapshot_c2w.clone()
        corrected_c2w[:, :3] *= scale  # Sim3(s, I, 0) applied to the pose
        ledger.apply_corrections(torch.full((2,), scale), corrected_c2w)

        result_c2w = SE3(ledger.trajectory_w2c().data).inv().data
        original_c2w = SE3(filled_w2c).inv().data
        torch.testing.assert_close(result_c2w[:, :3], original_c2w[:, :3] * scale, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result_c2w[:, 3:], original_c2w[:, 3:], atol=1e-5, rtol=1e-5)

    def test_spans_must_be_contiguous(self):
        ledger, _, _ = self._make_simple_ledger()
        bad_span = FilledSpan(
            frame_inds=torch.arange(6, 8),  # gap: previous span ended at 4
            poses_w2c=_random_se3(2, seed=5),
            anchor_a=torch.tensor([1, 1]),
            anchor_b=torch.tensor([1, 1]),
            interp_weight=torch.zeros(2),
            snapshot_kf_inds=torch.tensor([1]),
            snapshot_poses_w2c=_random_se3(1, seed=6),
        )
        with self.assertRaises(AssertionError):
            ledger.record_filled_span(bad_span)


class Sim3CorrectionTest(unittest.TestCase):
    def test_point_and_pose_corrections_are_consistent(self):
        # Applying the correction to a camera center (as part of the pose) must
        # equal applying it to that center as a world point.
        snapshot_c2w = _random_se3(3, seed=7)
        corrected_c2w = _random_se3(3, seed=8)
        scale = torch.tensor([1.0, 1.5, 0.7])
        correction = Sim3Correction.from_corrected_poses(scale, corrected_c2w, snapshot_c2w)

        centers = snapshot_c2w[:, :3]
        moved_poses = correction.apply_to_c2w(SE3(snapshot_c2w))
        for k in range(3):
            moved_center = correction.apply_to_points(k, centers[k : k + 1])
            torch.testing.assert_close(moved_poses.data[k, :3], moved_center[0], atol=1e-5, rtol=1e-5)
            # By construction, Delta maps the snapshot onto the corrected pose:
            # translation and rotation must match corrected_c2w exactly.
            torch.testing.assert_close(moved_poses.data[k, :3], corrected_c2w[k, :3], atol=1e-5, rtol=1e-5)


class Sim3PoseGraphTest(unittest.TestCase):
    def test_consistent_chain_is_a_fixed_point(self):
        # Edges measured exactly from the initial poses: optimization must return them.
        poses_w2c = _random_se3(6, seed=9)
        graph = Sim3PoseGraph()
        initial_c2w = SE3(poses_w2c).inv().data
        graph.set_nodes(initial_c2w)
        src, dst = torch.arange(5), torch.arange(1, 6)
        graph.add_edges(src, dst, graph.relative_from_w2c(poses_w2c, src, dst), torch.ones(5))

        scales, rigid = graph.optimize()
        torch.testing.assert_close(scales, torch.ones(6), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(rigid[:, :3], initial_c2w[:, :3], atol=1e-4, rtol=1e-4)

    def test_noisy_loop_is_reconciled(self):
        # A square loop whose chain edges are consistent but whose start is
        # noisily displaced: the loop-closure edge must pull the end back.
        true_c2w = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Drifted initial values: last node displaced.
        initial_c2w = true_c2w.clone()
        initial_c2w[3, 0] += 0.5

        graph = Sim3PoseGraph()
        graph.set_nodes(initial_c2w)
        poses_w2c = SE3(true_c2w).inv().data
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph.add_edges(src, dst, graph.relative_from_w2c(poses_w2c, src, dst), torch.ones(4))

        _, rigid = graph.optimize()
        torch.testing.assert_close(rigid[:, :3], true_c2w[:, :3], atol=1e-3, rtol=1e-3)


class RollingFrameCacheTest(unittest.TestCase):
    def test_push_pop_roundtrip(self):
        cache = RollingFrameCache()
        images = torch.rand(1, 3, 8, 8)
        masks = torch.zeros(1, 1, 1, dtype=torch.bool)
        cache.push(3, images, masks)
        self.assertEqual(len(cache), 1)
        out_images, out_masks = cache.pop(3, torch.device("cpu"))
        torch.testing.assert_close(out_images, images, atol=0, rtol=0)
        self.assertEqual(len(cache), 0)
        self.assertIsNotNone(out_masks)

    def test_pop_missing_raises(self):
        cache = RollingFrameCache()
        with self.assertRaises(KeyError):
            cache.pop(0, torch.device("cpu"))


class CapacityGuardTest(unittest.TestCase):
    def test_full_buffer_raises_actionable_error(self):
        system = LongSequenceSLAMSystem.__new__(LongSequenceSLAMSystem)
        system.buffer = _make_buffer(buffer_size=12)
        system.droid_net = None
        # The last usable slot is buffer_size - 2: the frontend pre-initializes
        # one slot past the newest keyframe (see _encode_frame_into_slot).
        with self.assertRaisesRegex(RuntimeError, "buffer full"):
            system._encode_frame_into_slot(frame_idx=0, slot_idx=11, images=None, buffer_masks=None)


class LongSequenceSLAMSystemInitTest(unittest.TestCase):
    def test_requires_window(self):
        cfg = OmegaConf.create(
            {"visualize": False, "infill_dense_disp": False, "frontend_window": 25, "window": None}
        )
        with self.assertRaises(ValueError):
            LongSequenceSLAMSystem(torch.device("cpu"), cfg)

    def test_window_too_small_for_frontend_raises(self):
        cfg = OmegaConf.create(
            {
                "visualize": False,
                "infill_dense_disp": False,
                "frontend_window": 25,
                "window": {"max_keyframes": 128, "retire_chunk": 120},
            }
        )
        with self.assertRaisesRegex(ValueError, "frontend"):
            LongSequenceSLAMSystem(torch.device("cpu"), cfg)

    def test_valid_window_constructs(self):
        cfg = OmegaConf.create(
            {
                "visualize": False,
                "infill_dense_disp": False,
                "frontend_window": 25,
                "window": {"max_keyframes": 128, "retire_chunk": 64},
            }
        )
        system = LongSequenceSLAMSystem(torch.device("cpu"), cfg)
        self.assertEqual(system.max_keyframes, 128)
        self.assertEqual(system.retire_chunk_size, 64)


if __name__ == "__main__":
    unittest.main()
