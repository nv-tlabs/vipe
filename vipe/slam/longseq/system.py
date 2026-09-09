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

"""Sliding-window SLAM recipe with bounded GPU memory for arbitrarily long sequences.

In short: the existing frontend/backend run unchanged over a live window of at
most ``slam.window.max_keyframes`` keyframes. When the window fills, the
oldest ``retire_chunk`` keyframes are *retired*: they get a final local BA,
the per-frame infill for their time span runs immediately (while their dense
state is still on GPU), their poses and pose-graph edges go to a CPU-side
:class:`TrajectoryLedger`, and the buffer is compacted. A final Sim3
pose-graph optimization over all keyframes restores global consistency, and
its corrections deform the infilled trajectory and map chunks.

GPU memory is O(window); host memory is a bounded rolling frame cache plus
bytes-per-frame ledger state.

This class deliberately does not extract shared logic out of
``vipe.slam.system.SLAMSystem`` -- it only reuses the methods already public
on that class (``_precompute_features``, ``_add_keyframe``,
``_build_components``, and the components it builds), and otherwise
re-implements the run loop locally, to keep ``vipe/slam/system.py`` completely
unmodified.
"""

import logging

import torch
from omegaconf import DictConfig

from vipe.ext import lietorch
from vipe.ext.lietorch import SE3
from vipe.streams.base import FrameAttribute, ProcessedVideoStream, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.model_cache import ModelCache

from ..interface import SLAMMap, SLAMOutput
from ..system import SLAMSystem, StandardResizeStreamProcessor
from .frame_cache import RollingFrameCache
from .ledger import FilledSpan, TrajectoryLedger
from .pose_graph import Sim3PoseGraph

logger = logging.getLogger(__name__)

# Retirement must never touch a keyframe the frontend still optimizes against.
# The frontend window is `frontend_window`; the margin covers keyframe-removal
# decrements and the frontend's small look-back (t1 - 5, init_pose at t1 - 2).
# Kept in sync with vipe.config.slam.SLAMConfig.validate_window.
_FRONTEND_SAFETY_MARGIN = 16


class LongSequenceSLAMSystem(SLAMSystem):
    """SLAMSystem variant whose GPU memory does not grow with sequence length.

    Selected by ``PoseOnlyLongAnnotationPipeline`` whenever the ``slam.window``
    config subtree is present. Everything hot (frontend, backend BA, filler,
    networks) is the inherited implementation; this class only re-orchestrates
    it around the retire/infill/pose-graph cycle.
    """

    def __init__(self, device: torch.device, config: DictConfig, model_cache: ModelCache | None = None) -> None:
        super().__init__(device, config, model_cache)

        window_config = self.config.window
        if window_config is None:
            raise ValueError("LongSequenceSLAMSystem requires slam.window to be set.")
        self.max_keyframes = int(window_config.max_keyframes)
        self.retire_chunk_size = int(window_config.retire_chunk)
        self.local_ba_steps = int(window_config.get("local_ba_steps", 5))
        self.max_cached_frames = int(window_config.get("max_cached_frames", 2000))
        self.keep_map = bool(window_config.get("keep_map", False))

        low_water = self.max_keyframes - self.retire_chunk_size
        min_low_water = int(self.config.frontend_window) + _FRONTEND_SAFETY_MARGIN
        if low_water < min_low_water:
            raise ValueError(
                f"slam.window: max_keyframes - retire_chunk = {low_water} keyframes would intrude on the "
                f"frontend optimization window; need at least frontend_window + {_FRONTEND_SAFETY_MARGIN} "
                f"= {min_low_water}."
            )
        if self.config.visualize:
            raise ValueError("slam.window: rerun visualization is not supported in long-sequence mode.")
        if self.config.infill_dense_disp:
            raise ValueError("slam.window: infill_dense_disp is not supported in long-sequence mode.")

        # Populated by run().
        self.frame_cache = RollingFrameCache()
        self.ledger = TrajectoryLedger()
        # Global keyframe index of buffer slot 0 (== number of retired keyframes).
        self.num_retired = 0
        # First frame index whose pose has not been infilled yet.
        self.infill_cursor = 0

    def _encode_frame_into_slot(
        self,
        frame_idx: int,
        slot_idx: int,
        images: torch.Tensor,
        buffer_masks: torch.Tensor | None,
    ) -> None:
        """Write a frame's image and DROID network state into a buffer slot.

        Reimplements the keyframe-encoding half of the inherited
        ``SLAMSystem._add_keyframe`` (tstamp/images/fmaps/nets/inps/masks only,
        no pose/metric-depth/frame_data_list handling) so the infill-at-
        retirement path can re-encode cached raw frames into spare buffer slots
        without needing a full ``VideoFrame`` (the rolling frame cache only
        keeps images + masks, not poses or metric depth).
        """
        buffer_size = self.buffer.tstamp.shape[0]
        if slot_idx >= buffer_size - 1:
            raise RuntimeError(
                f"Keyframe buffer full ({buffer_size} slots): the sliding window overflowed its own "
                f"allocation. This should not happen -- please report this as a bug."
            )
        self.buffer.tstamp[slot_idx] = frame_idx
        self.buffer.images[slot_idx] = images
        self.buffer.fmaps[slot_idx] = self.droid_net.encode_features(images)
        self.buffer.nets[slot_idx], self.buffer.inps[slot_idx] = self.droid_net.encode_context(images)
        if buffer_masks is not None:
            self.buffer.masks[slot_idx] = buffer_masks

    def _recover_original_intrinsics(self, resizers: list[StandardResizeStreamProcessor]) -> torch.Tensor:
        return torch.stack(
            [resizer.recover_intrinsics(self.buffer.intrinsics[v]) for v, resizer in enumerate(resizers)]
        )

    @torch.no_grad()
    def run(
        self,
        video_streams: list[VideoStream],
        rig: SE3 | None = None,
        camera_type: CameraType = CameraType.PINHOLE,
    ) -> SLAMOutput:
        assert len(video_streams) > 0
        resizers = [StandardResizeStreamProcessor() for _ in video_streams]
        video_streams = [
            ProcessedVideoStream(video_stream, [resizer]) for video_stream, resizer in zip(video_streams, resizers)
        ]

        frame_size = video_streams[0].frame_size()
        total_n_frames = len(video_streams[0])
        for vs in video_streams:
            assert vs.frame_size() == frame_size
            assert len(vs) == total_n_frames

        if rig is None:
            assert len(video_streams) == 1, "Need rig for multiple views"
            rig = SE3.Identity(1)
        self.rig = rig

        self.config.update(
            {
                "height": frame_size[0],
                "width": frame_size[1],
                "n_views": len(video_streams),
                "has_init_pose": FrameAttribute.POSE in video_streams[0].attributes(),
                "camera_type": camera_type,
            }
        )

        # The buffer only ever holds the live window plus the in-flight infill
        # chunk -- this is the O(window) GPU-memory contract.
        self.config.buffer = self.max_keyframes + int(self.config.infill_chunk_size) + 8
        self._build_components()

        # ── The single pass: frontend + windowed backends + retirement. ──
        frame_data_list: list[VideoFrame]
        for frame_idx, frame_data_list in pbar(
            enumerate(zip(*video_streams)), desc="SLAM Pass (long)", total=total_n_frames
        ):
            images, buffer_masks = self._precompute_features(frame_data_list)

            self.sparse_tracks.track_image(frame_data_list)

            if self.motion_filter.check(images, buffer_masks) or frame_idx == total_n_frames - 1:
                is_keyframe = True
                self._add_keyframe(frame_idx, images, buffer_masks, frame_data_list, phase=1)
            else:
                is_keyframe = False

            self.frontend.run()

            # Run the backend in between to correct intrinsics and extrinsics in advance
            # to avoid large errors and local minima.
            if self.buffer.n_frames in self.config.frontend_backend_iters and is_keyframe:
                self.backend.run_if_necessary(5, log=False)

            self.frame_cache.push(frame_idx, images, buffer_masks)

            if self.buffer.n_frames >= self.max_keyframes:
                self._retire_chunk()

            # Enforce the host-memory cap independently of retirement. The cache
            # retains every frame still pending infill; a frame can only be
            # infilled (and evicted) once it has a keyframe on each side. For
            # near-static footage the motion filter creates almost no keyframes,
            # so retirement never runs and the usual "settled below the
            # frontend window" boundary never advances -- the cache would then
            # grow ~O(num_frames) and OOM.
            #
            # Under cache pressure, drain everything up to the latest keyframe
            # instead. Those keyframes are still being optimized, but the final
            # Sim3 pose-graph deformation re-anchors the infilled poses once the
            # keyframes settle, and for near-static footage the keyframes barely
            # move so the pre-deformation error is negligible. This only fires
            # when the cache is already at the cap, so dense-keyframe footage
            # (where the boundary advances normally) is unaffected.
            if len(self.frame_cache) >= self.max_cached_frames:
                drain_slot = max(self.buffer.n_frames - 1, 1)
                self._infill_until(int(self.buffer.tstamp[drain_slot].item()))

        # ── Final polish of the live window: same two backend runs as the base system. ──
        self.backend.run(7, log=False)
        self.backend.run(self.config.backend_iters, update_depth=False, log=False)

        # ── Flush: infill the remaining span and record the remaining keyframes. ──
        self._infill_until(total_n_frames)
        self._record_keyframes_and_edges(self.buffer.n_frames, is_flush=True)
        if self.keep_map:
            self._record_map_chunk(self.buffer.n_frames)

        # ── Global consistency: Sim3 pose graph over all keyframes, then deform. ──
        scales, rigid_c2w = self._optimize_pose_graph()
        self.ledger.apply_corrections(scales, rigid_c2w)

        trajectory_w2c = self.ledger.trajectory_w2c()
        if trajectory_w2c.shape[0] != total_n_frames:
            raise ValueError("Your video might be malformed. Try using streams.cached=true in the config.")

        return SLAMOutput(
            trajectory=SE3(trajectory_w2c.data.to(self.device)).inv(),
            intrinsics=self._recover_original_intrinsics(resizers),
            rig=SE3(self.buffer.rig.clone()),
            slam_map=self.ledger.assemble_map(self.device),
        )

    def _retire_chunk(self) -> None:
        """The retirement transaction.

        Order matters: the local BA gives the leaving keyframes their final
        dense polish; the infill then runs against that polished state; the
        ledger snapshots poses and edges; the map chunk is extracted while the
        dense state is still resident; only then is the buffer compacted.
        """
        count = self.retire_chunk_size

        # 1. Final dense polish for the keyframes about to leave (bounded: the
        # graph covers at most `max_keyframes` keyframes).
        self.backend.run(self.local_ba_steps, log=False)

        # Intrinsics are global: refining them further would retroactively
        # invalidate retired geometry, so freeze from the first retirement on.
        # This also turns the mid-run `frontend_backend_iters` backends into
        # no-ops for the rest of the run.
        self.config.optimize_intrinsics = False
        self.config.optimize_rig_rotation = False

        # The frontend pre-initializes the *next* keyframe's pose and disparity
        # at slot n_frames (one past the live end, see SLAMFrontend.__update).
        # The infill below reuses that slot as scratch and compaction does not
        # shift one-past-the-end state, so save it and restore it at the slot's
        # post-shift position -- otherwise the first keyframe after retirement
        # starts from a pose a whole chunk behind and the frontend BA cannot
        # recover (GRU correlation finds nothing that far off).
        next_slot = self.buffer.n_frames
        next_keyframe_pose = self.buffer.poses[next_slot].clone()
        next_keyframe_disps = self.buffer.disps[next_slot].clone()

        # 2. Fill every frame older than the first surviving keyframe while the
        # retiring keyframes' dense state is still on GPU.
        first_surviving_tstamp = int(self.buffer.tstamp[count].item())
        self._infill_until(first_surviving_tstamp)

        # 3. Ledger: retiring keyframe poses + pose-graph edges.
        self._record_keyframes_and_edges(count, is_flush=False)

        # 4. Map points of the retiring chunk (world space, at current poses).
        if self.keep_map:
            self._record_map_chunk(count)

        # 5. Compact the buffer and re-index the frontend accordingly.
        self.buffer.retire_head(count)
        self.frontend.graph.shift_indices(count)
        self.frontend.t1 -= count
        self.num_retired += count

        # Restore the frontend's next-keyframe pre-initialization (see above).
        self.buffer.poses[next_slot - count] = next_keyframe_pose
        self.buffer.disps[next_slot - count] = next_keyframe_disps

        # Peak allocation is the memory-boundedness contract: it must stay flat
        # across retirements regardless of sequence length.
        peak_alloc_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        logger.info(
            f"Retired {count} keyframes (total {self.num_retired}); "
            f"live window {self.buffer.n_frames}, frame cache {len(self.frame_cache)}, "
            f"peak GPU alloc {peak_alloc_gib:.2f} GiB."
        )

    def _infill_until(self, span_end: int) -> None:
        """Fill per-frame poses for frames ``[infill_cursor, span_end)``.

        Reuses the inherited :class:`InnerFiller` exactly like the base pass 2:
        cached frames are re-encoded into spare buffer slots past the live
        keyframes, then interpolated + refined against the bracketing
        keyframes. The result goes to the ledger together with the anchors'
        pose snapshots, which the final deformation is computed against.
        """
        if span_end <= self.infill_cursor:
            return
        start_slot = self.buffer.n_frames
        assert start_slot >= 1, "cannot infill before the first keyframe exists"
        self.inner_filler.set_start_idx(start_slot)
        keyframe_tstamps = self.buffer.tstamp[:start_slot].detach().cpu().long()

        for frame_idx in range(self.infill_cursor, span_end):
            images, buffer_masks = self.frame_cache.pop(frame_idx, self.device)
            self._encode_frame_into_slot(frame_idx, self.buffer.n_frames, images, buffer_masks)
            self.buffer.n_frames += 1
            if self.inner_filler.check() or frame_idx == span_end - 1:
                self.inner_filler.compute()

        # Drain the filler (compute() appends one SE3 batch per chunk and
        # resets n_frames back to start_slot).
        poses_w2c = lietorch.cat(self.inner_filler.filled_poses, dim=0).data.detach().cpu()
        self.inner_filler.filled_poses.clear()
        assert self.buffer.n_frames == start_slot

        frame_inds = torch.arange(self.infill_cursor, span_end)
        # Mirror InnerFiller.compute's bracket search: left-inclusive nearest
        # keyframe and its successor (clamped at the ends).
        left_bracket = torch.searchsorted(keyframe_tstamps, frame_inds, right=True) - 1
        left_bracket = left_bracket.clamp(min=0)
        right_bracket = torch.where(left_bracket < start_slot - 1, left_bracket + 1, left_bracket)

        # Interpolation weight along [left, right] for the deformation blend.
        bracket_span = (keyframe_tstamps[right_bracket] - keyframe_tstamps[left_bracket]).float()
        interp_weight = (frame_inds - keyframe_tstamps[left_bracket]).float() / bracket_span.clamp(min=1.0)
        interp_weight = interp_weight.clamp(0.0, 1.0)

        snapshot_slots = torch.unique(torch.cat([left_bracket, right_bracket]))
        self.ledger.record_filled_span(
            FilledSpan(
                frame_inds=frame_inds,
                poses_w2c=poses_w2c,
                anchor_a=left_bracket + self.num_retired,
                anchor_b=right_bracket + self.num_retired,
                interp_weight=interp_weight,
                snapshot_kf_inds=snapshot_slots + self.num_retired,
                snapshot_poses_w2c=self.buffer.poses[snapshot_slots].detach().cpu().clone(),
            )
        )
        self.infill_cursor = span_end

    def _record_keyframes_and_edges(self, count: int, is_flush: bool) -> None:
        """Record buffer slots ``[0, count)`` as pose-graph nodes plus their edges.

        Edges are (a) the consecutive-keyframe chain -- including, at
        retirement, the boundary edge into the first surviving keyframe, whose
        node is recorded when *it* retires -- and (b) co-visibility edges of
        the last backend graph that touch the recorded slots. All measurements
        are taken at the current (just-polished) poses.
        """
        poses_w2c = self.buffer.poses[: self.buffer.n_frames]
        self.ledger.record_keyframes(poses_w2c[:count], self.buffer.tstamp[:count])

        # Chain edges (k, k+1). At flush there is no keyframe after the last.
        last_chain_src = count if not is_flush else count - 1
        src = torch.arange(0, last_chain_src, device=self.device)
        dst = src + 1

        # Co-visibility edges from the last backend graph (window-slot indices).
        if self.backend.last_graph is not None:
            graph_ii, graph_jj = self.backend.last_graph.unbind(-1)
            keep = graph_ii < graph_jj - 1  # one direction only; adjacency covered by the chain
            if not is_flush:
                keep &= graph_ii < count  # only edges touching the retiring chunk
            keep &= graph_jj < self.buffer.n_frames
            src = torch.cat([src, graph_ii[keep]])
            dst = torch.cat([dst, graph_jj[keep]])

        pose_graph = Sim3PoseGraph()
        relative_c2w = pose_graph.relative_from_w2c(poses_w2c, src, dst)
        self.ledger.record_edges(
            src_inds=src + self.num_retired,
            dst_inds=dst + self.num_retired,
            relative_c2w=relative_c2w,
            weights=torch.ones(src.shape[0]),
        )

    def _record_map_chunk(self, count: int) -> None:
        """Extract the retiring keyframes' map points and store them on CPU."""
        t_range = torch.arange(count, device=self.device)
        slam_map = self.buffer.extract_slam_map(
            filter_thresh=self.config.map_filter_thresh, t_range=t_range, is_local=False
        )
        cpu_map = SLAMMap(
            dense_disp_xyz=slam_map.dense_disp_xyz.cpu(),
            dense_disp_rgb=slam_map.dense_disp_rgb.cpu(),
            dense_disp_packinfo=slam_map.dense_disp_packinfo.cpu(),
            dense_disp_frame_inds=slam_map.dense_disp_frame_inds,
        )
        self.ledger.record_map_chunk(cpu_map, torch.arange(count) + self.num_retired)

    def _optimize_pose_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the Sim3 pose graph over all recorded keyframes."""
        pose_graph = Sim3PoseGraph()
        initial_c2w, edge_groups = self.ledger.pose_graph_inputs()
        pose_graph.set_nodes(initial_c2w)
        for src_inds, dst_inds, relative_c2w, weights in edge_groups:
            pose_graph.add_edges(src_inds, dst_inds, relative_c2w, weights)
        return pose_graph.optimize()
