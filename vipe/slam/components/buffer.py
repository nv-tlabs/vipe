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
# -------------------------------------------------------------------------------------------------
# This file includes code originally from the DROID-SLAM repository:
# https://github.com/cvg/DROID-SLAM
# Licensed under the MIT License. See THIRD_PARTY_LICENSES.md for details.
# -------------------------------------------------------------------------------------------------

import logging

import numpy as np
import rerun as rr
import torch

from einops import rearrange
from omegaconf.dictconfig import DictConfig

from vipe.ext import slam_ext
from vipe.ext.lietorch import SE3
from vipe.priors.depth import DepthEstimationInput, DepthEstimationModel
from vipe.priors.depth.base import DepthType
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.visualization import POINTS_STENCIL, draw_lines_batch, draw_points_batch

from ..ba.solver import Solver, SparseBlockVector
from ..ba.terms import DenseDepthFlowTerm, DispSensRegularizationTerm
from ..interface import SLAMMap
from ..maths import geom
from ..maths.retractor import DenseDispRetractor, IntrinsicsRetractor, PoseRetractor, RigRotationOnlyRetractor
from .sparse_tracks import SparseTracks


logger = logging.getLogger(__name__)


class GraphBuffer:
    def __init__(
        self,
        height: int,
        width: int,
        n_views: int,
        buffer_size: int,
        init_disp: float,
        cross_view_idx: list[int] | None,
        ba_config: DictConfig,
        sparse_tracks: SparseTracks,
        camera_type: CameraType,
        device: torch.device = torch.device("cuda"),
    ):
        if cross_view_idx is None:
            cross_view_idx = [(i + 1) % n_views for i in range(n_views)]

        self.n_frames: int = 0

        self.height = height
        self.width = width
        self.n_views = n_views
        self.device = device
        self.ba_config = ba_config
        self.sparse_tracks = sparse_tracks
        self.camera_type = camera_type

        assert self.height % 8 == 0 and self.width % 8 == 0

        # timestamp (frame index)
        self.tstamp = torch.zeros(buffer_size, device=device, dtype=torch.int)
        # Original image (full resolution) RGB 0-1.
        self.images = torch.zeros(
            buffer_size,
            self.n_views,
            3,
            self.height,
            self.width,
            device=device,
            dtype=torch.float16,
        )
        self.dirty = torch.zeros(buffer_size, device=device, dtype=torch.bool)
        # Rig pose defined as the 0-th view of each frame.
        self.poses = torch.zeros(buffer_size, 7, device=device, dtype=torch.float)
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.poses.device)
        # This will be the original intrinsics
        self.intrinsics = torch.zeros(
            self.n_views,
            self.camera_type.intrinsics_dim(),
            device=device,
            dtype=torch.float,
        )
        # rig pose in a multi-view setting.
        self.rig = torch.zeros(self.n_views, 7, device=device, dtype=torch.float)
        self.rig[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.rig.device)

        # Inferred inverse depth (resolution / 8)
        self.disps = (
            torch.ones(
                buffer_size,
                self.n_views,
                self.height // 8,
                self.width // 8,
                device=device,
                dtype=torch.float,
            )
            * init_disp
        )

        # Sensor inverse depth (resolution / 8), used to add MSE to BA: |disps - disps_sens|^2.
        self.disps_sens = torch.zeros(
            buffer_size,
            self.n_views,
            self.height // 8,
            self.width // 8,
            device=device,
            dtype=torch.float,
        )

        # Masks (resolution / 8), 1 = invalid, 0 = valid
        self.masks = torch.zeros(
            buffer_size,
            self.n_views,
            self.height // 8,
            self.width // 8,
            device=device,
            dtype=torch.bool,
        )

        # Droid attributes
        # The updated operator will take the correlation volume of fmaps, the nets, and the inps,
        # and update the nets (hidden state) to a new one.
        # - feature maps
        self.fmaps = torch.zeros(
            buffer_size,
            self.n_views,
            128,
            self.height // 8,
            self.width // 8,
            device=device,
            dtype=torch.half,
        )
        # - GRU update operator initial state
        self.nets = torch.zeros(
            buffer_size,
            self.n_views,
            128,
            self.height // 8,
            self.width // 8,
            device=device,
            dtype=torch.half,
        )
        # - GRU update operator inputs (context)
        self.inps = torch.zeros(
            buffer_size,
            self.n_views,
            128,
            self.height // 8,
            self.width // 8,
            device=device,
            dtype=torch.half,
        )

        # [..., 0] is time, [..., 1] is view
        assert len(cross_view_idx) == self.n_views
        self.cross_view_idx = torch.zeros(buffer_size, self.n_views, 2, device=device, dtype=torch.long)
        self.cross_view_idx[..., 0] = torch.arange(buffer_size, device=device)[:, None]
        self.cross_view_idx[..., 1] = torch.tensor(cross_view_idx, device=device).long()[None]

        # Used to store the intrinsics used to compute the sensor depth.
        self.last_depth_intrinsics: torch.Tensor | None = None

    @property
    def flattened_disps(self):
        return rearrange(self.disps, "n v h w -> (n v) h w")

    @property
    def flattened_disps_sens(self):
        return rearrange(self.disps_sens, "n v h w -> (n v) h w")

    @property
    def flattened_fmaps(self):
        return rearrange(self.fmaps, "n v c h w -> (n v) c h w")

    @property
    def flattened_nets(self):
        return rearrange(self.nets, "n v c h w -> (n v) c h w")

    @property
    def flattened_inps(self):
        return rearrange(self.inps, "n v c h w -> (n v) c h w")

    @property
    def K(self) -> np.ndarray:
        intr_np = self.camera_type.build_camera_model(self.intrinsics).pinhole().intrinsics.cpu().numpy()
        k_mat = np.eye(3)[None].repeat(self.n_views, axis=0)
        k_mat[:, 0, 0] = intr_np[:, 0]
        k_mat[:, 1, 1] = intr_np[:, 1]
        k_mat[:, 0, 2] = intr_np[:, 2]
        k_mat[:, 1, 2] = intr_np[:, 3]
        return k_mat

    @property
    def K_dense_disp(self) -> np.ndarray:
        k_mat = self.K
        k_mat[:, 0] /= 8
        k_mat[:, 1] /= 8
        return k_mat

    def remove_second_newest(self, ix: int):
        assert ix == self.n_frames - 2
        self.tstamp[ix] = self.tstamp[ix + 1]
        self.images[ix] = self.images[ix + 1]
        self.poses[ix] = self.poses[ix + 1]
        self.disps[ix] = self.disps[ix + 1]
        self.disps_sens[ix] = self.disps_sens[ix + 1]
        self.dirty[ix] = True
        self.nets[ix] = self.nets[ix + 1]
        self.inps[ix] = self.inps[ix + 1]
        self.fmaps[ix] = self.fmaps[ix + 1]
        self.masks[ix] = self.masks[ix + 1]
        self.cross_view_idx[ix] = self.cross_view_idx[ix + 1]
        self.n_frames -= 1

    def update_disps_sens(self, depth_model: DepthEstimationModel | None, frame_idx: int | None):
        if depth_model is None:
            return

        if frame_idx is not None:
            # Update only one frame for frontend.
            frames_to_update = [frame_idx]
        else:
            # Update all frames for backend.
            assert self.last_depth_intrinsics is not None
            if torch.allclose(self.last_depth_intrinsics, self.intrinsics):
                return
            # If we can update in an easier way, do it.
            if depth_model.depth_type == DepthType.METRIC_DEPTH:
                # Depth is already estimated and we can simply do the scaling
                self.disps_sens[: self.n_frames] *= (
                    self.last_depth_intrinsics[0][0].item() / self.intrinsics[0][0].item()
                )
                return

            frames_to_update = pbar(range(self.n_frames), desc="Update depth")

        assert self.n_views == 1

        for frame_idx in frames_to_update:
            focal_length = self.intrinsics[0][0].item()
            depth_input = DepthEstimationInput(
                rgb=self.images[frame_idx].moveaxis(1, -1).float(),
                focal_length=focal_length,
            )
            disp_sens = depth_model.estimate(depth_input).metric_depth
            disp_sens = disp_sens[:, 3::8, 3::8]
            disp_sens = torch.where(disp_sens > 0, disp_sens.reciprocal(), disp_sens)
            self.disps_sens[frame_idx] = disp_sens

        self.last_depth_intrinsics = self.intrinsics.clone()

    def build_adaptive_cross_view_idx(self, valid_thresh: float = 400.0):
        """
        Use the current buffer info to update the cross_view_idx, based on reprojection distances.
        """
        if self.n_views == 1 or self.n_frames < 2:
            return

        ix = torch.arange(self.n_frames).to(self.device)
        jx = torch.arange(self.n_frames).to(self.device)

        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii, jj = ii.reshape(-1), jj.reshape(-1)

        ds = []
        for offset in range(1, self.n_views):
            ds.append(
                self.frame_distance_dense_disp(ii, jj, beta=1.0, view_offset=offset, bidirectional=False)
                .reshape(self.n_frames, self.n_frames, -1)
                .permute(0, 2, 1)  # (source_edge, view, target_edge)
            )
        d_total = torch.stack(ds, dim=-1).reshape(self.n_frames, self.n_views, -1)
        d_min, inds_best = torch.min(d_total, dim=-1)
        t_best, off_best = inds_best // len(ds), inds_best % len(ds)
        tgt_view_best = (off_best + 1 + torch.arange(self.n_views, device=self.device)) % self.n_views

        old_inds = self.cross_view_idx[: self.n_frames]
        new_inds = torch.stack([t_best, tgt_view_best], dim=-1)
        update_mask = d_min < valid_thresh
        logger.info(f"Updating {update_mask.sum().item()} cross-view indices out of {update_mask.numel()}.")

        new_inds[~update_mask] = old_inds[~update_mask]
        self.cross_view_idx[: self.n_frames] = new_inds

    def expand_edge_multiview(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        cross: bool = True,
        view_offset: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
            Expand from edge (ii, jj) to (pi, qi, di, pj, qj, dj) which refers
        to the actual indices in the flattened buffer.
            In terms of ii == jj and cross=True this means to create cross-view edges of a single frame,
        where the number of edges will also be n_views, e.g. (0, 1), (1, 2), (2, 0) for n_views=3.

        Args:
            ii (torch.Tensor): edge source (M, )
            jj (torch.Tensor): edge target (M, )
            cross (bool): whether to create cross-view edges for the same frame

        Returns:
            pi (torch.Tensor): source pose index (M * n_views, )
            qi (torch.Tensor): source rig index (M * n_views, )
            di (torch.Tensor): source dense_disp index (M * n_views, )
            pj (torch.Tensor): target pose index (M * n_views, )
            qj (torch.Tensor): target rig index (M * n_views, )
            dj (torch.Tensor): target dense_disp index (M * n_views, )
        """
        qi = torch.arange(self.n_views, device=self.device).reshape(1, -1).to(self.device)
        qi = qi.repeat(ii.shape[0], 1)
        pi = ii.reshape(-1, 1).repeat(1, self.n_views).to(self.device)
        qj = torch.arange(self.n_views, device=self.device).reshape(1, -1).to(self.device)
        qj = qj.repeat(jj.shape[0], 1)
        pj = jj.reshape(-1, 1).repeat(1, self.n_views).to(self.device)

        if cross:
            cross_mask = ii == jj
            if torch.any(cross_mask):
                t, v = self.cross_view_idx[pi[cross_mask], qi[cross_mask]].unbind(-1)
                pj[cross_mask], qj[cross_mask] = t, v

        qj = (qj + view_offset) % self.n_views

        di = pi * self.n_views + qi
        dj = pj * self.n_views + qj

        return (
            pi.reshape(-1),
            qi.reshape(-1),
            di.reshape(-1),
            pj.reshape(-1),
            qj.reshape(-1),
            dj.reshape(-1),
        )

    def expand_tracks_edges(self, ii: torch.Tensor, tracks_length: int):
        iis = [ii] * (tracks_length - 1)
        jjs = [ii - m - 1 for m in range(tracks_length - 1)]
        ii_cat, jj_cat = torch.cat(iis), torch.cat(jjs)
        edge_mask = (ii_cat >= 0) & (ii_cat < self.n_frames) & (jj_cat >= 0) & (jj_cat < self.n_frames)
        ii_cat, jj_cat = ii_cat[edge_mask], jj_cat[edge_mask]
        pi, qi, di, pj, qj, dj = self.expand_edge_multiview(ii_cat, jj_cat, cross=False)
        ti = pi - pj - 1
        return pi, qi, di, pj, qj, dj, ti

    def bundle_adjustment(
        self,
        target: torch.Tensor,
        weight: torch.Tensor,
        disp_damping: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        t0: int,
        t1: int,
        n_iters: int,
        pose_damping: float,
        pose_ep: float,
        motion_only: bool,
        limited_disp: bool,
        optimize_intrinsics: bool,
        optimize_rig_rotation: bool,
        verbose: bool,
    ):
        """
        This function is just an extraction from the factor_graph.
        They have to be coupled together to work.
        """
        assert t0 <= t1
        weight_dense_disp, weight_tracks = 0.001, 0.001
        # weight_dense_disp, weight_tracks = 0.001, 0.0
        # weight_dense_disp, weight_tracks = 0.0, 0.001

        pi, qi, di, pj, qj, dj = self.expand_edge_multiview(ii, jj)
        di_unique = torch.unique(di)
        pi_unique = torch.unique(ii)  # Should be equivalent to unique(pi)

        solver = Solver(compute_energy=verbose)
        solver.add_term(
            DenseDepthFlowTerm(
                pose_i_inds=pi,
                pose_j_inds=pj,
                rig_i_inds=qi,
                rig_j_inds=qj,
                dense_disp_i_inds=di,
                target=target,
                weight=weight_dense_disp * weight,
                intrinsics=None,
                intrinsics_factor=8.0,
                rig=None,
                image_size=(self.height // 8, self.width // 8),
                camera_type=self.camera_type,
            )
        )

        if self.sparse_tracks.enabled:
            # This does not support cross-view tracking yet.
            sparse_target, sparse_weight = self.sparse_tracks.compute_dense_disp_target_weight(
                source_view_inds=qi,
                source_frame_inds=self.tstamp[pi],
                target_view_inds=qj,
                target_frame_inds=self.tstamp[pj],
                image_size=(self.height, self.width),
                dense_disp_size=(self.height // 8, self.width // 8),
            )
            sparse_target = sparse_target.flatten(1, 2)
            sparse_weight = sparse_weight.flatten(1, 2)
            solver.add_term(
                DenseDepthFlowTerm(
                    pose_i_inds=pi,
                    pose_j_inds=pj,
                    rig_i_inds=qi,
                    rig_j_inds=qj,
                    dense_disp_i_inds=di,
                    target=sparse_target,
                    weight=weight_tracks * sparse_weight,
                    intrinsics=None,
                    intrinsics_factor=8.0,
                    rig=None,
                    image_size=(self.height // 8, self.width // 8),
                    camera_type=self.camera_type,
                )
            )

        # self.debug_visualize_target_weight(
        #     target,
        #     weight,
        #     pi,
        #     pj,
        #     qi,
        #     qj,
        #     # other_target=sparse_target,
        #     # other_weight=sparse_weight,
        # )

        solver.set_fixed(
            "pose",
            (torch.cat([pi_unique[pi_unique < t0], pi_unique[pi_unique >= t1]]) if t0 < t1 else None),
        )
        solver.set_retractor("pose", PoseRetractor())
        solver.set_damping("pose", damping=pose_damping, ep=pose_ep)

        if not motion_only:
            disps_sens = rearrange(self.flattened_disps_sens, "nv h w -> nv (h w)")
            sens_i_inds = di_unique[disps_sens[di_unique].sum(1) > 0.0]
            if len(sens_i_inds) > 0:
                solver.add_term(
                    DispSensRegularizationTerm(
                        i_inds=sens_i_inds,
                        alpha=self.ba_config.dense_disp_alpha,
                        disps_sens=disps_sens,
                    )
                )
            solver.set_retractor("dense_disp", DenseDispRetractor())
            disp_damping = rearrange(disp_damping, "nv h w -> nv (h w)")
            solver.set_damping(
                "dense_disp",
                damping=SparseBlockVector(
                    inds=di_unique,
                    data=0.2 * disp_damping[di_unique] + 1e-7,
                ),
                ep=1e-7,
            )
            if limited_disp:
                solver.set_fixed("dense_disp", torch.cat([di[pi < t0], di[pi >= t1]]))
        else:
            solver.set_fixed("dense_disp")
        solver.set_marginilized("dense_disp")

        solver.set_retractor("intrinsics", IntrinsicsRetractor(self.camera_type))
        solver.set_damping("intrinsics", damping=1e-6, ep=1e-6)
        if not optimize_intrinsics:
            solver.set_fixed("intrinsics")

        solver.set_retractor("rig", RigRotationOnlyRetractor())
        solver.set_damping("rig", damping=1e-4, ep=1e-4)
        if not optimize_rig_rotation:
            solver.set_fixed("rig")
        else:
            solver.set_fixed("rig", torch.zeros(1, device=self.device).long())

        disps_flattened = rearrange(self.flattened_disps, "nv h w -> nv (h w)")

        ba_energy = []
        for _ in range(n_iters):
            cur_energy = solver.run_inplace(
                {
                    "pose": SE3(self.poses),
                    "dense_disp": disps_flattened,
                    "intrinsics": self.intrinsics,
                    "rig": SE3(self.rig),
                }
            )
            ba_energy.append(cur_energy)

        if verbose:
            logger.info(f"BA iters = {n_iters}, energy: {ba_energy[0]} -> {ba_energy[-1]}")

        self.disps.clamp_(min=0.001)

    def reproject_dense_disp(self, ii: torch.Tensor, jj: torch.Tensor):
        """project points from ii -> jj. This will be optical flow from ii to jj when subtracted coords0"""
        ii, jj = ii.reshape(-1), jj.reshape(-1)
        pi, qi, di, pj, qj, _ = self.expand_edge_multiview(ii, jj)
        assert self.disps is not None and self.intrinsics is not None
        coords, valid_mask, _, _, _ = geom.iproj_i_proj_j_disp(
            SE3(self.poses),
            self.flattened_disps,
            None,
            self.camera_type.build_camera_model(self.intrinsics).scaled(1 / 8.0).intrinsics,
            self.camera_type,
            SE3(self.rig),
            pi,
            pj,
            qi,
            qj,
            di,
            jacobian_p_d=False,
            jacobian_f=False,
            jacobian_r=False,
        )
        return coords, valid_mask

    def frame_distance_dense_disp(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        beta: float = 0.3,
        bidirectional=True,
        view_offset: int = 0,
    ):
        """frame distance metric"""
        pi, qi, di, pj, qj, dj = self.expand_edge_multiview(ii, jj, cross=False, view_offset=view_offset)
        poses = self.poses[: self.n_frames]
        intrinsics = self.camera_type.build_camera_model(self.intrinsics).scaled(1 / 8.0).intrinsics

        d = geom.frame_distance_dense_disp(
            SE3(poses),
            self.flattened_disps,
            intrinsics,
            self.camera_type,
            SE3(self.rig),
            pi,
            pj,
            qi,
            qj,
            di,
            beta,
        )

        if bidirectional:
            d2 = geom.frame_distance_dense_disp(
                SE3(poses),
                self.flattened_disps,
                intrinsics,
                self.camera_type,
                SE3(self.rig),
                pj,
                pi,
                qj,
                qi,
                dj,
                beta,
            )
            d = 0.5 * (d + d2)

        return d.view(-1, self.n_views)

    def extract_slam_map(
        self,
        filter_thresh: float,
        t_range: torch.Tensor | None = None,
        is_local: bool = False,
    ) -> SLAMMap:
        if t_range is None:
            t_range = torch.arange(self.n_frames, device=self.device)

        c2w_se3 = SE3(self.poses[t_range]).inv()
        images = rearrange(self.images[t_range][..., 3::8, 3::8], "n v c h w -> n v h w c")
        n_frames, n_views, ht, wd, _ = images.shape

        pts_list, mask_list = [], []
        for v in range(self.n_views):
            c2w_view: SE3 = c2w_se3 * SE3(self.rig[v])[None]  # type: ignore
            disps_v = self.disps[t_range, v].contiguous()  # (n_frames, ht, wd)
            camera_model = self.camera_type.build_camera_model(self.intrinsics[v])
            pts, _, _ = geom.iproj_disp(
                disps_v,
                None,
                camera_model.scaled(1 / 8.0).intrinsics[None].expand((n_frames, -1)),
                camera_type=self.camera_type,
            )
            if not is_local:
                pts: torch.Tensor = c2w_view[:, None, None].act(pts)  # type: ignore
            pts = pts[..., :3] / pts[..., 3:]

            # The threshold is applied on the depth differences.
            filter_thresh_v = filter_thresh * (1.0 / disps_v.mean().item())
            count = slam_ext.depth_filter(
                c2w_view.inv().data,
                disps_v,
                camera_model.pinhole().intrinsics / 8.0,
                torch.arange(n_frames, device=self.device),
                torch.full((n_frames,), filter_thresh_v, device=self.device),
            )
            masks = (
                (count >= min(2, n_frames - 1))
                & (disps_v > 0.5 * disps_v.mean(dim=[1, 2], keepdim=True))
                & (~self.masks[t_range, v])
            )
            pts_list.append(pts)
            mask_list.append(masks)

        return SLAMMap.from_masked_dense_disp(
            torch.stack(pts_list, dim=1),
            images,
            torch.stack(mask_list, dim=1),
            self.tstamp[t_range],
        )

    def log_tracks(self):
        if not self.sparse_tracks.enabled:
            return

        # Note that this vis does not reflect the factors we use
        frame_indices = self.tstamp[: self.n_frames].cpu().numpy()

        for kf_idx in range(self.n_frames):
            rr.set_time_sequence("frame", int(frame_indices[kf_idx]))

            for v in range(self.n_views):
                canvas = self.images[kf_idx, v].moveaxis(0, -1).cpu().numpy().astype(np.float32)
                canvas = (canvas * 255).astype(np.uint8)

                for delta_kf_cnt in range(10):
                    skf_idx = kf_idx - delta_kf_cnt
                    if (tkf_idx := kf_idx - delta_kf_cnt - 1) < 0:
                        break

                    kp_indices = self.sparse_tracks.get_correspondences(
                        v, frame_indices[kf_idx], frame_indices[tkf_idx]
                    )
                    kp_indices_subset = self.sparse_tracks.get_correspondences(
                        v, frame_indices[skf_idx], frame_indices[tkf_idx]
                    )  # Some immediate ones might be masked out.
                    kp_indices = torch.tensor(
                        list(set(kp_indices.numpy().tolist()).intersection(set(kp_indices_subset.numpy().tolist()))),
                    )
                    if len(kp_indices) == 0:
                        break

                    source = self.sparse_tracks.get_observations(v, frame_indices[skf_idx], kp_indices)
                    target = self.sparse_tracks.get_observations(v, frame_indices[tkf_idx], kp_indices)

                    bright_green = max(255 - 20 * delta_kf_cnt, 0)
                    dim_green = max(255 - 20 * (delta_kf_cnt + 1), 0)

                    canvas = draw_points_batch(
                        canvas,
                        source.cpu().numpy(),
                        (0, bright_green, 0),
                        stencil=POINTS_STENCIL,
                    )
                    canvas = draw_points_batch(
                        canvas,
                        target.cpu().numpy(),
                        (0, dim_green, 0),
                        stencil=POINTS_STENCIL,
                    )
                    canvas = draw_lines_batch(
                        canvas,
                        source.cpu().numpy(),
                        target.cpu().numpy(),
                        (0, bright_green, 0),
                    )

                rr.log(f"world/tracks_v{v}", rr.Image(canvas).compress())  # type: ignore

    def log(self, vis_thresh: float):
        (dirty_index,) = torch.where(self.dirty)
        self.dirty[dirty_index] = False

        dirty_index = dirty_index[dirty_index < self.n_frames]
        if len(dirty_index) == 0:
            return

        current_map = self.extract_slam_map(filter_thresh=vis_thresh, t_range=dirty_index, is_local=False)

        for di, didx in enumerate(dirty_index.cpu().numpy().tolist()):
            rr.set_time_sequence("frame", int(self.tstamp[int(didx)].item()))

            pose_mat = SE3(self.poses[int(didx)]).inv().matrix().cpu().numpy()
            rr.log(
                f"world/kf_{didx:04d}",
                rr.Transform3D(translation=pose_mat[:3, 3], mat3x3=pose_mat[:3, :3]),
            )

            for v in range(self.n_views):
                rig_mat = SE3(self.rig[v]).matrix().cpu().numpy()
                dd_ht, dd_wd = self.disps_sens.shape[-2:]

                pcd_xyz, pcd_rgb = current_map.get_dense_disp_pcd(di, v)
                image = self.images[int(didx), v, :, 3::8, 3::8].moveaxis(0, -1).cpu().numpy()

                rr.log(
                    f"world/kf_{didx:04d}/v{v}",
                    rr.Transform3D(translation=rig_mat[:3, 3], mat3x3=rig_mat[:3, :3]),
                    rr.Pinhole(
                        resolution=[dd_wd, dd_ht],
                        image_from_camera=self.K_dense_disp[v],
                        camera_xyz=rr.ViewCoordinates.RDF,
                    ),
                    rr.Image((image * 255).astype(np.uint8)).compress(),
                )
                rr.log(
                    f"world/kp_{didx:04d}/v{v}",
                    rr.Points3D(
                        pcd_xyz.cpu().numpy(),
                        colors=pcd_rgb.cpu().numpy().astype(np.float32),
                    ),
                )
