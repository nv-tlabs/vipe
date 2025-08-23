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

import warnings

import numpy as np
import rerun as rr
import torch

from einops import rearrange

from vipe.ext.lietorch import SE3

from ..networks.droid_net import AltCorrBlock, CorrBlock, DroidNet
from .buffer import GraphBuffer


# Disable all future warnings (mainly torch.cuda.amp related)
warnings.simplefilter(action="ignore", category=FutureWarning)


class FactorGraph:
    @staticmethod
    def coords_grid(ht, wd, **kwargs):
        y, x = torch.meshgrid(
            torch.arange(ht).to(**kwargs).float(),
            torch.arange(wd).to(**kwargs).float(),
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1)

    def __init__(
        self,
        net: DroidNet,
        buffer: GraphBuffer,
        device: torch.device,
        max_factors: int,
        incremental: bool,
        cross_view: bool,
    ):
        self.net = net
        self.buffer = buffer
        self.device = device
        self.max_factors = max_factors
        self.cross_view = cross_view and buffer.n_views > 1
        self.incremental = incremental

        # operator at 1/8 resolution
        ht = buffer.height // 8
        wd = buffer.width // 8

        self.coords0 = self.coords_grid(ht, wd, device=device)

        # edge connections are the same for all the views.
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.damping = 1e-6 * torch.ones_like(self.buffer.flattened_disps)

        # target_coords and weight of last update.
        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # corr is the correlation volume. Will not create if incremental=False.
        # inp is the input features for the update module. Will not create if incremental=False.
        # f_net is the hidden state of the GRU. Since this will be keep updated across update/update_lowmem calls,
        #   we will be storing the updated states no matter incremental is True or False.
        self.corr, self.f_net, self.inp = None, None, None

        # inactive and bad factors
        # - inactive factors are those who are removed by rm_factors(store=True)
        # They could be later revived in BA if use_inactive=True)
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """remove duplicate edges"""

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)]
            + [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)]
        )

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def get_edges_np(self) -> tuple[np.ndarray, np.ndarray]:
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()
        w = torch.mean(self.weight, dim=[0, 2, 3, 4]).cpu().numpy()

        ix = np.argsort(ii)
        ii, jj, w = ii[ix], jj[ix], w[ix]
        return np.stack([ii, jj], axis=1), w

    @torch.amp.autocast("cuda", enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """add edges to factor graph"""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if (
            self.max_factors > 0
            and self.ii.shape[0] + ii.shape[0] > self.max_factors
            and self.corr is not None
            and remove
        ):
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        pi, qi, _, pj, qj, _ = self.buffer.expand_edge_multiview(ii, jj)

        if self.incremental:
            # correlation volume for new edges (1, |E| V, 128, ht//8, wd//8)
            fmap1 = self.buffer.fmaps[pi, qi][None]
            fmap2 = self.buffer.fmaps[pj, qj][None]
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.buffer.inps[pi, qi][None]
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.buffer.reproject_dense_disp(ii, jj)
            target = target[None]
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # GRU initial hidden state (h_ij) is taken from the source edge.
        # (1, |E| V, 128, ht//8, wd//8)
        net = self.buffer.nets[pi, qi][None]
        self.f_net = net if self.f_net is None else torch.cat([self.f_net, net], 1)

        # reprojection factors initial state
        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.amp.autocast("cuda", enabled=True)
    def rm_factors(self, mask: torch.Tensor, store: bool = False):
        """drop edges from factor graph"""

        # store estimated factors
        exp_mask = mask.view(-1, 1).repeat(1, self.buffer.n_views).view(-1)

        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:, exp_mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:, exp_mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

        if self.corr is not None:
            self.corr = self.corr[~exp_mask]

        if self.f_net is not None:
            self.f_net = self.f_net[:, ~exp_mask]

        if self.inp is not None:
            self.inp = self.inp[:, ~exp_mask]

        self.target = self.target[:, ~exp_mask]
        self.weight = self.weight[:, ~exp_mask]

    @torch.amp.autocast("cuda", enabled=True)
    def rm_second_newest_keyframe(self, ix: int):
        """
        Remove the keyframe[ix] from the video and graph.
        There might be ix+1 in the video that are also available, so we have to shift them forward.
        """

        self.buffer.remove_second_newest(ix)

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            m_exp = m.view(-1, 1).repeat(1, self.buffer.n_views).view(-1)
            self.target_inac = self.target_inac[:, ~m_exp]
            self.weight_inac = self.weight_inac[:, ~m_exp]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)

    @torch.amp.autocast("cuda", enabled=True)
    def update(
        self,
        t0: int | None = None,  # will limit pose update to >= t0 if provided
        t1: int | None = None,  # will limit pose update to < t1 if provided
        itrs: int = 3,
        use_inactive: bool = False,
        motion_only: bool = False,
        fixed_motion: bool = False,
        limited_disp: bool = False,
    ):
        """run update operator on factor graph"""
        assert self.incremental
        assert self.corr is not None and self.inp is not None and self.f_net is not None
        assert not (motion_only and fixed_motion)

        if t0 is None:
            t0 = int(max(1, self.ii.min().item() + 1))

        if t1 is None:
            t1 = int(max(self.ii.max().item(), self.jj.max().item()) + 1)

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, _ = self.buffer.reproject_dense_disp(self.ii, self.jj)
            coords1 = coords1[None]  # NV, ht, wd, 2 -> 1, NV, ht, wd, 2
            # Prepare motion features:
            # - first 2 dimension is the current rigid flow.
            # - last 2 dimension residual flow of last update.
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            # (edge, view, ht, wd, 4) -> (edge, view, 4, ht, wd)
            motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        # correlation features
        corr = self.corr(coords1)

        # Apply network
        pi, qi, di, _, _, _ = self.buffer.expand_edge_multiview(self.ii, self.jj)
        di, dix = torch.unique(di, return_inverse=True)
        self.f_net, delta, weight, damping, _ = self.net.update.forward(  # type: ignore
            self.f_net, self.inp, corr, motn, ix=dix
        )
        weight[:, self.buffer.masks[pi, qi]] = 0.0

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)
            # Overwrite damping with newly computed values
            self.damping[di] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                exp_m = m.view(-1, 1).repeat(1, self.buffer.n_views).view(-1)
                target = torch.cat([self.target_inac[:, exp_m], self.target], 1)
                weight = torch.cat([self.weight_inac[:, exp_m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            ht, wd = self.coords0.shape[0:2]
            target = rearrange(target, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)
            weight = rearrange(weight, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)

            # dense bundle adjustment
            self.buffer.bundle_adjustment(
                target=target,
                weight=weight,
                disp_damping=self.damping,
                ii=ii,
                jj=jj,
                t0=t0,
                t1=t1 if not fixed_motion else t0,
                n_iters=itrs,
                pose_damping=1e-3,
                pose_ep=0.1,
                motion_only=motion_only,
                limited_disp=limited_disp,
                optimize_intrinsics=False,
                optimize_rig_rotation=False,
                verbose=False,
            )

        self.age += 1

    @torch.amp.autocast("cuda", enabled=False)
    def update_batch(
        self,
        itrs: int,
        steps: int,
        optimize_intrinsics: bool,
        optimize_rig_rotation: bool,
        solver_verbose: bool = False,
    ):
        """
        Suitable for batch processing of the factor graph, when incremental=False.
        This will reduce memory using the AltCorrBlock (where fmap.T @ fmap is not explicitly computed).

        Args:
            steps (int): number of steps for the update operator
            itrs (int): number of iterations for BA within each step
        """
        if self.incremental:
            warnings.warn("Calling update_batch with incremental=True could be slow.")
        assert self.f_net is not None

        # alternate corr implementation
        t = self.buffer.n_frames
        corr_op = AltCorrBlock(self.buffer.flattened_fmaps[None])

        for _ in range(steps):
            with torch.cuda.amp.autocast(enabled=False):
                coords1, _ = self.buffer.reproject_dense_disp(self.ii, self.jj)
                coords1 = coords1[None]  # NV, ht, wd, 2 -> 1, NV, ht, wd, 2
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            # Apply the update operator in batches of size s to reduce memory usage.
            s = 8
            assert self.jj.max() >= self.ii.max()
            for i in range(0, self.jj.max() + 1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis, jjs = self.ii[v], self.jj[v]
                v_exp = v.view(-1, 1).repeat(1, self.buffer.n_views).view(-1)
                pis, qis, dis, pjs, qjs, djs = self.buffer.expand_edge_multiview(iis, jjs)
                corr1 = corr_op(coords1[:, v_exp], dis, djs)
                dis, dixs = torch.unique(dis, return_inverse=True)

                with torch.cuda.amp.autocast(enabled=True):
                    net, delta, weight, damping, _ = self.net.update.forward(  # type: ignore
                        self.f_net[:, v_exp],
                        self.buffer.inps[pis, qis][None],
                        corr1,
                        motn[:, v_exp],
                        ix=dixs,
                    )
                    weight[:, self.buffer.masks[pis, qis]] = 0.0

                self.f_net[:, v_exp] = net
                self.target[:, v_exp] = coords1[:, v_exp] + delta.float()
                self.weight[:, v_exp] = weight.float()
                self.damping[dis] = damping

            ht, wd = self.coords0.shape[0:2]
            target = rearrange(self.target, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)
            weight = rearrange(self.weight, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)

            self.buffer.bundle_adjustment(
                target=target,
                weight=weight,
                disp_damping=self.damping,
                ii=self.ii,
                jj=self.jj,
                t0=1,
                t1=t,
                n_iters=itrs,
                pose_damping=1e-5,
                pose_ep=1e-2,
                motion_only=False,
                limited_disp=False,
                optimize_intrinsics=optimize_intrinsics,
                optimize_rig_rotation=optimize_rig_rotation,
                verbose=solver_verbose,
            )

    def add_neighborhood_factors(self, t0, t1, r: int = 3):
        """
        add edges between neighboring frames within radius r
        (note that the edges are uni-directional, hence both 0,1 and 1,0 are added)
        """

        ii, jj = torch.meshgrid(torch.arange(t0, t1), torch.arange(t0, t1), indexing="ij")
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = -1 if self.cross_view else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(
        self,
        t0: int = 0,
        t1: int = 0,
        rad: int = 2,
        nms: int = 2,
        beta: float = 0.25,
        thresh: float = 16.0,
        remove: bool = False,
    ):
        """
        Add the following two types of edges:
            - Neighborhood edges: Edges connecting i-rad, i-rad+1, ..., i-1 to i (where i is from t0 to t)
            - Proximity edges: All potential edges (i, j) connecting from [t0, t) to [t1, t):
                - Only considering i projected forward to j <= i - rad (forward movement assumption)
                - Will not add edges if i, j are already connected by neighborhood edges (NMS)
        Added edges will be then made bidirectional.

        (Note: This is borrowed from DROID-SLAM and is really weird. Consider re-writing completely.)
        """
        assert t0 >= t1, "t0 should be a subset of t1"

        t = self.buffer.n_frames
        ix = torch.arange(t0, t).to(self.device)
        jx = torch.arange(t1, t).to(self.device)

        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii, jj = ii.reshape(-1), jj.reshape(-1)

        d = self.buffer.frame_distance_dense_disp(ii, jj, beta=beta)
        d = d.mean(-1)

        def _suppress(i: int, j: int):
            if (t0 <= i < t) and (t1 <= j < t):
                d[(i - t0) * (t - t1) + (j - t1)] = np.inf

        def _suppress_nms(i: int, j: int):
            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        _suppress(i + di, j + dj)

        for i, j in zip(self.ii.cpu().numpy(), self.jj.cpu().numpy()):
            _suppress_nms(i, j)

        for i, j in zip(self.ii_inac.cpu().numpy(), self.jj_inac.cpu().numpy()):
            _suppress_nms(i, j)

        d[(ii - rad < jj) | (d > thresh)] = np.inf

        es = []
        for i in range(t0, t):
            if self.cross_view:
                es.append((i, i))
                _suppress(i, i)

            for j in range(max(i - rad - 1, 0), i):
                es.append((i, j))
                es.append((j, i))
                _suppress(i, j)

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i, j = int(ii[k].item()), int(jj[k].item())
            es.append((i, j))
            es.append((j, i))
            _suppress_nms(i, j)

        if len(es) == 0:
            return
        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)

    def log(self):
        center_pos = SE3(self.buffer.poses).inv().translation()[:, :3]
        active_edges = torch.stack([center_pos[self.ii], center_pos[self.jj]], dim=1)
        inactive_edges = torch.stack([center_pos[self.ii_inac], center_pos[self.jj_inac]], dim=1)
        rr.log("world/active_edges", rr.LineStrips3D(active_edges.cpu().numpy()))
        rr.log("world/inactive_edges", rr.LineStrips3D(inactive_edges.cpu().numpy()))
