# SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
# SPDX-License-Identifier: MIT
#
# Source: https://github.com/princeton-vl/DPVO
# See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

# Modified for MetricTrail integration.
# The following Apache-2.0 terms apply to NVIDIA modifications only.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .metric_depth import patch_metric_priors, prepare_metric_disparity
from .utils import *

mp.set_start_method('spawn', True)


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            raise ValueError("CLASSIC_LOOP_CLOSURE must be disabled")
        if float(self.cfg.STEREO_WEIGHT) <= 0.0:
            raise ValueError("STEREO_WEIGHT must be positive")
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0
        self.metric_prior_valid_patches = 0
        self.metric_prior_total_patches = 0
        self.metric_prior_valid_pixel_fractions = []
        self.metric_keyframes_seen = 0
        self.metric_prior_regularization_enabled = False
        self.metric_prior_ba_calls = 0
        self.global_ba_calls = 0
        self.stereo_baseline_m = None
        self.stereo_factor_count = 0

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)


        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)
        if self.cfg.STEREO:
            self.fmap1_right_ = torch.zeros(
                1, self.mem, 128, ht // 1, wd // 1, **kwargs
            )
            self.fmap2_right_ = torch.zeros(
                1, self.mem, 128, ht // 4, wd // 4, **kwargs
            )
            self.pyramid_right = (self.fmap1_right_, self.fmap2_right_)
        else:
            self.fmap1_right_ = None
            self.fmap2_right_ = None
            self.pyramid_right = None

        self.viewer = None
        if viz:
            self.start_viewer()


    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    @torch.no_grad()
    def terminate(self):


        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None, edge_is_stereo=None):
        """Local correlation against left-temporal or right-stereo features."""
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        stereo = (
            self.pg.edge_is_stereo
            if edge_is_stereo is None and indicies is None
            else torch.zeros(len(ii), dtype=torch.bool, device=ii.device)
            if edge_is_stereo is None
            else torch.as_tensor(edge_is_stereo, device=ii.device)
        )
        if stereo.dtype != torch.bool or stereo.ndim != 1 or len(stereo) != len(ii):
            raise ValueError("edge_is_stereo must be one boolean per correlation edge")
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        corr_left = torch.stack([corr1, corr2], -1).view(1, len(ii), -1)
        # Explicit-index callers (motion probes) are temporal-only.  Normal
        # mono runs also return before touching the right pyramid.  This avoids
        # a per-update CUDA ``any().item()`` synchronization.
        if not self.cfg.STEREO or (
            edge_is_stereo is None and indicies is not None
        ):
            return corr_left
        if self.pyramid_right is None:
            raise RuntimeError("stereo edges exist but the right feature pyramid is absent")
        corr1_right = altcorr.corr(
            self.gmap, self.pyramid_right[0], coords / 1, ii1, jj1, 3
        )
        corr2_right = altcorr.corr(
            self.gmap, self.pyramid_right[1], coords / 4, ii1, jj1, 3
        )
        corr_right = torch.stack([corr1_right, corr2_right], -1).view(
            1, len(ii), -1
        )
        return torch.where(stereo.view(1, -1, 1), corr_right, corr_left)

    def reproject(self, indicies=None, edge_is_stereo=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        stereo = None if not self.cfg.STEREO else (
            self.pg.edge_is_stereo
            if edge_is_stereo is None and indicies is None
            else edge_is_stereo
        )
        coords = pops.transform(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk,
            edge_is_stereo=stereo,
            stereo_baseline_m=(self.stereo_baseline_m or 0.0),
        )
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj, edge_is_stereo=None):
        if edge_is_stereo is None:
            edge_is_stereo = torch.zeros(
                len(ii), dtype=torch.bool, device=ii.device
            )
        else:
            edge_is_stereo = torch.as_tensor(edge_is_stereo, device=ii.device)
        if (
            edge_is_stereo.dtype != torch.bool
            or edge_is_stereo.ndim != 1
            or len(edge_is_stereo) != len(ii)
        ):
            raise ValueError("edge_is_stereo must be one boolean per factor")
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])
        self.pg.edge_is_stereo = torch.cat(
            [self.pg.edge_is_stereo, edge_is_stereo]
        )

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.edge_is_stereo_inac = torch.cat(
                (self.pg.edge_is_stereo_inac, self.pg.edge_is_stereo[m])
            )
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.edge_is_stereo = self.pg.edge_is_stereo[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        assert self.pg.ii.numel() == self.pg.edge_is_stereo.numel()

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.metric_disps_[i] = self.pg.metric_disps_[i+1]
                self.pg.metric_weights_[i] = self.pg.metric_weights_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]
                if self.cfg.STEREO:
                    self.fmap1_right_[0,i%self.mem] = self.fmap1_right_[0,(i+1)%self.mem]
                    self.fmap2_right_[0,i%self.mem] = self.fmap2_right_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M
            self.pg.metric_disps_[self.n].zero_()
            self.pg.metric_weights_[self.n].zero_()


        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        # Metric local-BA-only runs discard inactive factors. Metric loop runs
        # retain them because their global BA keeps physical scale and reapplies
        # the MoGe depth unary instead of using stock monocular normalization.
        self.remove_factors(
            to_remove,
            store=(not self.cfg.METRIC_DEPTH) or self.cfg.LOOP_CLOSURE,
        )

    def __run_global_BA(self):
        """Global BA over active/inactive edges while preserving metric scale."""
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))
        full_stereo = torch.cat(
            (self.pg.edge_is_stereo_inac, self.pg.edge_is_stereo)
        )

        # Stock DPV-SLAM removes the monocular gauge by normalizing depth and
        # translation.  That would destroy the metric frame supplied by MoGe
        # and the physical stereo baseline, so metric/stereo runs retain scale.
        if not self.cfg.METRIC_DEPTH and not self.cfg.STEREO:
            self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = full_ii.min().item()
        prior_lambda = (
            float(self.cfg.METRIC_DEPTH_PRIOR_WEIGHT)
            if self.cfg.METRIC_DEPTH and self.metric_prior_regularization_enabled
            else 0.0
        )
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk,
            t0, self.n, M=self.M, iterations=2, eff_impl=True,
            depth_prior=self.pg.metric_disps_.reshape(-1),
            depth_prior_weight=self.pg.metric_weights_.reshape(-1),
            depth_prior_lambda=prior_lambda,
            depth_prior_huber=float(self.cfg.METRIC_DEPTH_PRIOR_HUBER),
            edge_is_stereo=full_stereo,
            stereo_baseline_m=(self.stereo_baseline_m or 0.0))
        if prior_lambda > 0.0:
            self.metric_prior_ba_calls += 1
        self.global_ba_calls += 1
        self.ran_global_ba[self.n] = True

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=self.cfg.MIXED_PRECISION):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            if self.cfg.STEREO:
                weight[:, self.pg.edge_is_stereo] *= float(
                    self.cfg.STEREO_WEIGHT
                )
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    prior_lambda = (
                        float(self.cfg.METRIC_DEPTH_PRIOR_WEIGHT)
                        if self.cfg.METRIC_DEPTH
                        and self.metric_prior_regularization_enabled
                        else 0.0
                    )
                    fastba.BA(self.poses, self.patches, self.intrinsics, 
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk,
                        t0, self.n, M=self.M, iterations=2, eff_impl=False,
                        depth_prior=self.pg.metric_disps_.reshape(-1),
                        depth_prior_weight=self.pg.metric_weights_.reshape(-1),
                        depth_prior_lambda=prior_lambda,
                        depth_prior_huber=float(self.cfg.METRIC_DEPTH_PRIOR_HUBER),
                        edge_is_stereo=self.pg.edge_is_stereo,
                        stereo_baseline_m=(self.stereo_baseline_m or 0.0))
                    if prior_lambda > 0.0:
                        # Increment only after the CUDA BA call returns without
                        # raising, so diagnostics prove the unary was executed.
                        self.metric_prior_ba_calls += 1
            except Exception:
                # A silent fallback would make the metric experiment look as
                # though it used the prior while actually running no BA.
                if self.cfg.METRIC_DEPTH:
                    raise
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __edges_stereo(self):
        """One fixed-rig right-camera observation for every new left patch."""
        kk = torch.arange(self.m - self.M, self.m, device="cuda")
        jj = torch.full_like(kk, self.n - 1)
        return kk, jj

    def __call__(
        self,
        tstamp,
        image,
        intrinsics,
        metric_depth=None,
        metric_mask=None,
        *,
        right_image=None,
        stereo_baseline_m=None,
    ):
        """ track new frame """

        if self.cfg.STEREO:
            if right_image is None or stereo_baseline_m is None:
                raise ValueError(
                    "STEREO=True requires right_image and stereo_baseline_m "
                    "for every frame"
                )
            if not isinstance(right_image, torch.Tensor):
                raise TypeError("right_image must be a torch.Tensor")
            if tuple(right_image.shape) != tuple(image.shape):
                raise ValueError(
                    "rectified stereo images must have identical CHW shape: "
                    f"left={tuple(image.shape)} right={tuple(right_image.shape)}"
                )
            baseline = float(stereo_baseline_m)
            if not math.isfinite(baseline) or baseline <= 0.0:
                raise ValueError("stereo_baseline_m must be finite and positive")
            if self.stereo_baseline_m is None:
                self.stereo_baseline_m = baseline
            elif not math.isclose(
                baseline,
                self.stereo_baseline_m,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "stereo baseline changed within one sequence: "
                    f"{self.stereo_baseline_m} -> {baseline}"
                )
        elif right_image is not None or stereo_baseline_m is not None:
            raise ValueError(
                "right_image/stereo_baseline_m were provided while STEREO=False"
            )


        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        image_height, image_width = image.shape[-2:]
        metric_disps = None
        metric_valid = None
        metric_diagnostics = None
        metric_prior = None
        metric_weight = None
        if self.cfg.METRIC_DEPTH:
            if metric_depth is None:
                raise ValueError("METRIC_DEPTH=True requires metric_depth for every frame")
            metric_disps, metric_valid, metric_diagnostics = prepare_metric_disparity(
                metric_depth,
                metric_mask,
                image_hw=(image_height, image_width),
                min_depth=float(self.cfg.METRIC_DEPTH_MIN),
                max_depth=float(self.cfg.METRIC_DEPTH_MAX),
            )
            metric_disps = metric_disps.to(device=image.device)
            metric_valid = metric_valid.to(device=image.device)
            if not metric_diagnostics["has_valid_depth"]:
                raise ValueError(
                    "metric depth has no valid pixels; rejecting the frame "
                    "instead of injecting a 1 m initialization"
                )

        image = 2 * (image[None,None] / 255.0) - 0.5
        right_fmap = None
        if self.cfg.STEREO:
            right_network_input = 2 * (right_image[None, None] / 255.0) - 0.5
        
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    disps=metric_disps,
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)
            if self.cfg.STEREO:
                right_fmap = self.network.patchify.feature_only(
                    right_network_input
                )

        # Validate the sampled priors before mutating timestamps, poses, graph
        # indices, or diagnostic counters. A rejected frame therefore leaves
        # the tracker in the same state in which it entered this call.
        if self.cfg.METRIC_DEPTH:
            metric_prior, metric_weight = patch_metric_priors(patches, metric_valid)
            if not bool(metric_weight.any()):
                raise ValueError(
                    "metric depth has no valid sampled DPVO patches; rejecting "
                    "the frame instead of injecting an unobserved depth"
                )

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        if self.cfg.METRIC_DEPTH:
            patches[:, :, 2] = metric_prior[None, :, None, None]
            self.pg.metric_disps_[self.n] = metric_prior
            self.pg.metric_weights_[self.n] = metric_weight
            self.metric_prior_valid_pixel_fractions.append(
                metric_diagnostics["valid_pixel_fraction"]
            )
            self.metric_prior_valid_patches += int(metric_weight.sum().item())
            self.metric_prior_total_patches += int(metric_weight.numel())
        else:
            # Preserve the official DPVO initialization when no external prior
            # is requested.
            patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
            if self.is_initialized:
                s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
                patches[:,:,2] = s
            self.pg.metric_disps_[self.n].zero_()
            self.pg.metric_weights_[self.n].zero_()

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)
        if self.cfg.STEREO:
            self.fmap1_right_[:, self.n % self.mem] = F.avg_pool2d(
                right_fmap[0], 1, 1
            )
            self.fmap2_right_[:, self.n % self.mem] = F.avg_pool2d(
                right_fmap[0], 4, 4
            )

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M
        if self.cfg.METRIC_DEPTH:
            self.metric_keyframes_seen += 1
            enable_metric_prior_after_call = (
                not self.metric_prior_regularization_enabled
                and self.metric_keyframes_seen
                >= int(self.cfg.METRIC_DEPTH_PRIOR_WARMUP)
            )
        else:
            enable_metric_prior_after_call = False

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())
        if self.cfg.STEREO:
            stereo_kk, stereo_jj = self.__edges_stereo()
            self.append_factors(
                stereo_kk,
                stereo_jj,
                torch.ones_like(stereo_kk, dtype=torch.bool),
            )
            self.stereo_factor_count += int(stereo_kk.numel())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()


        # The first warm-up graph is optimized without the unary term. Once
        # the monotonic accepted-keyframe count reaches the threshold, latch
        # regularization on for subsequent updates even if compaction lowers n.
        if enable_metric_prior_after_call:
            self.metric_prior_regularization_enabled = True

    def force_global_ba_for_validation(self):
        """Exercise the native global-BA path in a truncated GPU smoke only.

        Formal inference never calls this method.  Natural loop factors remain
        data-dependent, so a short prefix can legitimately contain none; this
        hook validates the full active/inactive metric (and optional stereo)
        FastBA path without changing the production loop trigger.
        """
        if not self.cfg.LOOP_CLOSURE:
            raise RuntimeError("validation global BA requires LOOP_CLOSURE")
        if not self.is_initialized or self.n <= 1:
            raise RuntimeError("validation global BA requires an initialized graph")
        if self.pg.ii.numel() + self.pg.ii_inac.numel() <= 0:
            raise RuntimeError("validation global BA requires graph factors")
        self.__run_global_BA()

    def metric_prior_diagnostics(self):
        total = self.metric_prior_total_patches
        fractions = self.metric_prior_valid_pixel_fractions
        return {
            "valid_patches": int(self.metric_prior_valid_patches),
            "total_patches": int(total),
            "valid_patch_fraction": (
                float(self.metric_prior_valid_patches / total) if total else 0.0
            ),
            "mean_valid_pixel_fraction": (
                float(np.mean(fractions)) if fractions else 0.0
            ),
            "keyframes_seen": int(self.metric_keyframes_seen),
            "regularization_enabled": bool(
                self.metric_prior_regularization_enabled
            ),
            "successful_ba_calls_with_metric_prior": int(
                self.metric_prior_ba_calls
            ),
            "active_factors": int(self.pg.ii.numel()),
            "inactive_factors": int(self.pg.ii_inac.numel()),
            "active_stereo_factors": int(self.pg.edge_is_stereo.sum().item()),
            "inactive_stereo_factors": int(
                self.pg.edge_is_stereo_inac.sum().item()
            ),
            "stereo_factors_created": int(self.stereo_factor_count),
            "stereo_baseline_m": (
                float(self.stereo_baseline_m)
                if self.stereo_baseline_m is not None
                else None
            ),
            "global_ba_calls": int(self.global_ba_calls),
            "loop_closure_enabled": bool(self.cfg.LOOP_CLOSURE),
            "stereo_enabled": bool(self.cfg.STEREO),
        }
