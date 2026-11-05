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
import torch
from einops import asnumpy, reduce, repeat

from . import projective_ops as pops
from .lietorch import SE3
from .edge_utils import reduce_edges
from .utils import *


class PatchGraph:
    """ Dataclass for storing variables """

    def __init__(self, cfg, P, DIM, pmem, **kwargs):
        self.cfg = cfg
        self.P = P
        self.pmem = pmem
        self.DIM = DIM

        self.n = 0      # number of frames
        self.m = 0      # number of patches

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.tstamps_ = np.zeros(self.N, dtype=np.int64)
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        # Optional external inverse-depth observations, one value per patch.
        # A zero weight means that no metric prior is available for the patch.
        # Keeping the observations beside the patch graph is important because
        # keyframe removal compacts all frame-indexed state in-place.
        self.metric_disps_ = torch.zeros(self.N, self.M, dtype=torch.float, device="cuda")
        self.metric_weights_ = torch.zeros(self.N, self.M, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        ### edge information ###
        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        # True marks a same-timestamp left-to-right fixed-rig observation.
        # Stereo edges share the frame indices used by temporal edges, but
        # project through the right camera and never update a trajectory pose.
        self.edge_is_stereo = torch.as_tensor(
            [], dtype=torch.bool, device="cuda"
        )

        ### inactive edge information (i.e., no longer updated, but useful for BA) ###
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.edge_is_stereo_inac = torch.as_tensor(
            [], dtype=torch.bool, device="cuda"
        )
        self.weight_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")
        self.target_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")

    def edges_loop(self):
        """ Adding edges from old patches to new frames """
        lc_range = self.cfg.MAX_EDGE_AGE
        l = self.n - self.cfg.REMOVAL_WINDOW # l is the upper bound for "old" patches

        if l <= 0:
            return torch.empty(2, 0, dtype=torch.long, device='cuda')

        # create candidate edges
        jj, kk = flatmeshgrid(
            torch.arange(self.n - self.cfg.GLOBAL_OPT_FREQ, self.n - self.cfg.KEYFRAME_INDEX, device="cuda"),
            torch.arange(max(l - lc_range, 0) * self.M, l * self.M, device="cuda"), indexing='ij')
        ii = self.ix[kk]

        # Remove edges which have too large flow magnitude
        flow_mg, val = pops.flow_mag(SE3(self.poses), self.patches[...,1,1].view(1,-1,3,1,1), self.intrinsics, ii, jj, kk, beta=0.5)
        flow_mg_sum = reduce(flow_mg * val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).float()
        num_val = reduce(val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).clamp(min=1)
        flow_mag = torch.where(num_val > (self.M * 0.75), flow_mg_sum / num_val, torch.inf)

        mask = (flow_mag < self.cfg.BACKEND_THRESH)
        es = reduce_edges(asnumpy(flow_mag[mask]), asnumpy(ii[::self.M][mask]), asnumpy(jj[::self.M][mask]), max_num_edges=1000, nms=1)

        edges = torch.as_tensor(es, device=ii.device)
        ii, jj = repeat(edges, 'E ij -> ij E M', M=self.M, ij=2)
        kk = ii.mul(self.M) + torch.arange(self.M, device=ii.device)
        return kk.flatten(), jj.flatten()

    def normalize(self):
        """ normalize depth and poses """
        s = self.patches_[:self.n,:,2].mean()
        self.patches_[:self.n,:,2] /= s
        self.poses_[:self.n,:3] *= s
        for t, (t0, dP) in self.delta.items():
            self.delta[t] = (t0, dP.scale(s))
        self.poses_[:self.n] = (SE3(self.poses_[:self.n]) * SE3(self.poses_[[0]]).inv()).data

        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)
