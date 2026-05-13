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

import torch
from omegaconf import DictConfig

from vipe.ext.lietorch import SE3

from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


class SLAMFrontend:
    """
    Frontend is called given every new frame. Currently it is a no-op for non-keyframe frames.
    For keyframe, it handles the system initialization and partial update logic (i.e. use BA to get pose for this kf).
    """

    def __init__(self, net: DroidNet, video: GraphBuffer, args: DictConfig, device: torch.device):
        self.video = video
        self.graph = FactorGraph(
            net,
            video,
            device,
            max_factors=48,
            incremental=True,
            cross_view=args.cross_view,
        )

        # Number of frames that the frontend has so far optimized.
        self.t1 = 0

        # frontend variables
        self.is_initialized = False

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        # Number of frames to wait before initializing (default 8)
        self.args = args
        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius
        self.has_init_pose = args.has_init_pose

    def __init_pose(self):
        assert self.t1 > 1
        p1 = SE3(self.video.poses[self.t1 - 2])
        p2 = SE3(self.video.poses[self.t1 - 1])
        w = (p2 * p1.inv()).log() * 0.5
        self.video.poses[self.t1] = (SE3.exp(w) * p2).data
        # self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()

    def __update(self):
        """add edges, perform update"""

        self.t1 += 1

        # t1 - 1 is the new-added frame
        # t1 - 2 is the previous frame
        # t1 - 3 is the frame before the previous frame

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(
            self.t1 - 5,
            max(self.t1 - self.frontend_window, 0),
            rad=self.frontend_radius,
            nms=self.frontend_nms,
            thresh=self.frontend_thresh,
            beta=self.beta,
            remove=True,
        )

        for _ in range(self.iters1):
            self.graph.update(use_inactive=True, fixed_motion=self.has_init_pose)

        # remove frame t1-2 if it is too close to t1-3, so the new keyframes will be [t1-3, t1-1]
        d = self.video.frame_distance_dense_disp(
            torch.tensor([self.t1 - 3]),
            torch.tensor([self.t1 - 2]),
            beta=self.beta,
            bidirectional=True,
        )
        if d.max().item() < self.keyframe_thresh:
            self.graph.rm_second_newest_keyframe(self.t1 - 2)
            self.t1 -= 1
        else:
            for _ in range(self.iters2):
                self.graph.update(use_inactive=True, fixed_motion=self.has_init_pose)

        # set pose for next itration
        if not self.has_init_pose:
            self.__init_pose()
        for v in range(self.video.n_views):
            self.video.disps[self.t1, v] = self.video.disps[self.t1 - 1, v].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min() : self.t1] = True

    def __initialize(self):
        """initialize the SLAM system with keyframes idx [t0, t1)"""

        self.t1 = self.video.n_frames

        self.graph.add_neighborhood_factors(0, self.t1, r=1 if self.args.seq_init else 3)
        for _ in range(8):
            self.graph.update(t0=1, use_inactive=True, fixed_motion=self.has_init_pose)

        if not self.args.seq_init:
            self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
            for _ in range(8):
                self.graph.update(t0=1, use_inactive=True, fixed_motion=self.has_init_pose)

        if not self.has_init_pose:
            self.__init_pose()
        for v in range(self.video.n_views):
            self.video.disps[self.t1, v] = self.video.disps[self.t1 - 4 : self.t1, v].mean()
        self.video.dirty[: self.t1] = True

        # initialization complete
        self.is_initialized = True
        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def run(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.n_frames == self.warmup:
            self.__initialize()

        # do update if new keyframe is added.
        elif self.is_initialized and self.t1 < self.video.n_frames:
            self.__update()
