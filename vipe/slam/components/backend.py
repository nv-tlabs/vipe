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

from vipe.priors.depth import DepthEstimationModel

from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


class SLAMBackend:
    """
    Mainly used to run a pretty dense bundle adjustment for all the frames in the graph.
    """

    depth_model: DepthEstimationModel | None = None

    def __init__(self, net: DroidNet, video: GraphBuffer, args: DictConfig, device: torch.device):
        self.net = net
        self.video = video
        self.args = args
        self.device = device

    def _iterate_with_depth(self, graph: FactorGraph, steps: int, more_iters: bool):
        steps_preintr = steps // 2
        steps_postintr = steps - steps_preintr
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps_preintr,
            optimize_intrinsics=self.args.optimize_intrinsics,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )
        self.video.update_disps_sens(self.depth_model, frame_idx=None)
        # Don't update intrinsics again!
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps_postintr,
            optimize_intrinsics=False,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )

    def _iterate_without_depth(self, graph: FactorGraph, steps: int, more_iters: bool):
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps,
            optimize_intrinsics=self.args.optimize_intrinsics,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )

    @torch.no_grad()
    def run(self, steps: int = 12, update_depth: bool = True, log: bool = False):
        """main update (reset GRU state)"""

        t = self.video.n_frames

        print("build factor graph.")

        graph = FactorGraph(
            self.net,
            self.video,
            self.device,
            max_factors=16 * t,
            incremental=False,
            cross_view=self.args.cross_view,
        )

        print("add_proximity_factors.")

        graph.add_proximity_factors(
            rad=self.args.backend_radius,
            nms=self.args.backend_nms,
            thresh=self.args.backend_thresh,
            beta=self.args.beta,
        )

        print("build_adaptive_cross_view_idx.")

        if self.args.adaptive_cross_view:
            self.video.build_adaptive_cross_view_idx()


        if len(graph.ii) > 0:
            more_iters = self.args.optimize_intrinsics or self.args.optimize_rig_rotation
            if self.depth_model is not None:
                print("_iterate_with_depth.")
                self._iterate_with_depth(graph, steps, more_iters)
            else:
                print("_iterate_without_depth.")
                self._iterate_without_depth(graph, steps, more_iters)
        else:
            print("Empty graph with only one keyframe, assign sensor depth.")
            # Empty graph with only one keyframe, assign sensor depth
            self.video.disps[0] = torch.where(
                self.video.disps_sens[0] > 0,
                self.video.disps_sens[0],
                self.video.disps[0],
            )

        self.video.dirty[:t] = True

        print("over.")

        if log:
            self.video.log(self.args.map_filter_thresh)
            graph.log()

    @torch.no_grad()
    def run_if_necessary(self, steps: int = 12, log: bool = False):
        if self.args.optimize_intrinsics or self.args.optimize_rig_rotation:
            self.run(steps=steps, update_depth=True, log=log)
