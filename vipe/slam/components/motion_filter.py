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

from dataclasses import dataclass

import torch

from vipe.utils.nvtx import nvtx_range

from ..networks.droid_net import CorrBlock, DroidNet
from .sparse_tracks import SparseTracks


@dataclass(slots=True)
class MotionFilterResult:
    add_keyframe: bool
    fmap: torch.Tensor
    net: torch.Tensor | None = None
    inp: torch.Tensor | None = None


class MotionFilter:
    """
    This class is used to filter incoming frames and extract features.
    This module re-uses Droid's network to detect scene changes without considering the mask.
    For multi-view input, spawn keyframes if any of the views exceed the threshold.
    """

    def __init__(
        self,
        droid_net: DroidNet,
        sparse_tracks: SparseTracks,
        thresh: float,
        device: torch.device = torch.device("cuda"),
    ):
        self.net = droid_net
        self.thresh = thresh
        self.device = device
        self.sparse_tracks = sparse_tracks
        self.initialized = False

    @staticmethod
    def coords_grid(ht, wd, **kwargs):
        y, x = torch.meshgrid(
            torch.arange(ht).to(**kwargs).float(),
            torch.arange(wd).to(**kwargs).float(),
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1)

    @torch.amp.autocast("cuda", enabled=True)
    @torch.no_grad()
    def check(self, images: torch.Tensor, buffer_masks: torch.Tensor | None) -> MotionFilterResult:
        """
        main update operation - run on every frame in video

        Args:
            image (torch.Tensor): VCHW image RGB 0-1
            buffer_masks (torch.Tensor): Vhw mask 1-invalid, 0-valid

        Returns:
            Motion-filter decision and any DROID tensors already computed for the current frame.
        """

        with nvtx_range("slam.motion_filter.check"):
            num_views = images.shape[0]
            ht = images.shape[-2] // 8
            wd = images.shape[-1] // 8

            # extract features (subsequent depth will also work on this resolution)
            with nvtx_range("slam.motion_filter.check.encode_features"):
                gmap = self.net.encode_features(images)  # (V, 128, ht//8, wd//8)

            ### always add first frame to the depth video ###
            if not self.initialized:
                # (V, 128, ht//8, wd//8) x2
                with nvtx_range("slam.motion_filter.check.init_context"):
                    net, inp = self.net.encode_context(images)
                # Store features of the last keyframe.
                self.f_net, self.f_inp, self.f_fmap = net, inp, gmap
                self.f_mask = buffer_masks
                self.current_frame_idx = 0
                self.last_kf_frame_idx = 0
                self.last_n_sparse_tracks = 0
                self.initialized = True
                return MotionFilterResult(add_keyframe=True, fmap=gmap, net=net, inp=inp)

            ### only add new frame if there is enough motion ###
            else:
                self.current_frame_idx += 1

                # index correlation volume (1, 1, ht//8, wd//8, 2)
                coords0 = self.coords_grid(ht, wd, device=self.device)[None, None]
                coords0 = coords0.repeat(1, images.shape[0], 1, 1, 1)

                # compute cost volume using current frame and the last keyframe.
                # (1, V, 196, ht//8, wd//8)
                with nvtx_range("slam.motion_filter.check.corr"):
                    corr = CorrBlock(self.f_fmap[None], gmap[None])(coords0)

                # approximate flow magnitude using 1 update iteration
                with nvtx_range("slam.motion_filter.check.update_forward"):
                    _, delta, weight = self.net.update.forward(self.f_net[None], self.f_inp[None], corr)
                # flow-delta and weight: (1, V, ht//8, wd//8, 2)

                with nvtx_range("slam.motion_filter.check.dense_score"):
                    dense_flow = delta.norm(dim=-1)[0]
                    if self.f_mask is not None:
                        f_weight = (~self.f_mask).float()
                        dense_motion_score = (dense_flow * f_weight).mean([1, 2]) / (f_weight.mean([1, 2]) + 1e-6)
                    else:
                        dense_motion_score = dense_flow.mean([1, 2])
                    # Across views [max is the most conservative, while min don't add a lot of KFs]
                    dense_motion_score = dense_motion_score.min().item()

                # Mask for sparse tracks is already considered during tracking.
                sparse_motion_score: float = 0.0
                if self.sparse_tracks.enabled:
                    with nvtx_range("slam.motion_filter.check.sparse_score"):
                        n_sparse_tracks: int = 0

                        for view_idx in range(num_views):
                            kp_idx = self.sparse_tracks.get_correspondences(
                                view_idx, self.current_frame_idx, self.last_kf_frame_idx
                            )
                            n_sparse_tracks += len(kp_idx)
                            current_kp = self.sparse_tracks.get_observations(view_idx, self.current_frame_idx, kp_idx)
                            last_kp = self.sparse_tracks.get_observations(view_idx, self.last_kf_frame_idx, kp_idx)
                            sparse_delta = (current_kp - last_kp).norm(dim=-1).mean().item()
                            sparse_motion_score += sparse_delta

                        # If smaller than 0, either new kps are added or this is the next frame after kp is inserted
                        sparse_tracks_diff = n_sparse_tracks - self.last_n_sparse_tracks
                        # Force add keyframe if 20% of the sparse tracks are diminished
                        if sparse_tracks_diff < 0 and self.last_n_sparse_tracks > 0:
                            diff_ratio = -sparse_tracks_diff / self.last_n_sparse_tracks
                            if diff_ratio > 0.2:
                                sparse_motion_score += 100.0  # Arbitrary large number

                        self.last_n_sparse_tracks = n_sparse_tracks

                # check motion magnitue / add new frame to video
                if (
                    dense_motion_score > self.thresh
                    or sparse_motion_score > self.thresh * 2  # Larger threshold since we don't avg across pixels
                ):
                    with nvtx_range("slam.motion_filter.check.accept_context"):
                        net, inp = self.net.encode_context(images)
                    self.f_net, self.f_inp, self.f_fmap = net, inp, gmap
                    self.f_mask = buffer_masks
                    self.last_kf_frame_idx = self.current_frame_idx
                    self.last_n_sparse_tracks = 0
                    return MotionFilterResult(add_keyframe=True, fmap=gmap, net=net, inp=inp)

                else:
                    return MotionFilterResult(add_keyframe=False, fmap=gmap)
