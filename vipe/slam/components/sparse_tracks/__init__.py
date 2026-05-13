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

from abc import ABC, abstractmethod

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig

from vipe.streams.base import VideoFrame
from vipe.utils.depth import bilinear_splatting_inplace


class SparseTracks(ABC):
    """Note that the current design only supports single-camera for now"""

    # view_idx -> frame_idx -> {keypoint_idx -> uv}
    observations: list[list[dict[int, np.ndarray]]]
    enabled: bool = True

    def __init__(self, n_views: int):
        self.observations = [[] for _ in range(n_views)]

    @abstractmethod
    def track_image(self, frame_data_list: list[VideoFrame]) -> None: ...

    def get_correspondences(self, view_idx: int, source_frame_idx: int, target_frame_idx: int) -> torch.Tensor:
        """
        Returns:
            - keypoint_indices: The indices of the keypoints that are observed in both frames.
        """
        source_kps = set(self.observations[view_idx][source_frame_idx].keys())
        target_kps = set(self.observations[view_idx][target_frame_idx].keys())
        keypoint_indices = list(source_kps.intersection(target_kps))
        return torch.tensor(keypoint_indices)

    def get_observations(self, view_idx: int, frame_idx: int, keypoint_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_idx: The index of the view.
            frame_idx: The index of the frame.
            keypoint_indices: The indices of the keypoints to get the observations for.
        Returns:
            The observations for the given keypoints.
        """
        if len(keypoint_indices) == 0:
            return torch.empty(0, 2, device=keypoint_indices.device)
        uvs = self.observations[view_idx][frame_idx]
        return (
            torch.tensor(np.stack([uvs[kp_idx] for kp_idx in keypoint_indices.cpu().numpy()], axis=0))
            .to(keypoint_indices.device)
            .float()
        )

    def compute_dense_disp_target_weight(
        self,
        source_view_inds: torch.Tensor,
        source_frame_inds: torch.Tensor,
        target_view_inds: torch.Tensor,
        target_frame_inds: torch.Tensor,
        image_size: tuple[int, int],
        dense_disp_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_terms = len(source_view_inds)
        disp_h, disp_w = dense_disp_size
        uv_factor = torch.tensor(
            [disp_w / image_size[1], disp_h / image_size[0]],
            device=source_view_inds.device,
        )
        assert n_terms == len(target_view_inds) == len(source_frame_inds) == len(target_frame_inds), (
            "All indices must have the same length"
        )

        disp_value = torch.zeros(
            (n_terms, disp_h, disp_w, 2),
            dtype=torch.float32,
            device=source_view_inds.device,
        )
        disp_weight = torch.zeros(
            (n_terms, disp_h, disp_w),
            dtype=torch.float32,
            device=source_view_inds.device,
        )
        for term_idx, (s_vidx, s_fidx, t_vidx, t_fidx) in enumerate(
            zip(
                source_view_inds.cpu().numpy(),
                source_frame_inds.cpu().numpy(),
                target_view_inds.cpu().numpy(),
                target_frame_inds.cpu().numpy(),
            )
        ):
            assert s_vidx == t_vidx, "Only same view tracking is supported"
            kp_idx = self.get_correspondences(s_vidx, s_fidx, t_fidx)
            if len(kp_idx) == 0:
                continue
            uv_source = self.get_observations(s_vidx, s_fidx, kp_idx)
            uv_flow = self.get_observations(t_vidx, t_fidx, kp_idx) - uv_source

            bilinear_splatting_inplace(
                uv_flow.cuda() * uv_factor,
                uv_source.cuda() * uv_factor,
                disp_value[term_idx],
                disp_weight[term_idx],
            )

        disp_value /= disp_weight[..., None]
        disp_weight = disp_weight[..., None].repeat(1, 1, 1, 2)
        # If weight is 0, set to 0/whatever values since we don't care those positions.
        disp_value[torch.isnan(disp_value)] = 0.0
        # If weight is too small, then probably it's also not very reliable.
        disp_value[disp_weight < 0.1] = 0.0
        disp_weight[disp_weight < 0.1] = 0.0

        # We need to add original coordinates with the flow, so it becomes "target"
        y, x = torch.meshgrid(
            torch.arange(disp_h, device=disp_value.device),
            torch.arange(disp_w, device=disp_value.device),
            indexing="ij",
        )
        disp_value[..., 0] += x
        disp_value[..., 1] += y

        return disp_value, disp_weight


class DummySparseTracks(SparseTracks):
    enabled: bool = False

    def track_image(self, frame_data_list: list[VideoFrame]) -> None:
        for obs in self.observations:
            obs.append({})


def build_sparse_tracks(config: DictConfig, n_views: int) -> SparseTracks:
    if config.name == "dummy":
        return DummySparseTracks(n_views)

    if config.name == "cuvslam":
        from .cuvslam import CuVSLAMSparseTracks

        return CuVSLAMSparseTracks(n_views)

    raise ValueError(f"Unknown sparse tracks: {config.name}")
