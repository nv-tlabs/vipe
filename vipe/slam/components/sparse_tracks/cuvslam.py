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

import cuvslam as vslam
import numpy as np
import torch

from vipe.streams.base import VideoFrame
from vipe.utils.misc import unpack_optional

from . import SparseTracks


class CuVSLAMSparseTracks(SparseTracks):
    """Utilize the cuVSLAM backend for sparse tracking"""

    def __init__(self, n_views: int) -> None:
        super().__init__(n_views)
        self.tracker = None
        self.frame_idx = 0

    def _build_tracker(self, frame_data_list: list[VideoFrame]) -> None:
        assert len(frame_data_list) == 1, (
            "Only single-camera supported for now. Mainly due to rig transformations not properly set."
        )

        vslam_cameras = []
        for frame_data in frame_data_list:
            frame_height, frame_width = frame_data.size()
            fx, fy, cx, cy = unpack_optional(frame_data.intrinsics)

            cam = vslam.Camera()
            cam.distortion = vslam.Distortion()
            cam.distortion.model = vslam.Distortion.Model.Pinhole
            cam.focal = [float(fx), float(fy)]
            cam.principal = [float(cx), float(cy)]
            cam.size = [int(frame_width), int(frame_height)]
            cam.rig_from_camera = vslam.Pose()
            cam.rig_from_camera.rotation = np.array([0, 0, 0, 1])
            cam.rig_from_camera.translation = np.array([0, 0, 0])
            vslam_cameras.append(cam)

        rig = vslam.Rig()
        rig.cameras = vslam_cameras
        rig.imus = []

        cfg = vslam.Tracker.OdometryConfig()
        cfg.odometry_mode = vslam.Tracker.OdometryMode.Mono
        cfg.enable_observations_export = True
        tracker = vslam.Tracker(rig, cfg)

        self.tracker = tracker
        self.frame_idx = 0

    def track_image(self, frame_data_list: list[VideoFrame]) -> None:
        if self.tracker is None:
            self._build_tracker(frame_data_list)
        assert self.tracker is not None, "Tracker not initialized"

        invalid_masks = [
            (
                (~frame.mask).byte()
                if frame.mask is not None
                else torch.zeros(frame.size(), dtype=torch.uint8, device=frame.rgb.device)
            )
            for frame in frame_data_list
        ]
        self.tracker.track(
            self.frame_idx,
            [(frame.rgb * 255).byte().contiguous() for frame in frame_data_list],
            invalid_masks,
        )

        for camera_idx, observation in enumerate(self.observations):
            # Add new frame info
            observation.append({})
            # cuvslam will automatically prune observations being masked.
            for obs in self.tracker.get_last_observations(camera_idx):
                observation[self.frame_idx][obs.id] = np.array([obs.u, obs.v])
        self.frame_idx += 1
