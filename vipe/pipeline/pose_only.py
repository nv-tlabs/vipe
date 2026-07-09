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


import logging
import pickle

import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import MultiviewVideoList, VideoStream
from vipe.utils import io
from vipe.utils.cameras import CameraType

from . import AnnotationPipelineOutput, Pipeline
from .default import DefaultAnnotationPipeline

logger = logging.getLogger(__name__)


class PoseOnlyAnnotationPipeline(DefaultAnnotationPipeline):
    """Fast pipeline that estimates per-frame camera poses and intrinsics only.

    Compared to the default pipeline this skips all per-frame dense depth work:
    the depth-alignment post-processing pass and the depth/mask artifact dump.
    SLAM (including its keyframe-only metric depth prior and dynamic-object
    masking) is unchanged, so the recovered trajectory is identical to the one
    produced by the default pipeline.
    """

    def __init__(self, init: DictConfig, slam: DictConfig, output: DictConfig) -> None:
        super().__init__(init, slam, post=OmegaConf.create({"depth_align_model": None}), output=output)

    def _save_pose_artifacts(
        self, artifact_path: io.ArtifactPath, view_idx: int, slam_output: SLAMOutput, n_frames: int
    ) -> None:
        # Written directly from the SLAM output so no extra pass over the
        # video frames is needed. Formats match io.save_artifacts.
        trajectory = slam_output.get_view_trajectory(view_idx)
        pose_data = trajectory.matrix().cpu().numpy()
        pose_inds = np.arange(n_frames)
        artifact_path.pose_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(artifact_path.pose_path, data=pose_data, inds=pose_inds)

        intrinsics = slam_output.intrinsics[view_idx].cpu().numpy()
        intrinsics_data = np.repeat(intrinsics[None], n_frames, axis=0)
        artifact_path.intrinsics_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(artifact_path.intrinsics_path, data=intrinsics_data, inds=pose_inds)

        with artifact_path.camera_type_path.open("w") as f:
            for frame_idx in range(n_frames):
                f.write(f"{frame_idx}: {self.camera_type.name}\n")

        artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
        with artifact_path.meta_info_path.open("wb") as f:
            pickle.dump({"ba_residual": slam_output.ba_residual}, f)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            slam_rig = video_data.rig()
        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            slam_rig = None

        artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache(
                "process",
                online=True,
                async_prefetch=self.init_cfg.async_prefetch,
                prefetch_queue_size=self.init_cfg.prefetch_queue_size,
            )
            for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg, model_cache=self.model_cache)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        for view_idx, (slam_stream, artifact_path) in enumerate(zip(slam_streams, artifact_paths)):
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving pose artifacts to {artifact_path.pose_path}")
                self._save_pose_artifacts(artifact_path, view_idx, slam_output, len(slam_stream))

            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = [
                self._add_post_processors(view_idx, slam_stream, slam_output)
                for view_idx, slam_stream in enumerate(slam_streams)
            ]

        return annotate_output
