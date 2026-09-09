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

import torch

from vipe.slam.longseq.system import LongSequenceSLAMSystem
from vipe.streams.base import MultiviewVideoList, ProcessedVideoStream, VideoStream
from vipe.utils import io

from . import AnnotationPipelineOutput
from .pose_only import PoseOnlyAnnotationPipeline

logger = logging.getLogger(__name__)


class _StreamingProcessedVideoStream(ProcessedVideoStream):
    """A ``ProcessedVideoStream`` whose ``.cache()`` is a no-op passthrough.

    ``pose_only``'s ``run()`` calls ``.cache(..., online=True)`` to materialize
    the whole processed video as CPU-resident frames before handing it to SLAM
    -- this is O(sequence length) host memory. The long-sequence recipe reads
    the stream in a single pass (see ``LongSequenceSLAMSystem``), so nothing
    needs to be materialized: iterating this stream directly already overlaps
    correctly with the sliding-window SLAM consumer. Overriding ``cache()`` to
    return ``self`` keeps this class a drop-in ``VideoStream`` for the exact
    call shape ``PoseOnlyAnnotationPipeline.run()`` uses.
    """

    def cache(  # type: ignore[override]
        self,
        desc: str = "Caching",
        online: bool = False,
        async_prefetch: bool = False,
        prefetch_queue_size: int = 16,
    ) -> "_StreamingProcessedVideoStream":
        return self


class PoseOnlyLongAnnotationPipeline(PoseOnlyAnnotationPipeline):
    """Pose-only pipeline for arbitrarily long sequences with bounded GPU/CPU memory.

    Identical to ``PoseOnlyAnnotationPipeline`` except for two things: the
    per-frame init stream (intrinsics + instance masks) is consumed in a single
    streaming pass instead of being cached in full in CPU memory, and SLAM runs
    through ``LongSequenceSLAMSystem`` (a sliding-window recipe selected by
    ``slam.window``) instead of the standard ``SLAMSystem``. Everything else --
    keyframe metric-depth prior, dynamic-object masking, artifact saving -- is
    unchanged, so this class intentionally duplicates
    ``PoseOnlyAnnotationPipeline.run()`` rather than modifying it in place.
    """

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

        slam_streams: list[VideoStream] = []
        for video_stream in video_streams:
            init_stream = self._add_init_processors(video_stream)
            streaming_stream = _StreamingProcessedVideoStream(init_stream.stream, init_stream.processors)
            slam_streams.append(
                streaming_stream.cache(
                    "process",
                    online=True,
                    async_prefetch=self.init_cfg.async_prefetch,
                    prefetch_queue_size=self.init_cfg.prefetch_queue_size,
                )
            )

        slam_pipeline = LongSequenceSLAMSystem(
            device=torch.device("cuda"), config=self.slam_cfg, model_cache=self.model_cache
        )
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
