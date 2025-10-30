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
import os
import pickle

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.profiler import profile, record_function, ProfilerActivity

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import AdaptiveDepthProcessor, GeoCalibIntrinsicsProcessor, TrackAnythingProcessor


logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig, preloaded_gc_models: dict | None = None, preloaded_slam_models: dict | None = None, preloaded_unidepth=None, preloaded_sam=None, preloaded_aot=None) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)
        self.preloaded_gc_models = preloaded_gc_models  # For Modal optimization
        self.preloaded_slam_models = preloaded_slam_models  # For Modal optimization (DroidNet)
        self.preloaded_unidepth = preloaded_unidepth  # For Modal optimization (UniDepthV2)
        self.preloaded_sam = preloaded_sam  # For Modal optimization (SAM)
        self.preloaded_aot = preloaded_aot  # For Modal optimization (AOT)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                    preloaded_sam=self.preloaded_sam,
                    preloaded_aot=self.preloaded_aot,
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            post_processors.append(
                AdaptiveDepthProcessor(
                    slam_output, 
                    view_idx, 
                    depth_align_model,
                    preloaded_gc_models=self.preloaded_gc_models,
                    preloaded_unidepth=self.preloaded_unidepth
                )
            )
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        # Check if profiling is enabled
        enable_profiling = os.getenv("VIPE_PROFILE", "0") == "1"
        
        if enable_profiling:
            logger = logging.getLogger(__name__)
            logger.info("🔍 PyTorch profiling enabled for pipeline")
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(warmup=1, active=3, repeat=2),
                record_shapes=True,
                with_stack=True,
            ) as prof:
                return self._run_with_profiling(video_data, prof)
        else:
            return self._run_without_profiling(video_data)
    
    def _run_without_profiling(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        # Create SLAM system (pass preloaded models if available for Modal optimization)
        # Combine SLAM models (DroidNet) with UniDepthV2 for keyframe depth
        preloaded_slam = self.preloaded_slam_models.copy() if self.preloaded_slam_models else {}
        if self.preloaded_unidepth is not None:
            preloaded_slam['unidepth'] = self.preloaded_unidepth
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg, preloaded_models=preloaded_slam)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Dumping artifacts for all views in the streams
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(artifact_path, output_stream)
                with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            if self.out_cfg.save_viz:
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )

            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
    
    def _run_with_profiling(self, video_data: VideoStream | MultiviewVideoList, prof: torch.profiler.profile) -> AnnotationPipelineOutput:
        logger = logging.getLogger(__name__)
        
        with record_function("pipeline_initialization"):
            if isinstance(video_data, MultiviewVideoList):
                video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
                artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
                slam_rig = video_data.rig()

            else:
                assert isinstance(video_data, VideoStream)
                video_streams = [video_data]
                artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
                slam_rig = None

            annotate_output = AnnotationPipelineOutput()

            if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
                logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
                return annotate_output

        with record_function("initialization_processors"):
            slam_streams: list[VideoStream] = [
                self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
            ]

        with record_function("SLAM_pipeline_execution"):
            # Combine SLAM models (DroidNet) with UniDepthV2 for keyframe depth
            preloaded_slam = self.preloaded_slam_models.copy() if self.preloaded_slam_models else {}
            if self.preloaded_unidepth is not None:
                preloaded_slam['unidepth'] = self.preloaded_unidepth
            slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg, preloaded_models=preloaded_slam)
            slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        with record_function("post_processors"):
            output_streams = [
                self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
                for view_idx, slam_stream in enumerate(slam_streams)
            ]

        with record_function("artifact_saving"):
            # Dumping artifacts for all views in the streams
            for output_stream, artifact_path in zip(output_streams, artifact_paths):
                artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
                if self.out_cfg.save_artifacts:
                    logger.info(f"Saving artifacts to {artifact_path}")
                    io.save_artifacts(artifact_path, output_stream)
                    with artifact_path.meta_info_path.open("wb") as f:
                        pickle.dump({"ba_residual": slam_output.ba_residual}, f)

                if self.out_cfg.save_viz:
                    save_projection_video(
                        artifact_path.meta_vis_path,
                        output_stream,
                        slam_output,
                        self.out_cfg.viz_downsample,
                        self.out_cfg.viz_attributes,
                    )

                if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                    logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                    slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        # Log profiler statistics
        prof.step()
        self._log_profiler_stats(prof, logger)

        return annotate_output
    
    def _log_profiler_stats(self, prof: torch.profiler.profile, logger):
        """Log profiler statistics in a structured format"""
        logger.info("🔍 PyTorch Profiler Statistics (Pipeline):")
        logger.info("=" * 60)
        
        # Get profiler events
        events = prof.events()
        
        # Group events by function name
        function_stats = {}
        for event in events:
            if event.name not in function_stats:
                function_stats[event.name] = {
                    'total_time': 0,
                    'count': 0,
                    'cpu_time': 0,
                    'cuda_time': 0
                }
            
            function_stats[event.name]['total_time'] += event.cpu_time_total
            function_stats[event.name]['count'] += 1
            function_stats[event.name]['cpu_time'] += event.cpu_time
            function_stats[event.name]['cuda_time'] += event.cuda_time
        
        # Sort by total time
        sorted_functions = sorted(function_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        total_time = sum(stats['total_time'] for stats in function_stats.values())
        
        logger.info(f"{'Function':<30} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<12} {'%':<8}")
        logger.info("-" * 80)
        
        for func_name, stats in sorted_functions:
            if stats['total_time'] > 0:  # Only show functions with actual time
                percentage = (stats['total_time'] / total_time) * 100
                avg_time = stats['total_time'] / stats['count']
                logger.info(f"{func_name:<30} {stats['count']:<8} {stats['total_time']/1000:<12.2f} {avg_time/1000:<12.2f} {percentage:<8.1f}")
        
        logger.info("=" * 60)
