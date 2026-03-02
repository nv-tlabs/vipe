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
from pathlib import Path

import torch
from omegaconf import DictConfig

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
from .processors import (
    AdaptiveDepthProcessor,
    DynamicMaskProcessor,
    GeoCalibIntrinsicsProcessor,
    MultiviewDepthProcessor,
    TrackAnythingProcessor,
)

logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        init_processors.append(
            GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type, model_cache=self.model_cache)
        )
        if self.init_cfg.instance is not None:
            mode = self.init_cfg.instance.get("mode", "text")
            if mode == "motion":
                init_processors.append(
                    DynamicMaskProcessor(
                        sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                        sam_points_per_side=self.init_cfg.instance.get("sam_points_per_side", 16),
                        keep_motion_ratio=self.init_cfg.instance.get("keep_motion_ratio", 0.15),
                        flow_alpha=self.init_cfg.instance.get("flow_alpha", 0.5),
                        flow_beta=self.init_cfg.instance.get("flow_beta", 0.5),
                    )
                )
            else:
                init_processors.append(
                    TrackAnythingProcessor(
                        self.init_cfg.instance.phrases,
                        add_sky=self.init_cfg.instance.add_sky,
                        sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                        track_downscale=self.init_cfg.instance.get("track_downscale", 1),
                        track_stride=self.init_cfg.instance.get("track_stride", 1),
                        model_cache=self.model_cache,
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
            if depth_align_model.startswith("mvd_"):
                post_processors.append(MultiviewDepthProcessor(slam_output, model=depth_align_model))
            elif depth_align_model == "video_pi3x_moge2":
                from vipe.pipeline.processors import VideoPi3XDepthProcessor
                post_processors.append(VideoPi3XDepthProcessor())
            else:
                post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
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

        async_prefetch = self.init_cfg.async_prefetch
        prefetch_queue_size = self.init_cfg.prefetch_queue_size
        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache(
                "process",
                online=True,
                async_prefetch=async_prefetch,
                prefetch_queue_size=prefetch_queue_size,
            )
            for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg, model_cache=self.model_cache)
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

            if self.out_cfg.get("save_tsdf_ply", False):
                try:
                    import open3d as o3d
                    import numpy as np

                    logger.info(f"Generating TSDF volume for {artifact_path.artifact_name}")
                    volume = o3d.pipelines.integration.ScalableTSDFVolume(
                        voxel_length=self.out_cfg.get("tsdf_voxel_length", 0.005),
                        sdf_trunc=self.out_cfg.get("tsdf_sdf_trunc", 0.02),
                        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                    )

                    for frame_idx, frame_data in enumerate(output_stream):
                        if frame_data.pose is None or frame_data.metric_depth is None or frame_data.intrinsics is None:
                            continue

                        rgb = (frame_data.rgb.cpu().numpy() * 255).astype(np.uint8)
                        depth = frame_data.metric_depth.cpu().numpy()

                        # --- Dynamic Object Masking ---
                        if self.out_cfg.get("tsdf_apply_dynamic_mask", True):
                            # Check if ViPE generated a mask for moving objects
                            if hasattr(frame_data, 'mask') and frame_data.mask is not None:
                                # Convert mask to numpy (usually 1 for static, 0 for dynamic/ignored)
                                valid_mask = frame_data.mask.cpu().numpy().astype(bool)
                                # Force dynamic pixels to 0.0 so Open3D's TSDF completely ignores them
                                depth[~valid_mask] = 0.0
                        # ------------------------------

                        # --- Relative Depth Edge Pruning ---
                        if self.out_cfg.get("tsdf_apply_edge_pruning", False):
                            import cv2
                            grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
                            grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
                            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                            
                            # Calculate RELATIVE gradient (gradient divided by absolute depth)
                            # Adding 1e-6 prevents division by zero
                            relative_grad = grad_mag / (depth + 1e-6)
                            
                            prune_thresh = self.out_cfg.get("tsdf_pruning_threshold", 0.10)
                            edge_mask = relative_grad > prune_thresh  
                            depth[edge_mask] = 0.0
                        # -----------------------------------

                        rgb_o3d = o3d.geometry.Image(rgb)
                        depth_o3d = o3d.geometry.Image(depth)
                        
                        # Note: depth_scale MUST be 1.0 because ViPE outputs metric depth
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
                        )

                        fx, fy, cx, cy = frame_data.intrinsics.cpu().numpy()
                        h, w = rgb.shape[:2]
                        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

                        c2w = frame_data.pose.matrix().cpu().numpy()
                        w2c = np.linalg.inv(c2w)

                        volume.integrate(rgbd, intrinsic, w2c)

                        # --- Explicit Memory Management ---
                        # Force Python to release these arrays immediately to prevent OOM on long videos
                        del rgb, depth, rgb_o3d, depth_o3d, rgbd, intrinsic, c2w, w2c
                        # ----------------------------------

                    tsdf_mesh_path = artifact_path.meta_info_path.parent / f"{artifact_path.artifact_name}_tsdf.ply"
                    pcd = volume.extract_point_cloud()

                    # --- Statistical Outlier Removal (SOR) ---
                    if self.out_cfg.get("tsdf_apply_sor", True):
                        nb_neighbors = self.out_cfg.get("tsdf_sor_nb_neighbors", 50)
                        std_ratio = self.out_cfg.get("tsdf_sor_std_ratio", 2.0)
                        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
                        pcd = pcd.select_by_index(ind)
                    # -----------------------------------------

                    # --- Coordinate System Flip ---
                    # ViPE/Open3D use OpenCV format (Y-down). If the backend expects OpenGL (Y-up), we flip X by 180 degrees.
                    if self.out_cfg.get("tsdf_flip_x_axis", False):
                        logger.info("Flipping TSDF point cloud 180 degrees around X-axis for OpenGL compatibility")
                        # 180 degree rotation around X-axis: [1, 0, 0], [0, -1, 0], [0, 0, -1]
                        R_x_180 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
                        pcd.rotate(R_x_180, center=(0, 0, 0))
                    # ------------------------------

                    o3d.io.write_point_cloud(str(tsdf_mesh_path), pcd)
                    logger.info(f"Saved TSDF point cloud to {tsdf_mesh_path}")
                except ImportError:
                    logger.error("Open3D is not installed. Please install open3d to generate TSDF ply.")

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
