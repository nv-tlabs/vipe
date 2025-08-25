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

import numpy as np
import rerun as rr
import torch

from omegaconf import DictConfig

from vipe.ext import lietorch as lt
from vipe.priors.depth.base import DepthEstimationInput
from vipe.priors.depth.unik3d import Unik3DModel
from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import (
    CachedVideoStream,
    CameraType,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.geometry import project_points_to_panorama, se3_to_so3, so3_to_se3
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import EquirectProjectionProcessor


logger = logging.getLogger(__name__)


class MergedPanoramaVideoStream(VideoStream):
    # Determines how much height of the panorama is kept for depth estimation.
    # This way we crop the top and bottom of the input to avoid distortion (where depth estimation is not reliable).
    DEPTH_KEEP_RATIO = 0.6

    def __init__(
        self,
        pano_stream: VideoStream,
        projected_streams: list[VideoStream],
        slam_output: SLAMOutput,
        pano_depth_method: str | None,
    ):
        assert len(projected_streams) > 0
        self.pano_stream = pano_stream
        self.projected_streams = projected_streams
        self.slam_output = slam_output
        self.pano_depth_method = pano_depth_method

        if self.pano_depth_method == "unik3d":
            self.pano_depth_model = Unik3DModel()

        else:
            self.pano_depth_model = None  # type: ignore

    def frame_size(self) -> tuple[int, int]:
        return self.pano_stream.frame_size()

    def fps(self) -> float:
        return self.pano_stream.fps()

    def __len__(self) -> int:
        return len(self.pano_stream)

    def __iter__(self):
        if self.pano_depth_method is not None:
            xyz_global, _ = self.slam_output.slam_map.get_dense_disp_full_pcd()
        else:
            xyz_global = None

        last_inv_scale = 1.0

        for frame_idx, (pano_frame_data, *projected_frame_data) in enumerate(
            zip(self.pano_stream, *self.projected_streams)
        ):
            pano_frame_data.intrinsics = torch.zeros(4).float().cuda()
            pano_frame_data.pose = self.slam_output.trajectory[frame_idx]
            pano_frame_data.camera_type = CameraType.PANORAMA

            if self.pano_depth_model is not None:
                height_crop = int(pano_frame_data.size()[0] * (1 - self.DEPTH_KEEP_RATIO) / 2)
                full_distance = torch.zeros(pano_frame_data.size()).cuda()
                croped_distance = self.pano_depth_model.estimate(
                    DepthEstimationInput(
                        rgb=pano_frame_data.rgb[height_crop:-height_crop],
                    )
                ).metric_depth

                # Align distance map
                assert xyz_global is not None
                xyz = pano_frame_data.pose.inv()[None].act(xyz_global)
                uvd = project_points_to_panorama(xyz, return_depth=True)
                uvd[:, 0] *= pano_frame_data.size()[1]
                uvd[:, 1] *= pano_frame_data.size()[0]
                target_depth = torch.zeros(pano_frame_data.size(), device="cuda")

                # Filter out-of-bounds indices to prevent CUDA assertion errors
                height, width = pano_frame_data.size()
                u_indices = uvd[:, 0].floor().long()
                v_indices = uvd[:, 1].floor().long()

                # Create valid mask for in-bounds indices
                valid_mask = (u_indices >= 0) & (u_indices < width) & (v_indices >= 0) & (v_indices < height)

                if valid_mask.any():
                    # Only assign values for valid indices
                    target_depth[v_indices[valid_mask], u_indices[valid_mask]] = uvd[valid_mask, 2]

                target_depth = target_depth[height_crop:-height_crop]
                target_mask = target_depth > 0

                # Additional safety check to ensure target_mask is valid
                if not target_mask.any():
                    logger.warning(f"No valid pixels in pano frame {frame_idx}, skipping scale estimation.")
                    inv_scale = last_inv_scale
                else:
                    # Check if target_mask contains any invalid values
                    if torch.isnan(target_mask).any() or torch.isinf(target_mask).any():
                        logger.warning(
                            f"Invalid mask values detected in pano frame {frame_idx}, skipping scale estimation."
                        )
                        inv_scale = last_inv_scale
                    elif target_mask.float().sum() < 0.05 * target_mask.numel():
                        logger.warning(f"Too few valid pixels in pano frame {frame_idx}, skipping scale estimation.")
                        inv_scale = last_inv_scale
                    else:
                        # Ensure we only use valid values for median calculation
                        valid_depth_mask = (
                            target_mask
                            & (target_depth > 0)
                            & (~torch.isnan(target_depth))
                            & (~torch.isinf(target_depth))
                        )
                        if valid_depth_mask.any():
                            try:
                                inv_scale = torch.median(
                                    croped_distance[valid_depth_mask] / target_depth[valid_depth_mask]
                                ).item()
                                last_inv_scale = inv_scale
                            except (RuntimeError, ValueError):
                                logger.warning(
                                    f"Scale estimation failed in pano frame {frame_idx}, using previous scale."
                                )
                                inv_scale = last_inv_scale
                        else:
                            logger.warning(
                                f"No valid depth values for scale estimation in pano frame {frame_idx}, skipping."
                            )
                            inv_scale = last_inv_scale

                full_distance[height_crop:-height_crop] = croped_distance / inv_scale
                pano_frame_data.metric_depth = full_distance

            yield pano_frame_data

    def attributes(self) -> set[FrameAttribute]:
        return {FrameAttribute.POSE, FrameAttribute.INTRINSICS}

    def name(self) -> str:
        return self.pano_stream.name()


class PanoramaAnnotationPipeline(Pipeline):
    def __init__(
        self,
        virtual: DictConfig,
        slam: DictConfig,
        output: DictConfig,
        post: DictConfig,
    ) -> None:
        super().__init__()
        virtual_height = virtual.height
        virtual_focal = virtual_height / (2 * np.tan(np.deg2rad(virtual.fovx) / 2))
        virtual_width = int(virtual_focal * np.tan(np.deg2rad(virtual.fovx) / 2) * 2)
        virtual_width = virtual_width + (virtual_width % 2)
        self.virtual_intrinsics = (
            torch.tensor(
                [virtual_focal, virtual_focal, virtual_width // 2, virtual_height // 2],
            )
            .float()
            .cuda()
        )
        self.virtual_size = (virtual_height, virtual_width)

        self.virtual_cfg = virtual
        self.slam_cfg = slam
        self.out_cfg = output
        self.post_cfg = post
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)

    def run(self, video_stream: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        assert isinstance(video_stream, VideoStream), "Panorama pipeline only supports single video stream"
        annotate_output = AnnotationPipelineOutput()
        artifact_path = io.ArtifactPath(self.out_path, video_stream.name())

        # Check whether the sample has been processed
        if artifact_path.meta_info_path.exists() and self.out_cfg.skip_exists:
            logger.info(f"{video_stream.name()} has been proccessed already, skip it!!")
            return annotate_output

        rig_transforms = [
            so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(yaw, 0.0))
            for yaw in np.linspace(0, 2 * np.pi, self.virtual_cfg.num_views, endpoint=False)
        ]
        if self.virtual_cfg.top:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, np.pi / 2)))
        if self.virtual_cfg.bottom:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, -np.pi / 2)))
        rig_names = ["left"] * len(rig_transforms)

        # Cache video stream on the fly so no need to maintain several readers.
        cached_video_stream = CachedVideoStream(video_stream)

        slam_streams: list[VideoStream] = []
        for rig_transform, rig_name in zip(rig_transforms, rig_names):
            projectors: list[StreamProcessor] = [
                EquirectProjectionProcessor(
                    se3_to_so3(rig_transform),
                    self.virtual_size,
                    self.virtual_intrinsics,
                )
            ]

            slam_streams.append(ProcessedVideoStream(cached_video_stream, projectors).cache(online=True))
        rig_se3 = lt.stack(rig_transforms, dim=0)

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=rig_se3)

        # For visualization, we append the visualization of the full panorama.
        if self.slam_cfg.visualize:
            for frame_idx, frame_data in enumerate(cached_video_stream):
                rr.set_time_sequence("frame", frame_idx)
                image = frame_data.rgb.cpu().numpy()
                rr.log(
                    "world/camera_360",
                    rr.Image((image * 255).astype(np.uint8)).compress(),
                )

        depth_align_model = self.post_cfg.depth_align_model
        output_stream = CachedVideoStream(
            MergedPanoramaVideoStream(
                video_stream,
                slam_streams,
                slam_output,
                pano_depth_method=depth_align_model,
            )
        )

        if self.out_cfg.save_artifacts:
            io.save_artifacts(artifact_path, output_stream)

            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            with artifact_path.meta_info_path.open("wb") as f:
                pickle.dump({"finished": True}, f)

        if self.out_cfg.save_viz:
            save_projection_video(
                artifact_path.meta_vis_path,
                output_stream,
                slam_output,
                self.out_cfg.viz_downsample,
                self.out_cfg.viz_attributes,
            )

        return annotate_output
