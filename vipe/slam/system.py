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

import uuid

import numpy as np
import rerun as rr
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from vipe.ext.lietorch import SE3
from vipe.priors.depth import make_depth_model
from vipe.priors.depth.base import DepthType
from vipe.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional

from .components.backend import SLAMBackend
from .components.buffer import GraphBuffer
from .components.frontend import SLAMFrontend
from .components.inner_filler import FilledReturn, InnerFiller
from .components.motion_filter import MotionFilter
from .components.sparse_tracks import build_sparse_tracks
from .interface import SLAMOutput
from .networks.droid_net import DroidNet


class StandardResizeStreamProcessor(StreamProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.fac_x, self.fac_y = 1.0, 1.0

    def _compute_frame_size_crop(self, previous_frame_size: tuple[int, int]):
        h0, w0 = previous_frame_size
        scale_factor = np.sqrt((384 * 512) / (h0 * w0))
        h1 = int(h0 * scale_factor)
        w1 = int(w0 * scale_factor)

        crop_h, crop_w = h1 % 8, w1 % 8
        crop_top, crop_bottom = crop_h // 2, crop_h - crop_h // 2
        crop_left, crop_right = crop_w // 2, crop_w - crop_w // 2

        self.fac_x, self.fac_y = w0 / w1, h0 / h1
        self.scx, self.scy = crop_left, crop_top
        return (h1, w1), (crop_top, crop_bottom, crop_left, crop_right)

    def update_frame_size(self, previous_frame_size: tuple[int, int]):
        (h1, w1), (crop_top, crop_bottom, crop_left, crop_right) = self._compute_frame_size_crop(previous_frame_size)
        return h1 - (crop_top + crop_bottom), w1 - (crop_left + crop_right)

    def __call__(self, frame_idx: int, frame_data: VideoFrame) -> VideoFrame:
        (h1, w1), (crop_top, crop_bottom, crop_left, crop_right) = self._compute_frame_size_crop(frame_data.size())
        frame_data = frame_data.resize((h1, w1))
        frame_data = frame_data.crop(top=crop_top, bottom=crop_bottom, left=crop_left, right=crop_right)
        return frame_data

    def recover_intrinsics(self, after_intrinsics: torch.Tensor) -> torch.Tensor:
        new_intrinsics = after_intrinsics.clone()
        new_intrinsics[2] += self.scx
        new_intrinsics[3] += self.scy
        new_intrinsics[0:4:2] *= self.fac_x
        new_intrinsics[1:4:2] *= self.fac_y
        return new_intrinsics


class SLAMSystem:
    """Solver-defined SLAM"""

    def __init__(self, device: torch.device, config: DictConfig) -> None:
        self.device = device
        self.visualize = config.visualize
        self.config = config.copy()
        OmegaConf.set_struct(self.config, False)

    def _build_components(self):
        self.droid_net = DroidNet().to(self.device)
        self.sparse_tracks = build_sparse_tracks(self.config.sparse_tracks, self.config.n_views)
        self.buffer = GraphBuffer(
            height=self.config.height,
            width=self.config.width,
            n_views=self.config.n_views,
            buffer_size=self.config.buffer,
            init_disp=self.config.init_disp,
            cross_view_idx=self.config.get("cross_view_idx", None),
            ba_config=self.config.ba,
            sparse_tracks=self.sparse_tracks,
            camera_type=self.config.camera_type,
            device=self.device,
        )
        self.buffer.rig[:] = self.rig.to(self.device).data
        self.motion_filter = MotionFilter(
            self.droid_net,
            sparse_tracks=self.sparse_tracks,
            thresh=self.config.filter_thresh,
            device=self.device,
        )
        self.frontend = SLAMFrontend(self.droid_net, self.buffer, self.config, device=self.device)
        self.backend = SLAMBackend(self.droid_net, self.buffer, self.config, device=self.device)
        self.inner_filler = InnerFiller(self.droid_net, self.buffer, self.config, device=self.device)

        if self.config.keyframe_depth is not None:
            assert self.config.n_views == 1, """Currently the global scale lies in the null-space of the SLAM problem. 
            Adding more views requires adding factors to the graph to keep the null-space. 
            This is currently not supported for now."""

            self.metric_depth = make_depth_model(self.config.keyframe_depth)
            assert self.metric_depth.depth_type in [
                DepthType.METRIC_DEPTH,
                DepthType.MODEL_METRIC_DEPTH,
            ]

        else:
            self.metric_depth = None

        self.backend.depth_model = self.metric_depth

    def _add_keyframe(
        self,
        frame_idx: int,
        images: torch.Tensor,
        buffer_masks: torch.Tensor | None,
        frame_data_list: list[VideoFrame],
        phase: int,
    ):
        assert phase in [1, 2]
        kf_idx = self.buffer.n_frames
        self.buffer.tstamp[kf_idx] = frame_idx
        self.buffer.images[kf_idx] = images
        self.buffer.fmaps[kf_idx] = self.droid_net.encode_features(images)
        self.buffer.nets[kf_idx], self.buffer.inps[kf_idx] = self.droid_net.encode_context(images)
        if buffer_masks is not None:
            self.buffer.masks[kf_idx] = buffer_masks

        for view_idx, frame_data in enumerate(frame_data_list):
            if kf_idx == 0:
                self.buffer.intrinsics[view_idx] = unpack_optional(frame_data.intrinsics)

            if frame_data.metric_depth is not None:
                disp_sens = frame_data.metric_depth[3::8, 3::8]
                disp_sens = torch.where(disp_sens > 0, disp_sens.reciprocal(), disp_sens)
                assert not self.config.optimize_intrinsics
                self.buffer.disps_sens[kf_idx, view_idx] = disp_sens

            if frame_data.pose is not None and phase == 1:
                self.buffer.poses[kf_idx] = (SE3(self.buffer.rig[view_idx]) * frame_data.pose.inv()).data

        if phase == 1:
            self.buffer.update_disps_sens(self.metric_depth, frame_idx=kf_idx)
        self.buffer.n_frames += 1

    def _log_final(self, video_streams: list[VideoStream], filled_return: FilledReturn):
        trajectory = filled_return.poses.inv()
        for frame_idx, frame_data_list in enumerate(zip(*video_streams)):
            pose_mat = trajectory[frame_idx].matrix().cpu().numpy()
            rr.set_time_sequence("frame", frame_idx)
            for view_idx in range(len(frame_data_list)):
                rig_mat = pose_mat @ SE3(self.buffer.rig[view_idx]).matrix().cpu().numpy()
                image = frame_data_list[view_idx].rgb.cpu().numpy()
                rr.log(
                    f"world/camera_v{view_idx}",
                    rr.Transform3D(translation=rig_mat[:3, 3], mat3x3=rig_mat[:3, :3]),
                    rr.Pinhole(
                        resolution=[image.shape[1], image.shape[0]],
                        image_from_camera=self.buffer.K[view_idx],
                    ),
                    rr.Image((image * 255).astype(np.uint8)).compress(),
                )

    def _precompute_features(self, frame_data_list: list[VideoFrame]):
        images_list = []
        masks_list = []
        for frame_data in frame_data_list:
            images_list.append(frame_data.rgb)
            if frame_data.mask is not None:
                mask_height, mask_width = frame_data.mask.shape
                mask_height, mask_width = mask_height // 8, mask_width // 8
                mask = (
                    torch.nn.functional.interpolate(
                        frame_data.mask[None, None].float(),
                        (mask_height, mask_width),
                        mode="bilinear",
                    )[0, 0]
                    > 0.9
                )
                masks_list.append(~mask)
        images = rearrange(torch.stack(images_list), "n h w c -> n c h w")
        if masks_list:
            masks = torch.stack(masks_list)
        else:
            masks = None
        return images, masks

    @torch.no_grad()
    def run(
        self,
        video_streams: list[VideoStream],
        rig: SE3 | None = None,
        camera_type: CameraType = CameraType.PINHOLE,
    ) -> SLAMOutput:
        assert len(video_streams) > 0
        resizers = [StandardResizeStreamProcessor() for _ in video_streams]
        video_streams = [
            ProcessedVideoStream(video_stream, [resizer]) for video_stream, resizer in zip(video_streams, resizers)
        ]

        frame_size = video_streams[0].frame_size()
        total_n_frames = len(video_streams[0])
        for vs in video_streams:
            assert vs.frame_size() == frame_size
            assert len(vs) == total_n_frames

        if rig is None:
            assert len(video_streams) == 1, "Need rig for multiple views"
            rig = SE3.Identity(1)
        self.rig = rig

        self.config.update(
            {
                "height": frame_size[0],
                "width": frame_size[1],
                "n_views": len(video_streams),
                "has_init_pose": FrameAttribute.POSE in video_streams[0].attributes(),
                "camera_type": camera_type,
            }
        )

        self._build_components()

        if self.visualize:
            rr.init("ViPE Visualization", spawn=True, recording_id=uuid.uuid4())
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        # Run frontend to get attributes initialization. This will also populate attribute buffers.
        frame_data_list: list[VideoFrame]
        frame_idx: int = 0
        for frame_idx, frame_data_list in pbar(
            enumerate(zip(*video_streams)), desc="SLAM Pass (1/2)", total=total_n_frames
        ):
            images, buffer_masks = self._precompute_features(frame_data_list)

            self.sparse_tracks.track_image(frame_data_list)

            if self.motion_filter.check(images, buffer_masks) or frame_idx == total_n_frames - 1:
                is_keyframe = True
                self._add_keyframe(frame_idx, images, buffer_masks, frame_data_list, phase=1)
            else:
                is_keyframe = False

            self.frontend.run()

            if self.visualize:
                self.buffer.log(self.config.map_filter_thresh)
                self.frontend.graph.log()

            # Run the backend in between to correct intrinsics and extrinsics in advance
            # to avoid large errors and local minima.
            if self.buffer.n_frames in self.config.frontend_backend_iters and is_keyframe:
                self.backend.run_if_necessary(5, log=self.visualize)

        # Tracks can be determined earlier since it's fixed after frontend.
        if self.visualize:
            self.buffer.log_tracks()

        # Run the backend to perform a global BA over the keyframes.
        self.backend.run(7, log=self.visualize)

        # Run backend again with a new graph and cleared GRU states.
        self.backend.run(self.config.backend_iters, update_depth=False, log=self.visualize)

        # Infill poses and attributes for non-keyframe frames.
        self.inner_filler.set_start_idx(self.buffer.n_frames)
        for frame_idx, frame_data_list in pbar(
            enumerate(zip(*video_streams)), desc="SLAM Pass (2/2)", total=total_n_frames
        ):
            images, buffer_masks = self._precompute_features(frame_data_list)
            self._add_keyframe(frame_idx, images, buffer_masks, frame_data_list, phase=2)
            if self.inner_filler.check() or frame_idx == total_n_frames - 1:
                self.inner_filler.compute()

        filled_return = self.inner_filler.get_result()

        # This means the iterator is exhausted early than expected in the above loop.
        # Warn user to use cached video stream.
        if filled_return.poses.shape[0] != total_n_frames:
            raise ValueError("Your video might be malformed. Try using streams.cached=true in the config.")

        if self.visualize:
            self._log_final(video_streams, filled_return)

        slam_map = self.buffer.extract_slam_map(filter_thresh=self.config.map_filter_thresh)

        # Scale back the intrinsics to the original size.
        original_intrinsics = torch.stack(
            [resizer.recover_intrinsics(self.buffer.intrinsics[v]) for v, resizer in enumerate(resizers)]
        )

        return SLAMOutput(
            trajectory=filled_return.poses.inv(),
            intrinsics=original_intrinsics,
            rig=SE3(self.buffer.rig.clone()),
            slam_map=slam_map,
        )
