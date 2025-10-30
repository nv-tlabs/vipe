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

from typing import Iterator

import numpy as np
import torch

from vipe.priors.depth import DepthEstimationInput, make_depth_model
from vipe.priors.depth.alignment import align_inv_depth_to_depth
from vipe.priors.depth.priorda import PriorDAModel
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.depth.geometrycrafter import (
    GeometryCrafterDiffPipeline,
    GeometryCrafterDetermPipeline,
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid
)
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import CachedVideoStream, FrameAttribute, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional
from vipe.utils.morph import erode
from vipe.priors.depth.moge import MogeModel


logger = logging.getLogger(__name__)


class IntrinsicEstimationProcessor(StreamProcessor):
    """Override existing intrinsics with estimated intrinsics."""

    def __init__(self, video_stream: VideoStream, gap_sec: float = 1.0) -> None:
        super().__init__()
        gap_frame = int(gap_sec * video_stream.fps())
        gap_frame = min(gap_frame, (len(video_stream) - 1) // 2)
        self.sample_frame_inds = [0, gap_frame, gap_frame * 2]
        self.fov_y = -1.0
        self.camera_type = CameraType.PINHOLE
        self.distortion: list[float] = []

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert self.fov_y > 0, "FOV not set"
        frame_height, frame_width = frame.size()
        fx = fy = frame_height / (2 * np.tan(self.fov_y / 2))
        frame.intrinsics = torch.as_tensor(
            [fx, fy, frame_width / 2, frame_height / 2] + self.distortion,
        ).float()
        frame.camera_type = self.camera_type
        return frame


class GeoCalibIntrinsicsProcessor(IntrinsicEstimationProcessor):
    def __init__(
        self,
        video_stream: VideoStream,
        gap_sec: float = 1.0,
        camera_type: CameraType = CameraType.PINHOLE,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(video_stream, gap_sec)

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        model = GeoCalib(weights=weights).to(device)
        indexable_stream = CachedVideoStream(video_stream)

        if is_pinhole:
            sample_frames = torch.stack([indexable_stream[i].rgb.moveaxis(-1, 0) for i in self.sample_frame_inds])
            res = model.calibrate(
                sample_frames,
                shared_intrinsics=True,
            )
        else:
            # Use first frame for calibration
            camera_model = {
                CameraType.PINHOLE: "pinhole",
                CameraType.MEI: "simple_mei",
            }[camera_type]
            res = model.calibrate(
                indexable_stream[self.sample_frame_inds[0]].rgb.moveaxis(-1, 0)[None],
                camera_model=camera_model,
            )

        self.fov_y = res["camera"].vfov[0].item()
        self.camera_type = camera_type

        if not is_pinhole:
            # Assign distortion parameter
            self.distortion = [res["camera"].dist[0, 0].item()]

    def move_to_gpu(self):
        """Move model to GPU device"""
        if hasattr(self, 'model'):
            self.model = self.model.to("cuda")


class TrackAnythingProcessor(StreamProcessor):
    """
    A processor that tracks a mask caption in the video.
    """

    def __init__(
        self,
        mask_phrases: list[str],
        add_sky: bool,
        sam_run_gap: int = 30,
        mask_expand: int = 5,
        device: torch.device = torch.device("cuda"),
        preloaded_sam=None,
        preloaded_aot=None,
    ) -> None:
        self.mask_phrases = mask_phrases
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(
            self.mask_phrases, 
            sam_points_per_side=50, 
            sam_run_gap=self.sam_run_gap,
            preloaded_sam=preloaded_sam,
            preloaded_aot=preloaded_aot
        )
        self.mask_expand = mask_expand

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        frame.instance, frame.instance_phrases = self.tracker.track(frame)
        self.last_track_frame = frame.raw_frame_idx

        frame_instance_mask = frame.instance == 0
        if self.add_sky:
            # We won't mask out the sky.
            frame_instance_mask |= frame.sky_mask

        frame.mask = erode(frame_instance_mask, self.mask_expand)
        return frame

    def move_to_gpu(self):
        """Move tracker to GPU device"""
        if hasattr(self, 'tracker'):
            # TrackAnythingPipeline doesn't support device parameter
            pass


class AdaptiveDepthProcessor(StreamProcessor):
    """
    Compute projection of the SLAM map onto the current frames.
    If it's well-distributed, then use the fast map-prompted video depth model.
    If not, then use the slow metric depth + video depth alignment model.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        model: str = "adaptive_unidepth-l_svda",
        share_depth_model: bool = False,
        device: torch.device = torch.device("cuda"),
        preloaded_gc_models: dict | None = None,
        preloaded_unidepth=None,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Adaptive depth processor only supports view_idx=0"
        assert not share_depth_model, "Adaptive depth processor does not support shared depth model"
        self.require_cache = True
        self.model = model
        self.device = device

        try:
            prefix, metric_model, video_model = model.split("_")
            assert video_model in ["svda", "vda", "gc"]
            if video_model == "gc":
                # Use pre-loaded models if available (Modal optimization)
                if preloaded_gc_models is not None:
                    logger.info("Using pre-loaded GeometryCrafter models (Modal optimized)")
                    self.point_map_vae = preloaded_gc_models["point_map_vae"]
                    self.prior_model = preloaded_gc_models["prior_model"]
                    self.video_depth_model = preloaded_gc_models["video_depth_model"]
                else:
                    # Load models fresh (original behavior)
                    model_type = "diff" #TODO: provide flexibility to the user to choose the model type
                    cache_dir = "/home/afridi/Depth/GeometryCrafter/workspace/cache"
                    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
                        'TencentARC/GeometryCrafter',
                        subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        cache_dir=cache_dir
                    ).requires_grad_(False).to(device, dtype=torch.float16)
                    self.point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
                        'TencentARC/GeometryCrafter',
                        subfolder='point_map_vae',
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,
                        cache_dir=cache_dir
                    ).requires_grad_(False).to(device, dtype=torch.float32)
                    self.prior_model = MogeModel( #TODO: check if this is redundant
                        cache_dir=cache_dir,
                    ).requires_grad_(False).to(device, dtype=torch.float32)
                    self.video_depth_model = GeometryCrafterDiffPipeline.from_pretrained(
                        "stabilityai/stable-video-diffusion-img2vid-xt",
                        unet=unet,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        cache_dir=cache_dir
                    ).to(device)
            else:
                self.video_depth_model = VideoDepthAnythingDepthModel(model="vits" if video_model == "svda" else "vitl")

        except ValueError:
            prefix, metric_model = model.split("_")
            video_model = None
            self.video_depth_model = None

        assert prefix == "adaptive", "Model name should start with 'adaptive_'"

        # Use pre-loaded UniDepthV2 if available (Modal optimization)
        if preloaded_unidepth is not None:
            logger.info("Using pre-loaded UniDepthV2 model (Modal optimized)")
            self.depth_model = preloaded_unidepth
        else:
            self.depth_model = make_depth_model(metric_model)
        self.prompt_model = PriorDAModel()
        self.update_momentum = 0.99

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("AdaptiveDepthProcessor should not be called directly.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def _compute_uv_score(self, depth: torch.Tensor, patch_count: int = 10) -> float:
        h_shape = depth.size(0) // patch_count
        w_shape = depth.size(1) // patch_count
        depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
        depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
        depth_exist = depth_crop.any(dim=(1, 3))
        return depth_exist.float().mean().item()

    def _compute_video_da(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_list: list[np.ndarray] = []
        frame_data_list: list[VideoFrame] = []
        for frame in frame_iterator:
            frame_data_list.append(frame.cpu())
            frame_list.append(frame.rgb.cpu().numpy())

        video_depth_result: torch.Tensor = unpack_optional(
            self.video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
        )
        return video_depth_result, frame_data_list
    def _compute_video_gc(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_list: list[np.ndarray] = []
        frame_data_list: list[VideoFrame] = []
        for frame in frame_iterator:
            frame_data_list.append(frame.cpu())
            frame_list.append(frame.rgb)
        frames_tensor = torch.stack(frame_list).permute(0, 3, 1, 2)
        num_inference_steps = 5
        guidance_scale = 1.0
        window_size = 110
        decode_chunk_size = 8
        overlap = 25
        force_projection = True
        force_fixed_focal = True
        use_extract_interp = False
        track_time = False
        low_memory_usage = False
        height = frame.rgb.shape[0]
        width = frame.rgb.shape[1]
        assert height % 64 == 0
        assert width % 64 == 0
        with torch.inference_mode():
            rec_point_map, _ = self.video_depth_model(
                frames_tensor,
                self.point_map_vae,
                self.prior_model,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                window_size=window_size,
                decode_chunk_size=decode_chunk_size,
                overlap=overlap,
                force_projection=force_projection,
                force_fixed_focal=force_fixed_focal,
                use_extract_interp=use_extract_interp,
                track_time=track_time,
                low_memory_usage=low_memory_usage
            )
        
        print(f"Rec point map range: {rec_point_map[:,:,:,2].min()} to {rec_point_map[:,:,:,2].max()}")
        rec_depth = 1/(rec_point_map[:,:,:,2]) # Inverse depth
        print(f"Inverse Rec depth range: {rec_depth.min()} to {rec_depth.max()}")
        return rec_depth, frame_data_list

    def update_iterator(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias = None
        min_uv_score: float = 1.0

        if self.video_depth_model is not None:
            if isinstance(self.video_depth_model, VideoDepthAnythingDepthModel):
                video_depth_result, data_iterator = self._compute_video_da(previous_iterator)
            elif isinstance(self.video_depth_model, GeometryCrafterDiffPipeline):
                video_depth_result, data_iterator = self._compute_video_gc(previous_iterator)
            else:
                raise ValueError(f"Unknown video depth model: {self.video_depth_model}")
        else:
            video_depth_result = None
            data_iterator = previous_iterator

        for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning depth"):
            # Convert back to GPU if not already.
            frame = frame.cuda()

            # Compute the minimum UV score only once at the 0-th frame.
            if frame_idx == 0:
                for test_frame_idx in range(self.slam_output.trajectory.shape[0]):
                    if test_frame_idx % 10 != 0:
                        continue
                    depth_infilled = self.slam_output.slam_map.project_map(
                        test_frame_idx,
                        0,
                        frame.size(),
                        unpack_optional(frame.intrinsics),
                        self.infill_target_pose[test_frame_idx],
                        unpack_optional(frame.camera_type),
                        infill=False,
                    )
                    uv_score = self._compute_uv_score(depth_infilled)
                    if uv_score < min_uv_score:
                        min_uv_score = uv_score

                logger.info(f"Minimum UV score: {min_uv_score:.4f}")

            if min_uv_score < 0.3:
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type)
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = self.slam_output.slam_map.project_map(
                    frame_idx,
                    0,
                    frame.size(),
                    unpack_optional(frame.intrinsics),
                    self.infill_target_pose[frame_idx],
                    unpack_optional(frame.camera_type),
                    infill=False,
                )
                if frame.mask is not None:
                    depth_map = depth_map * frame.mask.float()
                prompt_result = self.prompt_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(),
                        prompt_metric_depth=depth_map,
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(SLAM)"

            if video_depth_result is not None:
                video_depth_inv_depth = video_depth_result[frame_idx]

                align_mask = video_depth_inv_depth > 1e-3
                if frame.mask is not None:
                    align_mask = align_mask & frame.mask & (~frame.sky_mask)

                try:
                    _, scale, bias = align_inv_depth_to_depth(
                        unpack_optional(video_depth_inv_depth),
                        prompt_result,
                        align_mask,
                    )
                except RuntimeError:
                    scale, bias = self.cache_scale_bias

                # momentum update
                if self.cache_scale_bias is None:
                    self.cache_scale_bias = (scale, bias)
                scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
                bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
                self.cache_scale_bias = (scale, bias)

                video_inv_depth = video_depth_inv_depth * scale + bias
                video_inv_depth[video_inv_depth < 1e-3] = 1e-3
                frame.metric_depth = video_inv_depth.reciprocal()

            else:
                frame.metric_depth = prompt_result

            yield frame
