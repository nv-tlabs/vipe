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
from vipe.priors.dynamic_mask import DynamicMaskPipeline
from vipe.priors.flow import SeaRaftModel
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import (CachedVideoStream, FrameAttribute,
                               StreamProcessor, VideoFrame, VideoStream)
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional
from vipe.utils.morph import erode

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
    ) -> None:
        super().__init__(video_stream, gap_sec)

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        model = GeoCalib(weights=weights).cuda()
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
    ) -> None:
        self.mask_phrases = mask_phrases
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(self.mask_phrases, sam_points_per_side=50, sam_run_gap=self.sam_run_gap)
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


class DynamicMaskProcessor(StreamProcessor):
    """Two-pass processor that auto-selects dynamic instances by optical-flow
    motion score (no text prompt required).

    Pass 0: cache RGB frames + run TrackAnything in auto-seg mode to get
    consistent class-agnostic instance ids across frames.
    Pass 1 (after the dynamic-mask pipeline has been resolved): emit each
    cached frame with `frame.instance` / `frame.instance_phrases` set to the
    auto-seg result, and `frame.mask` set to the binary "valid pixel" mask
    (True for static / sky pixels, False over instances scored as dynamic).
    """

    n_passes_required = 2

    def __init__(
        self,
        sam_run_gap: int = 30,
        sam_points_per_side: int = 30,
        mask_expand: int = 5,
        flow_iters: int | None = None,
        flow_alpha: float = 0.5,
        flow_beta: float = 0.5,
        keep_motion_ratio: float = 0.25,
        min_area_ratio: float = 1e-4,
        filter_time_thres: float = 1e-4,
        ransac_threshold: float = 0.01,
    ) -> None:
        self.mask_expand = mask_expand
        self.sam_run_gap = sam_run_gap
        self.sam_points_per_side = sam_points_per_side
        self.flow_iters = flow_iters
        self.flow_alpha = flow_alpha
        self.flow_beta = flow_beta
        self.keep_motion_ratio = keep_motion_ratio
        self.min_area_ratio = min_area_ratio
        self.filter_time_thres = filter_time_thres
        self.ransac_threshold = ransac_threshold

        self._instances: list[np.ndarray] | None = None
        self._instance_phrases: dict[int, str] | None = None
        self._dynamic_masks: list[torch.Tensor] | None = None
        self._cached_frames: list[VideoFrame] | None = None

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx == 0:
            tracker = TrackAnythingPipeline(
                mask_phrases=None,
                sam_points_per_side=self.sam_points_per_side,
                sam_run_gap=self.sam_run_gap,
            )
            cached_frames: list[VideoFrame] = []
            instance_maps: list[np.ndarray] = []
            instance_phrases: dict[int, str] = {}

            for frame in previous_iterator:
                inst, phrases = tracker.track(frame)
                instance_maps.append(inst.detach().cpu().numpy().astype(np.int32))
                instance_phrases.update({int(k): v for k, v in phrases.items()})
                cached_frames.append(frame)
                yield frame

            S = len(cached_frames)
            if S < 2:
                H, W = cached_frames[0].rgb.shape[:2] if S == 1 else (1, 1)
                self._dynamic_masks = [torch.zeros((H, W), dtype=torch.bool) for _ in range(S)]
            else:
                rgb_tensor = torch.stack(
                    [f.rgb.permute(2, 0, 1).float() for f in cached_frames], dim=0
                )
                instances_tensor = torch.from_numpy(np.stack(instance_maps, axis=0))
                flow_model = SeaRaftModel(iters=self.flow_iters)
                dyn = DynamicMaskPipeline(
                    flow_model=flow_model,
                    flow_alpha=self.flow_alpha,
                    flow_beta=self.flow_beta,
                    ransac_threshold=self.ransac_threshold,
                    keep_motion_ratio=self.keep_motion_ratio,
                    min_area_ratio=self.min_area_ratio,
                    filter_time_thres=self.filter_time_thres,
                )
                out = dyn.compute(rgb_tensor, instances_tensor)
                logger.info(
                    "DynamicMaskProcessor: motion scores=%s, selected=%s",
                    out.instance_motion_scores,
                    out.selected_instances,
                )
                self._dynamic_masks = [out.masks[i] for i in range(S)]

            self._instances = instance_maps
            self._instance_phrases = instance_phrases
            self._cached_frames = cached_frames
            return

        assert self._dynamic_masks is not None and self._instances is not None
        cached_frames = self._cached_frames or []
        for i, frame in enumerate(cached_frames):
            inst = torch.from_numpy(self._instances[i]).to(frame.rgb.device)
            frame.instance = inst.to(torch.uint8) if int(inst.max().item()) < 256 else inst
            frame.instance_phrases = dict(self._instance_phrases or {})

            dyn = self._dynamic_masks[i].to(frame.rgb.device)
            frame.mask = erode(~dyn, self.mask_expand)
            yield frame


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
    ):
        super().__init__()
        self.slam_output = slam_output
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Adaptive depth processor only supports view_idx=0"
        assert not share_depth_model, "Adaptive depth processor does not support shared depth model"
        self.require_cache = True
        self.model = model

        try:
            prefix, metric_model, video_model = model.split("_")
            assert video_model in ["svda", "vda"]
            self.video_depth_model = VideoDepthAnythingDepthModel(model="vits" if video_model == "svda" else "vitl")

        except ValueError:
            prefix, metric_model = model.split("_")
            video_model = None
            self.video_depth_model = None

        assert prefix == "adaptive", "Model name should start with 'adaptive_'"

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

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias = None
        min_uv_score: float = 1.0

        if self.video_depth_model is not None:
            video_depth_result, data_iterator = self._compute_video_da(previous_iterator)
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
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type
                    )
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


class MultiviewDepthProcessor(StreamProcessor):
    """
    Use multi-view depth model (e.g. DAv3, MapAnything, CAPA) to estimate depth map for each frame.
    To ensure that the depth maps are consistent with the SLAM map/pose (metric), we condition the depth model either with
    (a) sparse points, or (b) camera poses & intrinsics.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    To create enough parallex to improve estimation confidence, for each window we optionally also include
    neighboring keyframes, and their secondary neighboring keyframes.
    (Multi-view input video frames are currently not supported)
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        model: str = "mvd_dav3",
        window_size: int = 10,                  # Practically this should be as large as possible if memory permits.
        overlap_size: int = 3,
        secondary_keyframe: bool = False,       # This is found to cause jittering for some scenes due to abrupt context changes.
    ):
        super().__init__()
        self.slam_output = slam_output
        self.model = model
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.secondary_keyframe = secondary_keyframe

        self.keyframes_inds = unpack_optional(self.slam_output.slam_map).dense_disp_frame_inds
        self.keyframes_data: list[VideoFrame] = []
        self.n_frames = 0

        # Need two passes for this iterator to work.
        self.n_passes_required = 2

        if self.model == "mvd_dav3":
            try:
                from depth_anything_3.api import DepthAnything3
                from depth_anything_3.api import logger as dav3_logger
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "depth-anything-3 not found. Please reinstall vipe with `pip install --no-build-isolation -e .[dav3]`"
                )

            dav3_logger.level = 0  # Disable logging timing information
            self.dav3_api = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
            self.dav3_api = self.dav3_api.cuda().eval()

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("MultiviewDepthProcessor should not be called directly.")

    def _probe_keyframe_indices(self, frame_idx: int) -> list[int]:
        inds: list[int] = []
        left_idx = np.searchsorted(self.keyframes_inds, frame_idx, side="right").item() - 1
        inds.append(left_idx)
        if frame_idx < self.keyframes_inds[-1]:
            inds.append(left_idx + 1)
        # Pick the farthest secondary keyframe from the left keyframe.
        if self.secondary_keyframe:
            slam_graph = unpack_optional(self.slam_output.slam_map).backend_graph
            if slam_graph is not None:
                matching_secondary_j = slam_graph[slam_graph[:, 0] == left_idx, 1].tolist()
                picked_sj_idx = np.argmax([abs(self.keyframes_inds[j] - frame_idx) for j in matching_secondary_j])
                inds.append(matching_secondary_j[picked_sj_idx])
        return inds

    def record_keyframes(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        for frame_idx, frame in enumerate(previous_iterator):
            self.n_frames += 1
            if frame_idx in self.keyframes_inds:
                self.keyframes_data.append(frame)
            yield frame

    def estimate_depth_sliding_window(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        current_sliding_window: list[VideoFrame] = []
        current_sliding_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        for frame_idx, frame in pbar(enumerate(previous_iterator), desc="Estimating multi-view depth"):
            current_sliding_window.append(frame)
            current_sliding_window_idx.append(frame_idx)
            is_last_frame = frame_idx == self.n_frames - 1

            if len(current_sliding_window) == self.window_size or is_last_frame:
                # Grab all neighboring keyframes to anchor the current sliding window.
                # Note that we remove redundant keyframes that already exist in the current sliding window.
                sw_keyframe_inds = list(
                    set(sum([self._probe_keyframe_indices(i) for i in current_sliding_window_idx], []))
                )
                sw_keyframe_inds = [
                    t for t in sw_keyframe_inds if self.keyframes_inds[t] not in current_sliding_window_idx
                ]

                sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])

                if len(sw_keyframe_inds) > 0:
                    kf_images, kf_exts, kf_ints = zip(*[self.keyframes_data[t].dav3_conditions() for t in sw_keyframe_inds])
                else:
                    kf_images, kf_exts, kf_ints = tuple(), tuple(), tuple()

                # Perform inference
                dav3_inference_result = self.dav3_api.inference(
                    list(sw_images + kf_images),
                    extrinsics=np.stack(sw_exts + kf_exts, axis=0),
                    intrinsics=np.stack(sw_ints + kf_ints, axis=0),
                    process_res_method="lower_bound_resize",  # Keep aspect ratio
                )
                sw_depth = torch.from_numpy(dav3_inference_result.depth[: len(sw_images)]).float().cuda()
                sw_depth = torch.nn.functional.interpolate(sw_depth[:, None], frame.size(), mode="bilinear")[:, 0]

                n_frames_to_yield = (
                    self.window_size - self.overlap_size if not is_last_frame else len(current_sliding_window)
                )

                # Linearly interpolate the trailing depth with new depth
                if trailing_depth is not None:
                    n_interp_frames = len(trailing_depth)
                    alpha = torch.linspace(0, 1, n_interp_frames + 2)[1:-1].float().cuda()[:, None, None]
                    sw_depth[:n_interp_frames] = trailing_depth * (1 - alpha) + sw_depth[:n_interp_frames] * alpha

                for sw_idx, frame in enumerate(current_sliding_window[:n_frames_to_yield]):
                    frame.metric_depth = sw_depth[sw_idx]
                    yield frame

                trailing_depth = sw_depth[n_frames_to_yield:]
                current_sliding_window = current_sliding_window[n_frames_to_yield:]
                current_sliding_window_idx = current_sliding_window_idx[n_frames_to_yield:]

        assert len(current_sliding_window) == 0, "Current sliding window should be empty"

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx == 0:
            yield from self.record_keyframes(previous_iterator)
        elif pass_idx == 1:
            yield from self.estimate_depth_sliding_window(previous_iterator)
        else:
            raise ValueError(f"Invalid pass index: {pass_idx}")
