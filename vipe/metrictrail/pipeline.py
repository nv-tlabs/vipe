# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MetricTrailPipeline: the one MetricTrail pipeline (``pipeline=metrictrail``).

Per input video (the whole sequence; a stream whose tail was cut is refused):

0. input raster (``init.input_raster``): by default the in-the-wild policy --
   cover-fit to 280x504 and an integer centre crop, the source frame rate, one frame
   in k above 1500 frames; ``policy: resize`` resizes to the target without a crop;
   every resize is ViPE's bilinear resize (:mod:`vipe.metrictrail.raster`).
   ``policy: unchanged`` keeps the input as
   delivered (pre-prepared frames). ``init.tracking_stride`` k > 1: SharedK calibrates
   on every prepared frame and everything after it (tracking, focal refinement, outputs)
   sees frames 0, k, 2k, ... of the same prepared frames only (e.g. the 7-Scenes
   setup).
1. init: RGB-only SharedK calibration (:class:`SharedKIntrinsicsProcessor`), or the
   known K of ``pipeline.optional.intrinsics`` (:class:`KnownIntrinsicsProcessor`),
   gives one pinhole K for every frame; SharedK refuses inputs that carry intrinsics.
   Default SharedK policy: the in-the-wild one (provider-focal fallback).
2. slam: :class:`DPVOSLAMSystem` with the slam-group preset
   (``metrictrail_local`` default, ``metrictrail_gba``, ``metrictrail_gba_refine``),
   per-frame MoGe-2 metric depth in FP16 on the overlapped producer (depth_queue 4).
3. focal_refine (``pipeline.focal_refine.enabled``, off by default): route ->
   one stage-B attempt -> accept gate -> if accepted, rebind the focal and re-track;
   skipped (and recorded) with a known K.
4. outputs in ViPE's artifact layout: ``pose/<n>.npz`` (OpenCV camera-to-world,
   float64, plus ``keyframe_ids``, ``runtime`` and ``source_frame_ids``),
   ``intrinsics/<n>.npz`` + ``_camera.txt``, ``vipe/<n>_info.pkl``, the run
   record ``vipe/<n>_metrictrail.json`` and (``output.save_metric_depth: async``,
   the default) ``depth/<n>.zip``, the per-frame MoGe-2 metric depth of the final
   tracking pass written asynchronously during tracking; optionally the SLAM map
   and ViPE's visualization video.

Extension point: ``MetricTrailPipeline.post_slam_hooks`` (see :class:`PostSLAMHook`)
run after tracking and before the artifacts are written, in order; a hook with
``after_save = True`` runs after every artifact (pose, run record, SLAM map,
visualization) is written.
``pipeline.optional.loop_closure`` registers such a hook (:class:`LoopClosureHook`,
output ``pose_lc/``). The default pipeline has none. Stereo is a separate runner
(:mod:`vipe.metrictrail.stereo`).

ViPE's optional features, off by default and behaving as in ViPE's default
pipeline (``docs/metrictrail_dynamic_and_fusion.md``): ``init.instance`` runs
TrackAnything in the init stage and masks dynamic pixels out of DPVO
(:mod:`vipe.metrictrail.dynamic`); ``post.depth_align_model`` runs ViPE's
``AdaptiveDepthProcessor`` on the DPVO output and saves ViPE depth artifacts, which
then own ``depth/<n>.zip`` (the MoGe-2 metric depth is not written in that run).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

import numpy as np
import torch

from vipe.config.base_schema import config_to_primitive
from vipe.pipeline import AnnotationPipelineOutput, Pipeline
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoFrame,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType

from .calibration.processor import SharedKIntrinsicsProcessor, shared_device_k
from .config import PIPELINE_INSTANCE, MetricTrailPipelineConfig
from .depth_producer import TRACKING_OK, jsonable
from .dynamic import InstanceMasks, save_instance_artifacts
from .optional.intrinsics import KnownIntrinsicsProcessor, with_known_intrinsics
from .raster import prepare_input_raster
from .slam import DPVOSLAMOutput, DPVOSLAMSystem
from .streams import full_sequence_stream, tracking_stride_stream
from .torch_state import preserve_torch_state

logger = logging.getLogger(__name__)

RECORD_SCHEMA = "metrictrail_vipe_run_v1"

#: How a known K in source pixels follows each resizing input-raster policy (optional intrinsics' raster_map).
RASTER_K_MAP: dict[str, Literal["cover_crop", "stretch"]] = {"cover_crop": "cover_crop", "resize": "stretch"}


def record_path(artifact_path: io.ArtifactPath) -> Path:
    """``vipe/<name>_metrictrail.json`` beside ViPE's ``_info.pkl``."""

    return artifact_path.base_path / "vipe" / f"{artifact_path.artifact_name}_metrictrail.json"


class PostSLAMHook(Protocol):
    """Called after tracking (and focal refinement), before artifacts are written.

    It may return a new final output (e.g. a loop-closed trajectory) or None to
    keep ``final``; it may add entries to ``record`` (the run record).

    A hook with ``after_save = True`` is called instead after the artifacts
    (pose, run record, SLAM map, visualization) are written; only with
    ``output.save_artifacts``, never with ``return_payload``. It writes its own
    artifacts; its return value and any change to ``record`` are not saved.
    """

    def __call__(
        self,
        pipeline: "MetricTrailPipeline",
        video_stream: VideoStream,
        final: DPVOSLAMOutput,
        record: dict[str, Any],
    ) -> DPVOSLAMOutput | None: ...


class SharedIntrinsicsProcessor(StreamProcessor):
    """Assign one pinhole K to every frame (the rebound K of an accepted focal refinement)."""

    def __init__(self, k4: np.ndarray) -> None:
        super().__init__()
        self.k4 = np.asarray(k4, dtype=np.float32).reshape(4)
        self._device_k: dict[torch.device, torch.Tensor] = {}

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        # Cloned on the device from one copy per device: no per-frame host copy of K.
        frame.intrinsics = shared_device_k(self._device_k, self.k4, frame.rgb.device)
        frame.camera_type = CameraType.PINHOLE
        return frame


def published_tracking_record(record: dict[str, Any], depth_artifact: str | None) -> dict[str, Any]:
    """A tracking record as the run record keeps it: the metric-depth zip's staging path is replaced
    by the published artifact (``depth/<n>.zip``, relative to the output directory), or by null for
    a pass whose depth is not published (a superseded stage A)."""

    written = record.get("metric_depth_artifact")
    if not written:
        return record
    return {**record, "metric_depth_artifact": {**written, "path": depth_artifact}}


def run_runtime(stage_a: DPVOSLAMOutput, final: DPVOSLAMOutput) -> dict[str, Any] | None:
    """The run record's ``runtime`` block: the final tracking pass's (``vipe.metrictrail.runtime_record``).

    Tracking stage only (calibration and model load excluded); after an accepted focal
    refinement it is the re-track's, and stage A's block is kept beside it.
    """

    block = final.record.get("runtime")
    if block is None:
        return None
    retracked = final is not stage_a
    return {
        **block,
        "tracking_pass": "focal_refine_retrack" if retracked else "stage_a",
        "stage_a": stage_a.record.get("runtime") if retracked else None,
    }


def chw_float_frames(stream: VideoStream) -> Iterator[np.ndarray]:
    """The sequence as CHW float32 frames re-quantised to 8 bit: ``uint8 * (1/255)``."""

    for frame in stream:
        u8 = torch.round(frame.rgb.float() * 255.0).clamp_(0.0, 255.0).to(torch.uint8).cpu().numpy()
        value = np.ascontiguousarray(u8.transpose(2, 0, 1), dtype=np.float32)
        value *= 1.0 / 255.0
        yield value


class SavedMetricDepthProcessor(StreamProcessor):
    """Frame metric depth read back from the saved depth artifact (visualization without depth fusion).

    ``paths``: candidate ``depth/<n>.zip`` files; the first that exists when the stream is
    read is used (the artifact is moved to the output directory after the stream is built).
    Its pixels without depth are 0, which ViPE's depth visualization shows as invalid.
    """

    def __init__(self, paths: list[Path]) -> None:
        self.paths = [Path(path) for path in paths]

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        path = next((path for path in self.paths if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"no saved depth artifact among {[str(p) for p in self.paths]}")
        depths = io.read_depth_artifacts(path)
        for frame_idx, frame in enumerate(previous_iterator):
            depth_idx, depth = next(depths)
            if depth_idx != frame_idx:
                raise ValueError(f"{path}: depth entry {depth_idx} for frame {frame_idx}")
            frame.metric_depth = depth.to(frame.rgb.device)
            yield frame

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:  # update_iterator is used instead
        raise NotImplementedError


class MetricTrailPipeline(Pipeline):
    """RGB-only shared K -> DINO-fused DPVO with MoGe-2 metric depth -> optional focal refinement."""

    def __init__(
        self, init: Any, slam: Any, focal_refine: Any, output: Any, post: Any = None, optional: Any = None
    ) -> None:
        super().__init__()
        self.config = MetricTrailPipelineConfig.model_validate(
            {
                "instance": PIPELINE_INSTANCE,
                "init": config_to_primitive(init),
                "slam": config_to_primitive(slam),
                "focal_refine": config_to_primitive(focal_refine),
                "output": config_to_primitive(output),
                "post": config_to_primitive(post) if post is not None else {},
                "optional": config_to_primitive(optional) if optional is not None else {},
            }
        )
        self.post_cfg = self.config.post
        self.optional_cfg = self.config.optional
        self.init_cfg = self.config.init
        self.slam_cfg = self.config.slam
        self.refine_cfg = self.config.focal_refine
        self.out_cfg = self.config.output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.device = torch.device("cuda")
        #: Post-SLAM hooks (extension point), run in order; see :class:`PostSLAMHook`.
        self.post_slam_hooks: list[PostSLAMHook] = []
        if self.optional_cfg.loop_closure.enabled:
            from .optional.loop_closure import LoopClosureHook

            self.post_slam_hooks.append(LoopClosureHook(self.optional_cfg.loop_closure))

    # ----------------------------------------------------------------- stages

    def calibrate(self, video_stream: VideoStream) -> SharedKIntrinsicsProcessor:
        """Calibrate or reuse RGB-only SharedK before constructing any mask processor.

        Args:
            video_stream: Full-rate prepared calibration stream, without intrinsics.
        """

        return SharedKIntrinsicsProcessor(
            video_stream,
            self.init_cfg.calibration,
            model_cache=self.model_cache,
            device=self.device,
            calibration_cache=self.init_cfg.calibration_cache,
        )

    def instance_masks(self, video_stream: VideoStream) -> list[StreamProcessor]:
        """ViPE's TrackAnythingProcessor for ``init.instance`` (ViPE's own arguments), run once; [] when null."""

        instance = self.init_cfg.instance
        if instance is None:
            return []
        if {FrameAttribute.INSTANCE, FrameAttribute.MASK} & video_stream.attributes():
            raise ValueError("init.instance is on but the input stream already carries instance masks")
        from vipe.pipeline.processors import TrackAnythingProcessor

        return [
            InstanceMasks(
                TrackAnythingProcessor(
                    list(instance.phrases),
                    add_sky=instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * instance.kf_gap_sec),
                    model_cache=self.model_cache,
                )
            )
        ]

    def known_intrinsics(
        self, video_stream: VideoStream, input_raster: dict[str, Any] | None = None
    ) -> KnownIntrinsicsProcessor | None:
        """The known-K init processor, or None (SharedK) when pipeline.optional.intrinsics is null.

        ``input_raster`` (the raster stage's record): when the pipeline resized the input
        itself, a K in the source video's pixels is mapped the same way (``cover_crop``:
        ``cover_crop``; ``resize``: ``stretch``;
        :func:`~vipe.metrictrail.optional.intrinsics.resolve_known_intrinsics`).
        """

        prepared_from: tuple[tuple[int, int], Literal["cover_crop", "stretch"]] | None = None
        policy = input_raster.get("policy") if input_raster is not None else None
        if input_raster is not None and policy in RASTER_K_MAP:
            source_hw = input_raster["source_hw"]
            prepared_from = ((int(source_hw[0]), int(source_hw[1])), RASTER_K_MAP[policy])
        return with_known_intrinsics(video_stream, self.optional_cfg, prepared_from=prepared_from)

    def _post_slam_hooks(self, *, after_save: bool) -> list[PostSLAMHook]:
        return [hook for hook in self.post_slam_hooks if bool(getattr(hook, "after_save", False)) is after_save]

    def make_slam(self) -> DPVOSLAMSystem:
        return DPVOSLAMSystem(self.device, self.slam_cfg, model_cache=self.model_cache)

    def refine_focal(
        self, video_stream: VideoStream, stage_a: DPVOSLAMOutput, k4: np.ndarray, calibration: dict[str, Any]
    ) -> dict[str, Any]:
        from .focal_refine.refine import run_focal_refinement

        name = video_stream.name()
        workdir = (
            Path(self.refine_cfg.workdir) / name
            if self.refine_cfg.workdir
            else self.out_path / "metrictrail" / name / "stage_b"
        )
        return run_focal_refinement(
            self.refine_cfg,
            stage_a_c2w=stage_a.c2w,
            stage_a_k4=k4,
            calibration_record=calibration,
            tracking_status=stage_a.tracking_status,
            frames=lambda: chw_float_frames(video_stream),
            image_hw=(int(video_stream.frame_size()[0]), int(video_stream.frame_size()[1])),
            device=self.device,
            workdir=workdir,
        )

    # ------------------------------------------------------------------- run

    def should_filter(self, stream_name: str) -> bool:
        if not self.out_cfg.skip_exists:
            return False
        artifact_path = io.ArtifactPath(self.out_path, stream_name)
        return artifact_path.pose_path.is_file() and record_path(artifact_path).is_file()

    @preserve_torch_state  # the run's TF32-off / seeds / DPVO thread count do not outlive it
    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            raise NotImplementedError("the MetricTrail pipeline tracks one monocular video")
        video_stream = video_data
        name = video_stream.name()
        annotate_output = AnnotationPipelineOutput()
        if self.should_filter(name):
            logger.info(f"{name} has been processed already, skip it")
            return annotate_output
        started = time.perf_counter()
        video_stream, coverage = full_sequence_stream(video_stream, on_truncation=self.init_cfg.on_truncated_input)
        # After full_sequence_stream: the raster's frame count and stride use the counted length.
        video_stream, input_raster = prepare_input_raster(video_stream, self.init_cfg.input_raster, device=self.device)
        # init.tracking_stride: SharedK calibrates on every prepared frame, the rest sees every k-th one.
        calibration_stream = video_stream
        video_stream = tracking_stride_stream(video_stream, self.init_cfg.tracking_stride)
        try:
            return self._run(video_stream, name, coverage, input_raster, started, annotate_output, calibration_stream)
        finally:
            # Depth zips of passes that were not committed (failed run, superseded stage A).
            for staged in self._depth_staging(name).values():
                staged.unlink(missing_ok=True)

    def _depth_staging(self, name: str) -> dict[str, Path]:
        root = self.out_path / "metrictrail" / name
        return {stage: root / f"depth_{stage}.zip" for stage in ("stage_a", "retrack")}

    def _run(
        self,
        video_stream: VideoStream,
        name: str,
        coverage: dict[str, Any],
        input_raster: dict[str, Any],
        started: float,
        annotate_output: AnnotationPipelineOutput,
        calibration_stream: VideoStream | None = None,
    ) -> AnnotationPipelineOutput:
        """Run calibration, tracking and publication for one prepared sequence.

        Args:
            video_stream: Prepared stream at the tracking stride.
            name: Output sequence name.
            coverage: Source-frame coverage audit.
            input_raster: Prepared raster and resizing audit.
            started: Pipeline start from the monotonic performance counter.
            annotate_output: ViPE output accumulator.
            calibration_stream: Full-rate prepared RGB stream, or the tracking stream if omitted.
        """
        save_depth = self._saves_metric_depth()
        staging = self._depth_staging(name)
        if calibration_stream is None:
            calibration_stream = video_stream

        known = self.known_intrinsics(video_stream, input_raster)
        # A known K (optional input) replaces SharedK; SharedK still refuses inputs that carry K.
        init_processor = known if known is not None else self.calibrate(calibration_stream)
        k4 = init_processor.k4.copy()
        masks = self.instance_masks(video_stream)
        slam = self.make_slam()
        # ViPE's depth fusion prompts its depth model with the SLAM map: give it each keyframe's dense depth
        # too (vipe.metrictrail.fused_map). Without fusion the run is unchanged (no extra argument).
        fusion = {"dense_map": True} if self.post_cfg.depth_align_model is not None else {}
        stage_a = slam.run(
            ProcessedVideoStream(video_stream, [init_processor, *masks]),
            intrinsics=k4,
            name=name,
            depth_output=staging["stage_a"] if save_depth else None,
            **fusion,
        )
        final, final_k4 = stage_a, k4
        refinement: dict[str, Any] | None = None
        if self.refine_cfg.enabled and known is not None:
            # Calibrated mode: a known K has no focal to refine, so stage B is skipped.
            logger.info(f"{name}: known intrinsics given; focal refinement (stage B) skipped")
            refinement = {"enabled": False, "requested": True, "skipped_reason": "known_intrinsics"}
        elif self.refine_cfg.enabled:
            refinement = self.refine_focal(video_stream, stage_a, k4, init_processor.record)
            if refinement.get("accepted"):
                from .focal_refine.refine import rebind_focal

                final_k4 = rebind_focal(k4, float(refinement["focal"]))
                # The re-track recomputes the depth with the rebound focal; its depth is the one saved.
                final = slam.run(
                    ProcessedVideoStream(video_stream, [SharedIntrinsicsProcessor(final_k4), *masks]),
                    intrinsics=final_k4,
                    name=name,
                    depth_output=staging["retrack"] if save_depth else None,
                    **fusion,
                )

        # The depth of the final *tracking* pass (post-SLAM hooks may replace the trajectory, not the depth).
        depth_artifact = final.depth_artifact
        calibration_fallback = (init_processor.record.get("calibration_fallback") if known is None else None) or {}
        record = {
            "schema": RECORD_SCHEMA,
            "status": "success",
            "name": name,
            "pipeline": self.config.model_dump(),
            "input_coverage": coverage,
            "input_raster": input_raster,
            "tracking_stride": {
                "stride": self.init_cfg.tracking_stride,
                "calibration_frames": len(calibration_stream) if known is None else None,
                "tracked_frames": len(final.c2w),
                # raw_frame_idx of every tracked frame, as source_frame_ids in pose/<name>.npz.
                "source_frame_ids": np.asarray(final.source_frame_ids).tolist(),
            },
            "intrinsics_source": "known" if known is not None else "sharedk",
            "sensor_K_or_GT_read": known is not None,
            "selection_uses_GT": False,
            # SharedK's status; null with a known K (no calibration ran).
            "calibration_status": None
            if known is not None
            else init_processor.record.get("calibration_status", init_processor.record.get("selection_status")),
            "calibration_fallback_reason": calibration_fallback.get("reason"),
            "calibration": init_processor.record if known is None else None,
            # The nested calibration keeps its original timing; this is the cost spent by this run.
            "calibration_execution": getattr(init_processor, "execution", None) if known is None else None,
            "known_intrinsics": known.record if known is not None else None,
            "bound_K_fx_fy_cx_cy": final_k4.tolist(),
            "stage_a_K_fx_fy_cx_cy": k4.tolist(),
            "focal_refine": refinement if refinement is not None else {"enabled": False},
            # The depth zips' staging paths are moved or deleted before the record is saved.
            "stage_a_tracking": published_tracking_record(stage_a.record, None) if final is not stage_a else None,
            "tracking": published_tracking_record(final.record, f"depth/{name}.zip" if save_depth else None),
            "tracking_status": final.tracking_status,
            "dynamic_mask": final.record.get("dynamic_mask", {"enabled": False}),
            # slam.nonfinite_recovery: whether the FP32 recovery ran, its attempts and the kept seed.
            "nonfinite_recovery": final.record.get("nonfinite_recovery", {"enabled": False}),
            # The final tracking pass's wall clock, frames/s and CUDA peak memory (tracking stage only).
            "runtime": run_runtime(stage_a, final),
            "depth_align_model": self.post_cfg.depth_align_model,
            "loop_closure": {"enabled": self.optional_cfg.loop_closure.enabled},  # report: pose_lc/<name>.json
            "metric_depth": self._depth_record(name, final, save_depth, retracked=final is not stage_a),
            "runtime_seconds": time.perf_counter() - started,
        }
        if final.tracking_status != TRACKING_OK:
            logger.warning(f"{name}: {final.tracking_status} (DPVO never initialized; near-static trajectory)")

        # ---- post-SLAM hooks (extension point; none by default) ----
        for hook in self._post_slam_hooks(after_save=False):
            replaced = hook(self, video_stream, final, record)
            if replaced is not None:
                final = replaced
                record["tracking_status"] = final.tracking_status

        if self.return_payload:
            annotate_output.payload = final
            return annotate_output

        artifact_path = io.ArtifactPath(self.out_path, name)
        output_stream: VideoStream | None = None
        frame_artifacts = bool(masks) or self.post_cfg.depth_align_model is not None
        if self.out_cfg.save_viz or (self.out_cfg.save_artifacts and frame_artifacts):
            # Without depth fusion, the visualization shows the depth DPVO tracked with (depth/<n>.zip).
            saved = [artifact_path.depth_path, *([final.depth_artifact] if final.depth_artifact is not None else [])]
            output_stream = self._output_stream(
                video_stream,
                final,
                final_k4,
                masks,
                saved_depth=saved if self.out_cfg.save_viz and save_depth else None,
            )
        if self.out_cfg.save_artifacts:
            logger.info(f"Saving MetricTrail artifacts to {artifact_path}")
            if save_depth:
                if depth_artifact is None or len(final.c2w) != record["metric_depth"]["frames"]:
                    raise RuntimeError(f"{name}: the final tracking pass wrote no complete metric depth")
                artifact_path.depth_path.parent.mkdir(exist_ok=True, parents=True)
                os.replace(depth_artifact, artifact_path.depth_path)
            if frame_artifacts and output_stream is not None:
                # ViPE's depth (post.depth_align_model, which then owns depth/<n>.zip) and mask
                # artifacts, before the pose and the run record.
                if self.post_cfg.depth_align_model is not None:
                    io.save_depth_artifacts(artifact_path, output_stream)
                save_instance_artifacts(artifact_path, output_stream)
            self._save_pose_artifacts(artifact_path, final, final_k4, record)
        if self.out_cfg.save_viz and output_stream is not None:
            from vipe.utils.visualization import save_projection_video

            artifact_path.meta_vis_path.parent.mkdir(exist_ok=True, parents=True)
            save_projection_video(
                artifact_path.meta_vis_path,
                output_stream,
                final,
                self.out_cfg.viz_downsample,
                [[str(name) for name in group] for group in self.out_cfg.viz_attributes],
            )
        if self.out_cfg.save_slam_map and final.slam_map is not None:
            artifact_path.slam_map_path.parent.mkdir(exist_ok=True, parents=True)
            final.slam_map.save(artifact_path.slam_map_path)
        if self.out_cfg.save_artifacts:
            for hook in self._post_slam_hooks(after_save=True):  # e.g. loop closure: reads pose/, writes pose_lc/
                hook(self, video_stream, final, record)
        if self.return_output_streams:
            annotate_output.output_streams = [
                output_stream or self._output_stream(video_stream, final, final_k4, masks)
            ]
        return annotate_output

    # --------------------------------------------------------------- outputs

    def _saves_metric_depth(self) -> bool:
        """Whether this run writes the tracker's metric depth to ``depth/<n>.zip`` (``output.save_metric_depth``).

        Not with ``post.depth_align_model``: ViPE's fused depth is then the run's
        ``depth/<n>.zip``, as in ViPE's default pipeline (one depth artifact per run).
        """

        return (
            self.out_cfg.save_metric_depth == "async"
            and self.out_cfg.save_artifacts
            and not self.return_payload
            and self.post_cfg.depth_align_model is None
        )

    def _depth_record(self, name: str, final: DPVOSLAMOutput, save_depth: bool, *, retracked: bool) -> dict[str, Any]:
        if not save_depth:
            if self.out_cfg.save_metric_depth == "off":
                reason = "off"
            elif not self.out_cfg.save_artifacts or self.return_payload:
                reason = "artifacts_not_saved"
            else:
                reason = "depth_align_model"  # depth/<n>.zip holds ViPE's fused depth instead
            return {"mode": self.out_cfg.save_metric_depth, "saved": False, "reason": reason}
        written = dict(final.record.get("metric_depth_artifact") or {})
        written.pop("path", None)
        return {
            "mode": "async",
            "saved": True,
            "artifact": f"depth/{name}.zip",
            "pass": "focal_refine_retrack" if retracked else "stage_a",
            "depth_model": "the tracker's slam.keyframe_depth (MoGe-2 by default), K-conditioned on the bound K",
            **written,
        }

    def _output_stream(
        self,
        video_stream: VideoStream,
        final: DPVOSLAMOutput,
        k4: np.ndarray,
        masks: list[StreamProcessor] | None = None,
        saved_depth: list[Path] | None = None,
    ) -> VideoStream:
        """The input with the final pose and K assigned (visualization, ``return_output_streams``).

        With ``init.instance`` the frames carry the tracked masks again; with
        ``post.depth_align_model`` ViPE's AdaptiveDepthProcessor follows, as in
        ViPE's default pipeline (``_add_post_processors``). Without it, ``saved_depth``
        (the first existing of these depth artifacts) gives the frames the metric
        depth DPVO tracked with, for the visualization's depth panel.
        """

        intrinsics = torch.from_numpy(np.asarray(k4, dtype=np.float32)).to(self.device)
        processors: list[StreamProcessor] = [
            SharedIntrinsicsProcessor(k4),
            *(masks or []),
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: final.get_view_trajectory(0),  # type: ignore[dict-item]
                    FrameAttribute.INTRINSICS: [intrinsics] * len(video_stream),
                }
            ),
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            from vipe.pipeline.processors import AdaptiveDepthProcessor

            if final.slam_map is None:
                raise RuntimeError("post.depth_align_model needs DPVO's patch map, but DPVO kept no keyframe points")
            processors.append(AdaptiveDepthProcessor(final, 0, depth_align_model))
        elif saved_depth:
            processors.append(SavedMetricDepthProcessor(saved_depth))
        return ProcessedVideoStream(video_stream, processors).cache("metrictrail_output", online=True)

    def _save_pose_artifacts(
        self, artifact_path: io.ArtifactPath, final: DPVOSLAMOutput, k4: np.ndarray, record: dict[str, Any]
    ) -> None:
        count = len(final.c2w)
        inds = np.arange(count)
        keyframe_ids = (
            np.asarray(final.slam_map.dense_disp_frame_inds, dtype=np.int64)
            if final.slam_map is not None
            else np.zeros(0, dtype=np.int64)
        )
        transport = final.record.get("dpvo_transport", {})
        artifact_path.pose_path.parent.mkdir(exist_ok=True, parents=True)
        pose_fields: dict[str, Any] = {
            "data": final.c2w,
            "inds": inds,
            "keyframe_ids": keyframe_ids,
            "runtime": np.float64(final.record.get("runtime_seconds", 0.0)),
            "source_frame_ids": final.source_frame_ids,
        }
        if "transport_hw" in transport:
            pose_fields["dpvo_transport_hw"] = np.asarray(transport["transport_hw"], dtype=np.int32)
            pose_fields["dpvo_padding_ltrb"] = np.asarray(transport["padding_ltrb"], dtype=np.int32)
        np.savez(artifact_path.pose_path, **pose_fields)

        artifact_path.intrinsics_path.parent.mkdir(exist_ok=True, parents=True)
        k4_rows = np.repeat(np.asarray(k4, dtype=np.float32)[None], count, axis=0)
        np.savez(artifact_path.intrinsics_path, data=k4_rows, inds=inds)
        with artifact_path.camera_type_path.open("w") as handle:
            for frame_idx in range(count):
                handle.write(f"{frame_idx}: {CameraType.PINHOLE.name}\n")

        artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
        with artifact_path.meta_info_path.open("wb") as handle:
            pickle.dump({"ba_residual": final.ba_residual}, handle)
        path = record_path(artifact_path)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(jsonable(record), indent=2, sort_keys=True, allow_nan=True) + "\n")
        temporary.replace(path)
