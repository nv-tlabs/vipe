# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration of the MetricTrail pipeline (``pipeline=metrictrail``).

``MetricTrailPipelineConfig`` is registered in ViPE's ``PipelineConfig`` union
(``vipe/config/pipeline.py``), so this module must not import
``vipe.config.pipeline`` (or the ``vipe.config`` package namespace) itself; it
only uses ``vipe.config.base_schema`` and the SharedK schema.

Presets: ``configs/pipeline/metrictrail.yaml`` (the one pipeline) and the slam
group ``configs/slam/metrictrail_{local,gba,gba_refine}.yaml``, selected with
``slam@pipeline.slam=metrictrail_gba``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator

from vipe.config.base_schema import BaseConfigSchema, Field

from .calibration.config import SharedKCalibrationConfig
from .optional.config import MetricTrailOptionalConfig

#: Metric-depth validity range of MetricTrail's depth provider (``depth.py``); DPVO's
#: METRIC_DEPTH_MIN/MAX must equal it.
METRIC_DEPTH_RANGE_M = (0.2, 500.0)

PIPELINE_INSTANCE = "vipe.metrictrail.pipeline.MetricTrailPipeline"

#: Smallest DPVO BUFFER_SIZE of the automatic rule.
DPVO_BUFFER_FLOOR = 8192


def resolve_dpvo_buffer_size(configured: int | str, frame_count: int) -> int:
    """DPVO BUFFER_SIZE for a sequence of ``frame_count`` tracked frames.

    ``"auto"``: the smallest power of two >= ``frame_count + 2``, at least
    :data:`DPVO_BUFFER_FLOOR`. DPVO raises when ``n + 1 >= BUFFER_SIZE`` (``n`` keyframes
    held, at most the frames seen so far) and writes slots ``n`` and ``n + 1``, so
    ``frame_count + 2`` slots always suffice. The size is capacity only (about 13.4 kB
    of GPU memory per slot with 96 patches); it cannot change a tracked value. An
    explicit integer wins.
    """

    if configured == "auto":
        needed = int(frame_count) + 2
        if needed <= 2:
            raise ValueError("buffer_size auto needs a positive frame count")
        return max(DPVO_BUFFER_FLOOR, 1 << (needed - 1).bit_length())
    if isinstance(configured, bool):
        raise TypeError('buffer_size must be "auto" or a positive integer')
    size = int(configured)
    if size <= 0:
        raise ValueError("buffer_size must be positive")
    return size


class DPVOConfig(BaseConfigSchema):
    """Every key of DPVO's configuration (vendored ``vipe/metrictrail/dpvo/config.py``), with DPVO's defaults.

    The whole block is merged over DPVO's defaults with yacs. Keys not listed in a slam
    preset keep DPVO's default; add them on the command line with ``+``.
    """

    BUFFER_SIZE: int = Field(default=4096, ge=1, description="Keyframe buffer capacity (see slam.buffer_size).")
    CENTROID_SEL_STRAT: Literal["RANDOM", "GRADIENT_BIAS"] = Field(
        default="RANDOM", description="Patch centroid selection."
    )
    PATCHES_PER_FRAME: int = Field(default=80, ge=1, description="Patches sampled per frame.")
    REMOVAL_WINDOW: int = Field(default=20, ge=1, description="Frames after which factors leave the local window.")
    OPTIMIZATION_WINDOW: int = Field(default=12, ge=1, description="Keyframes optimised by the local BA.")
    PATCH_LIFETIME: int = Field(default=12, ge=1, description="Frames a patch is tracked into.")
    KEYFRAME_INDEX: int = Field(default=4, ge=1, description="Age of the keyframe tested for removal.")
    KEYFRAME_THRESH: float = Field(default=12.5, ge=0.0, description="Flow (px) below which a keyframe is removed.")
    MOTION_MODEL: str = Field(
        default="DAMPED_LINEAR", description="Pose initialisation of a new frame (anything else: constant pose)."
    )
    MOTION_DAMPING: float = Field(default=0.5, description="Damping of the linear motion model.")
    MIXED_PRECISION: bool = Field(default=True, description="Run DPVO's network under FP16 autocast.")
    METRIC_DEPTH: bool = Field(default=False, description="Metric depth initialises patches and adds a unary prior.")
    METRIC_DEPTH_MIN: float = Field(default=0.2, gt=0.0, description="Smallest metric depth used as a prior (m).")
    METRIC_DEPTH_MAX: float = Field(default=500.0, gt=0.0, description="Largest metric depth used as a prior (m).")
    METRIC_DEPTH_PRIOR_WEIGHT: float = Field(default=1.0, ge=0.0, description="Weight of the metric unary.")
    METRIC_DEPTH_PRIOR_HUBER: float = Field(default=0.3, gt=0.0, description="Huber width of the metric unary.")
    METRIC_DEPTH_PRIOR_WARMUP: int = Field(
        default=12, ge=0, description="Accepted keyframes before the metric unary is switched on."
    )
    LOOP_CLOSURE: bool = Field(default=False, description="Proximity loop edges (only with METRIC_GLOBAL_BA).")
    BACKEND_THRESH: float = Field(default=64.0, description="Flow (px) below which a proximity edge is added.")
    MAX_EDGE_AGE: int = Field(default=1000, ge=1, description="Oldest frame a proximity edge may reach.")
    GLOBAL_OPT_FREQ: int = Field(default=15, ge=1, description="Keyframes between proximity-edge searches.")
    CLASSIC_LOOP_CLOSURE: bool = Field(default=False, description="Retrieval loop closure (refused by MetricTrail).")
    LOOP_CLOSE_WINDOW_SIZE: int = Field(default=3, description="Retrieval loop-closure window (unused).")
    LOOP_RETR_THRESH: float = Field(default=0.04, description="Retrieval loop-closure threshold (unused).")
    METRIC_GLOBAL_BA: bool = Field(default=False, description="Metric-scale global BA over stored factors.")
    METRIC_GLOBAL_BA_T0: Literal["active_min", "all"] = Field(
        default="active_min", description="Poses freed by the global BA: all (every pose but the first) or active_min."
    )
    GBA_SCHEDULE_K: int = Field(default=0, ge=0, description="Also run the global BA every K keyframes (0 = off).")
    CULLED_INTERP: bool = Field(default=False, description="Spread keyframe corrections over culled frames.")
    GBA_REMEASURE_FINAL: Literal["off", "one", "two"] = Field(
        default="off", description="Final re-measured global BA passes at terminate."
    )
    FMAP_STORE: int = Field(default=512, ge=36, description="Feature-map ring length while re-measuring.")
    REMEASURE_MIN_AGE: int = Field(default=0, ge=0, description="Only re-measure edges at least this old.")
    REMEASURE_BURNIN: int = Field(default=0, ge=0, description="Refreshes before the first BA of each pass.")
    REMEASURE_WARM_ACTIVE: bool = Field(default=False, description="Warm-start active edges in pass 0.")
    REMEASURE_CARRY_STATE: bool = Field(default=False, description="Carry the update state across passes.")
    GBA_SCALE_LATENT: bool = Field(default=False, description="Per-keyframe log-scale latent of the metric prior.")
    GBA_SCALE_SMOOTH: float = Field(default=20.0, description="Smoothness weight of the scale latent.")
    GBA_SCALE_CLAMP: float = Field(default=0.35, description="Bound of the scale latent (log units).")
    GBA_SCALE_MIN_PATCHES: int = Field(default=32, ge=0, description="Minimum prior patches for a data term.")
    GBA_SCALE_MIN_PARALLAX_PX: float = Field(default=2.0, description="Minimum parallax (px) for a data term.")
    GBA_SCALE_LOCAL_USE: Literal["off", "frozen"] = Field(default="off", description="Local-BA use of the latent.")
    LOOP_REVERSE_EDGES: bool = Field(default=False, description="Also add the reverse proximity edges.")
    KEYFRAME_MAX_CULL: int = Field(default=0, ge=0, description="Keep a keyframe after this many culls (0 = off).")
    KEYFRAME_MAX_CULL_ROT_DEG: float = Field(
        default=10.0, ge=0.0, description="Rotation (deg) the culled run must also exceed."
    )


class MetricTrailSLAMConfig(BaseConfigSchema):
    """MetricTrail tracker: DINO-fused DPVO fed with per-frame metric depth (slam group ``metrictrail_*``)."""

    preset: Literal["local", "gba", "gba_refine", "custom"] = Field(
        description="Name of the DPVO backend preset (recorded with the run). local: fixed-lag local BA; gba: + "
        "proximity edges and metric global BA; gba_refine: + the full global-BA backend."
    )
    dpvo: DPVOConfig = Field(description="DPVO configuration merged over DPVO's defaults.")
    buffer_size: int | Literal["auto"] = Field(
        default="auto",
        description="DPVO BUFFER_SIZE applied after the preset (MetricTrail's dpvo_buffer_size; GPU memory only). "
        "auto: the smallest power of two >= frames + 2, at least 8192; an integer wins.",
    )
    tf32: bool = Field(
        default=False,
        description="TF32 for CUDA matmul and cuDNN during tracking. False (default): both off and float32 matmul "
        "precision highest; per-frame networks run under FP16/BF16 autocast.",
    )
    keyframe_depth: str = Field(
        default="moge2-l",
        description="Metric depth model for every frame DPVO receives (ViPE depth-registry name; moge2* uses "
        "MetricTrail's exact MoGe-2 call). It must return metric depth for a pinhole K.",
    )
    depth_fp16: bool = Field(default=True, description="Autocast the depth model to FP16 (weights stay FP32).")
    depth_queue: int = Field(
        default=4,
        ge=0,
        description="Frames the depth producer may run ahead of DPVO on an auxiliary CUDA stream; 0 runs depth "
        "and DPVO sequentially on the default stream (reference path).",
    )
    seed: int | None = Field(
        default=1234,
        description="Base tracking seed; the per-sequence seed is derived from the stream name (see seed_name), and "
        "the FP32 recovery of a non-finite trajectory (nonfinite_recovery) tracks at that seed + 0, 1, 2, 3. Null "
        "leaves the RNGs unseeded.",
    )
    seed_name: str | None = Field(
        default=None,
        description="Name the per-sequence seed is derived from. Null: the stream name with spaces replaced by "
        "'_'. Set it to reproduce a trajectory tracked under another name, e.g. demo for "
        "assets/examples/metrictrail_demo.mp4.",
    )
    nonfinite_recovery: bool = Field(
        default=True,
        description="Non-finite poses: the sequence is tracked once, at the per-sequence seed, with the preset as "
        "configured; only if a pose is non-finite is the cached DINO-DPVO network released (evicted from the model "
        "cache) and a freshly loaded one tracks it again in full FP32 (DPVO MIXED_PRECISION off, DINOv3 autocast "
        "off, TF32 off; every other preset setting, loop edges and global BA included, kept) at the per-sequence "
        "seed + 0, 1, 2, 3 in that order (unseeded with seed null), keeping the first all-finite trajectory; an "
        "attempt that runs out of CUDA memory is recorded and counts as failed; the run fails if none is kept. A "
        "finite first pass is kept as is. Every pass is recorded (nonfinite_recovery in the run record). False: a "
        "non-finite trajectory fails the run.",
    )
    rgb_normalization: Literal["metrictrail", "stream"] = Field(
        default="metrictrail",
        description="metrictrail re-quantises each frame to 8 bit and rebuilds float RGB as uint8 * (1/255) in "
        "float32; stream uses the stream's float RGB as is.",
    )
    weights: str | None = Field(
        default=None,
        description="DINO-DPVO checkpoint. Null: $VIPE_METRICTRAIL_DPVO_CHECKPOINT, else "
        "$TORCH_HOME/hub/metrictrail/dino_dpvo_multidataset15k.pth.",
    )
    dinov3_weights: str | None = Field(
        default=None,
        description="DINOv3 ViT-S/16 weights. Null: $VIPE_DINOV3_VITS16_CHECKPOINT, else "
        "$TORCH_HOME/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth.",
    )
    visualize: bool = Field(default=False, description="Accepted for `vipe infer -v`; DPVO has no viewer here.")

    @field_validator("buffer_size", mode="before")
    @classmethod
    def validate_buffer_size(cls, value: object) -> object:
        if value == "auto":
            return value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'buffer_size must be "auto" or a positive integer, got {value!r}')
        return value

    @model_validator(mode="after")
    def validate_metric_dpvo(self) -> MetricTrailSLAMConfig:
        dpvo = self.dpvo
        if not dpvo.METRIC_DEPTH:
            raise ValueError("MetricTrail needs dpvo.METRIC_DEPTH=true (metric-depth FastBA)")
        if dpvo.CLASSIC_LOOP_CLOSURE:
            raise ValueError("DPVO retrieval loop closure (dpvo.CLASSIC_LOOP_CLOSURE) must be disabled")
        if dpvo.LOOP_CLOSURE != dpvo.METRIC_GLOBAL_BA:
            raise ValueError(
                "dpvo.LOOP_CLOSURE (proximity loop edges) is allowed only together with dpvo.METRIC_GLOBAL_BA, and "
                f"METRIC_GLOBAL_BA needs it: LOOP_CLOSURE={dpvo.LOOP_CLOSURE}, METRIC_GLOBAL_BA={dpvo.METRIC_GLOBAL_BA}"
            )
        if (dpvo.METRIC_DEPTH_MIN, dpvo.METRIC_DEPTH_MAX) != METRIC_DEPTH_RANGE_M:
            raise ValueError(
                f"dpvo metric depth bounds ({dpvo.METRIC_DEPTH_MIN}, {dpvo.METRIC_DEPTH_MAX}) differ from the depth "
                f"provider's {METRIC_DEPTH_RANGE_M}"
            )
        return self


#: ViPE's ``init.instance`` defaults (``configs/pipeline/default.yaml``).
VIPE_INSTANCE_PHRASES = ("person", "animal", "vehicle", "ball", "balloon", "gun", "pet", "car", "bus")


class MetricTrailInstanceConfig(BaseConfigSchema):
    """ViPE's dynamic-object masking: the fields (and defaults) of ViPE's ``init.instance``."""

    kf_gap_sec: float = Field(
        default=2.0,
        gt=0.0,
        description="Minimum time gap, in seconds, between keyframes used to initialize instance segmentation.",
    )
    phrases: list[str] = Field(
        default=list(VIPE_INSTANCE_PHRASES),
        min_length=1,
        description="Text prompts passed to the open-vocabulary detector for objects that should be segmented.",
    )
    add_sky: bool = Field(
        default=True, description="Add a sky mask to the instance segmentation output (the sky is not masked out)."
    )


class InputRasterConfig(BaseConfigSchema):
    """Input raster and frame rate (in-the-wild default: cover-fit to 280x504 with an integer centre crop)."""

    # Every resize is the resampler's (default bicubic with antialias) followed by an integer crop; cover_crop keeps the
    # aspect ratio by cropping, resize changes it. Both keep the source frame rate unless the input has more than
    # max_frames frames, then keep one frame in k (smallest k that fits). See vipe/metrictrail/raster.py.
    policy: Literal["cover_crop", "resize", "unchanged"] = Field(
        default="cover_crop",
        description="cover_crop: resize (resampler) to cover height x width, then an integer centre crop (the "
        "aspect ratio changes by cropping, never by squashing); the geometry and frame selection of "
        "tools/prepare_video.py (in-the-wild video). resize: resize (resampler) to exactly height x width, no "
        "crop (7-Scenes). Both keep the source frame rate and keep one frame in k above max_frames. unchanged: the "
        "input raster and every frame as delivered (pre-prepared frames, e.g. benchmark frame directories).",
    )
    # Any size: DPVO reflect-pads a raster that is not divisible by 4 (e.g. 7-Scenes' 434x574).
    height: int = Field(default=280, ge=4, description="Prepared height (px).")
    width: int = Field(default=504, ge=4, description="Prepared width (px).")
    max_frames: int = Field(
        default=1500, ge=1, description="Inputs with more frames keep one frame in k, the smallest k that fits."
    )
    resampler: Literal["bicubic_antialias", "bilinear"] = Field(
        default="bicubic_antialias",
        description="Resize of cover_crop / resize: bicubic_antialias (default; torch interpolate, bicubic, "
        "antialias; closest to an ffmpeg Lanczos resize, about 2 ms more per frame once before tracking; a large "
        "downscale without antialiasing aliases) or bilinear (ViPE's VideoFrame.resize, no antialias; reproduces "
        "the validation numbers of the benchmarks resized on read).",
    )


class MetricTrailInitConfig(BaseConfigSchema):
    """Initialization: RGB-only shared intrinsics (SharedK) calibrated from the input video itself."""

    camera_type: Literal["pinhole"] = Field(description="Camera model (MetricTrail tracks pinhole video).")
    intrinsics: Literal["sharedk"] = Field(
        description="Source of the shared K. sharedk calibrates it from RGB only and refuses inputs that carry K "
        "(a known K is the optional input pipeline.optional.intrinsics, which replaces SharedK)."
    )
    calibration: SharedKCalibrationConfig = Field(description="SharedK calibration settings.")
    calibration_cache: str | None = Field(
        default=None,
        description="Optional frozen SharedK artifact path: create once, then reuse its model-estimated K after "
        "checking all prepared RGB frames and calibration settings. A mismatch raises; null recalibrates every run. "
        "Use a new path after changing calibration code or weights. Incompatible with optional known intrinsics.",
    )
    on_truncated_input: Literal["error", "warn"] = Field(
        default="error",
        description="What to do when the input stream stops before its source ends (e.g. frame_end set below the "
        "video length): MetricTrail processes full sequences.",
    )
    instance: MetricTrailInstanceConfig | None = Field(
        default=None,
        description="ViPE's dynamic-object masking (ViPE's init.instance): TrackAnything runs in the init stage, no "
        "DPVO patch is placed on a dynamic pixel, dynamic pixels get no metric prior and leave the SLAM map, and the "
        "masks are written as ViPE mask artifacts. Null (default) turns it off.",
    )
    input_raster: InputRasterConfig = Field(
        description="Input raster and frame-rate policy, applied before calibration (preset: in-the-wild cover_crop "
        "to 280x504; resize for a plain resize; unchanged for pre-prepared frames)."
    )
    tracking_stride: int = Field(
        default=1,
        ge=1,
        description="Track every k-th frame of the prepared input (frames 0, k, 2k, ...) while SharedK calibrates on "
        "every frame; the pose artifact's source_frame_ids name the tracked frames. 1 tracks every frame; 7-Scenes "
        "uses 5 on the stride-1 clip.",
    )


class FocalRefineConfig(BaseConfigSchema):
    """Focal refinement (stage B): one RoMa revisit-pair fit of the shared focal, then an accept gate.

    Flow: route (GT-free gates on stage A) -> one stage-B attempt -> accept gate
    -> if accepted, rebind the focal and re-track. The remaining fields are the
    stage-B settings.
    """

    enabled: bool = Field(default=False, description="Run focal refinement after the first tracking pass.")
    route_automatically: bool = Field(
        default=True, description="Enter stage B only when the routing gates on stage A's output say so."
    )
    seed: int = Field(default=7, description="Seed of RoMa's correspondence sampling.")
    roma_python: str | None = Field(
        default=None,
        description="Interpreter that runs stage B (RoMa). Null: $VIPE_METRICTRAIL_ROMA_PYTHON, else this one.",
    )
    roma_repo: str | None = Field(
        default=None, description="RoMa checkout put on sys.path of that interpreter; null: romatch must import."
    )
    roma_checkpoints: str | None = Field(
        default=None,
        description="Directory with roma_outdoor.pth and dinov2_vitl14_pretrain.pth. Null: "
        "$VIPE_METRICTRAIL_ROMA_CHECKPOINTS, else $TORCH_HOME/hub/metrictrail/roma.",
    )
    workdir: str | None = Field(
        default=None, description="Scratch directory of stage B; null: <output.path>/metrictrail/<name>/stage_b."
    )
    budget: int = Field(default=800, ge=1, description="Maximum revisit pairs matched.")
    radius_m: float = Field(default=15.0, gt=0.0, description="Revisit radius on stage A's trajectory (m).")
    min_frame_gap: int = Field(default=200, ge=0, description="Minimum frame gap of a revisit pair.")
    trajectory_stride: int = Field(default=5, ge=1, description="Frame stride of the revisit search.")
    candidate_band_deg: tuple[float, float] = Field(default=(25.0, 140.0), description="Heading band of candidates.")
    coarse_resolution: int = Field(default=560, ge=1, description="RoMa matching resolution.")
    upsample_predictions: bool = Field(default=False, description="RoMa upsampled refinement (off).")
    samples_per_pair: int = Field(default=3000, ge=1, description="Correspondences sampled per pair.")
    max_correspondences: int = Field(default=150, ge=1, description="Inliers kept per pair.")
    min_inliers: int = Field(default=100, ge=1, description="Minimum fundamental-matrix inliers per pair.")
    ransac_threshold_px: float = Field(default=0.6, gt=0.0, description="MAGSAC threshold (px).")
    grid_relative: tuple[float, float] = Field(default=(0.60, 1.30), description="Focal grid relative to stage A.")
    max_rail_fraction: float = Field(default=0.10, ge=0.0, description="Accept gate: bootstrap rail fraction.")
    max_deviation_pct: float = Field(default=25.0, ge=0.0, description="Accept gate: change from stage A (%).")
    max_estimator_disagreement_pct: float = Field(
        default=15.0, ge=0.0, description="Accept gate: centre-pinned vs profiled focal (%)."
    )
    centre_halfwidth_px: float = Field(default=40.0, ge=0.0, description="Principal-point column range (px).")
    centre_grid_points: int = Field(default=41, ge=1, description="Principal-point columns profiled.")
    profile_points_per_pair: int = Field(default=60, ge=1, description="Correspondences per pair in the profile.")
    bootstrap_draws: int = Field(default=400, ge=1, description="Bootstrap resamples.")
    fit_band_deg: tuple[float, float] = Field(default=(25.0, 140.0), description="Heading band of the fit.")
    gate_revisit_count: int = Field(default=1000, ge=0, description="Routing gate: revisit candidates.")
    gate_revisit_per_frame: float = Field(default=0.15, ge=0.0, description="Routing gate: candidates per frame.")
    gate_focal_drift_pct: float = Field(default=0.20, ge=0.0, description="Routing gate: minimum final focal drift.")
    gate_focal_drift_max_pct: float = Field(default=2.5, ge=0.0, description="Routing gate: maximum final focal drift.")


class MetricTrailPostConfig(BaseConfigSchema):
    """Depth post-processing after tracking: ViPE's AdaptiveDepthProcessor on the DPVO output."""

    depth_align_model: str | None = Field(
        default=None,
        description="ViPE's adaptive depth recipe adaptive_<metric>[_svda|_vda] (e.g. adaptive_unidepth-l_svda, "
        "adaptive_unidepth-l_vda, adaptive_moge_vda, adaptive_unidepth-l), run as in ViPE's default pipeline and "
        "saved as ViPE depth artifacts. mvd_* is not supported. Null (default): no depth fusion.",
    )

    @model_validator(mode="after")
    def validate_depth_align_model(self) -> MetricTrailPostConfig:
        model = self.depth_align_model
        if model is None:
            return self
        if model.startswith("mvd_"):
            raise ValueError(
                f"post.depth_align_model={model!r}: ViPE's multi-view depth (mvd_*) is not supported with the DPVO "
                "backend; use an adaptive_<metric>[_svda|_vda] recipe or null"
            )
        parts = model.split("_")
        if (
            parts[0] != "adaptive"
            or len(parts) not in (2, 3)
            or not parts[1]
            or parts[2:] not in ([], ["svda"], ["vda"])
        ):
            raise ValueError(
                f"post.depth_align_model={model!r} is not one of ViPE's adaptive_<metric>[_svda|_vda] recipes "
                "(e.g. adaptive_unidepth-l_svda, adaptive_moge_vda, adaptive_unidepth-l)"
            )
        return self


class MetricTrailOutputConfig(BaseConfigSchema):
    """Output artifacts (ViPE's layout) and the MetricTrail run record."""

    path: str = Field(description="Directory where artifacts are written.")
    skip_exists: bool = Field(description="Skip a sequence whose pose artifact and run record already exist.")
    save_artifacts: bool = Field(
        description="Save pose, intrinsics, camera type, the metric depth (save_metric_depth) and the MetricTrail run "
        "record."
    )
    save_slam_map: bool = Field(default=False, description="Save DPVO's patch point map as the ViPE SLAM map.")
    save_metric_depth: Literal["async", "off"] = Field(
        default="async",
        description="async: save the per-frame MoGe-2 metric depth of the final tracking pass as "
        "depth/<name>.zip (ViPE's depth artifact format), written by a background thread off the tracking "
        "path. off: pose-only.",
    )
    save_viz: bool = Field(description="Render ViPE's MP4 visualization of the output stream.")
    viz_downsample: int = Field(ge=1, description="Downsample factor of the visualization video.")
    viz_attributes: list[list[Literal["rgb", "instance", "depth", "pcd", "rectified"]]] = Field(
        min_length=1, description="Groups of frame attributes rendered into the visualization video."
    )


class MetricTrailPipelineConfig(BaseConfigSchema):
    """MetricTrail: RGB-only shared intrinsics, MoGe-2 metric depth and DINO-fused DPVO tracking."""

    instance: Literal["vipe.metrictrail.pipeline.MetricTrailPipeline"] = Field(
        description="Implementation class of the MetricTrail pipeline."
    )
    init: MetricTrailInitConfig = Field(description="Shared-intrinsics initialization.")
    slam: MetricTrailSLAMConfig = Field(description="DPVO tracker and its backend preset (slam group).")
    focal_refine: FocalRefineConfig = Field(description="Optional focal refinement (stage B).")
    post: MetricTrailPostConfig = Field(description="Optional ViPE depth fusion after tracking.")
    optional: MetricTrailOptionalConfig = Field(
        description="Optional inputs/outputs, all off by default: known intrinsics, loop closure "
        "(docs/metrictrail_optional.md)."
    )
    output: MetricTrailOutputConfig = Field(description="Output artifacts.")

    @model_validator(mode="after")
    def validate_calibration_cache(self) -> MetricTrailPipelineConfig:
        if self.init.calibration_cache is not None:
            if not self.init.calibration_cache.strip():
                raise ValueError("init.calibration_cache must be a nonempty artifact path")
            if self.optional.intrinsics is not None:
                raise ValueError("init.calibration_cache cannot be combined with optional.intrinsics")
        return self

    @model_validator(mode="after")
    def validate_loop_closure(self) -> MetricTrailPipelineConfig:
        if self.optional.loop_closure.enabled and (self.slam.dpvo.METRIC_GLOBAL_BA or self.slam.dpvo.LOOP_CLOSURE):
            # Global BA and loop closure are exclusive: loop closure needs a local-BA preset.
            raise ValueError("optional.loop_closure needs a local-BA trajectory (slam@pipeline.slam=metrictrail_local)")
        if self.optional.loop_closure.enabled and not self.output.save_artifacts:
            raise ValueError("optional.loop_closure reads pose/<name>.npz and needs output.save_artifacts=true")
        return self
