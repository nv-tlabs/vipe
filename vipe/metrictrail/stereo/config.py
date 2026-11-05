# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration of the MetricTrail stereo runner (``python -m vipe.metrictrail.stereo``).

The defaults are the KITTI stereo setup:

- ``dpvo``: :data:`STEREO_DPVO_PRESET` (local BA, rectified-stereo factors, metric
  depth), merged over the stereo DPVO tree's defaults (``dpvo_stereo/config.py``);
- the runner keys: MoGe-2 depth in FP16 (``depth_fp16``), ``depth_queue: 4``,
  ``seed: 1234`` and the default ``buffer_size: auto`` (the smallest power of two
  >= frames + 2, at least 8192). Known K and the baseline come from the command line.

``stereo.yaml`` beside this module holds the same values; :func:`load_stereo_config`
reads it (or another file with the same keys). This module imports nothing from the
pipeline registry (``vipe.config.pipeline``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml  # type: ignore[import-untyped]
from pydantic import field_validator, model_validator

from vipe.config.base_schema import BaseConfigSchema, Field

from ..config import METRIC_DEPTH_RANGE_M, resolve_dpvo_buffer_size

#: The packaged default configuration (equal to ``StereoRunnerConfig()``).
DEFAULT_CONFIG_PATH = Path(__file__).with_name("stereo.yaml")

#: The stereo DPVO preset (the ``dpvo`` default below and the ``dpvo`` block of ``stereo.yaml``).
STEREO_DPVO_PRESET: dict[str, Any] = {
    "BUFFER_SIZE": 8192,
    "PATCHES_PER_FRAME": 96,
    "REMOVAL_WINDOW": 22,
    "OPTIMIZATION_WINDOW": 10,
    "PATCH_LIFETIME": 13,
    "KEYFRAME_THRESH": 15.0,
    "MOTION_MODEL": "DAMPED_LINEAR",
    "MOTION_DAMPING": 0.5,
    "MIXED_PRECISION": True,
    "CENTROID_SEL_STRAT": "RANDOM",
    "LOOP_CLOSURE": False,
    "CLASSIC_LOOP_CLOSURE": False,
    "STEREO": True,
    "STEREO_WEIGHT": 1.0,
    "METRIC_DEPTH": True,
    "METRIC_DEPTH_MIN": 0.2,
    "METRIC_DEPTH_MAX": 500.0,
    "METRIC_DEPTH_PRIOR_WEIGHT": 1.0,
    "METRIC_DEPTH_PRIOR_HUBER": 0.3,
    "METRIC_DEPTH_PRIOR_WARMUP": 12,
}


class StereoDPVOConfig(BaseConfigSchema):
    """Every key of the stereo DPVO configuration (vendored ``dpvo_stereo/config.py``), with its defaults.

    The whole block is merged over those defaults with yacs.
    """

    BUFFER_SIZE: int = Field(default=4096, ge=1, description="Keyframe buffer capacity (see buffer_size).")
    CENTROID_SEL_STRAT: Literal["RANDOM", "GRADIENT_BIAS"] = Field(
        default="RANDOM", description="Patch centroid selection."
    )
    PATCHES_PER_FRAME: int = Field(default=80, ge=1, description="Patches sampled per frame.")
    REMOVAL_WINDOW: int = Field(default=20, ge=1, description="Frames after which factors leave the local window.")
    OPTIMIZATION_WINDOW: int = Field(default=12, ge=1, description="Keyframes optimised by the local BA.")
    PATCH_LIFETIME: int = Field(default=12, ge=1, description="Frames a patch is tracked into.")
    KEYFRAME_INDEX: int = Field(default=4, ge=1, description="Age of the keyframe tested for removal.")
    KEYFRAME_THRESH: float = Field(default=12.5, ge=0.0, description="Flow (px) below which a keyframe is removed.")
    MOTION_MODEL: str = Field(default="DAMPED_LINEAR", description="Pose initialisation of a new frame.")
    MOTION_DAMPING: float = Field(default=0.5, description="Damping of the linear motion model.")
    MIXED_PRECISION: bool = Field(default=True, description="Run DPVO's network under FP16 autocast.")
    STEREO: bool = Field(default=False, description="Rectified left-to-right factors from the right image.")
    STEREO_WEIGHT: float = Field(default=1.0, gt=0.0, description="Weight of the stereo factors.")
    METRIC_DEPTH: bool = Field(default=False, description="Metric depth initialises patches and adds a unary prior.")
    METRIC_DEPTH_MIN: float = Field(default=0.2, gt=0.0, description="Smallest metric depth used as a prior (m).")
    METRIC_DEPTH_MAX: float = Field(default=500.0, gt=0.0, description="Largest metric depth used as a prior (m).")
    METRIC_DEPTH_PRIOR_WEIGHT: float = Field(default=1.0, ge=0.0, description="Weight of the metric unary.")
    METRIC_DEPTH_PRIOR_HUBER: float = Field(default=0.3, gt=0.0, description="Huber width of the metric unary.")
    METRIC_DEPTH_PRIOR_WARMUP: int = Field(
        default=12, ge=0, description="Accepted keyframes before the metric unary is switched on."
    )
    LOOP_CLOSURE: bool = Field(default=False, description="Proximity loop edges (refused: local BA only).")
    BACKEND_THRESH: float = Field(default=64.0, description="Flow (px) below which a proximity edge is added.")
    MAX_EDGE_AGE: int = Field(default=1000, ge=1, description="Oldest frame a proximity edge may reach.")
    GLOBAL_OPT_FREQ: int = Field(default=15, ge=1, description="Keyframes between proximity-edge searches.")
    CLASSIC_LOOP_CLOSURE: bool = Field(default=False, description="Retrieval loop closure (refused).")
    LOOP_CLOSE_WINDOW_SIZE: int = Field(default=3, description="Retrieval loop-closure window (unused).")
    LOOP_RETR_THRESH: float = Field(default=0.04, description="Retrieval loop-closure threshold (unused).")


class StereoRunnerConfig(BaseConfigSchema):
    """MetricTrail stereo tracking: DINO-fused stereo DPVO with per-frame metric depth and a known K.

    The field names shared with ``MetricTrailSLAMConfig`` (``keyframe_depth``,
    ``depth_fp16``, ``depth_queue``, ``buffer_size``, ``tf32``, ``seed``,
    ``nonfinite_recovery``, ``rgb_normalization``, ``weights``, ``dinov3_weights``) have
    the same meaning there and here.
    """

    dpvo: StereoDPVOConfig = Field(
        default_factory=lambda: StereoDPVOConfig(**STEREO_DPVO_PRESET),
        description="Stereo DPVO configuration merged over the stereo tree's defaults (STEREO_DPVO_PRESET).",
    )
    buffer_size: int | Literal["auto"] = Field(
        default="auto",
        description="DPVO BUFFER_SIZE applied after dpvo (GPU memory only). auto: the "
        "smallest power of two >= frames + 2, at least 8192; an integer wins.",
    )
    tf32: bool = Field(
        default=False,
        description="TF32 for CUDA matmul and cuDNN during tracking. False (default): both off and float32 "
        "matmul precision highest; per-frame networks run under FP16/BF16 autocast.",
    )
    keyframe_depth: str = Field(
        default="moge2-l",
        description="Metric depth model for every left frame (ViPE depth-registry name; moge2* uses MetricTrail's "
        "exact MoGe-2 call).",
    )
    depth_fp16: bool = Field(default=True, description="Autocast the depth model to FP16 (weights stay FP32).")
    depth_queue: int = Field(
        default=4,
        ge=0,
        description="Frames the depth producer may run ahead of DPVO on an auxiliary CUDA stream; 0 = sequential.",
    )
    seed: int | None = Field(
        default=1234,
        description="Base tracking seed; the per-sequence seed adds the sequence name (int(name), else 4 bytes of "
        "its SHA-256), and the FP32 recovery of a non-finite trajectory (nonfinite_recovery) tracks at that seed + "
        "0, 1, 2, 3. Null: unseeded.",
    )
    nonfinite_recovery: bool = Field(
        default=True,
        description="Non-finite poses, as in the mono pipeline: the sequence is tracked once at the per-sequence "
        "seed with the configuration as given; only if a pose is non-finite is the cached network released and a "
        "freshly loaded DINO-DPVO network tracks it again in full FP32 (DPVO MIXED_PRECISION off, DINOv3 autocast "
        "off, TF32 off; nothing else changes) at the per-sequence seed + 0, 1, 2, 3, keeping the first all-finite "
        "trajectory; an attempt that runs out of CUDA memory counts as failed; the run fails if none is kept. "
        "Every pass is recorded (tracking.nonfinite_recovery). False: a non-finite trajectory fails the run.",
    )
    rgb_normalization: Literal["metrictrail", "stream"] = Field(
        default="metrictrail",
        description="metrictrail rebuilds each left frame as uint8 * (1/255) in float32; stream uses the stream's "
        "float RGB.",
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
    on_truncated_input: Literal["error", "warn"] = Field(
        default="error", description="What to do when an input stream stops before its source ends."
    )
    save_slam_map: bool = Field(default=False, description="Also save DPVO's patch point map as the ViPE SLAM map.")

    @field_validator("buffer_size", mode="before")
    @classmethod
    def validate_buffer_size(cls, value: object) -> object:
        if value == "auto":
            return value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'buffer_size must be "auto" or a positive integer, got {value!r}')
        return value

    @model_validator(mode="after")
    def validate_stereo_dpvo(self) -> StereoRunnerConfig:
        dpvo = self.dpvo
        if not dpvo.STEREO:
            raise ValueError("the stereo runner needs dpvo.STEREO=true")
        if not dpvo.METRIC_DEPTH:
            raise ValueError("the stereo runner needs dpvo.METRIC_DEPTH=true (metric-depth FastBA)")
        if dpvo.CLASSIC_LOOP_CLOSURE:
            raise ValueError("DPVO retrieval loop closure (dpvo.CLASSIC_LOOP_CLOSURE) must be disabled")
        if dpvo.LOOP_CLOSURE:
            # LOOP_CLOSURE is allowed only with the metric global BA, which the stereo tree does not have.
            raise ValueError("dpvo.LOOP_CLOSURE must be false: the stereo DPVO tree has no metric global BA")
        if (dpvo.METRIC_DEPTH_MIN, dpvo.METRIC_DEPTH_MAX) != METRIC_DEPTH_RANGE_M:
            raise ValueError(
                f"dpvo metric depth bounds ({dpvo.METRIC_DEPTH_MIN}, {dpvo.METRIC_DEPTH_MAX}) differ from the depth "
                f"provider's {METRIC_DEPTH_RANGE_M}"
            )
        return self


def load_stereo_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> StereoRunnerConfig:
    """``stereo.yaml`` (or ``path``), with top-level ``overrides`` applied, as a validated config.

    A ``dpvo`` mapping in the file or in ``overrides`` lists preset keys; keys it
    does not list keep the stereo tree's defaults (as with a yacs preset file).
    """

    source = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{source} must hold a mapping")
    value.update(overrides or {})
    return StereoRunnerConfig.model_validate(value)


def resolve_stereo_dpvo_config(
    config: StereoRunnerConfig, *, frame_count: int | None = None
) -> tuple[Any, dict[str, Any]]:
    """The stereo DPVO yacs config for ``config`` and its JSON form.

    ``dpvo`` is merged over the vendored stereo defaults, then ``BUFFER_SIZE`` is set
    to ``buffer_size`` (``auto`` is resolved for ``frame_count`` frames by the mono
    pipeline's rule, :func:`vipe.metrictrail.config.resolve_dpvo_buffer_size`), and the guards run on the
    result: metric-depth FastBA and stereo factors on, retrieval loop closure and
    proximity loop edges off, DPVO's depth bounds equal to the depth provider's.
    """

    from vipe.metrictrail.depth_producer import jsonable
    from vipe.metrictrail.dpvo_deps import CfgNode

    from .dpvo_stereo.config import cfg as base_cfg

    block = config.dpvo.model_dump()
    unknown = sorted(set(block) - set(base_cfg.keys()))
    absent = sorted(set(base_cfg.keys()) - set(block))
    if unknown or absent:
        raise RuntimeError(f"stereo dpvo schema and vendored config drifted: unknown={unknown}, absent={absent}")
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_other_cfg(CfgNode(block))
    if config.buffer_size == "auto" and frame_count is None:
        raise ValueError("buffer_size auto needs the number of frames to track")
    cfg.merge_from_list(["BUFFER_SIZE", str(resolve_dpvo_buffer_size(config.buffer_size, int(frame_count or 0)))])
    resolved = jsonable(cfg)
    if not bool(getattr(cfg, "METRIC_DEPTH", False)):
        raise ValueError("resolved DPVO config must set METRIC_DEPTH=True")
    if not bool(getattr(cfg, "STEREO", False)):
        raise RuntimeError("resolved DPVO stereo flag differs from pipeline mode")
    if bool(getattr(cfg, "CLASSIC_LOOP_CLOSURE", False)):
        raise ValueError("DPVO retrieval loop closure (CLASSIC_LOOP_CLOSURE) must be disabled")
    metric_global_ba = bool(getattr(cfg, "METRIC_GLOBAL_BA", False))
    if bool(getattr(cfg, "LOOP_CLOSURE", False)) != metric_global_ba:
        raise ValueError(
            "DPVO LOOP_CLOSURE (proximity loop edges) is allowed only together with METRIC_GLOBAL_BA: "
            f"LOOP_CLOSURE={bool(getattr(cfg, 'LOOP_CLOSURE', False))}, METRIC_GLOBAL_BA={metric_global_ba}"
        )
    if not np.isclose(float(cfg.METRIC_DEPTH_MIN), METRIC_DEPTH_RANGE_M[0]) or not np.isclose(
        float(cfg.METRIC_DEPTH_MAX), METRIC_DEPTH_RANGE_M[1]
    ):
        raise ValueError(
            f"runtime/config metric depth bounds disagree: runtime={METRIC_DEPTH_RANGE_M} "
            f"config=({cfg.METRIC_DEPTH_MIN},{cfg.METRIC_DEPTH_MAX})"
        )
    return cfg, resolved
