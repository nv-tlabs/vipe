# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SharedK calibration as a ViPE init stage.

:class:`SharedKIntrinsicsProcessor` is a ``StreamProcessor`` in the same role as
ViPE's own intrinsics estimators: built on the raw input stream, it calibrates
once at construction and then writes the shared K into every frame. It does not
touch ViPE's ``DefaultInitConfig``; MetricTrail pipelines add it to their own
init chain.

GT-free by construction: a stream that declares or carries intrinsics is
refused, calibration frames are rebuilt from RGB only, and the record states
``sensor_K_or_GT_read: false``.

With ``fallback: provider_focal`` (the default, the in-the-wild policy) a sweep that
does not converge, runs out of frames, keeps too few keyframes or raises binds the depth
provider's own focal with the principal point at the image centre instead
(:mod:`.fallback`); ``fallback: none`` keeps the benchmark behaviour.
``allow_unstable: true`` with ``provider_focal`` is the benchmark policy: a sweep that
reached its keyframe budget keeps its optimised K even when it fails the convergence gate.
"""

from __future__ import annotations

import gc
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch

from vipe.config.base_schema import config_to_primitive
from vipe.priors.depth.base import DepthEstimationModel, DepthType
from vipe.streams.base import FrameAttribute, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.model_cache import ModelCache

from .config import SharedKCalibrationConfig
from .driver import SharedFocalEngine, run_shared_focal_protocol
from .engine import AuditedDepthModel, ViPESharedFocalEngine, calibration_slam_config
from .fallback import (
    PROVIDER_FOCAL_FALLBACK,
    PROVIDER_FOCAL_STATUS,
    is_broken_installation,
    provider_focal_k4_values,
    unconverged_reason,
    vipe_error_reason,
)
from .initializer import estimate_initial_k
from .protocol import SCHEMA, plausible_k4, validate_k4

#: Largest distance (in 1/255 units) of ``rgb * 255`` from an integer accepted by
#: the metrictrail RGB normalisation.
UINT8_TOLERANCE = 1.0e-3


class GTFreeViolation(ValueError):
    """The calibration input carries intrinsics; SharedK is RGB-only."""


def _as_config(config: SharedKCalibrationConfig | Mapping[str, Any] | None) -> SharedKCalibrationConfig:
    """Accept the typed config, a plain dict, or a hydra config node."""

    if config is None:
        return SharedKCalibrationConfig()
    if isinstance(config, SharedKCalibrationConfig):
        return config
    return SharedKCalibrationConfig.model_validate(config_to_primitive(config))


def require_rgb_only(stream: VideoStream) -> None:
    if FrameAttribute.INTRINSICS in stream.attributes():
        raise GTFreeViolation(
            f"SharedK calibration is RGB-only, but stream {stream.name()!r} provides intrinsics; "
            "drop them or use a known-intrinsics configuration instead"
        )


def apply_calibration_determinism(*, seed: int, disable_tf32: bool) -> None:
    """The calibration precision/seed policy (process-wide): optional TF32 off, every RNG seeded."""

    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_calibration_frame(frame: VideoFrame, normalization: str) -> tuple[int, torch.Tensor]:
    """Return the exact RGB representation consumed by SharedK, refusing frame intrinsics.

    Args:
        frame: RGB-only source frame.
        normalization: ``metrictrail`` for uint8 quantization, or ``stream`` for float32 RGB.
    """
    if frame.intrinsics is not None:
        raise GTFreeViolation(f"frame {frame.raw_frame_idx} carries intrinsics; SharedK calibration is RGB-only")
    rgb = frame.rgb
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (H,W,3) RGB, got {tuple(rgb.shape)}")
    if normalization == "stream":
        return int(frame.raw_frame_idx), rgb.float()
    if normalization != "metrictrail":
        raise ValueError(f"unknown RGB normalization: {normalization}")
    scaled = rgb.float() * 255.0
    quantized = torch.round(scaled)
    if (
        not bool(torch.isfinite(scaled).all())
        or float((scaled - quantized).abs().max().item()) > UINT8_TOLERANCE
        or float(quantized.min().item()) < 0.0
        or float(quantized.max().item()) > 255.0
    ):
        raise ValueError(
            f"frame {frame.raw_frame_idx} is not 8-bit RGB; set rgb_normalization: stream for float inputs"
        )
    return int(frame.raw_frame_idx), quantized.to(torch.uint8)


class CalibrationFrames:
    """Reads the calibration stream once, keeping only the initializer prefix on the host."""

    def __init__(self, stream: VideoStream, *, device: torch.device, normalization: str) -> None:
        if normalization not in ("metrictrail", "stream"):
            raise ValueError(f"unknown RGB normalization: {normalization}")
        self.device = device
        self.normalization = normalization
        self._iterator: Iterator[VideoFrame] = iter(stream)
        self._head: list[tuple[int, torch.Tensor]] = []
        self._consumed_head = False

    def _canonical(self, frame: VideoFrame) -> tuple[int, torch.Tensor]:
        """Normalize one RGB-only frame for calibration.

        Args:
            frame: Input frame whose RGB and source index are retained.
        """
        return canonical_calibration_frame(frame, self.normalization)

    def device_rgb(self, canonical: torch.Tensor) -> torch.Tensor:
        """(H,W,3) float32 in [0,1] on the calibration device."""

        if canonical.dtype == torch.uint8:
            # uint8 -> float32 on the device, then an in-place division by 255 there.
            return canonical.to(device=self.device, dtype=torch.float32).div_(255.0).contiguous()
        return canonical.to(device=self.device, dtype=torch.float32).contiguous()

    def head(self, count: int) -> list[tuple[int, torch.Tensor]]:
        """The first ``count`` frames (fewer if the stream is shorter)."""

        if self._consumed_head:
            raise RuntimeError("the frame prefix was already handed to the sweep")
        while len(self._head) < count:
            try:
                source_id, canonical = self._canonical(next(self._iterator))
            except StopIteration:
                break
            self._head.append((source_id, canonical.cpu()))
        return list(self._head[:count])

    def initializer_images(self, count: int) -> "_InitializerImages":
        """The first ``count`` frames as (1,3,H,W) device images, converted one at a time."""

        return _InitializerImages(self, self.head(count))

    def sweep(self, initial_k: np.ndarray) -> Iterator[tuple[int, VideoFrame]]:
        """All frames from the first one, as RGB-only pinhole frames carrying ``initial_k``."""

        self._consumed_head = True
        k_tensor = torch.as_tensor(np.asarray(initial_k, dtype=np.float32), device=self.device)

        def make(source_id: int, canonical: torch.Tensor) -> tuple[int, VideoFrame]:
            return source_id, VideoFrame(
                raw_frame_idx=source_id,
                rgb=self.device_rgb(canonical),
                intrinsics=k_tensor.clone(),
                camera_type=CameraType.PINHOLE,
            )

        head, self._head = self._head, []
        last = -1

        def checked(source_id: int, canonical: torch.Tensor) -> tuple[int, VideoFrame]:
            nonlocal last
            if source_id <= last:
                raise RuntimeError(
                    f"calibration frames must be strictly increasing, got {source_id} after {last} "
                    "(was the stream restarted?)"
                )
            last = source_id
            return make(source_id, canonical)

        for source_id, canonical in head:
            yield checked(source_id, canonical)
        # Continue the SAME iterator with next(): `for frame in self._iterator` calls iter() on it again,
        # and ViPE's FrameDirStream / RawMp4Stream restart at frame 0 in __iter__, which replayed the
        # whole video after the initializer prefix (frames 0..31, then 0..N-1).
        while True:
            try:
                frame = next(self._iterator)
            except StopIteration:
                break
            yield checked(*self._canonical(frame))


class _InitializerImages:
    def __init__(self, frames: CalibrationFrames, head: list[tuple[int, torch.Tensor]]) -> None:
        self._frames = frames
        self._head = head

    def __len__(self) -> int:
        return len(self._head)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._frames.device_rgb(self._head[index][1]).permute(2, 0, 1).contiguous()[None]


def _initializer_source(depth_model: DepthEstimationModel) -> str:
    from vipe.metrictrail.depth import MetricTrailMoGe2

    if isinstance(depth_model, MetricTrailMoGe2):
        return "moge2_free_fov_component_median"
    return f"{type(depth_model).__name__.lower()}_free_fov_component_median"


def selection_status(
    record: Mapping[str, Any], *, allow_unstable: bool, fallback_reason: str | None = None
) -> dict[str, Any]:
    """Downstream selection fields of the calibration record.

    ``record`` is the sweep's result, or, after a provider-focal fallback
    (``fallback_reason`` set), the record whose selected K is the provider focal
    (``selected_K_source`` = :data:`PROVIDER_FOCAL_FALLBACK`). A sweep that raised
    has no ``convergence`` block and counts as not stable.
    """

    stable = bool((record.get("convergence") or {}).get("converged_without_GT", False))
    plausible = bool(record["K_plausible_without_GT"])
    if fallback_reason is not None:
        initializer_fallback = True
        status = PROVIDER_FOCAL_STATUS
    else:
        initializer_fallback = record.get("initializer_fallback_reason") is not None
        if stable:
            status = "stable_gt_free"
        elif initializer_fallback:
            status = "fallback_moge2_initializer_gt_free"
        else:
            status = "fallback_unconverged_final_optimized_K"
    return {
        "stable_for_downstream": stable and fallback_reason is None,
        "selected_for_downstream": bool(
            plausible and ((stable and fallback_reason is None) or initializer_fallback or allow_unstable)
        ),
        "selection_status": status,
        # Same value as selection_status; also repeated at the top of the run record.
        "calibration_status": status,
        "unstable_fallback_explicitly_allowed": bool(not stable and not initializer_fallback and allow_unstable),
    }


def select_shared_k(record: Mapping[str, Any], *, allow_unstable: bool) -> np.ndarray:
    """The K bound downstream (float32 ``[fx, fy, cx, cy]``); raises unless the record passes its gates."""

    if record.get("schema") != SCHEMA or record.get("status") != "success":
        raise ValueError("invalid SharedK calibration record")
    if record.get("sensor_K_or_GT_read") is not False:
        raise ValueError("SharedK record does not prove RGB-only calibration")
    if record.get("selection_uses_GT") is not False:
        raise ValueError("SharedK selection must not use GT")
    if record.get("K_plausible_without_GT") is not True:
        raise ValueError("SharedK final K failed its GT-free plausibility gate")
    stable = record.get("stable_for_downstream") is True
    # The provider's own K: the initializer (terminal policy) or the in-the-wild
    # provider-focal fallback (fallback provider_focal).
    initializer_fallback = (
        record.get("initializer_fallback_reason") is not None
        or record.get("selected_K_source") == PROVIDER_FOCAL_FALLBACK
    ) and record.get("selected_for_downstream") is True
    if not stable and not initializer_fallback and not allow_unstable:
        raise ValueError("SharedK did not pass its GT-free downstream stability gate")
    k4 = np.asarray(record["calibrated_K_fx_fy_cx_cy"], dtype=np.float32)
    if k4.shape != (4,) or not np.isfinite(k4).all() or np.any(k4[:2] <= 0):
        raise ValueError(f"K must be finite [fx,fy,cx,cy] with positive focal: {k4}")
    return k4


EngineFactory = Callable[..., SharedFocalEngine]


def calibrate_shared_k(
    stream: VideoStream,
    config: SharedKCalibrationConfig | Mapping[str, Any] | None = None,
    *,
    depth_model: DepthEstimationModel | None = None,
    model_cache: ModelCache | None = None,
    device: torch.device | None = None,
    engine_factory: EngineFactory | None = None,
) -> dict[str, Any]:
    """Calibrate one shared pinhole K for ``stream`` from RGB only; returns the calibration record.

    ``depth_model`` defaults to ``config.depth_model`` built through
    :func:`vipe.metrictrail.depth.make_metrictrail_depth_model`;
    ``engine_factory`` defaults to :class:`ViPESharedFocalEngine` (tests inject fakes).
    """

    started = time.perf_counter()
    cfg = _as_config(config)
    require_rgb_only(stream)
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    apply_calibration_determinism(seed=cfg.seed, disable_tf32=cfg.disable_tf32)
    image_hw = (int(stream.frame_size()[0]), int(stream.frame_size()[1]))

    load_seconds = 0.0
    if depth_model is None:
        from vipe.metrictrail.depth import make_metrictrail_depth_model

        load_started = time.perf_counter()
        depth_model = make_metrictrail_depth_model(
            cfg.depth_model, use_fp16=cfg.depth_use_fp16, model_cache=model_cache
        )
        load_seconds = time.perf_counter() - load_started
    if depth_model.depth_type != DepthType.MODEL_METRIC_DEPTH:
        raise TypeError("shared-K calibration requires a MODEL_METRIC_DEPTH provider (re-inferred per focal)")
    if cfg.fallback == "provider_focal" and not callable(getattr(depth_model, "estimate_free_fov_k4", None)):
        raise ValueError(
            "calibration fallback provider_focal needs a depth provider that predicts intrinsics; "
            f"{type(depth_model).__name__} does not"
        )
    source = _initializer_source(depth_model)
    audited = AuditedDepthModel(depth_model, device, model_load_seconds=load_seconds)

    frames = CalibrationFrames(stream, device=device, normalization=cfg.rgb_normalization)
    initial_k, initializer = estimate_initial_k(
        audited,
        frames.initializer_images(cfg.initializer_frames),
        image_hw,
        num_frames=cfg.initializer_frames,
        source=source,
        model_load_seconds=load_seconds,
    )
    initial_k32 = validate_k4(initial_k, image_hw).astype(np.float32)

    provider_focal = cfg.fallback == "provider_focal"
    calibrated: dict[str, Any] | None = None
    fallback_reason: str | None = None
    fallback_detail: dict[str, Any] = {}
    factory: EngineFactory = engine_factory or ViPESharedFocalEngine
    engine: SharedFocalEngine | None = None
    try:
        engine = factory(
            slam_config=calibration_slam_config(cfg.slam),
            depth_model=audited,
            image_hw=image_hw,
            device=device,
            model_cache=model_cache,
        )
        calibrated = run_shared_focal_protocol(
            engine,
            frames.sweep(initial_k32),
            initial_k,
            image_hw=image_hw,
            mode=cfg.budget,
            terminal_min_accepted_keyframes=cfg.terminal_min_accepted_keyframes,
            fallback_to_initial_k_on_terminal_failure=cfg.fallback_to_initializer,
            initializer_source=source,
        )
    except ImportError:
        # A broken installation is never an in-the-wild property of the video.
        raise
    except Exception as error:
        # GPU/toolchain failures (CUDA errors, no kernel image, out of memory) are never a fallback.
        if not provider_focal or is_broken_installation(error):
            raise
        fallback_reason, fallback_detail = vipe_error_reason(
            error, terminal_min_accepted_keyframes=cfg.terminal_min_accepted_keyframes
        )
    finally:
        del engine
    if calibrated is None:
        # Release whatever the failed sweep still holds before tracking.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    swept = calibrated or {}
    stable = bool((swept.get("convergence") or {}).get("converged_without_GT", False))
    # allow_unstable (benchmark policy) keeps the optimised K of a sweep that reached its keyframe
    # budget; an unconverged sweep that ran out of frames still binds the provider focal.
    keeps_unstable = cfg.allow_unstable and not swept.get("terminal_exhaustion") and swept.get("K_plausible_without_GT")
    if provider_focal and calibrated is not None and not stable and not keeps_unstable:
        fallback_reason, fallback_detail = unconverged_reason(calibrated)
    selected: dict[str, Any] = {}
    if fallback_reason is not None:
        selected_k = provider_focal_k4_values(initial_k, image_hw)
        selected = {
            "calibrated_K_fx_fy_cx_cy": selected_k,
            "selected_K_source": PROVIDER_FOCAL_FALLBACK,
            "K_plausible_without_GT": bool(plausible_k4(selected_k, image_hw)),
        }
    # Here the sweep completed, or it raised under provider_focal (fallback_reason set).
    status = selection_status({**swept, **selected}, allow_unstable=cfg.allow_unstable, fallback_reason=fallback_reason)
    record: dict[str, Any] = {
        "status": "success",
        "schema": SCHEMA,
        "stage": "moge2_backbone_sharedk_calibration_only",
        "mode": cfg.budget,
        "case": stream.name(),
        "sequence": stream.name(),
        "input_hw": list(image_hw),
        # No frame cap: the sweep stops only at the keyframe budget, a GT-free
        # early-convergence audit, or the end of the stream.
        "input_stream_frames": len(stream),
        "sensor_K_or_GT_read": False,
        "selection_uses_GT": False,
        "trajectory_pose_or_map_output": False,
        "initializer": initializer,
        **swept,
        **selected,
        **status,
        "calibration_fallback": {
            "policy": cfg.fallback,
            "applied": fallback_reason is not None,
            "reason": fallback_reason,
            "detail": fallback_detail,
            "vipe_stage_completed": calibrated is not None,
            "uses_GT": False,
        },
        "calibration_config": cfg.model_dump(exclude={"slam"}),
    }
    record["timing"] = {**(swept.get("timing") or {}), "end_to_end_wall_seconds": time.perf_counter() - started}
    return record


class SharedKIntrinsicsProcessor(StreamProcessor):
    """Init stage: calibrate a shared K from RGB at construction, then assign it to every frame.

    ``calibration_stream`` optionally supplies a different RGB view of the same
    clip (same resolution) to calibrate on, e.g. every raw frame when the
    tracked stream is subsampled.
    """

    def __init__(
        self,
        video_stream: VideoStream,
        config: SharedKCalibrationConfig | Mapping[str, Any] | None = None,
        *,
        calibration_stream: VideoStream | None = None,
        depth_model: DepthEstimationModel | None = None,
        model_cache: ModelCache | None = None,
        device: torch.device | None = None,
        engine_factory: EngineFactory | None = None,
        calibration_cache: str | Path | None = None,
    ) -> None:
        """Calibrate or load a validated, explicitly frozen RGB-only calibration.

        Args:
            video_stream: RGB-only input receiving shared intrinsics.
            config: SharedK calibration configuration, or defaults when omitted.
            calibration_stream: Optional full-rate RGB view used for calibration.
            depth_model: Optional injected metric depth provider.
            model_cache: Optional shared model cache for a fresh calibration.
            device: Calibration device, defaulting to CUDA.
            engine_factory: Optional calibration engine factory for tests.
            calibration_cache: Optional immutable artifact to create or validate and reuse.
        """
        super().__init__()
        require_rgb_only(video_stream)
        source = video_stream
        if calibration_stream is not None:
            if tuple(calibration_stream.frame_size()) != tuple(video_stream.frame_size()):
                raise ValueError(
                    f"calibration stream size {calibration_stream.frame_size()} differs from the input stream "
                    f"size {video_stream.frame_size()}"
                )
            source = calibration_stream
        self.config = _as_config(config)
        self.execution: dict[str, Any] | None = None

        def calibrate() -> dict[str, Any]:
            return calibrate_shared_k(
                source,
                self.config,
                depth_model=depth_model,
                model_cache=model_cache,
                device=device,
                engine_factory=engine_factory,
            )

        if calibration_cache is None:
            self.record = calibrate()
        else:
            from .cache import frozen_calibration

            self.record, self.execution = frozen_calibration(source, self.config, Path(calibration_cache), calibrate)
        self.k4 = select_shared_k(self.record, allow_unstable=self.config.allow_unstable)
        self.record["bound_K_fx_fy_cx_cy"] = self.k4.tolist()
        self._device_k: dict[torch.device, torch.Tensor] = {}

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        # One host-to-device copy per device; every frame gets a device-side clone, so
        # the producer thread makes no per-frame (synchronising) host copy of K.
        frame.intrinsics = shared_device_k(self._device_k, self.k4, frame.rgb.device)
        frame.camera_type = CameraType.PINHOLE
        return frame


def shared_device_k(cache: dict[torch.device, torch.Tensor], k4: np.ndarray, device: torch.device) -> torch.Tensor:
    """A fresh device tensor equal to ``k4`` (float32), cloned from a per-device copy made once."""

    device = torch.device(device)
    source = cache.get(device)
    if source is None:
        source = cache[device] = torch.as_tensor(np.asarray(k4, dtype=np.float32), device=device)
    return source.clone()
