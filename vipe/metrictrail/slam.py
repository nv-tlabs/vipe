# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DPVOSLAMSystem: DINO-fused DPVO with per-frame metric depth, returning ViPE's SLAMOutput.

The MetricTrail tracker for one ViPE video stream with a shared pinhole K (from
the SharedK init stage):

1. DPVO configuration: the slam preset's ``dpvo`` block merged over DPVO's
   defaults (yacs), then ``BUFFER_SIZE`` = ``slam.buffer_size`` (``auto``: the
   smallest power of two >= frames + 2, at least 8192);
   guards: metric-depth FastBA on, retrieval loop closure off, proximity loop edges
   only together with the metric global BA, DPVO depth bounds equal to the depth
   provider's. TF32 for CUDA matmul and cuDNN follows ``slam.tf32`` (off by default,
   as the runtime record sets it).
2. Models through ViPE's ``ModelCache``: the depth model from ViPE's depth
   registry (``slam.keyframe_depth``; ``moge2-l`` is MetricTrail's MoGe-2 call,
   :mod:`vipe.metrictrail.depth`) and the DINO-DPVO network (checkpoint digest
   d0aa57e1..., DINOv3 from ViPE's in-tree backbone).
3. Per sequence: seed = ``slam.seed`` + the seed name (``slam.seed_name``, by default
   the stream name with spaces replaced by ``_``;
   ``int(name)`` when it is an integer, else the first 4 bytes of its SHA-256); RNGs reset; a
   fresh DPVO; reflect-padded transport when the raster is not divisible by 4;
   :func:`vipe.metrictrail.depth_producer.track_sequence` (``slam.depth_queue``; DPVO's
   ``terminate``, with its final global-BA passes, runs under ``torch.no_grad()``).
   A trajectory with a non-finite pose row is re-tracked in full FP32
   (``slam.nonfinite_recovery``, on in every preset; :func:`track_with_nonfinite_recovery`:
   the cached network is released first, then a freshly loaded one tracks with DPVO mixed
   precision, DINOv3 autocast and TF32 off, every other preset setting kept, at the
   per-sequence seed + 0, 1, 2, 3; an attempt that runs out of CUDA memory counts as
   failed; every pass recorded); a finite trajectory is kept as tracked.
   A DPVO that never initialized (fewer than 8 keyframes) returns its near-static
   trajectory, labelled.
   With ``depth_output`` the per-frame metric depth DPVO tracked with is saved by
   an asynchronous writer (:mod:`vipe.metrictrail.depth_writer`); only the
   accepted attempt's file is kept.
4. Output: ``SLAMOutput`` (c2w trajectory, input-resolution K, identity rig) with
   DPVO's patch points of the retained keyframes as the ``SLAMMap``, plus the
   float64 c2w matrices and the run record, whose ``runtime`` block
   (:mod:`vipe.metrictrail.runtime_record`) times the tracking stage only and holds the
   CUDA peak memory since the models were loaded.

The pose convention is OpenCV camera-to-world (``c2w``) in DPVO's world frame (the
first frame is the identity).
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import logging
import os
import random
import time
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vipe.config.base_schema import config_to_primitive
from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel, DepthType
from vipe.slam.interface import SLAMMap, SLAMOutput
from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.model_cache import ModelCache

from .config import METRIC_DEPTH_RANGE_M, MetricTrailSLAMConfig, resolve_dpvo_buffer_size
from .depth_producer import (
    TRACKING_OK,
    CudaOps,
    PreparedFrame,
    jsonable,
    poses7_xyzw_to_c2w,
    track_sequence,
)
from .depth_writer import AsyncDepthWriter
from .dynamic import DynamicMaskTracking
from .fused_map import KeyframeDepthKeeper, fused_keyframe_points
from .runtime_record import TrackingRuntime
from .transport import ReflectPaddedDPVOTransport, dpvo_transport_padding, transport_hw

logger = logging.getLogger(__name__)

#: ``slam.nonfinite_recovery`` (on in every preset, and in the stereo runner): the seed offsets of the FP32
#: recovery attempts, and the policy and schema ids its record names.
FP32_RECOVERY_SEED_OFFSETS: tuple[int, ...] = (0, 1, 2, 3)
NONFINITE_RECOVERY_POLICY = "canonical_mixed_then_conditional_complete_fp32_fixed_seed_recovery_v1"
FP32_RECOVERY_POLICY = "complete_fp32_then_fixed_seed_restart_finite_gate_v1"
NONFINITE_RECOVERY_SCHEMA = "metrictrail_vipe_nonfinite_recovery_v1"
#: The DPVO keys the FP32 recovery changes (precision only); every other key keeps the preset's value.
FP32_RECOVERY_DPVO_CFG: dict[str, Any] = {"MIXED_PRECISION": False}
#: Preset keys the recovery record echoes to show they were kept.
RECOVERY_KEPT_CFG_KEYS = ("LOOP_CLOSURE", "CLASSIC_LOOP_CLOSURE", "METRIC_GLOBAL_BA")
#: Status of one FP32 recovery attempt in the record.
ATTEMPT_FINITE = "finite"
ATTEMPT_NONFINITE = "nonfinite"
ATTEMPT_OUT_OF_MEMORY = "cuda_out_of_memory"
#: Characters of a CUDA out-of-memory message kept in the record.
OOM_MESSAGE_CHARS = 500
GIB = float(1 << 30)

#: Largest distance (in 1/255 units) of ``rgb * 255`` from an integer for 8-bit input.
UINT8_TOLERANCE = 1.0e-3

# DPVO cfg keys echoed into the record of a global-BA run.
GBA_CFG_KEYS = (
    "LOOP_CLOSURE", "CLASSIC_LOOP_CLOSURE", "METRIC_GLOBAL_BA", "METRIC_GLOBAL_BA_T0",
    "BACKEND_THRESH", "MAX_EDGE_AGE", "GLOBAL_OPT_FREQ", "GBA_SCHEDULE_K",
    "CULLED_INTERP", "GBA_REMEASURE_FINAL", "FMAP_STORE", "REMEASURE_MIN_AGE",
    "REMEASURE_BURNIN", "REMEASURE_WARM_ACTIVE", "REMEASURE_CARRY_STATE",
    "GBA_SCALE_LATENT", "GBA_SCALE_SMOOTH", "GBA_SCALE_CLAMP", "GBA_SCALE_MIN_PATCHES",
    "GBA_SCALE_MIN_PARALLAX_PX", "GBA_SCALE_LOCAL_USE", "LOOP_REVERSE_EDGES",
    "KEYFRAME_MAX_CULL", "KEYFRAME_MAX_CULL_ROT_DEG", "METRIC_DEPTH",
    "METRIC_DEPTH_PRIOR_WEIGHT", "METRIC_DEPTH_PRIOR_HUBER", "METRIC_DEPTH_PRIOR_WARMUP",
    "REMOVAL_WINDOW", "OPTIMIZATION_WINDOW", "PATCH_LIFETIME", "PATCHES_PER_FRAME",
    "BUFFER_SIZE", "MIXED_PRECISION",
)  # fmt: skip
# Optional per-group diagnostic blocks DPVO adds when the group is on.
GBA_DIAGNOSTIC_BLOCKS = ("backend_variants", "burst_fixes", "backend_m", "scale_latent", "graph_g")


def case_seed(name: str, base_seed: int) -> int:
    """Per-sequence seed: ``base + int(name)``, else ``base`` + 4 bytes of SHA-256."""

    try:
        return base_seed + int(name)
    except ValueError:
        return base_seed + int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def seed_case_name(stream_name: str, seed_name: str | None = None) -> str:
    """The name the per-sequence seed is derived from: ``seed_name``, else the stream name with
    spaces replaced by ``_``."""

    return seed_name if seed_name is not None else stream_name.replace(" ", "_")


def reset_random_state(seed: int) -> None:
    """Put every RNG the tracker draws from into a known state.

    DPVO selects patch centroids at random (CENTROID_SEL_STRAT=RANDOM), so two
    runs of the same sequence differ unless this is called. It does not make the
    run bit-exact: MoGe-2's FoV head still varies at the 1e-7 level with GPU
    reduction order.
    """

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attempt_valid(tracked: Mapping[str, Any], *, count: int) -> tuple[bool, list[int]]:
    """Which pose rows came back non-finite, if any (a diverged BA returns NaN poses)."""

    poses = np.asarray(tracked["poses7_xyzw"])
    timestamps = np.asarray(tracked["timestamps"])
    if poses.shape != (count, 7):
        raise ValueError(f"DPVO pose inventory drifted: {poses.shape}")
    if timestamps.shape != (count,) or not np.array_equal(timestamps, np.arange(count)):
        raise ValueError("DPVO timestamps differ from the frame ordinals")
    bad = np.flatnonzero(~np.isfinite(poses).all(axis=1)).astype(int).tolist()
    return not bad, bad


@dataclass
class TrackingAttempt:
    """One whole-sequence tracking pass of :meth:`DPVOSLAMSystem.run`."""

    seed: int | None
    tracked: dict[str, Any]
    valid: bool
    bad_rows: list[int]
    source_ids: list[int]
    #: Set only for an all-finite attempt (the one a run may keep).
    depth_record: dict[str, Any] | None = None
    slam_map: SLAMMap | None = None
    dynamic_record: dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    #: With depth fusion: the dense keyframe map's record (:mod:`vipe.metrictrail.fused_map`).
    fused_map_record: dict[str, Any] | None = None

    def finite_summary(self) -> dict[str, Any]:
        return {
            "all_pose_rows_finite": self.valid,
            "nonfinite_pose_row_count": len(self.bad_rows),
            "first_nonfinite_pose_row": self.bad_rows[0] if self.bad_rows else None,
            "last_nonfinite_pose_row": self.bad_rows[-1] if self.bad_rows else None,
        }


# ------------------------------------------------------------ non-finite recovery


def fp32_recovery_dpvo_config(cfg: Any) -> Any:
    """DPVO's config for the FP32 recovery: the preset's own config with mixed precision off, nothing else.

    Only the keys of :data:`FP32_RECOVERY_DPVO_CFG` (``MIXED_PRECISION``) change; every
    other key keeps the preset's value, so a global-BA preset keeps its proximity loop
    edges (``LOOP_CLOSURE``) and its metric global BA. For ``metrictrail_local``, whose
    loop edges are already off, this is the same as also setting ``LOOP_CLOSURE`` and
    ``CLASSIC_LOOP_CLOSURE`` to false. Metric-depth FastBA must stay on.
    """

    value = cfg.clone()
    value.defrost()
    for key, item in FP32_RECOVERY_DPVO_CFG.items():
        setattr(value, key, item)
    if not bool(value.METRIC_DEPTH):
        raise ValueError("the FP32 recovery must retain metric-depth FastBA")
    value.freeze()
    return value


def evict_cached_model(cache: ModelCache, key: str) -> bool:
    """Remove the model cached under ``key`` from ViPE's ``ModelCache``; True if it was cached.

    ``ModelCache`` has no per-key eviction, so its dict (``_models``, a private attribute)
    is edited; should that attribute ever change, the
    whole cache is cleared instead (the next stream rebuilds what it needs).
    """

    models = getattr(cache, "_models", None)
    if isinstance(models, dict):
        return models.pop(key, None) is not None
    logger.warning("ModelCache has no _models dict; clearing the whole cache to release %s", key)
    cache.clear()
    return True


class NetworkSlot:
    """The canonical pass's DINO-DPVO network, held in one place so that the FP32 recovery can let it go.

    The caller hands over the slot and keeps no reference of its own (a reference passed as a
    call argument can outlive the callee's ``del``). :meth:`release` drops the slot's reference,
    calls ``evict`` (which removes the network from the model cache), collects garbage and
    empties the CUDA cache, so that the canonical network is gone before the recovery loads a
    fresh one: two DINO-DPVO networks are never on the GPU together.
    """

    def __init__(self, network: Any, evict: Callable[[], bool] | None = None) -> None:
        self._network: Any = network
        self._evict = evict

    def get(self) -> Any:
        if self._network is None:
            raise RuntimeError("the canonical DINO-DPVO network was released")
        return self._network

    def release(self) -> dict[str, Any]:
        """Let the network go; the record says whether it left the model cache and whether it was freed."""

        try:
            alive: Callable[[], Any] | None = weakref.ref(self._network)
        except TypeError:  # not weak-referenceable (test stand-ins): freedom cannot be observed
            alive = None
        self._network = None
        evicted = bool(self._evict()) if self._evict is not None else False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        freed = None if alive is None else alive() is None
        if freed is False:
            logger.warning("the canonical DINO-DPVO network is still referenced; the FP32 recovery loads a second one")
        return {"evicted_from_model_cache": evicted, "freed": freed}


def _cuda_peak_gib() -> float | None:
    return float(torch.cuda.max_memory_allocated()) / GIB if torch.cuda.is_available() else None


def _require_matching_context_network(network: Any) -> Any:
    if not all(
        bool(getattr(network, name, False))
        for name in ("_metrictrail_dino_enabled", "_metrictrail_dino_matching", "_metrictrail_dino_context")
    ):
        raise RuntimeError("numeric recovery requires the DINO-DPVO network")
    patchify = getattr(network, "patchify", None)
    if (
        patchify is None
        or not bool(getattr(patchify, "fuse_matching", False))
        or not bool(getattr(patchify, "fuse_context", False))
        or getattr(patchify, "match_fusion", None) is None
        or getattr(patchify, "context_fusion", None) is None
        or getattr(patchify, "dino", None) is None
    ):
        raise RuntimeError("matching and context DINOv3 fusion must both remain active")
    return patchify


@contextmanager
def complete_dino_fp32(network: Any) -> Iterator[dict[str, Any]]:
    """The DINOv3 backbone's own autocast and TF32 off for one attempt.

    DPVO's ``MIXED_PRECISION=False`` does not reach the frozen DINOv3 extractor, whose
    checkpoint config requests bfloat16 autocast; this scope sets it to float32 and
    turns TF32 off for CUDA matmul and cuDNN, and restores all three afterwards (a TF32
    switch that is already off is not written). The fusion weights and both DINOv3
    paths (matching and context) stay as they are. Refuses
    ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``. Yields the precision record.
    """

    tf32_override = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    if tf32_override == "1":
        raise RuntimeError("strict FP32 recovery refuses TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1")
    dino = _require_matching_context_network(network).dino
    previous_dtype = dino.amp_dtype
    previous_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    dino.amp_dtype = torch.float32
    if previous_matmul_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    if previous_cudnn_tf32:
        torch.backends.cudnn.allow_tf32 = False
    audit = {
        "dpvo_autocast_mixed_precision": False,
        "dpvo_graph_storage": "float32",
        "fastba_storage_and_solve": "float32",
        "dinov3_backbone_autocast": False,
        "dinov3_backbone_dtype": "float32",
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "torch_allow_tf32_cublas_override": tf32_override,
        "dinov3_matching_active": True,
        "dinov3_context_active": True,
        "dinov3_fusion_scale": 1.0,
        "checkpoint_parameters_modified": False,
    }
    try:
        yield audit
    finally:
        dino.amp_dtype = previous_dtype
        if previous_matmul_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        if previous_cudnn_tf32:
            torch.backends.cudnn.allow_tf32 = True


def track_with_nonfinite_recovery(
    attempt: Callable[[Any, Any, int | None], TrackingAttempt],
    *,
    cfg: Any,
    canonical_network: NetworkSlot,
    base_seed: int | None,
    fresh_network: Callable[[], Any],
    name: str,
) -> tuple[TrackingAttempt, dict[str, Any]]:
    """``slam.nonfinite_recovery`` (precision only); the kept pass and its record.

    ``attempt(cfg, network, seed)`` tracks the whole sequence once. The canonical pass
    runs once, at ``base_seed``, with the preset's ``cfg`` and the cached network of
    ``canonical_network`` exactly as configured (DPVO mixed precision and the DINOv3
    checkpoint's bfloat16 autocast in every shipped preset); there is no seed restart.
    Only if it returns a non-finite pose row is the canonical network released
    (:meth:`NetworkSlot.release`: dropped and evicted from the model cache, so two
    networks are never on the GPU together) and a freshly loaded network
    (``fresh_network()``; nothing of the canonical pass is reused) tracks the whole
    sequence again with :func:`fp32_recovery_dpvo_config` under :func:`complete_dino_fp32`,
    at ``base_seed`` + 0, 1, 2, 3 in that order (four unseeded attempts when ``slam.seed``
    is null); the first attempt whose pose rows are all finite is kept. An attempt that
    raises ``torch.cuda.OutOfMemoryError`` (FP32 doubles DPVO's half-precision buffers) is
    recorded and counts as failed, like a non-finite one. If no attempt is kept, the run
    fails. Only the precision changes: every other preset setting, loop edges and global
    BA included, is the canonical pass's. The gate reads neither ground truth nor a
    metric. Every pass is recorded.
    """

    mixed = bool(cfg.MIXED_PRECISION)
    canonical_dino_dtype = str(
        getattr(getattr(getattr(canonical_network.get(), "patchify", None), "dino", None), "amp_dtype", None)
    )
    seeds: tuple[int | None, ...]
    if base_seed is None:
        seeds = tuple(None for _ in FP32_RECOVERY_SEED_OFFSETS)
    else:
        base_seed = int(base_seed)
        if base_seed < 0:
            raise ValueError("the recovery base seed must be non-negative")
        seeds = tuple(base_seed + offset for offset in FP32_RECOVERY_SEED_OFFSETS)
    canonical_mode = "canonical_mixed_precision" if mixed else "canonical_preset_precision"

    canonical = attempt(cfg, canonical_network.get(), base_seed)
    canonical_attempt = {
        "attempt_index": 0,
        "seed": base_seed,
        "canonical_mixed_precision": mixed,
        "dpvo_cfg_mixed_precision": mixed,
        "dinov3_checkpoint_amp_dtype": canonical_dino_dtype,
        **canonical.finite_summary(),
        "exact_timestamp_inventory": True,
        "metric_prior_status": canonical.tracked.get("metric_prior_status"),
    }
    attempts: list[dict[str, Any]] = []
    selected: TrackingAttempt | None = canonical if canonical.valid else None
    selected_index: int | None = None
    selected_precision: dict[str, Any] | None = None
    released: dict[str, Any] | None = None
    load_error: str | None = None
    if not canonical.valid:
        logger.warning(
            "%s: the canonical pass returned %d non-finite pose rows; FP32 recovery at seeds %s",
            name,
            len(canonical.bad_rows),
            list(seeds),
        )
        fp32_cfg = fp32_recovery_dpvo_config(cfg)
        # The canonical network leaves the GPU before the fresh one is loaded.
        released = canonical_network.release()
        recovery_network: Any = None
        try:
            recovery_network = fresh_network()
        except torch.cuda.OutOfMemoryError as error:
            load_error = str(error)[:OOM_MESSAGE_CHARS]
        try:
            for attempt_index, seed in enumerate(seeds if recovery_network is not None else ()):
                precision: dict[str, Any] = {}
                out_of_memory: str | None = None
                try:
                    with complete_dino_fp32(recovery_network) as audit:
                        precision = dict(audit)
                        result = attempt(fp32_cfg, recovery_network, seed)
                except torch.cuda.OutOfMemoryError as error:
                    out_of_memory = str(error)[:OOM_MESSAGE_CHARS]
                if out_of_memory is not None:
                    # Outside the handler: the traceback (and the failed pass's DPVO it references) is gone.
                    peak = _cuda_peak_gib()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.warning("%s: FP32 recovery attempt %d ran out of CUDA memory", name, attempt_index)
                    attempts.append(
                        {
                            "attempt_index": attempt_index,
                            "seed": seed,
                            "status": ATTEMPT_OUT_OF_MEMORY,
                            "all_pose_rows_finite": False,
                            "error": out_of_memory,
                            "max_memory_allocated_gib": peak,
                            "numeric_precision": precision,
                        }
                    )
                    continue
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "seed": seed,
                        "status": ATTEMPT_FINITE if result.valid else ATTEMPT_NONFINITE,
                        **result.finite_summary(),
                        "exact_timestamp_inventory": True,
                        "metric_prior_status": result.tracked.get("metric_prior_status"),
                        "numeric_precision": precision,
                    }
                )
                if result.valid:
                    selected, selected_index, selected_precision = result, attempt_index, precision
                    break
        finally:
            del recovery_network
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    triggered = not canonical.valid
    record = {
        "schema": NONFINITE_RECOVERY_SCHEMA,
        "enabled": True,
        "status": "success" if selected is not None else "failed_closed",
        "policy": NONFINITE_RECOVERY_POLICY,
        "recovery_policy": FP32_RECOVERY_POLICY,
        # Only the precision changes; the preset's other DPVO keys are those of the canonical pass.
        "recovery_scope": "precision_only",
        "recovery_triggered": triggered,
        "trigger": (
            ("nonfinite_canonical_mixed_precision_trajectory" if mixed else "nonfinite_canonical_trajectory")
            if triggered
            else None
        ),
        "base_seed": base_seed,
        "seeded": base_seed is not None,
        "canonical_attempt_count": 1,
        "canonical_attempt": canonical_attempt,
        "predeclared_seed_offsets": list(FP32_RECOVERY_SEED_OFFSETS),
        "predeclared_seed_candidates": list(seeds),
        "fp32_dpvo_cfg": dict(FP32_RECOVERY_DPVO_CFG),
        "preset_dpvo_cfg_kept": {key: bool(getattr(cfg, key, False)) for key in RECOVERY_KEPT_CFG_KEYS},
        "selected_seed": selected.seed if selected is not None else None,
        "selected_attempt_index": selected_index,
        "selected_mode": "complete_fp32_recovery" if triggered else canonical_mode,
        "selection_gate": "all_pose_rows_finite_and_exact_timestamp_inventory",
        "selection_reads_ground_truth_poses": False,
        "selection_reads_benchmark_metrics": False,
        "attempts": attempts,
        "out_of_memory_attempts": sum(item["status"] == ATTEMPT_OUT_OF_MEMORY for item in attempts),
        "selected_numeric_precision": selected_precision,
        # The canonical network was dropped and evicted from the model cache before the fresh load.
        "canonical_network_released": released,
        "fresh_network_load_error": load_error,
        "fresh_exact_checkpoint_reload_before_attempts": triggered,
        "fresh_network_reused_only_across_predeclared_seed_restarts": triggered,
        "all_canonical_frames_replayed_each_attempt": True,
    }
    if selected is None:
        raise RuntimeError(
            f"{name}: every predeclared FP32 recovery attempt was non-finite or ran out of CUDA memory: {record}"
        )
    return selected, record


def gba_record(resolved_cfg: Mapping[str, Any], tracked: Mapping[str, Any], preset: str) -> dict[str, Any]:
    """Record fields of a run with a global-BA DPVO preset.

    Checks that DPVO reported its global-BA diagnostics (and never normalized the
    depths, which would discard the metric scale). A DPVO that never initialized
    never ran its proximity edges or global BA; the block then names the reason.
    """

    diagnostics = dict(tracked["metric_prior_diagnostics"] or {})
    if diagnostics.get("metric_global_ba") is not True:
        raise RuntimeError("DPVO did not report METRIC_GLOBAL_BA diagnostics")
    for key in ("global_ba_calls", "loop_edges_added", "global_ba_calls_with_metric_prior"):
        value = diagnostics.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError(f"DPVO METRIC_GLOBAL_BA diagnostics lack {key}")
    if diagnostics.get("normalize_called") is not False:
        raise RuntimeError("METRIC_GLOBAL_BA path must never normalize the depths")
    status = tracked.get("tracking_status", TRACKING_OK)
    return {
        "loop_closure": False,
        "global_ba": True,
        "dpvo_native_loop_disabled": False,
        "dpvo_metric_global_ba": {
            # An uninitialised graph is labelled skipped only when DPVO really ran no
            # global BA (gba_refine still runs its final passes inside terminate()).
            "skipped_reason": (None if status == TRACKING_OK or diagnostics["global_ba_calls"] > 0 else status),
            "tracking_status": status,
            "preset": preset,
            "cfg": {key: resolved_cfg[key] for key in GBA_CFG_KEYS if key in resolved_cfg},
            "t0_mode": diagnostics.get("metric_global_ba_t0_mode"),
            "keyframes_kept_at_terminate": diagnostics.get("keyframes_kept_at_terminate"),
            "global_ba_calls": diagnostics["global_ba_calls"],
            "global_ba_calls_with_metric_prior": diagnostics["global_ba_calls_with_metric_prior"],
            "global_ba_t0_min": diagnostics.get("global_ba_t0_min"),
            "global_ba_edges_max": diagnostics.get("global_ba_edges_max"),
            "loop_edges_added": diagnostics["loop_edges_added"],
            "loop_edge_batches": diagnostics.get("loop_edge_batches"),
            "inactive_factors_peak": diagnostics.get("inactive_factors_peak"),
            "inactive_factors_at_terminate": diagnostics.get("inactive_factors_at_terminate"),
            "inactive_factors_released_at_terminate": diagnostics.get("inactive_factors_released_at_terminate"),
            "pmem": diagnostics.get("pmem"),
            "normalize_called": False,
            "blocks": {key: diagnostics[key] for key in GBA_DIAGNOSTIC_BLOCKS if key in diagnostics},
        },
    }


def canonical_rgb_chw_deferred(
    rgb_hwc: torch.Tensor, *, normalization: str, device: torch.device, pinned: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(3,H,W) float32 RGB on ``device`` as the tracker consumes it, plus its 8-bit check.

    ``metrictrail``: the frame is re-quantised to 8 bit and rebuilt as
    ``uint8 * (1/255)`` in float32 (a multiplication by the reciprocal; ViPE's
    streams divide by 255, which differs in the last bit for about half of the
    256 values). ``stream``: the stream's float RGB as is.

    The check is returned unread, as the number of values that are not 8-bit (an
    int64 tensor on ``device``, also for a frame that arrives on the host; ``None``
    for ``stream``), so the producer does not synchronise; the consumer reads it
    (``PreparedFrame.checks``).
    """

    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError(f"expected (H,W,3) RGB, got {tuple(rgb_hwc.shape)}")
    if normalization == "stream":
        return rgb_hwc.to(device=device, dtype=torch.float32).permute(2, 0, 1).contiguous(), None
    if normalization != "metrictrail":
        raise ValueError(f"unknown RGB normalization: {normalization}")
    scaled = rgb_hwc.float() * 255.0
    quantized = torch.round(scaled)
    bad = ((scaled - quantized).abs() > UINT8_TOLERANCE) | (quantized < 0.0) | (quantized > 255.0)
    # On the tracking device: _prepare stacks it with the other (device) checks, also for a host frame.
    not_8bit = bad.sum().to(device=device)
    # The clamp is the identity on 8-bit input; it only keeps a rejected frame defined.
    u8 = quantized.clamp_(0.0, 255.0).to(torch.uint8).permute(2, 0, 1).contiguous()
    if u8.device.type == "cpu" and device.type == "cuda":
        u8 = (u8.pin_memory() if pinned else u8).to(device=device, non_blocking=pinned)
    else:
        u8 = u8.to(device=device)
    return u8.to(torch.float32).mul_(1.0 / 255.0), not_8bit


def canonical_rgb_chw(rgb_hwc: torch.Tensor, *, normalization: str, device: torch.device, pinned: bool) -> torch.Tensor:
    """:func:`canonical_rgb_chw_deferred` with the 8-bit check read at once (raises on float input)."""

    rgb, not_8bit = canonical_rgb_chw_deferred(rgb_hwc, normalization=normalization, device=device, pinned=pinned)
    if not_8bit is not None and int(not_8bit.item()) > 0:
        raise ValueError("input RGB is not 8-bit; set slam.rgb_normalization: stream for float inputs")
    return rgb


def tf32_flags() -> dict[str, Any]:
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


def apply_tf32_policy(enabled: bool) -> dict[str, Any]:
    """TF32 for CUDA matmul *and* cuDNN on or off, process-wide.

    Off (``slam.tf32: false``, the default): strict FP32 for both and float32 matmul
    precision ``highest``, as the runtime record and the SharedK stage set them
    (PyTorch's own default leaves cuDNN TF32 on). Every per-frame convolution of the
    shipped presets runs under FP16/BF16 autocast, where these flags do not apply.
    """

    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    return tf32_flags()


def bgr_uint8(rgb_chw: torch.Tensor) -> torch.Tensor:
    """DPVO's BGR uint8 image from the canonical RGB."""

    return rgb_chw[[2, 1, 0]].mul(255.0).round().clamp_(0.0, 255.0).to(torch.uint8).contiguous()


def slam_map_from_keyframe_points(
    xyz: torch.Tensor, rgb: torch.Tensor, valid: torch.Tensor, frame_ordinals: np.ndarray
) -> SLAMMap:
    """ViPE ``SLAMMap`` from per-keyframe patch points: ``xyz``/``rgb`` (N,M,3), ``valid`` (N,M)."""

    n, m = int(xyz.shape[0]), int(xyz.shape[1])
    tstamps = torch.as_tensor(np.asarray(frame_ordinals, dtype=np.int64))
    return SLAMMap.from_masked_dense_disp(
        xyz.reshape(n, 1, 1, m, 3),
        rgb.reshape(n, 1, 1, m, 3),
        valid.reshape(n, 1, 1, m),
        tstamps,
    )


def _dpvo_patch_points(dpvo: Any, patch_static: Any = None) -> tuple[Any, ...] | None:
    """(xyz, rgb, valid, ordinals) of DPVO's retained keyframes' patch points, or None (see :func:`dpvo_slam_map`)."""

    from vipe.metrictrail.dpvo import projective_ops as pops
    from vipe.metrictrail.dpvo.lietorch import SE3 as DPVOSE3

    n, m, patches_per_frame = int(dpvo.n), int(dpvo.m), int(dpvo.M)
    if n == 0 or m != n * patches_per_frame:
        return None
    with torch.inference_mode():
        points = pops.point_cloud(DPVOSE3(dpvo.poses), dpvo.patches[:, :m], dpvo.intrinsics, dpvo.ix[:m])
        disparity = points[..., 1, 1, 3:]
        xyz = (points[..., 1, 1, :3] / disparity).reshape(n, patches_per_frame, 3).float()
        valid = (torch.isfinite(xyz).all(dim=-1) & (disparity.reshape(n, patches_per_frame) > 0)).reshape(
            n, patches_per_frame
        )
        rgb = dpvo.pg.colors_[:n].float().div(255.0)
    ordinals = np.asarray(dpvo.pg.tstamps_[:n], dtype=np.int64)
    if patch_static is not None:
        centre = int(dpvo.patches.shape[-1]) // 2
        centres = dpvo.patches[0, :m, :2, centre, centre].reshape(n, patches_per_frame, 2)
        valid = valid & patch_static(ordinals, centres).to(valid.device)
    return xyz, rgb, valid, ordinals


def dpvo_slam_map(dpvo: Any, patch_static: Any = None) -> SLAMMap | None:
    """DPVO's retained keyframes and their patch points (world frame) as a ``SLAMMap``.

    ``patch_static(ordinals, centres)``, when given, returns which patch centres
    (n,M,2 on DPVO's 1/4 grid) lie on static pixels; the others are left out, as
    ViPE leaves dynamic pixels out of its map (:mod:`vipe.metrictrail.dynamic`).
    """

    points = _dpvo_patch_points(dpvo, patch_static)
    if points is None:
        return None
    xyz, rgb, valid, ordinals = points
    return slam_map_from_keyframe_points(xyz.clone(), rgb.clone(), valid.clone(), ordinals)


def dpvo_fused_slam_map(
    dpvo: Any, keeper: KeyframeDepthKeeper, k4: np.ndarray, patch_static: Any = None
) -> tuple[SLAMMap | None, dict[str, Any]]:
    """The patch map of :func:`dpvo_slam_map` plus each keyframe's dense provider depth scaled to its patches.

    For ViPE's depth fusion (``post.depth_align_model``); see :mod:`vipe.metrictrail.fused_map`.
    """

    from vipe.metrictrail.dpvo.lietorch import SE3 as DPVOSE3

    points = _dpvo_patch_points(dpvo, patch_static)
    if points is None:
        return None, {"enabled": True, "keyframes": 0}
    xyz, rgb, valid, ordinals = points
    n = int(xyz.shape[0])
    with torch.inference_mode():
        w2c = DPVOSE3(dpvo.poses).matrix()[0, :n].float()
        dense = fused_keyframe_points(keeper, ordinals, w2c, k4, xyz, valid)
        all_xyz = torch.cat((xyz.to(dense.xyz.device), dense.xyz), dim=1)
        all_rgb = torch.cat((rgb.to(dense.rgb.device), dense.rgb), dim=1)
        all_valid = torch.cat((valid.to(dense.valid.device), dense.valid), dim=1)
    dense.record["patch_points"] = int(valid.sum().item())
    return slam_map_from_keyframe_points(all_xyz.clone(), all_rgb.clone(), all_valid.clone(), ordinals), dense.record


@dataclass(kw_only=True)
class DPVOSLAMOutput(SLAMOutput):
    """``SLAMOutput`` plus what MetricTrail writes beside the ViPE artifacts."""

    #: (N,4,4) float64 OpenCV camera-to-world matrices (DPVO's world frame).
    c2w: np.ndarray
    #: Source frame index of every tracked frame (``VideoFrame.raw_frame_idx``).
    source_frame_ids: np.ndarray
    #: The tracking record (configuration, statuses, diagnostics, timing).
    record: dict[str, Any] = field(default_factory=dict)
    #: The metric depth zip this pass wrote (``run(depth_output=...)``), if any.
    depth_artifact: Path | None = None

    @property
    def tracking_status(self) -> str:
        return str(self.record.get("tracking_status", TRACKING_OK))


def _slam_config(config: Any) -> MetricTrailSLAMConfig:
    if isinstance(config, MetricTrailSLAMConfig):
        return config
    return MetricTrailSLAMConfig.model_validate(config_to_primitive(config))


def resolve_dpvo_config(config: MetricTrailSLAMConfig, *, frame_count: int | None = None) -> tuple[Any, dict[str, Any]]:
    """DPVO's yacs config for ``config`` and its JSON form.

    ``frame_count`` (frames to track) resolves ``buffer_size: auto``; an integer
    ``buffer_size`` does not need it.
    """

    from vipe.metrictrail.dpvo.config import cfg as base_cfg
    from vipe.metrictrail.dpvo_deps import CfgNode

    block = config.dpvo.model_dump()
    unknown = sorted(set(block) - set(base_cfg.keys()))
    absent = sorted(set(base_cfg.keys()) - set(block))
    if unknown or absent:
        raise RuntimeError(f"slam.dpvo schema and vendored DPVO config drifted: unknown={unknown}, absent={absent}")
    cfg = base_cfg.clone()
    cfg.defrost()
    # The path yacs' merge_from_file takes for a preset file.
    cfg.merge_from_other_cfg(CfgNode(block))
    if config.buffer_size == "auto" and frame_count is None:
        raise ValueError("slam.buffer_size auto needs the number of frames to track")
    buffer_size = resolve_dpvo_buffer_size(config.buffer_size, int(frame_count or 0))
    cfg.merge_from_list(["BUFFER_SIZE", str(buffer_size)])
    resolved = jsonable(cfg)
    if not bool(getattr(cfg, "METRIC_DEPTH", False)):
        raise ValueError("resolved DPVO config must set METRIC_DEPTH=True")
    if bool(getattr(cfg, "CLASSIC_LOOP_CLOSURE", False)):
        raise ValueError("DPVO retrieval loop closure (CLASSIC_LOOP_CLOSURE) must be disabled")
    metric_global_ba = bool(getattr(cfg, "METRIC_GLOBAL_BA", False))
    if bool(getattr(cfg, "LOOP_CLOSURE", False)) != metric_global_ba:
        raise ValueError(
            "DPVO LOOP_CLOSURE (proximity loop edges) is allowed only together with METRIC_GLOBAL_BA, and "
            f"METRIC_GLOBAL_BA needs it: LOOP_CLOSURE={bool(getattr(cfg, 'LOOP_CLOSURE', False))}, "
            f"METRIC_GLOBAL_BA={metric_global_ba}"
        )
    if not np.isclose(float(cfg.METRIC_DEPTH_MIN), METRIC_DEPTH_RANGE_M[0]) or not np.isclose(
        float(cfg.METRIC_DEPTH_MAX), METRIC_DEPTH_RANGE_M[1]
    ):
        raise ValueError(
            f"runtime/config metric depth bounds disagree: runtime={METRIC_DEPTH_RANGE_M} "
            f"config=({cfg.METRIC_DEPTH_MIN},{cfg.METRIC_DEPTH_MAX})"
        )
    return cfg, resolved


class DPVOSLAMSystem:
    """MetricTrail's tracker on one ViPE video stream with a shared pinhole K."""

    #: The config key that switches the FP32 recovery (messages only).
    recovery_option = "slam.nonfinite_recovery"

    def __init__(self, device: torch.device, config: Any, model_cache: ModelCache | None = None) -> None:
        self.device = torch.device(device)
        self.config = _slam_config(config)
        self.model_cache = model_cache if model_cache is not None else ModelCache()
        self.ops = CudaOps(self.device)
        if self.config.visualize:
            logger.warning("slam.visualize is ignored: the MetricTrail DPVO tracker has no viewer")

    # ------------------------------------------------------------------ models

    def depth_model(self) -> DepthEstimationModel:
        from .depth import make_metrictrail_depth_model

        model = make_metrictrail_depth_model(
            self.config.keyframe_depth, use_fp16=self.config.depth_fp16, model_cache=self.model_cache
        )
        if model.depth_type not in (DepthType.MODEL_METRIC_DEPTH, DepthType.METRIC_DEPTH):
            raise TypeError(f"slam.keyframe_depth={self.config.keyframe_depth!r} does not predict metric depth")
        return model

    def _network_files(self) -> tuple[Path, Path]:
        from .checkpoint import resolve_dino_dpvo_checkpoint, resolve_dinov3_weights

        return resolve_dino_dpvo_checkpoint(self.config.weights), resolve_dinov3_weights(self.config.dinov3_weights)

    def dpvo_network_cache_key(self) -> str:
        """The model-cache key of the DINO-DPVO network."""

        checkpoint, dinov3 = self._network_files()
        return f"metrictrail/dino_dpvo/{checkpoint.resolve()}|{dinov3.resolve()}"

    def fresh_dpvo_network(self) -> torch.nn.Module:
        """The DINO-DPVO network loaded anew from its checkpoint (the model cache's builder; the FP32 recovery's network)."""

        from vipe.metrictrail.dpvo.net import VONet

        from .checkpoint import load_dino_dpvo_network

        checkpoint, dinov3 = self._network_files()
        network = load_dino_dpvo_network(VONet, checkpoint, dinov3_weights=dinov3)
        return network.to(self.device).eval()

    def recovery_dpvo_network(self) -> torch.nn.Module:
        """The FP32 recovery's network: loaded anew, outside the model cache."""

        return self.fresh_dpvo_network()

    def evict_dpvo_network(self) -> bool:
        """Remove the cached DINO-DPVO network from the model cache (before the FP32 recovery loads its own)."""

        return evict_cached_model(self.model_cache, self.dpvo_network_cache_key())

    def dpvo_network(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        from .checkpoint import checkpoint_identity

        checkpoint, dinov3 = self._network_files()
        network = self.model_cache.get(self.dpvo_network_cache_key(), self.fresh_dpvo_network)
        identity = {"dpvo": checkpoint_identity(checkpoint), "dinov3": checkpoint_identity(dinov3)}
        return network, identity

    def track_passes(
        self,
        attempt: Callable[[Any, Any, int | None], TrackingAttempt],
        *,
        cfg: Any,
        canonical_network: NetworkSlot,
        base_seed: int | None,
        runtime: TrackingRuntime,
        name: str,
    ) -> tuple[TrackingAttempt, dict[str, Any]]:
        """The kept tracking pass and the ``nonfinite_recovery`` record (shared by the mono and stereo trackers).

        With ``nonfinite_recovery`` on: :func:`track_with_nonfinite_recovery`, whose fresh
        network load is left out of the tracking wall clock. Off: one pass, and a
        non-finite trajectory fails the run.
        """

        if self.config.nonfinite_recovery:

            def load_recovery_network() -> torch.nn.Module:
                with runtime.paused("fp32_recovery_network_load"):
                    return self.recovery_dpvo_network()

            # A non-finite canonical pass is re-tracked in FP32 (precision only, fresh network, fixed seeds).
            return track_with_nonfinite_recovery(
                attempt,
                cfg=cfg,
                canonical_network=canonical_network,
                base_seed=base_seed,
                fresh_network=load_recovery_network,
                name=name,
            )
        selected = attempt(cfg, canonical_network.get(), base_seed)
        if not selected.valid:
            raise ValueError(
                f"DPVO returned {len(selected.bad_rows)} non-finite pose rows on {name} (first "
                f"{selected.bad_rows[0]}) and {self.recovery_option} is off: {selected.finite_summary()}"
            )
        return selected, {"enabled": False}

    # --------------------------------------------------------------- per frame

    def _prepare(
        self,
        index: int,
        frame: VideoFrame,
        pinned: bool,
        *,
        depth_model: DepthEstimationModel,
        k4: np.ndarray,
        k_device: torch.Tensor,
        image_hw: tuple[int, int],
        source_ids: list[int],
        depth_writer: AsyncDepthWriter | None = None,
        dynamic: DynamicMaskTracking | None = None,
        map_depth: KeyframeDepthKeeper | None = None,
    ) -> PreparedFrame:
        """One frame's metric depth and DPVO inputs, on the current CUDA stream, without a host synchronisation.

        The frame's checks (valid depth pixels, 8-bit input, K equal to the shared K)
        are left on the device for the consumer (:func:`resolve_frame_checks`).
        """

        from .depth import MetricTrailMoGe2

        zero = torch.zeros((), dtype=torch.int64, device=self.device)
        frame_k: torch.Tensor | None = None
        k_mismatch = zero
        if frame.intrinsics is not None:
            frame_k = frame.intrinsics.detach()
            if tuple(frame_k.shape) != (4,):
                raise ValueError(f"frame {frame.raw_frame_idx} carries K of shape {tuple(frame_k.shape)}, not (4,)")
            k_mismatch = (frame_k.to(device=self.device, dtype=torch.float32) != k_device).sum()
        source_ids.append(int(frame.raw_frame_idx))
        rgb, not_8bit = canonical_rgb_chw_deferred(
            frame.rgb, normalization=self.config.rgb_normalization, device=self.device, pinned=pinned
        )
        if tuple(rgb.shape[-2:]) != image_hw:
            raise ValueError(f"frame {frame.raw_frame_idx} is {tuple(rgb.shape[-2:])}, expected {image_hw}")
        depth_start = self.ops.event()
        ready = self.ops.event()
        depth_start.record()
        with torch.inference_mode():
            if isinstance(depth_model, MetricTrailMoGe2):
                prediction = depth_model.predict_metric(rgb[None], focal_px=float(k4[0]))
                depth = prediction.depth[0]
                metric_mask = prediction.valid[0].contiguous()
            else:
                result = depth_model.estimate(
                    DepthEstimationInput(
                        rgb=rgb.permute(1, 2, 0),
                        intrinsics=torch.from_numpy(k4).to(self.device),
                        camera_type=CameraType.PINHOLE,
                    )
                )
                if result.metric_depth is None:
                    raise RuntimeError(f"{type(depth_model).__name__} returned no metric depth")
                depth = result.metric_depth.to(device=self.device, dtype=torch.float32)
                metric_mask = (
                    torch.isfinite(depth) & (depth >= METRIC_DEPTH_RANGE_M[0]) & (depth <= METRIC_DEPTH_RANGE_M[1])
                ).contiguous()
        if tuple(depth.shape) != image_hw:
            raise ValueError(f"depth shape {tuple(depth.shape)} does not match RGB {image_hw}")
        provider_mask = metric_mask
        if dynamic is not None:
            # ViPE's dynamic mask (init.instance): no metric prior on dynamic pixels.
            metric_mask = dynamic.prepare(index, frame, metric_mask)
        # Same reduction as before; its host read moved to the consumer.
        checks = torch.stack(
            (metric_mask.sum().to(torch.int64), zero if not_8bit is None else not_8bit.to(torch.int64), k_mismatch)
        )
        # Keep every sampled value finite even though metric_mask is the source of
        # truth for whether a unary depth observation is active.
        metric_depth = torch.where(metric_mask, depth, torch.ones_like(depth)).contiguous()
        image_bgr = bgr_uint8(rgb)
        # The saved depth map is built before ``ready``: without dynamic masks provider_mask is
        # the tensor DPVO receives as metric_mask, and DPVO's metric-prior preparation ANDs into
        # it in place on the default stream once ``ready`` releases it. Invalid pixels are 0
        # ("no depth"), as MetricTrail's depth model returns them to ViPE. The provider's
        # validity mask, not the dynamic one: with init.instance, dynamic pixels keep their
        # depth here (as in ViPE's depth artifacts) and the masks are saved as ViPE mask
        # artifacts.
        saved_depth = torch.where(provider_mask, depth, torch.zeros_like(depth)) if depth_writer is not None else None
        if map_depth is not None:
            # Depth fusion only: the dense keyframe map (fused_map), without the dynamic pixels.
            map_depth.keep(int(index), depth, metric_mask, rgb)
        ready.record()
        if depth_writer is not None and saved_depth is not None:
            # After ``ready``: DPVO never waits for the device-to-host copy.
            depth_writer.submit(int(index), saved_depth)
        del rgb, depth, saved_depth
        return PreparedFrame(
            index=int(index),
            frame_id=int(index),
            image_bgr=image_bgr,
            metric_depth=metric_depth,
            metric_mask=metric_mask,
            checks=checks,
            depth_start=depth_start,
            ready=ready,
            source_frame_id=int(frame.raw_frame_idx),
            frame_k=frame_k,
        )

    # ------------------------------------------------------------ per sequence

    def run(
        self,
        video_stream: VideoStream,
        *,
        intrinsics: np.ndarray,
        name: str | None = None,
        depth_output: str | Path | None = None,
        dense_map: bool = False,
    ) -> DPVOSLAMOutput:
        """Track ``video_stream`` (every frame, in order) with the shared pinhole ``intrinsics`` [fx,fy,cx,cy].

        ``depth_output``: write the per-frame metric depth DPVO tracked with to this
        zip (ViPE's depth artifact format) through the asynchronous writer; ``None``
        writes nothing. ``dense_map`` (ViPE's depth fusion): add every keyframe's dense
        provider depth, scaled to its refined patches, to the ``SLAMMap``
        (:mod:`vipe.metrictrail.fused_map`); off, the map is DPVO's patch points.
        """

        from vipe.ext.lietorch import SE3
        from vipe.metrictrail.dpvo.dpvo import DPVO
        from vipe.utils.geometry import se3_matrix_to_se3

        from .depth import MetricTrailMoGe2

        started = time.perf_counter()
        name = video_stream.name() if name is None else name
        count = len(video_stream)
        if count <= 0:
            raise ValueError(f"stream {name!r} has no frames")
        image_hw = (int(video_stream.frame_size()[0]), int(video_stream.frame_size()[1]))
        k4 = np.asarray(intrinsics, dtype=np.float32).reshape(4)
        if not np.isfinite(k4).all() or np.any(k4[:2] <= 0):
            raise ValueError(f"DPVO intrinsics must be finite [fx,fy,cx,cy] with positive focal: {k4}")
        if not {"metric_depth", "metric_mask"} <= set(inspect.signature(DPVO.__call__).parameters):
            raise RuntimeError("vendored DPVO does not expose the metric API (metric_depth, metric_mask)")
        cfg, resolved = resolve_dpvo_config(self.config, frame_count=count)
        tf32 = apply_tf32_policy(self.config.tf32)
        depth_model = self.depth_model()
        network, identities = self.dpvo_network()
        # This frame keeps no reference of its own: the FP32 recovery must be able to free it.
        canonical_network = NetworkSlot(network, evict=self.evict_dpvo_network)
        del network
        padding = dpvo_transport_padding(image_hw)
        dpvo_hw = transport_hw(image_hw, padding)
        padded = any(padding)
        masked = FrameAttribute.MASK in video_stream.attributes()
        seed_name = seed_case_name(name, self.config.seed_name)
        base_seed = case_seed(seed_name, int(self.config.seed)) if self.config.seed is not None else None
        k_device = torch.from_numpy(k4.copy()).to(self.device)

        def attempt(attempt_cfg: Any, attempt_network: Any, seed: int | None) -> TrackingAttempt:
            """Track the whole sequence once with a fresh DPVO (and, if given, the depth writer)."""

            if seed is not None:
                reset_random_state(seed)
            source_ids: list[int] = []
            dpvo = DPVO(attempt_cfg, attempt_network, ht=dpvo_hw[0], wd=dpvo_hw[1], viz=False)
            slam: Any = ReflectPaddedDPVOTransport(dpvo, image_hw, padding) if padded else dpvo
            dynamic = (
                DynamicMaskTracking(
                    image_hw=image_hw,
                    padding_ltrb=padding,
                    transport_hw=dpvo_hw,
                    patches_per_frame=int(attempt_cfg.PATCHES_PER_FRAME),
                )
                if masked
                else None
            )
            if dynamic is not None:
                slam = dynamic.transport(slam, attempt_network.patchify, self.ops)
            writer = AsyncDepthWriter(depth_output) if depth_output is not None else None
            keeper = KeyframeDepthKeeper(count, image_hw, torch.device(self.device)) if dense_map else None

            def prepare(
                index: int,
                frame: VideoFrame,
                pinned: bool,
                _ids: list[int] = source_ids,
                _writer: AsyncDepthWriter | None = writer,
                _dynamic: DynamicMaskTracking | None = dynamic,
                _keeper: KeyframeDepthKeeper | None = keeper,
            ) -> PreparedFrame:
                return self._prepare(
                    index,
                    frame,
                    pinned,
                    depth_model=depth_model,
                    k4=k4,
                    k_device=k_device,
                    image_hw=image_hw,
                    source_ids=_ids,
                    depth_writer=_writer,
                    dynamic=_dynamic,
                    map_depth=_keeper,
                )

            try:
                candidate = track_sequence(
                    iter(video_stream),
                    count,
                    prepare,
                    slam,
                    k4,
                    ops=self.ops,
                    depth_queue=self.config.depth_queue,
                    metric_prior_warmup=int(attempt_cfg.METRIC_DEPTH_PRIOR_WARMUP),
                )
                valid, bad_rows = attempt_valid(candidate, count=count)
                result = TrackingAttempt(
                    seed=seed, tracked=candidate, valid=valid, bad_rows=bad_rows, source_ids=source_ids
                )
                if valid:
                    if writer is not None:
                        # Joins the writer, re-raises its error, publishes the zip.
                        result.depth_record = {**writer.finish(expected_frames=count), "path": str(writer.path)}
                    if keeper is None:
                        result.slam_map = (
                            dpvo_slam_map(dpvo) if dynamic is None else dpvo_slam_map(dpvo, dynamic.static_patches)
                        )
                    else:
                        result.slam_map, result.fused_map_record = dpvo_fused_slam_map(
                            dpvo, keeper, k4, None if dynamic is None else dynamic.static_patches
                        )
                    if dynamic is not None:
                        result.dynamic_record = dynamic.record()
                return result
            finally:
                if writer is not None:
                    writer.abort()  # a no-op after finish(); otherwise joins and deletes the partial file
                del slam, dpvo, writer, keeper
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Tracking stage only: the models are loaded; the CUDA peak statistics restart here.
        runtime = TrackingRuntime(self.device, frames=count)
        runtime.start()
        selected, nonfinite_recovery = self.track_passes(
            attempt,
            cfg=cfg,
            canonical_network=canonical_network,
            base_seed=base_seed,
            runtime=runtime,
            name=name,
        )
        runtime_block = runtime.stop(passes=1 + len(nonfinite_recovery.get("attempts", ())))
        tracked = selected.tracked
        depth_record = selected.depth_record
        source_ids = selected.source_ids
        slam_map = selected.slam_map

        c2w = poses7_xyzw_to_c2w(tracked["poses7_xyzw"])
        if c2w.shape != (count, 4, 4) or not np.isfinite(c2w).all():
            raise ValueError("DINO-DPVO returned an invalid trajectory")
        describe = getattr(depth_model, "describe", None)
        record: dict[str, Any] = {
            "schema": "metrictrail_vipe_tracking_v1",
            "status": "success",
            "case": name,
            "num_frames": count,
            "input_hw": list(image_hw),
            "intrinsics_fx_fy_cx_cy": k4.tolist(),
            "depth_provider_K_equals_dpvo_K": True,
            "preset": self.config.preset,
            "tracking_mode": "mono",
            "dinov3_matching": True,
            "dinov3_context": True,
            "metric_depth_fastba": True,
            "runtime_resize_or_stretch": False,
            "dpvo_transport": (
                {"source_hw": list(image_hw), "padding_ltrb": list(padding), "transport_hw": list(dpvo_hw)}
                if padded
                else {"source_hw": list(image_hw), "padding_ltrb": [0, 0, 0, 0]}
            ),
            "rgb_normalization": self.config.rgb_normalization,
            "depth_provider": describe()
            if callable(describe) and isinstance(depth_model, MetricTrailMoGe2)
            else {"registry_name": self.config.keyframe_depth, "depth_type": depth_model.depth_type.name},
            "dpvo_buffer_size": int(cfg.BUFFER_SIZE),
            "dpvo_buffer_size_config": self.config.buffer_size,
            "tf32_flags": tf32,
            "metric_depth_artifact": depth_record,
            "tracking_seed": self.config.seed,
            "seed_name": seed_name,
            "case_seed": base_seed,
            "nonfinite_recovery": nonfinite_recovery,
            "tracking_status": tracked["tracking_status"],
            "dpvo_initialized": tracked["dpvo_initialized"],
            "metric_prior_status": tracked["metric_prior_status"],
            "metric_prior_diagnostics": tracked["metric_prior_diagnostics"],
            "timing": tracked["timing"],
            "execution": tracked["execution"],
            "valid_depth_fractions": tracked["valid_depth_fractions"],
            "dynamic_mask": selected.dynamic_record,
            **({"fused_slam_map": selected.fused_map_record} if dense_map else {}),
            "resolved_dpvo_cfg": resolved,
            "checkpoint_identities": identities,
            "loop_closure": False,
            "global_ba": False,
            "dpvo_native_loop_disabled": True,
            "runtime": runtime_block,
            "runtime_seconds": time.perf_counter() - started,
        }
        if bool(cfg.METRIC_GLOBAL_BA):
            record.update(gba_record(resolved, tracked, self.config.preset))
        record["end_to_end_fps"] = count / record["runtime_seconds"] if record["runtime_seconds"] > 0 else None

        trajectory = se3_matrix_to_se3(torch.from_numpy(c2w).float().to(self.device))
        return DPVOSLAMOutput(
            trajectory=trajectory,
            intrinsics=torch.from_numpy(k4)[None].to(self.device),
            rig=SE3.Identity(1, device=self.device),
            slam_map=slam_map,
            ba_residual=0.0,
            c2w=c2w,
            source_frame_ids=np.asarray(source_ids, dtype=np.int64),
            record=record,
            depth_artifact=Path(depth_record["path"]) if depth_record is not None else None,
        )
