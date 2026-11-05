# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``runtime`` block of the tracking and run records: tracking wall clock, frames/s, CUDA peak memory.

Definitions (those of the KITTI-02 runtime record, without its warm-up):

- ``tracking_wall_seconds``: wall clock of the tracking stage only. It starts after the
  models (depth model and DINO-DPVO network) are loaded, right before the first tracking
  pass, and stops when the kept pass has returned, with a CUDA synchronisation at both ends.
  Calibration and model loading are outside it; so is the network load of an FP32
  recovery (:meth:`TrackingRuntime.paused`). Every tracking pass is inside it (the run
  record's ``tracking_passes`` counts them), including DPVO's construction and
  ``terminate``, the depth model and the asynchronous depth writer.
- ``tracking_fps`` = ``frames`` / ``tracking_wall_seconds``.
- ``max_memory_allocated_gib`` / ``max_memory_reserved_gib``: ``torch.cuda.max_memory_allocated``
  and ``torch.cuda.max_memory_reserved`` over the tracking stage, in GiB (2^30 bytes). The
  peak statistics are reset (after ``empty_cache``) once the models are loaded, right before
  tracking starts, so the resident weights count and loading transients do not.
- No warm-up pass runs (the first frames include CUDA's lazy initialisation).

On a non-CUDA device every CUDA call is skipped and the memory fields are ``None``. The
CUDA functions come from an injectable namespace (``torch.cuda`` by default), so the
bookkeeping is testable on CPU.
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

RUNTIME_SCHEMA = "metrictrail_vipe_runtime_v1"
GIB = float(1 << 30)
WALL_SCOPE = (
    "tracking stage only: from after model load to the end of the kept tracking pass (every pass, FP32 recovery "
    "attempts included), CUDA-synchronised; excludes calibration, model load and the FP32 recovery's network load"
)
PEAK_RESET = "after model load, right before tracking: empty_cache + reset_peak_memory_stats (no warm-up)"


def attention_backends() -> dict[str, Any]:
    """The attention kernels of the networks whose backend depends on the environment.

    MoGe-2's DINOv2 (``vipe.priors.depth.moge2.dinov2``, ``MemEffAttention``) calls
    ``xformers.ops.memory_efficient_attention`` when xformers imports and
    ``XFORMERS_DISABLED`` is unset, otherwise torch's ``scaled_dot_product_attention``;
    the two differ at FP16 rounding level. DINO-DPVO's DINOv3 always uses torch SDPA.
    """

    record: dict[str, Any] = {
        "dinov3_dino_dpvo": "torch_sdpa",
        "xformers_disabled_env": os.environ.get("XFORMERS_DISABLED") is not None,
    }
    try:
        from vipe.priors.depth.moge2.dinov2.layers import attention as moge_attention
    except Exception as error:  # noqa: BLE001 - a record must not fail the run
        return {**record, "moge2_dinov2": "unknown", "error": f"{type(error).__name__}: {error}"[:300]}
    uses_xformers = bool(getattr(moge_attention, "XFORMERS_AVAILABLE", False))
    version = None
    if uses_xformers:
        try:
            version = str(getattr(importlib.import_module("xformers"), "__version__", None))
        except Exception:  # noqa: BLE001
            version = None
    return {**record, "moge2_dinov2": "xformers" if uses_xformers else "torch_sdpa", "xformers_version": version}


def _gib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / GIB


class TrackingRuntime:
    """Times the tracking stage and reads the CUDA peak memory of one tracking call.

    ``start()`` once the models are loaded, ``paused(reason)`` around work that is not
    tracking (the FP32 recovery's network load), ``stop(passes=...)`` after the kept pass;
    ``stop`` returns the ``runtime`` block.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        frames: int,
        cuda: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.device = torch.device(device)
        self.frames = int(frames)
        if cuda is None and self.device.type == "cuda" and torch.cuda.is_available():
            cuda = torch.cuda
        self.cuda = cuda if self.device.type == "cuda" else None
        self.clock = clock
        self._started: float | None = None
        self._excluded: dict[str, float] = {}
        self._allocated_at_reset: int | None = None

    def _synchronize(self) -> None:
        if self.cuda is not None:
            self.cuda.synchronize(self.device)

    def start(self) -> None:
        """Reset the CUDA peak statistics and start the clock (after model load, before the first pass)."""

        if self.cuda is not None:
            self.cuda.synchronize(self.device)
            self.cuda.empty_cache()
            self.cuda.reset_peak_memory_stats(self.device)
            self._allocated_at_reset = int(self.cuda.memory_allocated(self.device))
        self._started = self.clock()

    @contextmanager
    def paused(self, reason: str) -> Iterator[None]:
        """Leave the enclosed work out of ``tracking_wall_seconds`` (its memory still counts in the peak)."""

        self._synchronize()
        began = self.clock()
        try:
            yield
        finally:
            self._synchronize()
            self._excluded[reason] = self._excluded.get(reason, 0.0) + (self.clock() - began)

    def stop(self, *, passes: int = 1) -> dict[str, Any]:
        """The ``runtime`` block (stops the clock, after a CUDA synchronisation)."""

        if self._started is None:
            raise RuntimeError("TrackingRuntime.stop() before start()")
        self._synchronize()
        elapsed = self.clock() - self._started
        wall = max(elapsed - sum(self._excluded.values()), 0.0)
        cuda = self.cuda
        return {
            "schema": RUNTIME_SCHEMA,
            "tracking_wall_seconds": wall,
            "frames": self.frames,
            "tracking_fps": self.frames / wall if wall > 0 else None,
            "tracking_passes": int(passes),
            "excluded_seconds": dict(self._excluded),
            "wall_scope": WALL_SCOPE,
            "warmup": False,
            "device": str(self.device),
            "max_memory_allocated_gib": _gib(cuda.max_memory_allocated(self.device)) if cuda is not None else None,
            "max_memory_reserved_gib": _gib(cuda.max_memory_reserved(self.device)) if cuda is not None else None,
            "memory_allocated_at_peak_reset_gib": _gib(self._allocated_at_reset),
            "peak_memory_reset": PEAK_RESET if cuda is not None else None,
            "gib_bytes": int(GIB),
            "attention_backend": attention_backends(),
        }
