# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-frame metric depth feeding DPVO, sequentially or overlapped (``slam.depth_queue``).

It also checks DPVO's metric-prior contract after ``terminate``. ``depth_queue=0``
is the sequential reference path: depth and DPVO alternate on the default CUDA stream.
``depth_queue>=1`` runs the depth model in one producer thread on an auxiliary
CUDA stream, at most ``depth_queue`` prepared frames ahead of DPVO, which stays
on the default stream (its FastBA kernels launch there). The overlap is correct
because of two lines in the consumer:

* ``default.wait_event(item.ready)`` -- DPVO kernels may not start before the
  producer's kernels for that frame have finished;
* ``tensor.record_stream(default)`` on the BGR image, metric depth and metric
  mask -- the caching allocator may not recycle those blocks for a later frame
  while DPVO kernels on the default stream still read them.

Both paths hand DPVO the same frames, in the same order (checked) and with the
same synchronisation contract; RGB, K and the validity mask are identical, and
the depth values are equal up to MoGe-2's GPU nondeterminism (a few ulp on some
frames when it runs on the auxiliary stream while DPVO runs on the default one).
A producer exception is re-raised in the calling thread, and the
producer thread is always joined. The CUDA calls go through :class:`CudaOps`, so
the ordering and error logic can be exercised on CPU with a stand-in.

The producer makes no host synchronisation of its own: the per-frame checks it
computes -- valid metric-depth pixels, input values that are not 8-bit, K entries
that differ from the shared K -- stay on the device as one small tensor
(:attr:`PreparedFrame.checks`) that the consumer reads once, after
``wait_event``, right before DPVO (:func:`resolve_frame_checks`), where DPVO's own
metric-prior preparation synchronises on the same frame anyway.

ViPE's own ``init.async_prefetch`` is not equivalent (a CPU-thread prefetch
without a dedicated CUDA stream and with an unbounded host list), so the
MetricTrail pipeline does not use it.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
import torch

#: Statuses of DPVO's metric-prior contract after terminate. Both "not" statuses are
#: deterministic properties of the input (too short or too static), never a reason
#: to retry.
METRIC_PRIOR_APPLIED = "applied"
METRIC_PRIOR_TOO_FEW_KEYFRAMES = "not_applicable_too_few_keyframes"
TRACKING_OK = "tracked"
TRACKING_NOT_INITIALIZED = "not_initialized_static_or_too_short"


def poses7_xyzw_to_c2w(poses7: np.ndarray) -> np.ndarray:
    """DPVO's terminate poses ``[t, q_xyzw]`` (camera-to-world) as (N,4,4) float64 matrices."""

    poses = np.asarray(poses7, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError(f"DPVO terminate poses must be finite [S,7], got {poses.shape}")
    translation = poses[:, :3]
    quaternion = poses[:, 3:7]
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm < 1e-8):
        raise ValueError("DPVO returned a zero quaternion")
    quaternion = quaternion / norm[:, None]
    x, y, z, w = quaternion.T
    rotation = np.empty((len(poses), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    c2w = np.tile(np.eye(4, dtype=np.float64), (len(poses), 1, 1))
    c2w[:, :3, :3] = rotation
    c2w[:, :3, 3] = translation
    return c2w


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def valid_metric_prior_diagnostics(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    successful_calls = value.get("successful_ba_calls_with_metric_prior")
    inactive_factors = value.get("inactive_factors")
    return (
        isinstance(successful_calls, int)
        and not isinstance(successful_calls, bool)
        and successful_calls > 0
        and isinstance(inactive_factors, int)
        and not isinstance(inactive_factors, bool)
        and inactive_factors == 0
        and value.get("regularization_enabled") is True
    )


def _plain_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def metric_prior_status(value: Any, *, warmup: int | None) -> str:
    """Check DPVO's metric-prior contract after terminate.

    ``applied``: at least one successful BA call carried the metric unary and no
    inactive factor is retained. ``not_applicable_too_few_keyframes``: the
    sequence never reached ``warmup`` accepted keyframes, so DPVO never turned
    the unary on (no call, nothing retained). Anything else -- in particular
    ``>= warmup`` keyframes with zero successful prior BA calls -- raises.
    ``warmup=None`` keeps the strict contract (only ``applied`` passes).
    """

    if valid_metric_prior_diagnostics(value):
        return METRIC_PRIOR_APPLIED
    if warmup is not None and isinstance(value, Mapping):
        seen = value.get("keyframes_seen")
        calls = value.get("successful_ba_calls_with_metric_prior")
        inactive = value.get("inactive_factors")
        if (
            _plain_int(seen)
            and seen < int(warmup)
            and _plain_int(calls)
            and calls == 0
            and _plain_int(inactive)
            and inactive == 0
            and value.get("regularization_enabled") is False
        ):
            return METRIC_PRIOR_TOO_FEW_KEYFRAMES
    raise RuntimeError(f"DPVO metric-prior BA contract failed: warmup={warmup}, diagnostics={value}")


@dataclass
class PreparedFrame:
    """One frame as DPVO consumes it (all tensors on the DPVO device)."""

    index: int
    frame_id: int
    image_bgr: torch.Tensor
    metric_depth: torch.Tensor
    metric_mask: torch.Tensor
    #: ``[valid metric-depth pixels, values that are not 8-bit, K entries != the shared K]``
    #: as an int64 device tensor (or a plain sequence), read by :func:`resolve_frame_checks`.
    checks: Any
    depth_start: Any
    ready: Any
    #: Source frame index (messages only).
    source_frame_id: int | None = None
    #: The K the frame carried, if any (messages only).
    frame_k: torch.Tensor | None = None


def resolve_frame_checks(item: PreparedFrame, shared_k: np.ndarray) -> int:
    """Read a frame's deferred checks (one host read) and raise on a failed one; returns the valid pixel count.

    Called by the consumer before DPVO, after ``wait_event`` when the depth runs ahead.
    """

    checks = item.checks
    values = checks.tolist() if torch.is_tensor(checks) else list(checks)
    if len(values) != 3:
        raise ValueError(f"expected 3 frame checks, got {values}")
    valid_pixels, not_8bit, k_mismatch = (int(value) for value in values)
    source = item.source_frame_id if item.source_frame_id is not None else item.frame_id
    if k_mismatch:
        frame_k = item.frame_k.tolist() if item.frame_k is not None else "another K"
        raise ValueError(f"frame {source} carries K {frame_k}, not the shared {np.asarray(shared_k).tolist()}")
    if not_8bit:
        raise ValueError("input RGB is not 8-bit; set slam.rgb_normalization: stream for float inputs")
    if valid_pixels == 0:
        raise RuntimeError(f"the depth model produced no valid metric depth for frame {source}")
    return valid_pixels


class CudaOps:
    """The CUDA stream/event operations of the producer, on one device."""

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)

    def current_stream(self) -> Any:
        return torch.cuda.current_stream(self.device)

    def new_stream(self) -> Any:
        return torch.cuda.Stream(device=self.device)

    def event(self) -> Any:
        return torch.cuda.Event(enable_timing=True)

    @contextmanager
    def producer_context(self, stream: Any) -> Iterator[None]:
        # Device, stream and inference mode are thread-local, so all three are
        # entered inside the producer thread.
        with torch.cuda.device(self.device), torch.cuda.stream(stream), torch.inference_mode():
            yield

    def record_stream(self, tensor: torch.Tensor, stream: Any) -> None:
        tensor.record_stream(stream)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    @staticmethod
    def elapsed_seconds(start: Any, end: Any) -> float:
        return float(start.elapsed_time(end)) / 1000.0


_SENTINEL = object()

PrepareFn = Callable[[int, Any, bool], PreparedFrame]


def _elapsed(ops: CudaOps, pairs: list[tuple[Any, Any]]) -> list[float]:
    return [ops.elapsed_seconds(start, end) for start, end in pairs]


def track_sequence(
    frames: Iterable[Any],
    count: int,
    prepare: PrepareFn,
    slam: Any,
    intrinsics: np.ndarray,
    *,
    ops: CudaOps,
    depth_queue: int = 4,
    metric_prior_warmup: int | None = None,
) -> dict[str, Any]:
    """Run per-frame metric depth and DPVO's metric API over one sequence.

    ``frames`` is iterated exactly once (in the producer thread when
    ``depth_queue >= 1``); ``prepare(index, frame, pinned)`` turns frame
    ``index`` into a :class:`PreparedFrame` on the current CUDA stream. DPVO
    receives the ordinal ``index`` as its timestamp.

    ``terminate`` runs under ``torch.no_grad()`` (see the call). After it, the
    metric-prior contract is checked with
    :func:`metric_prior_status`. A DPVO that never initialized (fewer than 8
    accepted keyframes: static or too short) still returns ``terminate``'s
    near-static trajectory, labelled ``tracking_status =
    "not_initialized_static_or_too_short"``.
    """

    if int(depth_queue) < 0:
        raise ValueError("depth_queue must be >= 0")
    intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics_np.shape != (4,) or not np.isfinite(intrinsics_np).all():
        raise ValueError(f"DPVO intrinsics must be finite [fx,fy,cx,cy], got {intrinsics_np}")
    intrinsics_tensor = torch.from_numpy(intrinsics_np).to(ops.device)
    frame_ids = np.arange(int(count))
    depth_pairs: list[tuple[Any, Any]] = []
    dpvo_pairs: list[tuple[Any, Any]] = []
    valid_depth_pixels: list[int] = []
    valid_depth_fractions: list[float] = []
    default = ops.current_stream()
    wall_started = time.perf_counter()
    pipeline_start = ops.event()
    pipeline_start.record(default)

    def consume(item: PreparedFrame) -> None:
        valid_pixels = resolve_frame_checks(item, intrinsics_np)
        depth_pairs.append((item.depth_start, item.ready))
        valid_depth_pixels.append(valid_pixels)
        valid_depth_fractions.append(valid_pixels / item.metric_mask.numel())
        dpvo_start = ops.event()
        dpvo_end = ops.event()
        dpvo_start.record(default)
        with torch.inference_mode():
            slam(
                int(item.frame_id),
                item.image_bgr,
                intrinsics_tensor,
                metric_depth=item.metric_depth,
                metric_mask=item.metric_mask,
            )
        dpvo_end.record(default)
        dpvo_pairs.append((dpvo_start, dpvo_end))

    consumed = 0
    if int(depth_queue) == 0:
        # Reference path: the depth model and DPVO each under inference mode,
        # preprocessing between them, no per-frame device synchronisation.
        for index, frame in enumerate(frames):
            item = prepare(index, frame, False)
            consume(item)
            consumed += 1
            del item
    else:
        auxiliary = ops.new_stream()
        ready_queue: queue.Queue[Any] = queue.Queue(maxsize=int(depth_queue))
        stop = threading.Event()
        producer_errors: list[BaseException] = []

        def put(item: Any) -> bool:
            # Bounded waits so that a stop request can never leave the producer
            # blocked on a full queue.
            while not stop.is_set():
                try:
                    ready_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def producer() -> None:
            try:
                with ops.producer_context(auxiliary):
                    auxiliary.wait_event(pipeline_start)
                    for index, frame in enumerate(frames):
                        if stop.is_set():
                            return
                        item = prepare(index, frame, True)
                        if not put(item):
                            return
                        del item
            except BaseException as error:  # re-raised in the calling thread
                producer_errors.append(error)
            finally:
                put(_SENTINEL)

        worker = threading.Thread(target=producer, name="metrictrail-depth", daemon=True)
        worker.start()
        try:
            with torch.inference_mode():
                while True:
                    item = ready_queue.get()
                    if item is _SENTINEL:
                        break
                    if item.index != consumed:
                        raise AssertionError(f"depth queue order drifted: {item.index} != {consumed}")
                    default.wait_event(item.ready)
                    for tensor in (item.image_bgr, item.metric_depth, item.metric_mask):
                        ops.record_stream(tensor, default)
                    consume(item)
                    consumed += 1
                    del item
        finally:
            stop.set()
            while True:
                try:
                    ready_queue.get_nowait()
                except queue.Empty:
                    break
            worker.join()
            while True:
                try:
                    ready_queue.get_nowait()
                except queue.Empty:
                    break
        if producer_errors:
            raise producer_errors[0]
    if consumed != count:
        raise RuntimeError(f"tracking consumed {consumed}/{count} frames")

    terminate_start = ops.event()
    terminate_end = ops.event()
    terminate_start.record(default)
    # DPVO's terminate (12 final update passes, plus the final global-BA /
    # re-measurement passes of the gba presets) is not decorated with no_grad
    # upstream; outside inference mode it keeps the autograd graph of those passes
    # alive (about 16 GiB instead of 3.5 GiB peak on KITTI). Grad is disabled
    # here, at the call site; the vendored DPVO is unchanged.
    with torch.no_grad():
        terminated = slam.terminate()
    terminate_end.record(default)
    ops.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    if not isinstance(terminated, tuple) or len(terminated) != 2:
        raise TypeError("DPVO terminate must return (poses7_xyzw, timestamps)")
    poses7, timestamps = terminated
    diagnostic_fn = getattr(slam, "metric_prior_diagnostics", None)
    metric_prior_diagnostics = jsonable(diagnostic_fn()) if callable(diagnostic_fn) else None
    prior_status = metric_prior_status(metric_prior_diagnostics, warmup=metric_prior_warmup)
    # DPVO sets is_initialized at its 8th accepted keyframe (the transport
    # wrapper forwards the attribute).
    initialized = bool(getattr(slam, "is_initialized", True))
    poses7 = np.asarray(poses7)
    timestamps = np.asarray(timestamps)
    if len(poses7) != len(frame_ids) or timestamps.shape != frame_ids.shape:
        raise ValueError(
            "DPVO terminate frame count mismatch: "
            f"poses={len(poses7)}, timestamps={timestamps.shape}, input={len(frame_ids)}"
        )
    if not np.allclose(timestamps.astype(np.float64), frame_ids, rtol=0.0, atol=1e-8):
        raise ValueError("DPVO terminate timestamps do not match the frame ordinals")
    depth_frame_seconds = _elapsed(ops, depth_pairs)
    dpvo_frame_seconds = _elapsed(ops, dpvo_pairs)
    timing = {
        "depth_frame_gpu_seconds": depth_frame_seconds,
        "dpvo_frame_gpu_seconds": dpvo_frame_seconds,
        "depth_stage_gpu_seconds": float(sum(depth_frame_seconds)),
        "dpvo_stage_gpu_seconds": float(sum(dpvo_frame_seconds)),
        "dpvo_terminate_gpu_seconds": ops.elapsed_seconds(terminate_start, terminate_end),
        "critical_path_gpu_seconds": ops.elapsed_seconds(pipeline_start, terminate_end),
        "wall_seconds": float(wall_seconds),
        "wall_fps": float(count / wall_seconds) if wall_seconds > 0 else None,
        "measurement": "CUDA events per stage; wall clock includes frame decode and DPVO terminate",
    }
    execution = {
        "depth_queue": int(depth_queue),
        "mode": "sequential" if int(depth_queue) == 0 else "overlapped_producer_thread",
        "depth_stream": "default" if int(depth_queue) == 0 else "auxiliary",
        "dpvo_stream": "default",
        "per_frame_event_dependency": int(depth_queue) > 0,
        "record_stream_on_dpvo_inputs": int(depth_queue) > 0,
        "frames": int(count),
    }
    return {
        "poses7_xyzw": poses7,
        "timestamps": timestamps,
        "timing": timing,
        "execution": execution,
        "valid_depth_fractions": valid_depth_fractions,
        "valid_depth_pixels": valid_depth_pixels,
        "metric_prior_diagnostics": metric_prior_diagnostics,
        "metric_prior_status": prior_status,
        "dpvo_initialized": initialized,
        "tracking_status": TRACKING_OK if initialized else TRACKING_NOT_INITIALIZED,
    }
