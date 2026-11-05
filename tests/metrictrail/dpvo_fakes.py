# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU stand-ins for the CUDA stream/event operations and for DPVO (producer and tracker tests)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch

from vipe.metrictrail.depth_producer import PreparedFrame


class FakeEvent:
    def __init__(self, log: list[tuple[str, Any]]) -> None:
        self.log = log
        self.time: float | None = None
        self.stream: Any = None

    def record(self, stream: Any = None) -> None:
        self.time = time.perf_counter()
        self.stream = stream if stream is not None else FakeOps.current.get()
        self.log.append(("record", self))


class FakeStream:
    def __init__(self, name: str, log: list[tuple[str, Any]]) -> None:
        self.name = name
        self.log = log

    def wait_event(self, event: FakeEvent) -> None:
        self.log.append((f"{self.name}.wait_event", event))


class _CurrentStream(threading.local):
    def __init__(self) -> None:
        self.stream: FakeStream | None = None

    def get(self) -> FakeStream | None:
        return self.stream


class FakeOps:
    """:class:`vipe.metrictrail.depth_producer.CudaOps` on CPU, logging every call."""

    current = _CurrentStream()

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.log: list[tuple[str, Any]] = []
        self.default = FakeStream("default", self.log)
        self.streams: list[FakeStream] = []
        self.record_stream_calls: list[tuple[int, FakeStream]] = []
        self.synchronized = 0
        self.producer_threads: list[str] = []
        FakeOps.current.stream = self.default

    def current_stream(self) -> FakeStream:
        return self.default

    def new_stream(self) -> FakeStream:
        stream = FakeStream(f"aux{len(self.streams)}", self.log)
        self.streams.append(stream)
        return stream

    def event(self) -> FakeEvent:
        return FakeEvent(self.log)

    @contextmanager
    def producer_context(self, stream: FakeStream) -> Iterator[None]:
        self.producer_threads.append(threading.current_thread().name)
        previous = FakeOps.current.stream
        FakeOps.current.stream = stream
        try:
            with torch.inference_mode():
                yield
        finally:
            FakeOps.current.stream = previous

    def record_stream(self, tensor: torch.Tensor, stream: FakeStream) -> None:
        self.record_stream_calls.append((id(tensor), stream))

    def synchronize(self) -> None:
        self.synchronized += 1

    @staticmethod
    def elapsed_seconds(start: FakeEvent, end: FakeEvent) -> float:
        return float((end.time or 0.0) - (start.time or 0.0))


def make_prepare(
    hw: tuple[int, int] = (4, 6),
    *,
    fail_at: int | None = None,
    wrong_index_at: int | None = None,
    produced: list[int] | None = None,
    delay: float = 0.0,
):
    """``prepare(index, frame, pinned)`` building deterministic tensors from the frame value."""

    def prepare(index: int, frame: Any, pinned: bool) -> PreparedFrame:
        if delay:
            time.sleep(delay)
        if fail_at is not None and index == fail_at:
            raise RuntimeError(f"depth failed at {index}")
        if produced is not None:
            produced.append(index)
        value = float(frame)
        start, ready = FakeOps.current_event_pair()
        depth = torch.full(hw, value + 1.0)
        return PreparedFrame(
            index=index if index != wrong_index_at else index + 1,
            frame_id=index,
            image_bgr=torch.full((3, *hw), int(frame) % 256, dtype=torch.uint8),
            metric_depth=depth,
            metric_mask=torch.ones(hw, dtype=torch.bool),
            checks=torch.tensor([hw[0] * hw[1], 0, 0]),
            depth_start=start,
            ready=ready,
        )

    return prepare


def _event_pair() -> tuple[FakeEvent, FakeEvent]:
    log: list[tuple[str, Any]] = []
    start, ready = FakeEvent(log), FakeEvent(log)
    start.record()
    ready.record()
    return start, ready


FakeOps.current_event_pair = staticmethod(_event_pair)  # type: ignore[attr-defined]


class FakeDPVO:
    """DPVO's tracking API: records every call; ``terminate`` returns identity poses."""

    def __init__(
        self,
        *,
        initialized: bool = True,
        diagnostics: dict[str, Any] | None = None,
        nan_rows: tuple[int, ...] = (),
        fail_at: int | None = None,
        fail_with: BaseException | None = None,
        consumed: list[int] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.is_initialized = initialized
        self.diagnostics = diagnostics if diagnostics is not None else applied_diagnostics()
        self.nan_rows = nan_rows
        self.fail_at = fail_at
        self.fail_with = fail_with
        self.consumed = consumed
        self.delay = delay
        self.terminated = False

    def __call__(self, tstamp, image, intrinsics, metric_depth=None, metric_mask=None):
        if self.delay:
            time.sleep(self.delay)
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise self.fail_with if self.fail_with is not None else RuntimeError("DPVO failed")
        if self.consumed is not None:
            self.consumed.append(int(tstamp))
        self.calls.append(
            {
                "tstamp": tstamp,
                "image": image.clone(),
                "intrinsics": intrinsics.clone(),
                "metric_depth": metric_depth.clone(),
                "metric_mask": metric_mask.clone(),
                "inference_mode": torch.is_inference_mode_enabled(),
            }
        )

    def terminate(self):
        self.terminated = True
        count = len(self.calls)
        poses = np.zeros((count, 7), dtype=np.float32)
        poses[:, 6] = 1.0
        poses[:, 0] = np.arange(count, dtype=np.float32)
        for row in self.nan_rows:
            if row < count:
                poses[row] = np.nan
        return poses, np.arange(count, dtype=np.float64)

    def metric_prior_diagnostics(self):
        return dict(self.diagnostics)


def applied_diagnostics() -> dict[str, Any]:
    return {
        "keyframes_seen": 20,
        "regularization_enabled": True,
        "successful_ba_calls_with_metric_prior": 7,
        "inactive_factors": 0,
    }


def too_few_keyframes_diagnostics(seen: int = 5) -> dict[str, Any]:
    return {
        "keyframes_seen": seen,
        "regularization_enabled": False,
        "successful_ba_calls_with_metric_prior": 0,
        "inactive_factors": 0,
    }
