# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted stand-ins for the calibration engine, depth provider and video stream (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream

FocalRule = Callable[[int, int, bool, int], float | None]


class FakeEngine:
    """A keyframe SLAM whose keyframe decisions and focal updates are scripted.

    - ``keyframe(frame_index) -> bool`` is the motion-filter verdict.
    - ``focal(call_index, steps, update_depth, accepted) -> new fx or None`` runs at
      every backend call; a changed focal triggers one depth inference per retained
      keyframe minus ``relatch_deficit`` (ViPE's relatch).
    - ``replace_after`` lists frame indices after whose frontend run the newest
      keyframe is merged into its predecessor (``remove_second_newest``).
    """

    internal_hw = (384, 512)

    def __init__(
        self,
        *,
        keyframe: Callable[[int], bool] = lambda index: True,
        focal: FocalRule | None = None,
        relatch_deficit: int = 0,
        replace_after: tuple[int, ...] = (),
        start_accepted: int = 0,
        **factory_kwargs: Any,
    ) -> None:
        self.keyframe = keyframe
        self.focal = focal or (lambda call, steps, update_depth, accepted: None)
        self.relatch_deficit = relatch_deficit
        self.replace_after = set(replace_after)
        self.accepted = start_accepted
        self.factory_kwargs = factory_kwargs
        self.k: np.ndarray | None = None
        self.depth = 0
        self.frames: list[Any] = []
        self.keyframe_indices: list[int] = []
        self.backend_calls: list[tuple[int, bool, int]] = []
        self._last_index: int | None = None
        self.benchmark_started = False

    def candidate(self, frame_index: int, frame: Any) -> bool:
        self.frames.append(frame)
        self._last_index = frame_index
        return bool(self.keyframe(frame_index))

    def add_keyframe(self, frame_index: int) -> None:
        assert frame_index == self._last_index
        if self.k is None:
            intrinsics = getattr(self.frames[-1], "intrinsics", None)
            self.k = (
                np.asarray(intrinsics.cpu().numpy(), dtype=np.float32)
                if intrinsics is not None
                else np.asarray([100.0, 100.0, 50.0, 40.0], dtype=np.float32)
            )
        self.accepted += 1
        self.keyframe_indices.append(frame_index)

    def run_frontend(self) -> None:
        if self._last_index in self.replace_after:
            self.accepted -= 1

    def run_backend(self, steps: int, *, update_depth: bool) -> None:
        assert self.k is not None
        call = len(self.backend_calls)
        self.backend_calls.append((steps, update_depth, self.accepted))
        new_focal = self.focal(call, steps, update_depth, self.accepted)
        if new_focal is not None and not np.isclose(new_focal, self.k[0], rtol=0.0, atol=0.0):
            self.k = self.k.copy()
            self.k[0] = self.k[1] = np.float32(new_focal)
            self.depth += max(self.accepted - self.relatch_deficit, 0)

    def accepted_keyframes(self) -> int:
        return self.accepted

    def recovered_k(self) -> np.ndarray:
        assert self.k is not None
        return self.k.copy()

    def depth_calls(self) -> int:
        return self.depth

    def begin_benchmark(self) -> None:
        self.benchmark_started = True

    def end_benchmark(self) -> dict[str, float]:
        return {"gpu_event_seconds": 0.0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}

    def depth_diagnostics(self) -> dict[str, Any]:
        return {"metric_depth_calls": self.depth}

    def backend_contract(self) -> dict[str, Any]:
        return {"network": "fake", "keyframe_depth": "fake"}


def converging_focal(start: float, final: float, *, final_jitter: float = 0.001) -> FocalRule:
    """Milestone calls move the focal towards ``final``; the audit calls change it by ``final_jitter``."""

    def rule(call: int, steps: int, update_depth: bool, accepted: int) -> float:
        if steps == 5:
            return start + (final - start) * min(1.0, 0.5 * (call + 1))
        if steps == 7:
            return final * (1.0 + final_jitter)
        return final

    return rule


INITIAL_K = np.asarray([300.0, 300.0, 252.0, 140.0], dtype=np.float32)
IMAGE_HW = (280, 504)


def frames(count: int, *, start_id: int = 0, k4: np.ndarray = INITIAL_K) -> list[tuple[int, Any]]:
    """``count`` opaque frames carrying ``k4`` as their (initial) intrinsics."""

    return [
        (start_id + index, SimpleNamespace(intrinsics=torch.as_tensor(np.asarray(k4, dtype=np.float32))))
        for index in range(count)
    ]


def u8_rgb(height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (height, width, 3), generator=generator, dtype=torch.uint8)


HW = (24, 40)
CPU = torch.device("cpu")


class FakeStream(VideoStream):
    def __init__(
        self,
        count: int,
        *,
        hw: tuple[int, int] = HW,
        step: int = 1,
        attributes: set[FrameAttribute] | None = None,
        with_intrinsics: bool = False,
        float_rgb: bool = False,
    ) -> None:
        self.count = count
        self.hw = hw
        self.step = step
        self._attributes = attributes or set()
        self.with_intrinsics = with_intrinsics
        self.float_rgb = float_rgb
        self.u8 = [u8_rgb(*hw, seed=index) for index in range(count)]

    def frame_size(self) -> tuple[int, int]:
        return self.hw

    def fps(self) -> float:
        return 30.0

    def name(self) -> str:
        return "clip"

    def __len__(self) -> int:
        return self.count

    def attributes(self) -> set[FrameAttribute]:
        return self._attributes

    def rgb(self, index: int) -> torch.Tensor:
        if self.float_rgb:
            return torch.rand(*self.hw, 3, generator=torch.Generator().manual_seed(index))
        return self.u8[index].float() / 255.0  # ViPE streams normalise on the host

    def __iter__(self):
        for index in range(self.count):
            yield VideoFrame(
                raw_frame_idx=index * self.step,
                rgb=self.rgb(index),
                intrinsics=torch.tensor([1.0, 1.0, 1.0, 1.0]) if self.with_intrinsics else None,
            )


class FakeFreeFovDepth(DepthEstimationModel):
    """A MODEL_METRIC_DEPTH provider whose free-FoV focal is scripted per call."""

    def __init__(self, focals: list[float], *, centre_offset: float = 0.0, depth_type=DepthType.MODEL_METRIC_DEPTH):
        self.focals = focals
        self.centre_offset = centre_offset
        self._depth_type = depth_type
        self.images: list[torch.Tensor] = []
        self.precision = "FP32"

    @property
    def depth_type(self) -> DepthType:
        return self._depth_type

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        assert src.rgb is not None
        return DepthEstimationResult(metric_depth=torch.ones(src.rgb.shape[:-1]))

    def estimate_free_fov_k4(self, rgb_bchw: torch.Tensor) -> np.ndarray:
        self.images.append(rgb_bchw)
        focal = self.focals[(len(self.images) - 1) % len(self.focals)]
        height, width = rgb_bchw.shape[-2:]
        return np.asarray([focal, focal, width / 2 + self.centre_offset, height / 2 - self.centre_offset])


class EngineFactory:
    def __init__(self, **engine_kwargs) -> None:
        self.engine_kwargs = engine_kwargs
        self.engines: list[FakeEngine] = []
        self.factory_kwargs: list[dict] = []

    def __call__(self, **kwargs) -> FakeEngine:
        self.factory_kwargs.append(kwargs)
        engine = FakeEngine(**self.engine_kwargs)
        self.engines.append(engine)
        return engine


def calibrate_with_fakes(stream: VideoStream, depth: DepthEstimationModel, factory: EngineFactory, **config):
    """SharedKIntrinsicsProcessor on CPU with a fake depth provider and engine."""

    config.setdefault("disable_tf32", False)
    from vipe.metrictrail.calibration.processor import SharedKIntrinsicsProcessor

    return SharedKIntrinsicsProcessor(stream, config, depth_model=depth, device=CPU, engine_factory=factory)
