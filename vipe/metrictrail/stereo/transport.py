# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stereo transport: the synchronized right raster handed to DPVO next to each left frame.

:class:`StereoSlamProxy` wraps DPVO, leaves the mono depth loop
(:func:`vipe.metrictrail.depth_producer.track_sequence`) as it is, and adds
``right_image`` (BGR uint8, CHW) and ``stereo_baseline_m`` to every DPVO call,
taking the right frame with DPVO's timestamp (the frame ordinal).

:class:`RightFrames` presents a ViPE right stream as a ``Sequence`` of CHW float32
``uint8 * (1/255)`` frames, read once, in order: DPVO receives the ordinals 0..N-1 in
FIFO order, and a request out of that order is refused instead of silently re-decoding.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vipe.streams.base import VideoStream


class StereoSlamProxy:
    """Inject a synchronized right raster without changing the mono depth loop."""

    def __init__(
        self,
        slam: Any,
        right: Sequence[np.ndarray],
        *,
        baseline_m: float,
        device: torch.device,
    ) -> None:
        if baseline_m <= 0 or len(right) == 0:
            raise ValueError("stereo requires positive baseline and right frames")
        self.slam = slam
        self.right = right
        self.baseline_m = float(baseline_m)
        self.device = device

    def __call__(
        self,
        timestamp: int,
        image_bgr: torch.Tensor,
        intrinsics: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        index = int(timestamp)
        rgb = np.asarray(self.right[index])
        if rgb.dtype != np.uint8:
            if not np.issubdtype(rgb.dtype, np.floating) or not np.isfinite(rgb).all():
                raise TypeError("right RGB must be uint8 or finite unit float")
            rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        right_rgb = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=self.device, dtype=torch.uint8)
        return self.slam(
            timestamp,
            image_bgr,
            intrinsics,
            right_image=right_rgb[[2, 1, 0]].contiguous(),
            stereo_baseline_m=self.baseline_m,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.slam, name)


def chw_float_frames(stream: VideoStream) -> Iterator[np.ndarray]:
    """``stream`` as CHW float32 ``uint8 * (1/255)`` (step 2's pipeline helper of the same name)."""

    from vipe.metrictrail.pipeline import chw_float_frames as frames

    return frames(stream)


class RightFrames(Sequence[np.ndarray]):
    """A ViPE right stream as CHW float32 ``uint8 * (1/255)`` frames, read once and in order."""

    def __init__(self, stream: VideoStream, image_hw: Sequence[int]) -> None:
        self.stream = stream
        self.frame_count = len(stream)  # not `count`: Sequence.count is a method
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self._frames: Iterator[np.ndarray] | None = None
        self._next = 0

    def __len__(self) -> int:
        return self.frame_count

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.frame_count, 3, *self.image_hw)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            raise TypeError("right frames are read one at a time, in order")
        value = int(index)
        if not 0 <= value < self.frame_count:
            raise IndexError(index)
        if value != self._next:
            raise ValueError(f"right frames are read in order: requested {value}, next is {self._next}")
        if self._frames is None:
            self._frames = chw_float_frames(self.stream)
        try:
            frame = next(self._frames)
        except StopIteration:
            raise ValueError(f"right stream ended after {self._next} of {self.frame_count} frames") from None
        if tuple(frame.shape) != (3, *self.image_hw):
            raise ValueError(f"right RGB shape drifted: {tuple(frame.shape)} != {(3, *self.image_hw)}")
        self._next += 1
        return frame
