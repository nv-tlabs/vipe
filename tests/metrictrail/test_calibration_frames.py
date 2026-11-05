# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SharedK sweep reads every frame exactly once, in order, after the initializer prefix.

ViPE's FrameDirStream / RawMp4Stream restart at frame 0 whenever ``__iter__`` is called. A sweep that
continued with ``for frame in iterator`` therefore replayed the video after the 32-frame prefix
(0..31, 0..N-1), which corrupted the calibration graph on KITTI.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vipe.metrictrail.calibration.processor import CalibrationFrames
from vipe.streams.base import VideoFrame


class RestartingStream:
    """Mimics ViPE's frame-dir / mp4 streams: ``__iter__`` resets to frame 0 and returns ``self``."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.index = 0

    def __iter__(self) -> "RestartingStream":
        self.index = 0
        return self

    def __next__(self) -> VideoFrame:
        if self.index >= self.count:
            raise StopIteration
        value = (self.index % 256) / 255.0
        frame = VideoFrame(raw_frame_idx=self.index, rgb=torch.full((4, 6, 3), value))
        self.index += 1
        return frame


@pytest.mark.parametrize("count, prefix", [(80, 32), (20, 32), (32, 32)])
def test_sweep_reads_each_frame_once_in_order(count: int, prefix: int) -> None:
    frames = CalibrationFrames(RestartingStream(count), device=torch.device("cpu"), normalization="metrictrail")
    head = frames.head(prefix)
    assert [source_id for source_id, _ in head] == list(range(min(prefix, count)))
    ids = [source_id for source_id, _ in frames.sweep(np.array([10.0, 10.0, 3.0, 2.0]))]
    assert ids == list(range(count))


def test_sweep_refuses_a_restarted_sequence() -> None:
    class Replaying(RestartingStream):
        def __next__(self) -> VideoFrame:
            frame = super().__next__()
            if self.index == 5:  # jump back once, as a restarted reader would
                self.index = 0
            return frame

    frames = CalibrationFrames(Replaying(10), device=torch.device("cpu"), normalization="metrictrail")
    frames.head(2)
    with pytest.raises(RuntimeError, match="strictly increasing"):
        list(frames.sweep(np.array([10.0, 10.0, 3.0, 2.0])))
