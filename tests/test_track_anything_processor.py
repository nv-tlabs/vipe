# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from vipe.pipeline.processors import TrackAnythingProcessor
from vipe.streams.base import VideoFrame


class _FakeTracker:
    """Mimics TrackAnythingPipeline's batching/tracking interface."""

    def __init__(self, sam_run_gap: int) -> None:
        self.sam_run_gap = sam_run_gap
        self.frame_idx = 0
        self.tracked_sizes: list[tuple[int, int]] = []
        self.encoded_batches: list[int] = []

    def should_batch_aot_frame(self, pending_frame_count: int = 0) -> bool:
        frame_idx = self.frame_idx + pending_frame_count
        return frame_idx > 0 and frame_idx % self.sam_run_gap != 0

    def encode_aot_frames(self, frame_data_list):
        self.encoded_batches.append(len(frame_data_list))
        return [None] * len(frame_data_list)

    def track(self, frame_data, aot_img_embs=None):
        self.tracked_sizes.append(frame_data.size())
        self.frame_idx += 1
        # Instance id equals the number of tracked frames so far, so each
        # output frame records which tracker invocation produced its mask.
        instance = torch.full(frame_data.size(), self.frame_idx, dtype=torch.uint8)
        return instance, {0: "background"}


def _make_processor(track_downscale: int, track_stride: int, sam_run_gap: int, batch_size: int):
    processor = TrackAnythingProcessor.__new__(TrackAnythingProcessor)
    processor.mask_phrases = ["person"]
    processor.add_sky = False
    processor.track_downscale = track_downscale
    processor.track_stride = track_stride
    processor.sam_run_gap = max(1, sam_run_gap // track_stride)
    processor.tracker = _FakeTracker(processor.sam_run_gap)
    processor.mask_expand = 1
    processor.aot_encoder_batch_size = batch_size
    processor._last_instance = None
    processor._last_phrases = None
    processor._last_mask = None
    return processor


def _frames(n: int, size=(16, 24)):
    return [VideoFrame(raw_frame_idx=i, rgb=torch.rand(*size, 3)) for i in range(n)]


def _run(processor, frames):
    out = list(processor.update_iterator(iter(frames), 0))
    return out


def test_plain_processing_matches_frame_count_and_order():
    processor = _make_processor(track_downscale=1, track_stride=1, sam_run_gap=4, batch_size=3)
    out = _run(processor, _frames(11))
    assert [f.raw_frame_idx for f in out] == list(range(11))
    assert all(f.instance is not None and f.mask is not None for f in out)
    # Every frame tracked individually.
    assert processor.tracker.frame_idx == 11


def test_stride_reuses_previous_mask_in_order():
    processor = _make_processor(track_downscale=1, track_stride=2, sam_run_gap=4, batch_size=3)
    out = _run(processor, _frames(10))
    assert [f.raw_frame_idx for f in out] == list(range(10))
    # Only every other frame is tracked.
    assert processor.tracker.frame_idx == 5
    for idx, frame in enumerate(out):
        # Strided frames reuse the mask of the preceding tracked frame.
        expected_track_ordinal = idx // 2 + 1
        assert int(frame.instance[0, 0]) == expected_track_ordinal


def test_downscale_tracks_small_and_outputs_full_resolution():
    processor = _make_processor(track_downscale=2, track_stride=1, sam_run_gap=4, batch_size=1)
    out = _run(processor, _frames(4, size=(16, 24)))
    assert all(size == (8, 12) for size in processor.tracker.tracked_sizes)
    assert all(frame.instance.shape == (16, 24) for frame in out)
    assert all(frame.mask.shape == (16, 24) for frame in out)


def test_batched_stride_preserves_order_and_batches():
    processor = _make_processor(track_downscale=2, track_stride=2, sam_run_gap=8, batch_size=2)
    out = _run(processor, _frames(16))
    assert [f.raw_frame_idx for f in out] == list(range(16))
    assert processor.tracker.frame_idx == 8
    assert len(processor.tracker.encoded_batches) > 0
    for idx, frame in enumerate(out):
        assert int(frame.instance[0, 0]) == idx // 2 + 1
