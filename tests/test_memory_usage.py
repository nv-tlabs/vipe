from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from vipe.streams.base import CachedVideoStream, FrameAttribute, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.io import ArtifactPath, read_depth_artifacts, save_artifacts


class SinglePassStream(VideoStream):
    def __init__(self, n_frames: int = 2) -> None:
        self.n_frames = n_frames
        self.n_iterations = 0

    def frame_size(self) -> tuple[int, int]:
        return (8, 8)

    def name(self) -> str:
        return "single_pass"

    def fps(self) -> float:
        return 30.0

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator[VideoFrame]:
        if self.n_iterations > 0:
            raise AssertionError("stream was iterated more than once")
        self.n_iterations += 1
        for frame_idx in range(self.n_frames):
            yield VideoFrame(
                raw_frame_idx=frame_idx,
                rgb=torch.full((8, 8, 3), frame_idx / 10.0, dtype=torch.float32),
                intrinsics=torch.tensor([4.0, 4.0, 4.0, 4.0]),
                camera_type=CameraType.PINHOLE,
                metric_depth=torch.ones((8, 8), dtype=torch.float32) * (frame_idx + 1),
                instance=torch.full((8, 8), frame_idx + 1, dtype=torch.uint8),
                instance_phrases={frame_idx + 1: f"object-{frame_idx}"},
            )

    def attributes(self) -> set[FrameAttribute]:
        return {
            FrameAttribute.INTRINSICS,
            FrameAttribute.CAMERA_TYPE,
            FrameAttribute.METRIC_DEPTH,
            FrameAttribute.INSTANCE,
        }


def test_save_artifacts_streams_in_single_pass(tmp_path) -> None:
    stream = SinglePassStream(n_frames=2)
    artifact_path = ArtifactPath(tmp_path, stream.name())

    save_artifacts(artifact_path, stream)

    assert stream.n_iterations == 1
    assert artifact_path.rgb_path.exists()
    assert artifact_path.intrinsics_path.exists()
    assert artifact_path.camera_type_path.exists()
    assert artifact_path.depth_path.exists()
    assert artifact_path.mask_path.exists()
    assert artifact_path.mask_phrase_path.read_text().splitlines() == ["1: object-0", "2: object-1"]
    assert [frame_idx for frame_idx, _ in read_depth_artifacts(artifact_path.depth_path)] == [0, 1]


def test_compact_cpu_frame_storage_uses_uint8_rgb() -> None:
    frame = VideoFrame(raw_frame_idx=0, rgb=torch.rand((8, 8, 3), dtype=torch.float32))

    compact = frame.cpu(compact_rgb=True)

    assert compact.rgb.dtype == torch.uint8
    restored = compact.rgb.float() / 255.0
    assert torch.allclose(restored, frame.rgb, atol=1 / 255)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CachedVideoStream restores cached frames on CUDA")
def test_cached_video_stream_compacts_rgb_storage() -> None:
    stream = SinglePassStream(n_frames=1)
    cached = CachedVideoStream(stream)

    _ = cached[0]

    assert cached.data[0].rgb.dtype == torch.uint8
