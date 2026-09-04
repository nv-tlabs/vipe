# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for user-provided calibration on the standard input paths.

`pipeline.init.intrinsics=gt` was already exposed by the typed configuration, but neither
`vipe infer video.mp4` nor `vipe infer --image-dir` had a way to ingest a calibration, so
the option could not be exercised from the CLI. These tests pin the contract that wires it
up on both paths:

- `FrameDirStream` and `RawMp4Stream` accept one pinhole `[fx, fy, cx, cy]` vector,
  validate it, stamp it on every frame with `CameraType.PINHOLE`, and advertise exactly
  those attributes;
- `DefaultAnnotationPipeline` skips GeoCalib under `init.intrinsics=gt` and refuses a
  stream that does not carry intrinsics, instead of silently estimating them;
- the CLI validates the calibration JSON before any model is loaded, applies one
  calibration to every input it is given (a frame directory, a video, or a directory of
  videos) and checks each input's frame size against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from vipe.config import parse_typed_config
from vipe.pipeline import make_pipeline
from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.streams.base import FrameAttribute, ProcessedVideoStream, VideoStream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.cameras import CameraType

cv2 = pytest.importorskip("cv2")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FrameDirStream moves frames to CUDA")

WIDTH, HEIGHT = 64, 48
CALIBRATION = {"width": WIDTH, "height": HEIGHT, "fx": 80.0, "fy": 80.0, "cx": 32.0, "cy": 24.0}
INTRINSICS = torch.tensor([80.0, 80.0, 32.0, 24.0])


def _write_frames(directory: Path, count: int = 3) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        image = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.imwrite(str(directory / f"frame-{index:05d}.png"), image)
    return directory


def _write_calibration(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _write_video(path: Path, count: int = 3) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 in this environment")
    rng = np.random.default_rng(0)
    for _ in range(count):
        writer.write(rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()
    return path


class _EmptyStream(VideoStream):
    """A stream that carries no frame attributes at all."""

    def frame_size(self) -> tuple[int, int]:
        return (HEIGHT, WIDTH)

    def fps(self) -> float:
        return 30.0

    def name(self) -> str:
        return "empty"

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())


class _CalibratedStream(_EmptyStream):
    def attributes(self) -> set[FrameAttribute]:
        return {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}


# --- FrameDirStream -------------------------------------------------------------------


def test_uncalibrated_frame_dir_stream_advertises_no_attributes(tmp_path: Path) -> None:
    stream = FrameDirStream(_write_frames(tmp_path / "frames"))
    assert stream.attributes() == set()
    assert ProcessedVideoStream(stream, []).attributes() == set()


def test_calibrated_frame_dir_stream_advertises_intrinsics_and_camera_type(tmp_path: Path) -> None:
    stream = FrameDirStream(_write_frames(tmp_path / "frames"), intrinsics=INTRINSICS)
    assert stream.attributes() == {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}
    # The CLI wraps the stream in ProcessedVideoStream(...).cache(); the pipeline reads
    # the attribute set through that wrapper, so the advertisement must survive it.
    assert ProcessedVideoStream(stream, []).attributes() == {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}


@pytest.mark.parametrize(
    "intrinsics",
    [
        torch.tensor([80.0, 80.0, 32.0]),
        torch.tensor([[80.0, 80.0, 32.0, 24.0]]),
        torch.tensor([80.0, float("nan"), 32.0, 24.0]),
        torch.tensor([80.0, float("inf"), 32.0, 24.0]),
    ],
)
def test_frame_dir_stream_rejects_malformed_intrinsics(tmp_path: Path, intrinsics: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite \\[fx, fy, cx, cy\\]"):
        FrameDirStream(_write_frames(tmp_path / "frames"), intrinsics=intrinsics)


@pytest.mark.parametrize("intrinsics", [torch.tensor([0.0, 80.0, 32.0, 24.0]), torch.tensor([80.0, -1.0, 32.0, 24.0])])
def test_frame_dir_stream_rejects_non_positive_focal_lengths(tmp_path: Path, intrinsics: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="focal lengths must be positive"):
        FrameDirStream(_write_frames(tmp_path / "frames"), intrinsics=intrinsics)


@requires_cuda
def test_calibrated_frames_carry_a_copy_of_the_pinhole_intrinsics(tmp_path: Path) -> None:
    stream = FrameDirStream(_write_frames(tmp_path / "frames"), intrinsics=INTRINSICS)
    frames = list(stream)
    assert len(frames) == 3
    for frame in frames:
        assert frame.camera_type is CameraType.PINHOLE
        assert frame.intrinsics is not None
        assert frame.intrinsics.device.type == "cuda"
        assert torch.equal(frame.intrinsics.cpu(), INTRINSICS)
        assert frame.attributes() >= {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}
    # Each frame owns its own tensor: mutating one must not leak into the next.
    frames[0].intrinsics[0] = 1.0
    assert torch.equal(frames[1].intrinsics.cpu(), INTRINSICS)


@requires_cuda
def test_uncalibrated_frames_carry_no_intrinsics(tmp_path: Path) -> None:
    for frame in FrameDirStream(_write_frames(tmp_path / "frames")):
        assert frame.intrinsics is None
        assert frame.camera_type is None


# --- RawMp4Stream ----------------------------------------------------------------------------


def test_uncalibrated_mp4_stream_advertises_no_attributes(tmp_path: Path) -> None:
    stream = RawMp4Stream(_write_video(tmp_path / "video.mp4"))
    assert stream.attributes() == set()
    assert stream.frame_size() == (HEIGHT, WIDTH)


def test_calibrated_mp4_stream_advertises_intrinsics_and_camera_type(tmp_path: Path) -> None:
    stream = RawMp4Stream(_write_video(tmp_path / "video.mp4"), intrinsics=INTRINSICS)
    assert stream.attributes() == {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}
    assert ProcessedVideoStream(stream, []).attributes() == {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}


@pytest.mark.parametrize("intrinsics", [torch.tensor([80.0, 80.0, 32.0]), torch.tensor([0.0, 80.0, 32.0, 24.0])])
def test_mp4_stream_rejects_malformed_intrinsics(tmp_path: Path, intrinsics: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        RawMp4Stream(_write_video(tmp_path / "video.mp4"), intrinsics=intrinsics)


@requires_cuda
def test_calibrated_mp4_frames_carry_the_pinhole_intrinsics(tmp_path: Path) -> None:
    frames = list(RawMp4Stream(_write_video(tmp_path / "video.mp4"), intrinsics=INTRINSICS))
    assert len(frames) == 3
    for frame in frames:
        assert frame.camera_type is CameraType.PINHOLE
        assert torch.equal(frame.intrinsics.cpu(), INTRINSICS)


# --- Typed configuration -------------------------------------------------------------------


def _config(tmp_path: Path, *extra: str):
    return parse_typed_config(
        "default",
        [
            "pipeline=default",
            "streams=frame_dir_stream",
            f"streams.base_path={tmp_path / 'frames'}",
            f"pipeline.output.path={tmp_path / 'out'}",
            *extra,
        ],
    )


def test_gt_intrinsics_disable_intrinsic_optimization(tmp_path: Path) -> None:
    config = _config(tmp_path, "pipeline.init.intrinsics=gt")
    assert config.pipeline.init.intrinsics == "gt"
    assert config.pipeline.slam.optimize_intrinsics is False


def test_geocalib_intrinsics_keep_intrinsic_optimization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.pipeline.init.intrinsics == "geocalib"
    assert config.pipeline.slam.optimize_intrinsics is True


# --- DefaultAnnotationPipeline ----------------------------------------------------------------


def test_gt_pipeline_refuses_a_stream_without_intrinsics(tmp_path: Path) -> None:
    pipeline = make_pipeline(_config(tmp_path, "pipeline.init.intrinsics=gt").pipeline)
    assert isinstance(pipeline, DefaultAnnotationPipeline)
    with pytest.raises(ValueError, match="requires every input frame to provide intrinsics"):
        pipeline._add_init_processors(_EmptyStream())


def test_gt_pipeline_does_not_add_geocalib_for_a_calibrated_stream(tmp_path: Path) -> None:
    pipeline = make_pipeline(_config(tmp_path, "pipeline.init.intrinsics=gt", "pipeline.init.instance=null").pipeline)
    processed = pipeline._add_init_processors(_CalibratedStream())
    assert processed.processors == []
    assert processed.attributes() == {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}


def test_geocalib_pipeline_refuses_a_stream_that_already_carries_intrinsics(tmp_path: Path) -> None:
    pipeline = make_pipeline(_config(tmp_path, "pipeline.init.instance=null").pipeline)
    with pytest.raises(AssertionError):
        pipeline._add_init_processors(_CalibratedStream())


# --- CLI ------------------------------------------------------------------------------------


def _infer():
    from vipe.cli.main import infer

    return infer


def test_cli_accepts_intrinsics_on_the_video_path(tmp_path: Path) -> None:
    """A calibration applies to a video exactly as to a frame directory: the malformed
    case is refused for its content, never for the input type."""
    video = _write_video(tmp_path / "video.mp4")
    calibration = _write_calibration(tmp_path / "calibration.json", {**CALIBRATION, "fx": 0.0})
    result = CliRunner().invoke(_infer(), [str(video), "--intrinsics", str(calibration)])
    assert result.exit_code != 0
    assert "requires --image-dir" not in result.output
    assert "intrinsics" in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in CALIBRATION.items() if k != "cy"},
        {**CALIBRATION, "k1": 0.0},
        {**CALIBRATION, "fx": 0.0},
        {**CALIBRATION, "fy": float("nan")},
    ],
)
def test_cli_rejects_malformed_calibration_before_loading_models(tmp_path: Path, payload: dict) -> None:
    frames = _write_frames(tmp_path / "frames")
    calibration = _write_calibration(tmp_path / "calibration.json", payload)
    result = CliRunner().invoke(
        _infer(), ["--image-dir", str(frames), "--intrinsics", str(calibration), "-o", str(tmp_path / "out")]
    )
    assert result.exit_code != 0
    assert "intrinsics" in result.output
    assert not (tmp_path / "out").exists()  # refused before the pipeline created anything


@requires_cuda
def test_cli_rejects_a_calibration_whose_dimensions_do_not_match_the_frames(tmp_path: Path) -> None:
    frames = _write_frames(tmp_path / "frames")
    calibration = _write_calibration(tmp_path / "calibration.json", {**CALIBRATION, "width": WIDTH + 1})
    result = CliRunner().invoke(
        _infer(), ["--image-dir", str(frames), "--intrinsics", str(calibration), "-o", str(tmp_path / "out")]
    )
    assert result.exit_code != 0
    assert "do not match frames" in result.output


@requires_cuda
def test_cli_rejects_a_video_calibration_whose_dimensions_do_not_match(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "video.mp4")
    calibration = _write_calibration(tmp_path / "calibration.json", {**CALIBRATION, "height": HEIGHT + 2})
    result = CliRunner().invoke(_infer(), [str(video), "--intrinsics", str(calibration), "-o", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "do not match frames" in result.output


@requires_cuda
def test_cli_applies_one_calibration_to_every_video_in_a_directory(tmp_path: Path) -> None:
    """A directory of videos shares one camera: the calibration is checked against each
    video, and a mismatch names the video it failed on."""
    videos = tmp_path / "videos"
    videos.mkdir()
    _write_video(videos / "a.mp4")
    _write_video(videos / "b.mp4")
    calibration = _write_calibration(tmp_path / "calibration.json", {**CALIBRATION, "width": WIDTH + 1})
    result = CliRunner().invoke(_infer(), [str(videos), "--intrinsics", str(calibration), "-o", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "do not match frames" in result.output
    assert "a.mp4" in result.output
