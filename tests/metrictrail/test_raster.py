# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The input raster (CPU).

- geometry and stride: :func:`plan_raster` against a reference copy of the cover-fit arithmetic
  (``cover_crop``: scale to cover, integer centre crop, ceil stride), and a plain resize (``resize``);
- frame selection: :func:`fps_filter_source_indices` against ffmpeg's ``fps`` filter as measured
  on constant-rate clips;
- pixels: ViPE's bilinear resize (``VideoFrame.resize`` itself) then the integer crop, on frames
  decoded as ViPE's streams decode them;
- rotated (portrait phone) clips: planned on the raster ViPE's ``RawMp4Stream`` decodes (OpenCV applies
  the display rotation), cropped, never squashed, and recorded (``video_orientation``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from metrictrail_fakes import FakeStream

import vipe.metrictrail.raster as raster_module
from vipe.metrictrail.raster import (
    PreparedRasterStream,
    fps_filter_source_indices,
    plan_raster,
    prepare_input_raster,
    raster_frame,
    raster_uint8,
)
from vipe.streams.base import FrameAttribute, ProcessedVideoStream, VideoFrame
from vipe.streams.raw_mp4_stream import RawMp4Stream

CPU = torch.device("cpu")


def raster_config(**overrides) -> SimpleNamespace:
    values = {"policy": "cover_crop", "height": 280, "width": 504, "max_frames": 1500}
    values.update(overrides)
    return SimpleNamespace(**values)


def prepare_video_arithmetic(info: dict, height_out: int = 280, width_out: int = 504, max_frames: int = 1500):
    """Reference cover-fit arithmetic (geometry and stride): scale to cover, integer centre crop, ceil stride."""

    scale = max(width_out / info["width"], height_out / info["height"])
    width = int(round(info["width"] * scale))
    height = int(round(info["height"] * scale))
    off_x, off_y = (width - width_out) // 2, (height - height_out) // 2
    stride = max(1, -(-info["frames"] // max_frames))
    out_fps = info["fps"] / stride
    return scale, (height, width), (off_x, off_y), stride, out_fps


def vipe_bilinear(rgb_u8: np.ndarray, size_hw: tuple[int, int]) -> torch.Tensor:
    """ViPE's VideoFrame.resize written out: interpolate(mode=bilinear), align_corners and antialias at default."""

    rgb = torch.from_numpy(rgb_u8).float() / 255.0
    return torch.nn.functional.interpolate(rgb.permute(2, 0, 1)[None], size_hw, mode="bilinear")[0].permute(1, 2, 0)


def to_u8(rgb: torch.Tensor) -> np.ndarray:
    return torch.round(rgb * 255.0).clamp(0, 255).to(torch.uint8).numpy()


@pytest.mark.parametrize("max_frames", [1500, 100])
def test_plan_equals_prepare_video_arithmetic(max_frames: int) -> None:
    sources = [(720, 1280), (1080, 1920), (1920, 1080), (480, 640), (280, 504), (100, 100), (2160, 3840)]
    sources += [(1080, 1440), (359, 641), (72, 128), (96, 160), (283, 505)]
    for height, width in sources:
        for frames in (1, 16, 99, 100, 101, 819, 1500, 1501, 3000, 3001, 4661, 10000):
            for fps in (30.0, 30000 / 1001, 25.0, 59.94):
                info = {"width": width, "height": height, "frames": frames, "fps": fps}
                plan = plan_raster((height, width), frames, fps, max_frames=max_frames)
                scale, resized, offsets, stride, out_fps = prepare_video_arithmetic(info, max_frames=max_frames)
                assert plan.policy == "cover_crop" and plan.scale == scale
                assert plan.resized_hw == resized and plan.crop_left_top == offsets
                assert plan.frame_stride == stride and plan.prepared_fps == out_fps
                assert plan.target_hw == (280, 504) and plan.resized_hw[0] >= 280 and plan.resized_hw[1] >= 504
                # resize: the same stride rule, the target raster itself and no crop.
                plain = plan_raster((height, width), frames, fps, max_frames=max_frames, policy="resize")
                assert (plain.scale, plain.resized_hw, plain.crop_left_top) == (None, (280, 504), (0, 0))
                assert (plain.frame_stride, plain.prepared_fps) == (stride, out_fps)
    # The demo video (examples/demo.mp4, 1280x720, 819 frames at 30 fps).
    demo = plan_raster((720, 1280), 819, 30.0)
    assert (demo.scale, demo.resized_hw, demo.crop_left_top) == (0.39375, (284, 504), (0, 2))
    assert (demo.frame_stride, demo.prepared_fps) == (1, 30.0)
    # 7-Scenes: 480x640 frames stretched to 434x574 (plain resize), no crop.
    seven = plan_raster((480, 640), 1000, 30.0, target_hw=(434, 574), policy="resize")
    assert (seven.resized_hw, seven.crop_left_top, seven.frame_stride, seven.resize) == ((434, 574), (0, 0), 1, True)
    with pytest.raises(ValueError, match="invalid prepared raster"):
        plan_raster((720, 1280), 10, 30.0, target_hw=(0, 504))
    with pytest.raises(ValueError, match="unknown input raster policy"):
        plan_raster((720, 1280), 10, 30.0, policy="stretch")


# (source frames, stride) -> frames kept by prepare_video.py (ffmpeg 4.4.2 fps filter), measured on
# constant-rate H.264 clips whose frames carry their index: count, first three, last.
MEASURED_FPS_FILTER = {
    (1501, 2): (751, [0, 2, 4], 1500),
    (3001, 3): (1000, [1, 4, 7], 2998),
    (301, 4): (75, [1, 5, 9], 297),
    (1000, 4): (250, [1, 5, 9], 997),
}


def test_frame_selection_equals_the_measured_fps_filter() -> None:
    for (frames, stride), (count, head, last) in MEASURED_FPS_FILTER.items():
        kept = fps_filter_source_indices(frames, stride)
        assert (len(kept), kept[:3], kept[-1]) == (count, head, last)
    assert fps_filter_source_indices(819, 1) == list(range(819))
    # Never the plain range(0, n, k) for k >= 3: the filter rounds timestamps to the nearest output tick.
    assert fps_filter_source_indices(12, 3) == [1, 4, 7, 10]


# ------------------------------------------------------------------ pixels


def test_pixels_are_vipe_bilinear_resize_then_the_integer_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    image = FakeStream(1, hw=(72, 128)).u8[0].numpy()
    calls: list[tuple[tuple[int, int], tuple[int, int]]] = []
    original = VideoFrame.resize

    def spy(self: VideoFrame, size: tuple[int, int]) -> VideoFrame:
        calls.append((tuple(self.rgb.shape[:2]), tuple(size)))
        return original(self, size)

    monkeypatch.setattr(VideoFrame, "resize", spy)
    cover = plan_raster((72, 128), 20, 30.0, target_hw=(28, 48))
    assert (cover.resized_hw, cover.crop_left_top) == ((28, 50), (1, 0))
    prepared = raster_uint8(image, cover)
    np.testing.assert_array_equal(prepared, to_u8(vipe_bilinear(image, (28, 50))[:, 1:49]))
    plain = plan_raster((72, 128), 20, 30.0, target_hw=(28, 48), policy="resize")
    np.testing.assert_array_equal(raster_uint8(image, plain), to_u8(vipe_bilinear(image, (28, 48))))
    # ViPE's own resize was called for both (not a reimplementation).
    assert calls == [((72, 128), (28, 50)), ((72, 128), (28, 48))]
    # Not an antialiased or a corner-aligned bilinear.
    rgb = torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255.0
    for other in (
        torch.nn.functional.interpolate(rgb, (28, 48), mode="bilinear", antialias=True),
        torch.nn.functional.interpolate(rgb, (28, 48), mode="bilinear", align_corners=True),
    ):
        assert not np.array_equal(raster_uint8(image, plain), to_u8(other[0].permute(1, 2, 0)))
    # raster_uint8 is raster_frame on ViPE's stream convention (uint8 / 255).
    assert torch.equal(
        raster_frame(torch.from_numpy(image).float() / 255.0, plain), torch.from_numpy(raster_uint8(image, plain))
    )
    with pytest.raises(ValueError, match="expected"):
        raster_uint8(image[:10], plain)


def test_cover_crop_prepares_a_stream_with_vipe_bilinear() -> None:
    stream = FakeStream(20, hw=(72, 128))
    prepared, record = prepare_input_raster(stream, raster_config(), device=CPU)
    assert isinstance(prepared, PreparedRasterStream)
    assert len(prepared) == 20 and prepared.frame_size() == (280, 504) and prepared.fps() == 30.0
    assert record["policy"] == "cover_crop" and record["interpolation"]["name"] == "vipe_bilinear"
    assert record["interpolation"]["call"] == "vipe.streams.base.VideoFrame.resize"
    assert "resampler" not in record and "ffmpeg_filters" not in record
    assert record["crop_left_top"] == [0, 2] and record["resized_hw"] == [284, 504] and record["frame_stride"] == 1
    assert record["frames"] == 20 and record["scale"] == 3.9375
    frames = list(prepared)
    assert [frame.raw_frame_idx for frame in frames] == list(range(20))
    first = frames[0]
    assert first.rgb.dtype == torch.float32 and tuple(first.rgb.shape) == (280, 504, 3)
    # ViPE's stream convention (uint8 / 255) round-trips to the prepared bytes.
    assert torch.equal(torch.round(first.rgb * 255).to(torch.uint8), prepared.uint8_frame(0))
    assert first.intrinsics is None and prepared.attributes() == set()
    for index in (0, 7, 19):
        expected = to_u8(vipe_bilinear(stream.u8[index].numpy(), (284, 504))[2:282])
        np.testing.assert_array_equal(prepared.uint8_frame(index).numpy(), expected)


def test_resize_policy_stretches_without_a_crop() -> None:
    stream = FakeStream(17, hw=(48, 64))
    prepared, record = prepare_input_raster(stream, raster_config(policy="resize", height=43, width=57), device=CPU)
    assert prepared.frame_size() == (43, 57) and len(prepared) == 17
    assert record["policy"] == "resize" and record["scale"] is None
    assert record["resized_hw"] == [43, 57] and record["crop_left_top"] == [0, 0] and record["prepared_hw"] == [43, 57]
    assert record["reference"].startswith("7-Scenes loaders")
    for index in (0, 16):
        expected = to_u8(vipe_bilinear(stream.u8[index].numpy(), (43, 57)))
        np.testing.assert_array_equal(prepared.uint8_frame(index).numpy(), expected)


@pytest.mark.parametrize("policy", ["cover_crop", "resize"])
def test_prepared_frames_pass_through_exactly(policy: str) -> None:
    stream = FakeStream(30, hw=(280, 504))
    prepared, record = prepare_input_raster(stream, raster_config(policy=policy), device=CPU)
    assert record["resized_hw"] == [280, 504] and record["crop_left_top"] == [0, 0]
    for index, frame in enumerate(prepared):
        assert torch.equal(prepared.uint8_frame(index), stream.u8[index])
        assert torch.equal(frame.rgb, stream.u8[index].float() / 255.0)


@pytest.mark.parametrize("policy", ["cover_crop", "resize"])
def test_long_input_keeps_the_fps_filter_frames(policy: str) -> None:
    stream = FakeStream(70, hw=(280, 504), step=3)  # raw frame ids 0, 3, 6, ...
    prepared, record = prepare_input_raster(stream, raster_config(policy=policy, max_frames=20), device=CPU)
    expected = fps_filter_source_indices(70, 4)
    assert record["frame_stride"] == 4 and record["prepared_fps"] == 7.5 and len(prepared) == len(expected) == 18
    assert prepared.source_frame_ids == [3 * index for index in expected]
    assert torch.equal(prepared.uint8_frame(0), stream.u8[expected[0]])


def test_unchanged_policy_keeps_the_stream() -> None:
    stream = FakeStream(5, hw=(26, 42))
    same, record = prepare_input_raster(stream, raster_config(policy="unchanged"), device=CPU)
    assert same is stream
    assert record == {
        "schema": "metrictrail_input_raster_v2",
        "policy": "unchanged",
        "source_hw": [26, 42],
        "prepared_hw": [26, 42],
        "frames": 5,
        "frame_stride": 1,
        "interpolation": None,
        "video_orientation": None,  # not a video file
    }


def test_refusals() -> None:
    with pytest.raises(ValueError, match="RGB-only"):
        prepare_input_raster(FakeStream(20, attributes={FrameAttribute.INTRINSICS}), raster_config(), device=CPU)
    with pytest.raises(ValueError, match="RGB-only"):
        prepare_input_raster(FakeStream(20, with_intrinsics=True), raster_config(policy="resize"), device=CPU)
    with pytest.raises(ValueError, match="not 8-bit"):
        prepare_input_raster(FakeStream(20, float_rgb=True), raster_config(), device=CPU)
    with pytest.raises(ValueError, match="at least 16"):
        prepare_input_raster(FakeStream(15), raster_config(), device=CPU)
    with pytest.raises(ValueError, match="unknown input raster policy"):
        prepare_input_raster(FakeStream(20), raster_config(policy="stretch"), device=CPU)
    # ViPE's bilinear resize is the only one: the ffmpeg and Pillow resamplers are gone.
    for name in ("decode_with_ffmpeg", "ffmpeg_available", "probe_video", "ffmpeg_source", "resize_cover_crop"):
        assert not hasattr(raster_module, name)


# ------------------------------------------------------------- video input


def _write_clip(path: Path, frames: int, hw: tuple[int, int]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (hw[1], hw[0]))
    if not writer.isOpened():
        pytest.skip("OpenCV has no mp4v encoder")
    rows, cols = np.meshgrid(np.arange(hw[0]), np.arange(hw[1]), indexing="ij")
    for index in range(frames):
        rgb = np.stack([(2 * rows + 4 * index) % 256, (cols + 8 * index) % 256, (rows + cols) % 256], -1)
        writer.write(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def test_video_is_decoded_by_vipes_raw_mp4_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)  # RawMp4Stream calls .cuda()
    clip = _write_clip(tmp_path / "clip.mp4", 40, (96, 160))
    stream = ProcessedVideoStream(RawMp4Stream(clip), [])  # what run.py hands the pipeline
    prepared, record = prepare_input_raster(stream, raster_config(height=28, width=52, max_frames=20), device=CPU)
    assert record["frame_stride"] == 2 and prepared.source_frame_ids[:3] == [0, 2, 4]
    plan = plan_raster((96, 160), 40, record["source_fps"], target_hw=(28, 52), max_frames=20)
    capture = cv2.VideoCapture(str(clip))
    decoded = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    for index, source in enumerate(prepared.source_frame_ids):
        np.testing.assert_array_equal(prepared.uint8_frame(index).numpy(), raster_uint8(decoded[source], plan))
    orientation = record["video_orientation"]  # a clip without a display rotation
    assert orientation["decoded_hw"] == orientation["stored_hw"] == [96, 160]
    assert orientation["orientation_meta_deg"] in (0.0, None)


def _rotated_clip(tmp_path: Path) -> Path:
    """A portrait phone clip: frames stored landscape (96 x 160) with a 90-degree display rotation."""

    stored = _write_clip(tmp_path / "stored.mp4", 20, (96, 160))
    if shutil.which("ffmpeg") is None:
        pytest.skip("writing a display rotation needs ffmpeg")
    clip = tmp_path / "portrait.mp4"
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(stored), "-c", "copy", "-metadata:s:v:0", "rotate=90", str(clip),
    ]  # fmt: skip
    if subprocess.run(command, capture_output=True).returncode != 0:
        pytest.skip("this ffmpeg cannot write a display rotation")
    return clip


def _decode(path: Path, *, auto_orientation: bool = True) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not auto_orientation:
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def test_rotated_phone_clip_follows_vipes_decoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ViPE's RawMp4Stream (cv2.VideoCapture at OpenCV's defaults) applies the display rotation and reports the
    # displayed raster; the input raster plans on what the stream delivers, so the portrait clip is cover-cropped
    # (never squashed), the record states what the decoder did, and the point-cloud export decodes the same frames.
    from vipe.metrictrail import pointcloud

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)  # RawMp4Stream calls .cuda()
    clip = _rotated_clip(tmp_path)
    if RawMp4Stream(clip).frame_size() != (160, 96):
        pytest.skip("this OpenCV does not apply the display rotation")
    stream = ProcessedVideoStream(RawMp4Stream(clip), [])  # what run.py hands the pipeline
    prepared, record = prepare_input_raster(stream, raster_config(), device=CPU)
    # Cover-fit of the portrait 160x96 raster to 280x504 (scale 5.25) and a 280-row centre crop; a squash would
    # have resized straight to 280x504.
    assert record["source_hw"] == [160, 96] and record["scale"] == 5.25
    assert record["resized_hw"] == [840, 504] and record["crop_left_top"] == [0, 280]
    orientation = record["video_orientation"]
    assert orientation["rotation_applied_by_decoder"] is True and orientation["orientation_meta_deg"] in (90.0, 270.0)
    assert orientation["decoded_hw"] == [160, 96] and orientation["stored_hw"] == [96, 160]
    assert orientation["decoder"].startswith("cv2.VideoCapture")
    decoded, stored = _decode(clip), _decode(clip, auto_orientation=False)
    assert len(decoded) == len(stored) == len(prepared) == 20
    # The decoded frames are the stored ones turned by a quarter turn, exactly.
    turns = [k for k in (1, 3) if np.array_equal(np.rot90(stored[0], k), decoded[0])]
    assert len(turns) == 1 and all(np.array_equal(np.rot90(a, turns[0]), b) for a, b in zip(stored, decoded))
    plan = plan_raster((160, 96), 20, record["source_fps"])
    for index in range(20):
        np.testing.assert_array_equal(prepared.uint8_frame(index).numpy(), raster_uint8(decoded[index], plan))
    rgb, _, _ = pointcloud.open_rgb_source(
        (280, 504), video=clip, input_raster=record, source_frame_ids=np.asarray(prepared.source_frame_ids)
    )
    for index in (0, 19):
        np.testing.assert_array_equal(rgb(index), prepared.uint8_frame(index).numpy())


def test_ten_bit_clip_is_taken_as_vipes_decoder_delivers_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenCV (RawMp4Stream) converts a 10-bit video to 8-bit BGR while decoding; the input raster takes those frames.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)  # RawMp4Stream calls .cuda()
    if shutil.which("ffmpeg") is None:
        pytest.skip("writing a 10-bit clip needs ffmpeg")
    clip = tmp_path / "clip10.mp4"
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-t", "0.6",
        "-pix_fmt", "yuv420p10le", "-c:v", "libx264", "-profile:v", "high10", str(clip),
    ]  # fmt: skip
    if subprocess.run(command, capture_output=True).returncode != 0:
        pytest.skip("this ffmpeg cannot write 10-bit H.264")
    prepared, record = prepare_input_raster(
        ProcessedVideoStream(RawMp4Stream(clip), []), raster_config(height=28, width=52), device=CPU
    )
    decoded = _decode(clip)
    assert len(prepared) == len(decoded) == 18 and decoded[0].dtype == np.uint8
    plan = plan_raster((180, 320), 18, record["source_fps"], target_hw=(28, 52))
    for index in (0, 9, 17):
        np.testing.assert_array_equal(prepared.uint8_frame(index).numpy(), raster_uint8(decoded[index], plan))


DEMO_VIDEO = Path(__file__).resolve().parents[2] / "assets" / "examples" / "metrictrail_demo.mp4"


@pytest.mark.skipif(not DEMO_VIDEO.is_file(), reason="assets/examples/metrictrail_demo.mp4 is not present")
@pytest.mark.parametrize(("policy", "hw"), [("cover_crop", (280, 504)), ("resize", (434, 574))])
def test_demo_video_frames_are_vipe_bilinear_resize_exactly(
    policy: str, hw: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real 1280x720 H.264 frames: each prepared frame is VideoFrame.resize of the frame RawMp4Stream decodes,
    # cropped and rounded, byte for byte; the point-cloud export's colours are the same bytes.
    from vipe.metrictrail import pointcloud

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)  # RawMp4Stream calls .cuda()
    head = range(0, 16, 1)
    stream = ProcessedVideoStream(RawMp4Stream(DEMO_VIDEO, seek_range=head), [])
    prepared, record = prepare_input_raster(stream, raster_config(policy=policy, height=hw[0], width=hw[1]), device=CPU)
    plan = plan_raster((720, 1280), len(head), record["source_fps"], target_hw=hw, policy=policy)
    assert (list(plan.resized_hw), list(plan.crop_left_top)) == (record["resized_hw"], record["crop_left_top"])
    left, top = plan.crop_left_top
    for index, frame in enumerate(RawMp4Stream(DEMO_VIDEO, seek_range=head)):
        expected = to_u8(frame.resize(plan.resized_hw).rgb[top : top + hw[0], left : left + hw[1]])
        np.testing.assert_array_equal(prepared.uint8_frame(index).numpy(), expected)
    rgb, _, rule = pointcloud.open_rgb_source(
        hw, video=DEMO_VIDEO, input_raster=record, source_frame_ids=np.asarray(prepared.source_frame_ids)
    )
    assert rule == f"ViPE bilinear {policy}"
    for index in (0, 7, 15):
        np.testing.assert_array_equal(rgb(index), prepared.uint8_frame(index).numpy())


def test_bicubic_antialias_resampler_and_its_record():
    """The in-the-wild default: torch bicubic with antialias, recorded as the record's interpolation."""

    import torch.nn.functional as F

    from vipe.metrictrail.raster import INTERPOLATIONS, raster_uint8, resampler_of_record

    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, size=(144, 256, 3), dtype=np.uint8)
    plan = plan_raster((144, 256), 30, 30.0, target_hw=(28, 50), resampler="bicubic_antialias")
    assert plan.resize and "resampler" not in plan.record()
    expected = (
        F.interpolate(
            torch.from_numpy(raw).float().div(255.0).permute(2, 0, 1)[None],
            size=plan.resized_hw,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
    )
    left, top = plan.crop_left_top
    expected = expected[top : top + 28, left : left + 50]
    expected_u8 = torch.round(expected * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    assert np.array_equal(raster_uint8(raw, plan), expected_u8)
    bilinear = plan_raster((144, 256), 30, 30.0, target_hw=(28, 50))
    assert not np.array_equal(raster_uint8(raw, bilinear), expected_u8)  # a different resize
    assert resampler_of_record({"interpolation": INTERPOLATIONS["bicubic_antialias"]}) == "bicubic_antialias"
    assert resampler_of_record({"interpolation": INTERPOLATIONS["bilinear"]}) == "bilinear"
    assert resampler_of_record({}) == "bilinear"  # records before the option: ViPE's bilinear
    with pytest.raises(ValueError, match="unknown resampler"):
        plan_raster((144, 256), 30, 30.0, resampler="lanczos")


def test_prepare_input_raster_records_the_configured_resampler():
    from vipe.metrictrail.raster import raster_uint8

    stream = FakeStream(20, hw=(72, 128))
    prepared, record = prepare_input_raster(stream, raster_config(resampler="bicubic_antialias"), device=CPU)
    assert record["interpolation"]["name"] == "torch_bicubic_antialias" and "resampler" not in record
    plan = plan_raster((72, 128), 20, 30.0, resampler="bicubic_antialias")
    assert np.array_equal(prepared.uint8_frame(3).cpu().numpy(), raster_uint8(stream.u8[3].numpy(), plan))
