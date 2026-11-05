# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-cloud export (``vipe.metrictrail.pointcloud``) on synthetic scenes, CPU only."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from metrictrail_fakes import FakeStream

from vipe.metrictrail import pointcloud as pc
from vipe.metrictrail.raster import plan_raster, prepare_input_raster, raster_uint8
from vipe.streams.base import ProcessedVideoStream, VideoFrame, VideoStream

PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
CPU = torch.device("cpu")
# A source raster that the in-the-wild rule both scales (0.325, to 32x52) and crops (2 rows off the top).
SOURCE_HW, RASTER_HW = (100, 160), (28, 52)


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    rotation, _ = cv2.Rodrigues(vector / np.linalg.norm(vector) * angle)
    return rotation


def _pixel_coded_rgb(height: int, width: int) -> np.ndarray:
    """Colour (row, col, 7) at each pixel, so a fused point names the pixel it came from."""

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack([rows, cols, np.full_like(rows, 7)], axis=-1).astype(np.uint8)


def _plane_scene(frames: int, height: int, width: int, *, depth_m: float = 3.0, step_m: float = 0.05):
    """A fronto-parallel plane at world z = depth_m, seen by a camera translating along +x."""

    k4 = np.tile([50.0, 50.0, (width - 1) / 2.0, (height - 1) / 2.0], (frames, 1))
    c2w = np.tile(np.eye(4), (frames, 1, 1))
    c2w[:, 0, 3] = step_m * np.arange(frames)
    depth = np.full((frames, height, width), depth_m, dtype=np.float32)
    return depth, k4, c2w


def _no_filter(**overrides) -> pc.FusionSettings:
    values = dict(pixel_stride=1, voxel_size_m=1e-3, min_depth_m=1e-3, max_depth_m=1e3, min_consistent=0)
    values.update(overrides)
    return pc.FusionSettings(**values)


def _read_ply(path: Path) -> tuple[bytes, np.ndarray]:
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    return data[:end], np.frombuffer(data[end:], dtype=PLY_DTYPE)


def _raster_config(**overrides) -> SimpleNamespace:
    values = {"policy": "cover_crop", "height": RASTER_HW[0], "width": RASTER_HW[1], "max_frames": 20}
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_frames(directory: Path, frames) -> Path:
    directory.mkdir()
    for index, frame in enumerate(frames):
        cv2.imwrite(str(directory / f"{index:05d}.png"), cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    return directory


def _textured_frame(index: int, hw: tuple[int, int]) -> np.ndarray:
    """Smooth enough for a lossy codec, and different for every frame index."""

    rows, cols = np.meshgrid(np.arange(hw[0]), np.arange(hw[1]), indexing="ij")
    return np.stack([(2 * rows + 4 * index) % 256, (cols + 8 * index) % 256, (rows + cols) % 256], -1).astype(np.uint8)


def _write_clip(path: Path, frames: int, hw: tuple[int, int]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (hw[1], hw[0]))
    if not writer.isOpened():
        pytest.skip("OpenCV has no mp4v encoder")
    for index in range(frames):
        writer.write(cv2.cvtColor(_textured_frame(index, hw), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


class _OpenCvVideo(VideoStream):
    """ViPE's video-stream decode (OpenCV, uint8 / 255) on the CPU, through the point-cloud reader."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.reader = pc.VideoReader(path)

    def frame_size(self) -> tuple[int, int]:
        return SOURCE_HW

    def fps(self) -> float:
        return 30.0

    def name(self) -> str:
        return "clip"

    def __len__(self) -> int:
        return len(self.reader)

    def attributes(self):
        return set()

    def __iter__(self):
        for index in range(len(self.reader)):
            yield VideoFrame(raw_frame_idx=index, rgb=torch.from_numpy(self.reader(index)).float() / 255.0)


def test_defaults_are_the_visualization_setting():
    assert pc.FusionSettings() == pc.FusionSettings(
        frame_stride=1,
        pixel_stride=2,
        voxel_size_m=0.02,
        min_depth_m=0.2,
        max_depth_m=15.0,
        max_input_points=8_000_000,
        min_consistent=2,
        consistency_window=5,
        consistency_max_views=6,
        pixel_threshold=1.0,
        relative_depth_threshold=0.05,
    )
    parse = pc.build_parser().parse_args
    args = parse(["out", "--name", "n", "--frames", "f"])
    assert args.filter == "default" and pc.settings_from_args(args) == pc.FusionSettings()
    assert args.world_frame is False  # the z-up frame unless --world-frame


def test_strict_filter_preset_and_overrides():
    strict = pc.FusionSettings(
        min_consistent=4,
        consistency_window=9,
        consistency_max_views=10,
        pixel_threshold=0.5,
        relative_depth_threshold=0.02,
    )
    assert pc.fusion_settings("strict") == strict
    assert (
        pc.settings_from_args(pc.build_parser().parse_args(["o", "--name", "n", "--frames", "f", "--filter", "strict"]))
        == strict
    )
    # an explicit option wins over the preset, for the filter and for every other setting
    args = pc.build_parser().parse_args(
        ["o", "--name", "n", "--frames", "f", "--filter", "strict", "--min-consistent", "3", "--max-depth", "100",
         "--frame-stride", "12", "--voxel-size", "0.05"]
    )  # fmt: skip
    assert pc.settings_from_args(args) == dataclasses.replace(
        strict, min_consistent=3, max_depth_m=100.0, frame_stride=12, voxel_size_m=0.05
    )
    args = pc.build_parser().parse_args(["o", "--name", "n", "--frames", "f", "--max-depth", "100"])
    assert pc.settings_from_args(args) == pc.FusionSettings(max_depth_m=100.0)
    with pytest.raises(ValueError, match="unknown filter"):
        pc.fusion_settings("other")
    with pytest.raises(SystemExit):
        pc.build_parser().parse_args(["o", "--name", "n", "--frames", "f", "--filter", "other"])


def test_unprojection_uses_pinhole_rays_and_c2w():
    height, width = 15, 20
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    depth = (1.0 + 0.1 * rows + 0.03 * cols).astype(np.float32)
    k4 = np.array([40.0, 55.0, 9.5, 7.0])
    c2w = np.eye(4)
    c2w[:3, :3] = _rotation((0.3, -1.0, 0.2), 0.7)
    c2w[:3, 3] = [0.5, -1.2, 2.0]
    rgb = _pixel_coded_rgb(height, width)

    cloud = pc.fuse_point_cloud(depth[None], k4[None], c2w[None], lambda _: rgb, _no_filter(), world_frame=True)

    assert len(cloud.points) == height * width
    row = cloud.colors[:, 0].astype(np.float64)
    col = cloud.colors[:, 1].astype(np.float64)
    z = depth[cloud.colors[:, 0], cloud.colors[:, 1]].astype(np.float64)
    camera = np.stack([(col - k4[2]) * z / k4[0], (row - k4[3]) * z / k4[1], z], axis=1)
    np.testing.assert_allclose(cloud.points, camera @ c2w[:3, :3].T + c2w[:3, 3], atol=1e-5)

    local, out_rows, out_cols = pc.backproject_metric_depth(
        depth, np.array([[40.0, 0, 9.5], [0, 55.0, 7.0], [0, 0, 1]]), np.ones_like(depth, bool), pixel_stride=1
    )
    np.testing.assert_allclose(local[:, 2], depth[out_rows, out_cols])
    np.testing.assert_allclose(local[:, 0], (out_cols - 9.5) * depth[out_rows, out_cols] / 40.0, rtol=1e-6)


def test_default_frame_is_the_original_z_up_frame():
    depth, k4, c2w = _plane_scene(4, 20, 30)
    tilt = _rotation((1.0, 0.2, 0.0), 0.3)  # cameras pitched, so the frame's up is not a world axis
    c2w[:, :3, :3] = tilt
    rgb = _pixel_coded_rgb(20, 30)
    world = pc.fuse_point_cloud(depth, k4, c2w, lambda _: rgb, _no_filter(), world_frame=True)
    z_up = pc.fuse_point_cloud(depth, k4, c2w, lambda _: rgb, _no_filter())

    # The original visualization's rotation: the mean camera up (-Y of the OpenCV poses) goes to +Z.
    rotation = np.asarray(z_up.stats["world_to_output_rotation"])
    np.testing.assert_allclose(rotation @ -tilt[:, 1], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation, pc.up_to_z_rotation(pc.up_from_poses(c2w)))
    np.testing.assert_allclose(z_up.points, world.points @ rotation.T, atol=1e-5)
    np.testing.assert_array_equal(z_up.colors, world.colors)
    assert world.stats["world_to_output_rotation"] == np.eye(3).tolist()


def test_pixel_stride_samples_every_nth_row_and_column():
    height, width, stride = 15, 20, 3
    depth = np.full((1, height, width), 2.0, dtype=np.float32)
    k4 = np.array([[30.0, 30.0, 10.0, 7.0]])
    cloud = pc.fuse_point_cloud(
        depth, k4, np.eye(4)[None], lambda _: _pixel_coded_rgb(height, width), _no_filter(pixel_stride=stride)
    )

    pixels = {(int(r), int(c)) for r, c, _ in cloud.colors}
    assert pixels == {(r, c) for r in range(0, height, stride) for c in range(0, width, stride)}


def test_depth_range_is_inclusive_and_drops_invalid_depth():
    values = [0.1, 0.2, 5.0, 15.0, 15.01, np.nan, 0.0, np.inf, -3.0]
    depth = np.asarray(values, dtype=np.float32)[None, None, :]
    k4 = np.array([[10.0, 10.0, 4.0, 0.0]])
    settings = pc.FusionSettings(pixel_stride=1, voxel_size_m=1e-3, min_consistent=0)  # default 0.2-15 m
    cloud = pc.fuse_point_cloud(
        depth, k4, np.eye(4)[None], lambda _: _pixel_coded_rgb(1, len(values)), settings, world_frame=True
    )

    assert sorted(int(c) for c in cloud.colors[:, 1]) == [1, 2, 3]
    np.testing.assert_allclose(np.sort(cloud.points[:, 2]), [0.2, 5.0, 15.0], rtol=1e-6)


def test_consistency_filter_keeps_consistent_points_and_rejects_a_flying_point():
    frames, height, width = 15, 40, 60
    depth, k4, c2w = _plane_scene(frames, height, width)
    source, flying = 7, (20, 30)
    depth[source][flying] = 2.0  # a stray depth between camera and plane, seen by this frame only
    rows, cols = np.meshgrid(np.arange(5, 35), np.arange(10, 50), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    defaults = pc.FusionSettings()

    keep = pc.multiview_consistency_mask(
        rows,
        cols,
        depth[source][rows, cols],
        source,
        depth=depth,
        intrinsics=k4,
        c2w=c2w,
        window=defaults.consistency_window,
        max_views=defaults.consistency_max_views,
        min_consistent=defaults.min_consistent,
        pixel_threshold=defaults.pixel_threshold,
        relative_depth_threshold=defaults.relative_depth_threshold,
    )
    is_flying = (rows == flying[0]) & (cols == flying[1])
    assert not keep[is_flying].any()
    assert keep[~is_flying].all()

    # Through the fusion: without the filter the stray point is in the cloud, with the defaults it is not.
    rgb = _pixel_coded_rgb(height, width)
    unfiltered = pc.fuse_point_cloud(
        depth, k4, c2w, lambda _: rgb, pc.FusionSettings(min_consistent=0), world_frame=True
    )
    filtered = pc.fuse_point_cloud(depth, k4, c2w, lambda _: rgb, pc.FusionSettings(), world_frame=True)
    assert (unfiltered.points[:, 2] < 2.5).sum() == 1
    assert (filtered.points[:, 2] < 2.5).sum() == 0
    assert filtered.stats["points_rejected_by_filters"] > 0


def test_consistency_filter_needs_min_consistent_agreeing_views():
    depth, k4, c2w = _plane_scene(3, 20, 30)  # frame 1 has exactly two neighbours
    rows, cols = np.array([10]), np.array([15])
    kwargs = dict(depth=depth, intrinsics=k4, c2w=c2w, window=7, max_views=8)
    assert pc.multiview_consistency_mask(rows, cols, depth[1][rows, cols], 1, min_consistent=2, **kwargs).all()
    assert not pc.multiview_consistency_mask(rows, cols, depth[1][rows, cols], 1, min_consistent=3, **kwargs).any()
    assert pc.neighbour_indices(10, 20, window=7, max_views=8) == [9, 11, 8, 12, 7, 13, 6, 14]
    assert pc.neighbour_indices(0, 20, window=7, max_views=8) == [1, 2, 3, 4, 5, 6, 7]


def test_voxel_downsampling_averages_position_and_colour():
    accumulator = pc.VoxelAccumulator(0.1)
    accumulator.add(
        np.array([[0.01, 0.01, 0.01], [0.09, 0.05, 0.03], [-0.01, 0.0, 0.0], [0.15, 0.0, 0.0]]),
        np.array([[10, 20, 30], [20, 40, 60], [1, 2, 3], [4, 5, 6]]),
    )
    for _ in range(40):  # more adds than compact_every, so pending batches are merged into existing voxels
        accumulator.add(np.array([[0.05, 0.05, 0.02]]), np.array([[15, 30, 45]]))
    points, colors, counts = accumulator.finish()

    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points, colors, counts = points[order], colors[order], counts[order]
    np.testing.assert_allclose(points[0], [-0.01, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(
        points[1],
        (np.array([0.01, 0.01, 0.01]) + [0.09, 0.05, 0.03] + 40 * np.array([0.05, 0.05, 0.02])) / 42,
        atol=1e-6,
    )
    np.testing.assert_allclose(points[2], [0.15, 0.0, 0.0], atol=1e-7)
    np.testing.assert_array_equal(colors, [[1, 2, 3], [15, 30, 45], [4, 5, 6]])
    np.testing.assert_array_equal(counts, [1, 42, 1])

    # Two frames seeing the same surface fuse into the same voxels as one frame does.
    depth, k4, c2w = _plane_scene(2, 20, 30, step_m=0.0)
    rgb = np.full((20, 30, 3), 90, np.uint8)
    one = pc.fuse_point_cloud(depth[:1], k4[:1], c2w[:1], lambda _: rgb, _no_filter(voxel_size_m=0.2))
    two = pc.fuse_point_cloud(depth, k4, c2w, lambda _: rgb, _no_filter(voxel_size_m=0.2))
    assert len(two.points) == len(one.points) < 20 * 30
    assert two.stats["observations_per_voxel_mean"] == pytest.approx(2 * one.stats["observations_per_voxel_mean"])


def test_ply_header_and_bytes(tmp_path):
    points = np.array([[1.0, -2.5, 3.25], [0.0, 1e-3, -7.0]], dtype=np.float32)
    colors = np.array([[255, 0, 12], [1, 2, 3]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"
    pc.write_binary_point_cloud(path, points, colors)

    header, records = _read_ply(path)
    assert header == (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    assert path.stat().st_size == len(header) + 2 * 15
    body = path.read_bytes()[len(header) :]
    assert body[:15] == points[0].astype("<f4").tobytes() + colors[0].tobytes()
    np.testing.assert_array_equal(np.stack([records["x"], records["y"], records["z"]], 1), points)
    np.testing.assert_array_equal(np.stack([records["red"], records["green"], records["blue"]], 1), colors)
    with pytest.raises(ValueError):
        pc.write_binary_point_cloud(path, points, colors[:1])


# ------------------------------------------------------------------ colours


def test_frames_without_a_cover_crop_record_are_used_as_given(tmp_path):
    frames = [_textured_frame(index, RASTER_HW) for index in range(3)]
    on_raster = pc.FrameDirectoryReader(_write_frames(tmp_path / "on", frames))
    rgb, input_frames, rule = pc.open_rgb_source(RASTER_HW, frames=tmp_path / "on")
    assert (input_frames, rule) == (3, "as given")
    for index in range(3):
        np.testing.assert_array_equal(rgb(index), frames[index])
        np.testing.assert_array_equal(rgb(index), on_raster(index))

    # Off the depth raster: the one read-time resize, ViPE's bilinear cover-crop onto it (raster.py).
    larger = [_textured_frame(index, SOURCE_HW) for index in range(3)]
    _write_frames(tmp_path / "off", larger)
    unchanged = {"policy": "unchanged", "source_hw": list(SOURCE_HW), "frame_stride": 1}
    rgb, _, rule = pc.open_rgb_source(RASTER_HW, frames=tmp_path / "off", input_raster=unchanged)
    assert rule == "as given"
    expected = raster_uint8(larger[2], plan_raster(SOURCE_HW, 1, 1.0, target_hw=RASTER_HW, max_frames=1))
    np.testing.assert_array_equal(rgb(2), expected)
    assert not np.array_equal(
        rgb(2), cv2.resize(larger[2], (RASTER_HW[1], RASTER_HW[0]), interpolation=cv2.INTER_LINEAR)
    )


def test_rgb_source_follows_source_frame_ids(tmp_path):
    for index in range(6):
        cv2.imwrite(str(tmp_path / f"{index:06d}.png"), np.full((8, 12, 3), 40 * index, np.uint8))
    reader = pc.FrameDirectoryReader(tmp_path)
    assert len(reader) == 6
    rgb = pc.make_rgb_source(reader, (4, 6), source_frame_ids=np.array([0, 2, 4]))
    assert rgb(1).shape == (4, 6, 3)
    assert (rgb(1) == 80).all() and (rgb(2) == 160).all()
    assert (pc.make_rgb_source(reader, (4, 6))(1) == 40).all()
    _, input_frames, _ = pc.open_rgb_source((4, 6), frames=tmp_path, source_frame_ids=np.array([0, 2, 4]))
    assert input_frames is None  # the ids say which frames were tracked, not the frame count


def test_cover_crop_record_gives_the_tracked_frames_of_a_frame_directory(tmp_path):
    stream = FakeStream(60, hw=SOURCE_HW)  # 60 frames, max_frames 20: one frame in 3 (1, 4, 7, ...)
    prepared, record = prepare_input_raster(stream, _raster_config(), device=CPU)
    assert record["interpolation"]["name"] == "vipe_bilinear" and record["frame_stride"] == 3
    assert record["crop_left_top"] == [0, 2]
    _write_frames(tmp_path / "frames", [frame.numpy() for frame in stream.u8])

    plan = pc.raster_plan(record)
    assert plan is not None and plan.record() == {key: record[key] for key in plan.record()}
    rgb, input_frames, rule = pc.open_rgb_source(
        RASTER_HW,
        frames=tmp_path / "frames",
        input_raster=record,
        source_frame_ids=np.asarray(prepared.source_frame_ids),
    )
    assert (input_frames, rule) == (None, "ViPE bilinear cover_crop")
    for index in range(len(prepared)):
        np.testing.assert_array_equal(rgb(index), prepared.uint8_frame(index).numpy())

    # A run that tracked one prepared frame in 5: its source_frame_ids name the tracked frames.
    tracked = np.asarray(prepared.source_frame_ids[::5])
    rgb, _, _ = pc.open_rgb_source(RASTER_HW, frames=tmp_path / "frames", input_raster=record, source_frame_ids=tracked)
    for index in range(len(tracked)):
        np.testing.assert_array_equal(rgb(index), prepared.uint8_frame(5 * index).numpy())


def test_video_colours_are_the_frames_the_pipeline_tracked(tmp_path):
    clip = _write_clip(tmp_path / "clip.mp4", 60, SOURCE_HW)
    prepared, record = prepare_input_raster(_OpenCvVideo(clip), _raster_config(), device=CPU)
    assert record["frame_stride"] == 3 and prepared.source_frame_ids[:3] == [1, 4, 7]

    rgb, _, rule = pc.open_rgb_source(
        RASTER_HW, video=clip, input_raster=record, source_frame_ids=np.asarray(prepared.source_frame_ids)
    )
    assert rule == "ViPE bilinear cover_crop"
    for index in range(len(prepared)):
        np.testing.assert_array_equal(rgb(index), prepared.uint8_frame(index).numpy())


def test_video_colours_are_the_frames_raw_mp4_stream_gave_the_pipeline(tmp_path, monkeypatch):
    from vipe.streams.raw_mp4_stream import RawMp4Stream

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)  # RawMp4Stream calls .cuda()
    clip = _write_clip(tmp_path / "clip.mp4", 60, SOURCE_HW)
    stream = ProcessedVideoStream(RawMp4Stream(clip), [])  # what run.py hands the pipeline
    prepared, record = prepare_input_raster(stream, _raster_config(), device=CPU)
    assert record["interpolation"]["name"] == "vipe_bilinear" and record["frame_stride"] == 3
    ids = np.asarray(prepared.source_frame_ids)

    rgb, input_frames, rule = pc.open_rgb_source(RASTER_HW, video=clip, input_raster=record, source_frame_ids=ids)
    assert input_frames is None and rule == "ViPE bilinear cover_crop"
    for index in range(len(prepared)):
        np.testing.assert_array_equal(rgb(index), prepared.uint8_frame(index).numpy())
    np.testing.assert_array_equal(rgb(0), prepared.uint8_frame(0).numpy())  # read again from the start

    # A run that tracked one prepared frame in 3 (tracking_stride): its source_frame_ids name those frames.
    tracked = ids[::3]
    strided, _, _ = pc.open_rgb_source(RASTER_HW, video=clip, input_raster=record, source_frame_ids=tracked)
    for index in range(len(tracked)):
        np.testing.assert_array_equal(strided(index), prepared.uint8_frame(3 * index).numpy())


def test_resize_record_colours_follow_the_plain_resize(tmp_path):
    stream = FakeStream(20, hw=SOURCE_HW)
    prepared, record = prepare_input_raster(stream, _raster_config(policy="resize"), device=CPU)
    assert (
        record["policy"] == "resize" and record["crop_left_top"] == [0, 0] and record["resized_hw"] == list(RASTER_HW)
    )
    _write_frames(tmp_path / "frames", [frame.numpy() for frame in stream.u8])
    plan = pc.raster_plan(record)
    assert plan is not None and plan.policy == "resize"
    tracked = np.asarray(prepared.source_frame_ids[::5])  # tracking_stride 5
    rgb, input_frames, rule = pc.open_rgb_source(
        RASTER_HW, frames=tmp_path / "frames", input_raster=record, source_frame_ids=tracked
    )
    assert (input_frames, rule) == (None, "ViPE bilinear resize")
    for index in range(len(tracked)):
        np.testing.assert_array_equal(rgb(index), prepared.uint8_frame(5 * index).numpy())
    # Not the cover-crop of the same frames.
    cover = raster_uint8(stream.u8[0].numpy(), plan_raster(SOURCE_HW, 20, 30.0, target_hw=RASTER_HW))
    assert not np.array_equal(rgb(0), cover)


def test_a_record_that_disagrees_with_the_raster_arithmetic_is_refused():
    _, record = prepare_input_raster(FakeStream(20, hw=SOURCE_HW), _raster_config(), device=CPU)
    assert pc.raster_plan({**record, "policy": "unchanged"}) is None
    with pytest.raises(ValueError, match="does not match"):
        pc.raster_plan({**record, "crop_left_top": [0, 0]})


def test_a_record_of_an_earlier_raster_schema_warns(caplog):
    _, record = prepare_input_raster(FakeStream(20, hw=SOURCE_HW), _raster_config(), device=CPU)
    with caplog.at_level("WARNING", logger=pc.logger.name):
        assert pc.raster_plan(record) is not None
    assert not caplog.records
    # A run prepared before ViPE's bilinear resize (schema v1, another resampler): same plan, a warning.
    earlier = {**record, "schema": "metrictrail_input_raster_v1", "resampler": "pil"}
    with caplog.at_level("WARNING", logger=pc.logger.name):
        assert pc.raster_plan(earlier) == pc.raster_plan(record)
    assert any("metrictrail_input_raster_v1" in item.getMessage() for item in caplog.records)


def test_video_reader_seeks_forward_and_back(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV has no mp4v encoder")
    for index in range(6):
        writer.write(np.full((48, 64, 3), 40 * index, np.uint8))
    writer.release()

    reader = pc.VideoReader(path)
    assert len(reader) == 6
    assert abs(float(reader(3).mean()) - 120) < 4
    assert abs(float(reader(4).mean()) - 160) < 4
    assert abs(float(reader(1).mean()) - 40) < 4  # earlier frame: decode again from the start
    with pytest.raises(IndexError):
        reader(6)


# ------------------------------------------------------------------ ViPE artifacts and CLI


class _Pose:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = torch.from_numpy(matrix)

    def matrix(self) -> torch.Tensor:
        return self._matrix


class _AttributeStream:
    """Just what ViPE's artifact writers read from a stream."""

    def __init__(self, attributes: dict) -> None:
        self._attributes = attributes

    def get_stream_attribute(self, attribute):
        return self._attributes[attribute]


def _save_run(base: Path, depth, k4, c2w, *, no_pose=(), no_depth=(), source_frame_ids=None, input_raster=None):
    """A ViPE output directory written by ViPE's own artifact writers (plus MetricTrail's extras, if given)."""

    from vipe.streams.base import FrameAttribute
    from vipe.utils import io
    from vipe.utils.cameras import CameraType

    frames = len(depth)
    stream = _AttributeStream(
        {
            FrameAttribute.POSE: [None if i in no_pose else _Pose(c2w[i]) for i in range(frames)],
            FrameAttribute.METRIC_DEPTH: [None if i in no_depth else torch.from_numpy(depth[i]) for i in range(frames)],
            FrameAttribute.INTRINSICS: [torch.from_numpy(row.astype(np.float32)) for row in k4],
            FrameAttribute.CAMERA_TYPE: [CameraType.PINHOLE] * frames,
        }
    )
    run = io.ArtifactPath(base, "clip")
    io.save_pose_artifacts(run, stream)
    io.save_intrinsics_artifacts(run, stream)
    io.save_depth_artifacts(run, stream)
    if source_frame_ids is not None:
        with np.load(run.pose_path) as archive:
            fields = dict(archive)
        np.savez(run.pose_path, **fields, source_frame_ids=np.asarray(source_frame_ids, dtype=np.int64))
    if input_raster is not None:
        (base / "vipe").mkdir(parents=True, exist_ok=True)
        (base / "vipe" / "clip_metrictrail.json").write_text(json.dumps({"input_raster": input_raster}))
    return run


def test_round_trip_through_vipe_io(tmp_path, capsys):
    from vipe.utils import io
    from vipe.utils.cameras import CameraType

    frames, height, width = 12, 28, 50
    depth, k4, c2w = _plane_scene(frames, height, width, depth_m=3.01)  # 3.01 m is not a half-precision value
    stored_m = float(np.float16(3.01))
    assert stored_m != np.float32(3.01)
    run = _save_run(tmp_path / "out", depth, k4, c2w, no_pose=(1,), no_depth=(2,))  # ViPE writes no row for these

    loaded = pc.load_vipe_artifacts(run.base_path, "clip")
    vipe_depth = dict(io.read_depth_artifacts(run.depth_path))
    assert sorted(vipe_depth) == [i for i in range(frames) if i != 2]
    for index, frame in vipe_depth.items():
        np.testing.assert_array_equal(loaded.depth[index], frame.numpy())
        np.testing.assert_array_equal(loaded.depth[index], depth[index].astype(np.float16).astype(np.float32))
    assert np.isnan(loaded.depth[2]).all()
    inds, vipe_k4, camera_types = io.read_intrinsics_artifacts(run.intrinsics_path, run.camera_type_path)
    np.testing.assert_array_equal(loaded.intrinsics[inds], vipe_k4.double().numpy())
    assert camera_types == [CameraType.PINHOLE] * frames
    np.testing.assert_array_equal(loaded.c2w[[i for i in range(frames) if i != 1]], np.delete(c2w, 1, axis=0))
    assert np.isnan(loaded.c2w[1]).all()
    assert loaded.source_frame_ids is None and loaded.input_raster is None

    # The CLI on the same artifacts. No run record: the frames are used as given and, being larger
    # than the depth raster, cover-cropped onto it.
    frame_dir = _write_frames(tmp_path / "frames", [np.full((60, 100, 3), (200, 120, 10), np.uint8)] * frames)
    ply = tmp_path / "cloud.ply"
    summary = pc.main([str(run.base_path), "--name", "clip", "--frames", str(frame_dir), "--output", str(ply)])
    assert summary["input_hw"] == [height, width]
    assert summary["frames"] == frames
    assert (summary["frame"], summary["rgb"]) == ("z_up", "as given")
    header, records = _read_ply(ply)
    assert f"element vertex {len(records)}\n".encode() in header
    assert summary["voxels"] == len(records) > 0
    # Level cameras: the camera's -Y (world -y) becomes +Z, so the plane at world z = 3.01 m lies at y = 3.01 m.
    np.testing.assert_allclose(records["y"], stored_m, atol=1e-6)  # the plane, at the depth ViPE stored
    assert (records["red"] == 200).all() and (records["green"] == 120).all() and (records["blue"] == 10).all()
    assert '"voxels"' in capsys.readouterr().out

    summary = pc.main([str(run.base_path), "--name", "clip", "--frames", str(frame_dir), "--world-frame"])
    assert summary["frame"] == "vipe_world" and summary["output"] == str(run.base_path / "pointcloud" / "clip.ply")
    _, records = _read_ply(run.base_path / "pointcloud" / "clip.ply")
    np.testing.assert_allclose(records["z"], stored_m, atol=1e-6)


def test_cli_colours_with_the_run_record(tmp_path, monkeypatch):
    stream = FakeStream(60, hw=SOURCE_HW)
    prepared, record = prepare_input_raster(stream, _raster_config(), device=CPU)
    depth, k4, c2w = _plane_scene(len(prepared), *RASTER_HW)
    run = _save_run(
        tmp_path / "out",
        depth,
        k4,
        c2w,
        source_frame_ids=prepared.source_frame_ids,
        input_raster=json.loads(json.dumps(record)),
    )
    frame_dir = _write_frames(tmp_path / "frames", [frame.numpy() for frame in stream.u8])
    seen = {}

    def fake_fusion(depth, intrinsics, c2w, rgb, settings, *, world_frame, progress):
        seen.update(frames=len(depth), images=[rgb(index) for index in range(len(depth))], world_frame=world_frame)
        return pc.FusedCloud(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), {"frames": len(depth)})

    monkeypatch.setattr(pc, "fuse_point_cloud", fake_fusion)
    summary = pc.main([str(run.base_path), "--name", "clip", "--frames", str(frame_dir)])
    assert summary["rgb"] == "ViPE bilinear cover_crop" and seen["world_frame"] is False
    assert seen["frames"] == len(prepared) == 20
    for index, image in enumerate(seen["images"]):
        np.testing.assert_array_equal(image, prepared.uint8_frame(index).numpy())
