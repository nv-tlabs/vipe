# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in export of a fused, coloured point cloud from a ViPE output directory.

    python -m vipe.metrictrail.pointcloud vipe_results/ --name clip --video clip.mp4

Reads ``pose/<name>.npz`` (OpenCV camera-to-world), ``intrinsics/<name>.npz``
(``fx, fy, cx, cy``), ``depth/<name>.zip`` (ViPE's zipped half-float EXR
metric depth) and, if the run wrote it, the run record
``vipe/<name>_metrictrail.json``; takes RGB from the input video or frame
directory, and writes one binary little-endian PLY (``x y z`` float32,
``red green blue`` uint8). Nothing in ViPE calls this module; it only reads
finished artifacts.

Per fused frame: back-project every ``pixel_stride``-th pixel whose depth lies
in ``[min_depth_m, max_depth_m]``, cap the frame at an even share of
``max_input_points``, keep a pixel only if at least ``min_consistent`` nearby
frames reproduce it (the multi-view consistency filter that removes flying
points at depth discontinuities), move it to world with the frame's pose, and
average everything into ``voxel_size_m`` voxels (position and colour).

The colours come from the frames tracking used (:func:`open_rgb_source`). When
the run record's ``input_raster`` says the pipeline resized its input
(``cover_crop`` or ``resize``), each input frame goes through
:mod:`vipe.metrictrail.raster` with the recorded plan (the recorded resampler, then
the integer crop), and artifact frame ``i`` is input frame ``source_frame_ids[i]``.
Otherwise the frames are used as given. A frame that is still not on the depth
raster gets ViPE's bilinear resize and centre crop onto it (``cover_crop``
geometry), the one read-time resize of :mod:`vipe.metrictrail`.

The cloud is written in the original visualization's z-up frame: ViPE's world
frame rotated so that the mean camera up is +Z. ``world_frame`` keeps ViPE's
world frame, the frame of ``pose/<name>.npz``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .raster import (
    LEGACY_RESAMPLER,
    RASTER_SCHEMA,
    RESIZE_POLICIES,
    RasterPlan,
    plan_raster,
    raster_uint8,
    resampler_of_record,
)

logger = logging.getLogger(__name__)

FrameSource = Callable[[int], np.ndarray]
ProgressCallback = Callable[[int, int], None]

# ViPE's frame-directory stream lists these, so a frame index here is ViPE's ``raw_frame_idx``.
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


@dataclass(frozen=True)
class FusionSettings:
    """Fusion parameters. Every field can be overridden on its own."""

    frame_stride: int = 1
    pixel_stride: int = 2
    voxel_size_m: float = 0.02
    min_depth_m: float = 0.2
    max_depth_m: float = 15.0
    max_input_points: int = 8_000_000
    min_consistent: int = 2
    consistency_window: int = 5
    consistency_max_views: int = 6
    pixel_threshold: float = 1.0
    relative_depth_threshold: float = 0.05


#: Consistency-filter presets (``--filter``): ``default`` is :class:`FusionSettings`' own filter, the one of
#: the original MetricTrail visualization of in-the-wild clips; ``strict`` is the stronger filter its figure
#: of the demo clip used, which removes more flying points along depth discontinuities and keeps less of a
#: fast-moving or distant scene. Explicit settings override the preset.
FILTER_PRESETS: dict[str, dict[str, int | float]] = {
    "default": {},
    "strict": {
        "min_consistent": 4,
        "consistency_window": 9,
        "consistency_max_views": 10,
        "pixel_threshold": 0.5,
        "relative_depth_threshold": 0.02,
    },
}


def fusion_settings(filter: str = "default", **overrides: int | float) -> FusionSettings:
    """:class:`FusionSettings` of a ``FILTER_PRESETS`` entry, with ``overrides`` applied on top."""

    if filter not in FILTER_PRESETS:
        raise ValueError(f"unknown filter preset {filter!r}; choose one of {sorted(FILTER_PRESETS)}")
    values: dict[str, Any] = {**FILTER_PRESETS[filter], **overrides}
    return FusionSettings(**values)


@dataclass
class FusedCloud:
    points: np.ndarray  # [M,3] float32
    colors: np.ndarray  # [M,3] uint8
    stats: dict


# ----------------------------------------------------------------- back-projection


def _finite_matrix(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {matrix.shape}")
    return matrix


def backproject_metric_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project Z-depth through calibrated pinhole rays.

    Returns camera-space points and their integer row/column indices, so callers
    can sample RGB without duplicating coordinate logic.
    """

    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    intrinsics = _finite_matrix(intrinsics, (3, 3), "intrinsics")
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(f"depth/mask shapes are incompatible: {depth.shape}, {mask.shape}")
    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    rows = np.arange(0, depth.shape[0], pixel_stride, dtype=np.int64)
    cols = np.arange(0, depth.shape[1], pixel_stride, dtype=np.int64)
    grid_col, grid_row = np.meshgrid(cols, rows)
    sampled_depth = depth[grid_row, grid_col]
    valid = mask[grid_row, grid_col] & np.isfinite(sampled_depth) & (sampled_depth > 0)
    row = grid_row[valid]
    col = grid_col[valid]
    z = sampled_depth[valid].astype(np.float64, copy=False)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (col.astype(np.float64) - cx) * z / fx
    y = (row.astype(np.float64) - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float32), row, col


# ------------------------------------------------------- multi-view consistency


def _unproject(uv: np.ndarray, depth: np.ndarray, k4: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = k4
    return np.stack([(uv[:, 0] - cx) * depth / fx, (uv[:, 1] - cy) * depth / fy, depth], axis=1)


def _project(points_camera: np.ndarray, k4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = k4
    z = points_camera[:, 2]
    safe = np.where(np.abs(z) > 1e-8, z, 1e-8)
    uv = np.stack([points_camera[:, 0] / safe * fx + cx, points_camera[:, 1] / safe * fy + cy], axis=1)
    return uv, z


def _to_camera(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return (points_world - c2w[:3, 3]) @ c2w[:3, :3]


def _to_world(points_camera: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return points_camera @ c2w[:3, :3].T + c2w[:3, 3]


def neighbour_indices(index: int, count: int, window: int, max_views: int) -> list[int]:
    """Frames within ``window`` of ``index``, closest first, at most ``max_views``."""

    candidates = [other for other in range(max(0, index - window), min(count, index + window + 1)) if other != index]
    candidates.sort(key=lambda other: (abs(other - index), other))
    return candidates[:max_views]


def multiview_consistency_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    depth_values: np.ndarray,
    index: int,
    *,
    depth: Sequence[np.ndarray] | np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    window: int = 7,
    max_views: int = 8,
    min_consistent: int = 3,
    pixel_threshold: float = 0.75,
    relative_depth_threshold: float = 0.03,
) -> np.ndarray:
    """Boolean keep-mask over the given pixels of frame ``index``.

    A pixel is unprojected to world and projected into a neighbour; the
    neighbour's own depth at that (rounded) pixel is unprojected and brought back
    into this frame. The pixel agrees with that neighbour when the round trip
    lands within ``pixel_threshold`` pixels and ``relative_depth_threshold`` in
    depth; it is kept when at least ``min_consistent`` neighbours agree.
    ``depth`` is ``[N,H,W]``, ``intrinsics`` ``[N,4]``, ``c2w`` ``[N,4,4]``.
    """

    count = min(len(depth), c2w.shape[0], intrinsics.shape[0])
    neighbours = neighbour_indices(index, count, window, max_views)
    agreements = np.zeros(rows.shape[0], dtype=np.int32)
    if not neighbours:
        return agreements >= 0 if min_consistent <= 0 else agreements > 0

    source_k = np.asarray(intrinsics[index], dtype=np.float64)
    source_c2w = np.asarray(c2w[index], dtype=np.float64)
    source_uv = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
    world = _to_world(_unproject(source_uv, depth_values, source_k), source_c2w)

    for other in neighbours:
        other_c2w = np.asarray(c2w[other], dtype=np.float64)
        other_k = np.asarray(intrinsics[other], dtype=np.float64)
        if not np.isfinite(other_c2w).all() or not np.isfinite(other_k).all():
            continue
        other_depth = np.asarray(depth[other], dtype=np.float32)
        height, width = other_depth.shape

        camera = _to_camera(world, other_c2w)
        uv, z = _project(camera, other_k)
        x = np.rint(uv[:, 0]).astype(np.int64)
        y = np.rint(uv[:, 1]).astype(np.int64)
        inside = (z > 1e-8) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not inside.any():
            continue
        sampled = np.zeros_like(z)
        sampled[inside] = other_depth[y[inside], x[inside]]
        inside &= np.isfinite(sampled) & (sampled > 1e-8)
        if not inside.any():
            continue

        # Round trip: the neighbour's own surface, brought back to this frame.
        back_world = _to_world(_unproject(uv[inside], sampled[inside], other_k), other_c2w)
        back_camera = _to_camera(back_world, source_c2w)
        back_uv, back_depth = _project(back_camera, source_k)
        reprojection = np.linalg.norm(back_uv - source_uv[inside], axis=1)
        relative = np.abs(back_depth - depth_values[inside]) / np.maximum(depth_values[inside], 1e-8)
        agree = (reprojection <= pixel_threshold) & (relative <= relative_depth_threshold)
        agreements[np.flatnonzero(inside)[agree]] += 1

    return agreements >= min_consistent


# ---------------------------------------------------------------- voxel fusion


def _merge_voxel_records(
    keys: np.ndarray,
    point_sums: np.ndarray,
    color_sums: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge already-quantized voxel records without changing membership."""

    if len(keys) == 0:
        return keys, point_sums, color_sums, counts
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    group_start = np.empty(len(order), dtype=bool)
    group_start[0] = True
    group_start[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    starts = np.flatnonzero(group_start)
    return (
        sorted_keys[starts],
        np.add.reduceat(point_sums[order], starts, axis=0),
        np.add.reduceat(color_sums[order], starts, axis=0),
        np.add.reduceat(counts[order], starts),
    )


@dataclass
class VoxelAccumulator:
    """Streaming voxel fusion: each voxel keeps the mean position and mean colour of its points."""

    voxel_size: float
    compact_every: int = 16
    keys: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    point_sums: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    color_sums: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    counts: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    _pending: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        if self.compact_every <= 0:
            raise ValueError("compact_every must be positive")

    def add(self, points: np.ndarray, colors: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
            raise ValueError(f"invalid point/color shapes: {points.shape}, {colors.shape}")
        finite = np.isfinite(points).all(axis=1)
        if finite.any():
            self._pending.append((points[finite], colors[finite]))
        if len(self._pending) >= self.compact_every:
            self._compact()

    def _compact(self) -> None:
        if not self._pending:
            return
        points = np.concatenate([item[0] for item in self._pending], axis=0)
        colors = np.concatenate([item[1] for item in self._pending], axis=0)
        new_keys = np.floor(points.astype(np.float64) / self.voxel_size).astype(np.int64)
        keys = np.concatenate((self.keys, new_keys), axis=0)
        point_sums = np.concatenate((self.point_sums, points.astype(np.float64)), axis=0)
        color_sums = np.concatenate((self.color_sums, colors.astype(np.float64)), axis=0)
        counts = np.concatenate((self.counts, np.ones(len(points), dtype=np.int64)), axis=0)
        self.keys, self.point_sums, self.color_sums, self.counts = _merge_voxel_records(
            keys, point_sums, color_sums, counts
        )
        self._pending.clear()

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._compact()
        if len(self.counts) == 0:
            raise RuntimeError("reconstruction contains no valid points")
        points = (self.point_sums / self.counts[:, None]).astype(np.float32)
        colors = np.rint(self.color_sums / self.counts[:, None]).clip(0, 255).astype(np.uint8)
        return points, colors, self.counts.copy()


# ---------------------------------------------------------------------- fusion


def up_to_z_rotation(up_world: np.ndarray) -> np.ndarray:
    """Rotation taking a world up direction onto +Z."""

    source = np.asarray(up_world, dtype=np.float64)
    source = source / np.linalg.norm(source)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(source @ target)
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sine
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sine * cross + (1 - cosine) * (cross @ cross)


def up_from_poses(c2w: np.ndarray) -> np.ndarray:
    """World up implied by OpenCV camera-to-world poses (minus the mean camera +Y)."""

    valid = c2w[np.isfinite(c2w).all(axis=(1, 2))]
    if valid.shape[0] == 0:
        raise ValueError("no finite pose to derive world up from")
    up = -np.mean(valid[:, :3, 1], axis=0)
    return up / np.linalg.norm(up)


def fuse_point_cloud(
    depth: Sequence[np.ndarray] | np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    rgb: FrameSource,
    settings: FusionSettings = FusionSettings(),
    *,
    world_frame: bool = False,
    progress: ProgressCallback | None = None,
) -> FusedCloud:
    """Fuse per-frame metric depth into one coloured, voxel-averaged cloud.

    ``depth`` is ``[N,H,W]`` metric Z-depth, ``intrinsics`` ``[N,4]`` as
    ``fx, fy, cx, cy``, ``c2w`` ``[N,4,4]`` OpenCV camera-to-world (non-finite
    rows are skipped), and ``rgb(index)`` returns frame ``index`` as ``HxWx3``
    uint8 on the depth raster. The points are returned in the z-up frame (the
    fused cloud rotated so the mean camera up is +Z, as the original
    visualization wrote it), or in the frame of ``c2w`` with ``world_frame``.
    """

    s = settings
    started = time.perf_counter()
    count = min(len(depth), c2w.shape[0], intrinsics.shape[0])
    indices = np.arange(0, count, s.frame_stride, dtype=np.int64)
    # Spread the budget evenly so a long sequence is not dominated by whichever
    # frames happen to come first.
    per_frame_cap = max(1, s.max_input_points // max(1, len(indices)))
    accumulator = VoxelAccumulator(s.voxel_size_m)
    generator = np.random.default_rng(0)
    sampled = 0
    rejected = 0

    for position, index in enumerate(indices):
        pose = c2w[index]
        fx, fy, cx, cy = intrinsics[index]
        if not np.isfinite(pose).all() or not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0:
            continue
        frame_depth = np.asarray(depth[int(index)], dtype=np.float32)
        mask = (frame_depth >= s.min_depth_m) & (frame_depth <= s.max_depth_m)
        if not mask.any():
            continue
        k3 = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        local, rows, cols = backproject_metric_depth(frame_depth, k3, mask, pixel_stride=s.pixel_stride)
        if local.shape[0] == 0:
            continue
        raw_count = local.shape[0]
        if local.shape[0] > per_frame_cap:
            choice = np.sort(generator.choice(local.shape[0], per_frame_cap, replace=False))
            local, rows, cols = local[choice], rows[choice], cols[choice]
        image = rgb(int(index))
        if image.shape[:2] != frame_depth.shape:
            raise ValueError(f"frame {index}: RGB {image.shape[:2]} is not on the depth raster {frame_depth.shape}")
        colors = image[rows, cols]

        if s.min_consistent > 0:
            keep = multiview_consistency_mask(
                rows,
                cols,
                local[:, 2],
                int(index),
                depth=depth,
                intrinsics=intrinsics,
                c2w=c2w,
                window=s.consistency_window,
                max_views=s.consistency_max_views,
                min_consistent=s.min_consistent,
                pixel_threshold=s.pixel_threshold,
                relative_depth_threshold=s.relative_depth_threshold,
            )
            local, colors, rows, cols = local[keep], colors[keep], rows[keep], cols[keep]
        if local.shape[0] == 0:
            continue

        world = local @ pose[:3, :3].T + pose[:3, 3]
        accumulator.add(world.astype(np.float32), colors.astype(np.uint8))
        sampled += local.shape[0]
        rejected += int(raw_count - local.shape[0])
        if progress is not None:
            progress(position, len(indices))

    points, colors, counts = accumulator.finish()
    rotation = np.eye(3)
    if not world_frame:
        # The original visualization rotated the fused cloud by this before writing its PLY.
        rotation = up_to_z_rotation(up_from_poses(c2w[:count]))
        points = (points.astype(np.float64) @ rotation.T).astype(np.float32)
    stats = {
        "frames": int(len(indices)),
        "voxels": int(points.shape[0]),
        "sampled_points_before_voxel": int(sampled),
        "points_rejected_by_filters": int(rejected),
        "observations_per_voxel_mean": float(counts.mean()),
        "world_to_output_rotation": rotation.tolist(),
        "fusion_seconds": round(time.perf_counter() - started, 3),
        "settings": dataclasses.asdict(s),
    }
    return FusedCloud(points=points, colors=colors, stats=stats)


# ------------------------------------------------------------------------- PLY


def write_binary_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY: ``x y z`` float32 then ``red green blue`` uint8 per vertex."""

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("PLY point/color arrays must both be [N,3]")
    record_type = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    records = np.empty(len(points), dtype=record_type)
    for index, name in enumerate(("x", "y", "z")):
        records[name] = points[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with Path(path).open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


# --------------------------------------------------------------------- inputs


def list_frame_files(directory: Path) -> list[Path]:
    """Frame files in the order ViPE's frame-directory stream indexes them."""

    files: list[Path] = []
    for extension in FRAME_EXTENSIONS:
        files.extend(sorted(directory.glob(f"*{extension}")))
        files.extend(sorted(directory.glob(f"*{extension.upper()}")))
    return sorted(set(files))


class FrameDirectoryReader:
    """Source frame ``i`` of a frame directory, as RGB uint8."""

    def __init__(self, directory: Path) -> None:
        self.files = list_frame_files(Path(directory))
        if not self.files:
            raise FileNotFoundError(f"no image files in {directory}")

    def __len__(self) -> int:
        return len(self.files)

    def __call__(self, raw_index: int) -> np.ndarray:
        if not 0 <= raw_index < len(self.files):
            raise IndexError(f"frame {raw_index} is outside the {len(self.files)} frames of the directory")
        image = cv2.imread(str(self.files[raw_index]))
        if image is None:
            raise ValueError(f"could not read frame {self.files[raw_index]}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class VideoReader:
    """Source frame ``i`` of a video, decoded sequentially with OpenCV as ViPE's video stream does."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        capture = cv2.VideoCapture(str(self.path))
        self.reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        self._capture: cv2.VideoCapture | None = None
        self._next = 0

    def __len__(self) -> int:
        return self.reported_frames

    def __call__(self, raw_index: int) -> np.ndarray:
        if self._capture is None or raw_index < self._next:
            if self._capture is not None:
                self._capture.release()
            self._capture = cv2.VideoCapture(str(self.path))
            self._next = 0
        while self._next < raw_index:
            if not self._capture.grab():
                raise IndexError(f"{self.path} ends before frame {raw_index}")
            self._next += 1
        ok, frame = self._capture.read()
        if not ok:
            raise IndexError(f"{self.path} ends before frame {raw_index}")
        self._next += 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fit_to_raster(image: np.ndarray, raster_hw: tuple[int, int], plan: RasterPlan | None = None) -> np.ndarray:
    """An input frame as tracking used it, on the depth raster ``raster_hw``.

    With ``plan`` the frame goes through the pipeline's input raster,
    :func:`~vipe.metrictrail.raster.raster_uint8` (the recorded resampler, then the
    integer crop). A frame that is still not on the depth raster gets ViPE's bilinear
    resize and centre crop onto it (``cover_crop`` geometry, the one read-time resize of
    :mod:`vipe.metrictrail`); a frame on the raster is returned unchanged.
    """

    if plan is not None:
        image = raster_uint8(image, plan)
    if image.shape[:2] != tuple(raster_hw):
        target_hw = (int(raster_hw[0]), int(raster_hw[1]))
        image = raster_uint8(image, plan_raster(image.shape[:2], 1, 1.0, target_hw=target_hw, max_frames=1))
    return image


def make_rgb_source(
    read_raw: FrameSource,
    raster_hw: tuple[int, int],
    *,
    plan: RasterPlan | None = None,
    source_frame_ids: np.ndarray | None = None,
) -> FrameSource:
    """``rgb(i)``: input frame ``source_frame_ids[i]`` (frame ``i`` without ids) through :func:`fit_to_raster`."""

    def rgb(index: int) -> np.ndarray:
        raw_index = int(source_frame_ids[index]) if source_frame_ids is not None else int(index)
        return fit_to_raster(read_raw(raw_index), raster_hw, plan)

    return rgb


def raster_plan(input_raster: dict | None) -> RasterPlan | None:
    """The run's raster plan if its record says the pipeline resized the input (``cover_crop``, ``resize``).

    Recomputed with :func:`~vipe.metrictrail.raster.plan_raster` from the recorded
    policy, source raster, frame count, frame rate, target and ``max_frames``, and
    checked against the recorded geometry and frame stride, with the recorded
    resampler (its ``interpolation``; ``bilinear`` for a record without one). None for ``unchanged`` or no
    record. A record of an earlier raster schema (frames resized by another
    interpolation) is coloured with ViPE's bilinear resize, with a warning.
    """

    if not input_raster or input_raster.get("policy") not in RESIZE_POLICIES:
        return None
    resampler = LEGACY_RESAMPLER
    if input_raster.get("schema") == RASTER_SCHEMA:
        resampler = resampler_of_record(input_raster)
    else:
        logger.warning(
            "the run record's input raster is %s, not %s (resampler %r): the run's frames were not resized by "
            "ViPE's bilinear resize, which the colours use, so they can differ from the tracked frames by a few levels",
            input_raster.get("schema"),
            RASTER_SCHEMA,
            input_raster.get("resampler"),
        )
    plan = plan_raster(
        (int(input_raster["source_hw"][0]), int(input_raster["source_hw"][1])),
        int(input_raster["source_frames"]),
        float(input_raster["source_fps"]),
        target_hw=(int(input_raster["target_hw"][0]), int(input_raster["target_hw"][1])),
        max_frames=int(input_raster["max_frames"]),
        policy=str(input_raster["policy"]),
        resampler=resampler,
    )
    recorded = {key: input_raster[key] for key in ("resized_hw", "crop_left_top", "frame_stride")}
    if {key: plan.record()[key] for key in recorded} != recorded:
        raise ValueError(f"the run record's input raster {recorded} does not match the raster plan {plan}")
    return plan


def open_rgb_source(
    raster_hw: tuple[int, int],
    *,
    video: Path | None = None,
    frames: Path | None = None,
    input_raster: dict | None = None,
    source_frame_ids: np.ndarray | None = None,
) -> tuple[FrameSource, int | None, str]:
    """``(rgb, input_frames, rule)``: ``rgb(i)`` is the frame tracking used for artifact frame ``i``.

    ``video`` or ``frames`` is the run's input, decoded as ViPE's video or
    frame-directory stream decodes it (OpenCV), and ``input_raster`` the run record's
    entry. With policy ``cover_crop`` or ``resize`` every input frame goes through the
    recorded plan (:func:`~vipe.metrictrail.raster.raster_uint8`: the recorded resampler,
    then the integer crop), and artifact frame ``i`` is input frame
    ``source_frame_ids[i]``. Without such a record the frames are used as given.
    ``input_frames`` is the input's frame count when artifact frame ``i`` is input frame
    ``i``, otherwise None.
    """

    if (video is None) == (frames is None):
        raise ValueError("pass exactly one of video and frames")
    plan = raster_plan(input_raster)
    reader: VideoReader | FrameDirectoryReader
    if video is not None:
        reader = VideoReader(video)
    else:
        assert frames is not None
        reader = FrameDirectoryReader(frames)
    rgb = make_rgb_source(reader, raster_hw, plan=plan, source_frame_ids=source_frame_ids)
    names = {"bilinear": "ViPE bilinear", "bicubic_antialias": "bicubic antialias"}
    rule = f"{names.get(plan.resampler, plan.resampler)} {plan.policy}" if plan is not None else "as given"
    return rgb, (len(reader) if source_frame_ids is None else None), rule


@dataclass
class VipeArtifacts:
    depth: np.ndarray  # [N,H,W] float32, NaN where a frame has no depth
    c2w: np.ndarray  # [N,4,4] float64 OpenCV camera-to-world, NaN where a frame has no pose
    intrinsics: np.ndarray  # [N,4] float64 fx, fy, cx, cy, NaN where missing
    source_frame_ids: np.ndarray | None  # [N] input-frame index of each artifact frame, if recorded
    input_raster: dict | None = None  # the run record's ``input_raster``, if the run wrote a record


def read_depth_zip(path: Path) -> dict[int, np.ndarray]:
    """Frame index -> float32 depth from ViPE's zipped half-float EXR (``vipe.utils.io`` format).

    Mirrors ``vipe.utils.io.read_depth_artifacts``, including its all-NaN map for
    an EXR that fails to decode, keyed by the frame index in the entry name.
    """

    import OpenEXR

    frames: dict[int, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    with zipfile.ZipFile(path, "r") as archive:
        for file_name in sorted(archive.namelist()):
            index = int(file_name.split(".")[0])
            with archive.open(file_name) as handle:
                try:
                    exr = OpenEXR.InputFile(handle)
                except OSError:
                    if shape is None:
                        raise
                    logger.warning("failed to load EXR %s-%s; using an all-NaN map", path, file_name)
                    frames[index] = np.full(shape, np.nan, dtype=np.float32)
                    continue
                window = exr.header()["dataWindow"]
                shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
                channel = exr.channels(["Z"])[0]
                frames[index] = np.frombuffer(channel, dtype=np.float16).reshape(shape).astype(np.float32)
    return frames


def _check_pinhole(camera_type_path: Path) -> None:
    if not camera_type_path.is_file():
        return
    for line in camera_type_path.read_text().splitlines():
        if line.strip() and line.split(":", 1)[1].strip() != "PINHOLE":
            raise ValueError(f"{camera_type_path}: only PINHOLE cameras can be unprojected, got {line.strip()!r}")


def load_vipe_artifacts(output_dir: Path, name: str) -> VipeArtifacts:
    """Depth, poses, intrinsics and input raster of run ``name`` in a ViPE output directory.

    Frames are indexed as ViPE indexes its artifacts; the count is that of the
    depth artifact, and pose or intrinsics rows beyond it are dropped. The input
    raster is the ``input_raster`` entry of the MetricTrail run record
    ``vipe/<name>_metrictrail.json``, None for a run without one.
    """

    output_dir = Path(output_dir)
    depth_path = output_dir / "depth" / f"{name}.zip"
    pose_path = output_dir / "pose" / f"{name}.npz"
    intrinsics_path = output_dir / "intrinsics" / f"{name}.npz"
    for path in (depth_path, pose_path, intrinsics_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing ViPE artifact {path}")
    _check_pinhole(output_dir / "intrinsics" / f"{name}_camera.txt")

    frames = read_depth_zip(depth_path)
    if not frames:
        raise ValueError(f"{depth_path} holds no frames")
    count = max(frames) + 1
    height, width = next(iter(frames.values())).shape
    depth = np.full((count, height, width), np.nan, dtype=np.float32)
    for index, frame in frames.items():
        depth[index] = frame

    def scatter(path: Path, row_shape: tuple[int, ...]) -> np.ndarray:
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float64)
            inds = np.asarray(archive["inds"], dtype=np.int64)
        if data.shape[1:] != row_shape:
            raise ValueError(f"{path}: expected rows of shape {row_shape}, got {data.shape[1:]}")
        out = np.full((count, *row_shape), np.nan, dtype=np.float64)
        keep = inds < count
        out[inds[keep]] = data[keep]
        return out

    c2w = scatter(pose_path, (4, 4))
    intrinsics = scatter(intrinsics_path, (4,))
    # MetricTrail records which input frame each artifact frame came from (ViPE's raw_frame_idx).
    source_frame_ids = None
    with np.load(pose_path) as archive:
        if "source_frame_ids" in archive.files:
            ids = np.asarray(archive["source_frame_ids"], dtype=np.int64)
            if len(ids) >= count:
                source_frame_ids = ids[:count]
            else:
                logger.warning("%s: %d source_frame_ids for %d frames; assuming identity", pose_path, len(ids), count)
    record_path = output_dir / "vipe" / f"{name}_metrictrail.json"
    input_raster = json.loads(record_path.read_text()).get("input_raster") if record_path.is_file() else None
    return VipeArtifacts(
        depth=depth, c2w=c2w, intrinsics=intrinsics, source_frame_ids=source_frame_ids, input_raster=input_raster
    )


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    defaults = FusionSettings()
    parser = argparse.ArgumentParser(
        prog="python -m vipe.metrictrail.pointcloud",
        description="Export a fused, coloured binary PLY point cloud from a ViPE output directory.",
    )
    parser.add_argument("output_dir", type=Path, help="ViPE output directory (holds pose/, intrinsics/, depth/)")
    parser.add_argument("--name", required=True, help="artifact name, i.e. pose/<name>.npz")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="the input video of the run")
    source.add_argument("--frames", type=Path, help="the input frame directory of the run")
    parser.add_argument("--output", type=Path, help="PLY path (default: <output_dir>/pointcloud/<name>.ply)")
    strict = FILTER_PRESETS["strict"]

    def setting(flag: str, kind: type, text: str, name: str | None = None) -> None:
        # None = not given: the --filter preset's value, else FusionSettings' default (settings_from_args).
        name = name or flag[2:].replace("-", "_")
        default = getattr(defaults, name)
        extra = f"; strict: {strict[name]}" if name in strict else ""
        parser.add_argument(flag, dest=name, type=kind, default=None, help=f"{text} (default: {default}{extra})")

    setting(
        "--frame-stride",
        int,
        "fuse every N-th frame; the consistency filter still compares against every frame. Use N > 1 when a "
        "subject moves, which is otherwise fused once per frame along its path",
    )
    setting("--pixel-stride", int, "take every N-th pixel in both directions")
    setting("--voxel-size", float, "voxel size in metres; each voxel becomes one point", "voxel_size_m")
    setting("--min-depth", float, "nearest depth kept, metres", "min_depth_m")
    setting("--max-depth", float, "farthest depth kept, metres (the depth cap)", "max_depth_m")
    setting("--max-input-points", int, "point budget before voxelization, spread evenly over the fused frames")
    parser.add_argument(
        "--filter",
        choices=sorted(FILTER_PRESETS),
        default="default",
        help="flying-point filter preset: 'default' (the original visualization's) or 'strict' (fewer flying "
        "points, sparser); the five options below override single values",
    )
    setting(
        "--min-consistent",
        int,
        "neighbouring frames that must reproduce a point to keep it; 0 disables the flying-point filter",
    )
    setting("--consistency-window", int, "neighbour search window (frames)")
    setting("--consistency-max-views", int, "neighbouring frames checked per point (nearest first)")
    setting("--pixel-threshold", float, "round-trip reprojection tolerance (px)")
    setting("--relative-depth-threshold", float, "round-trip relative depth tolerance")
    parser.add_argument(
        "--world-frame",
        action="store_true",
        help="write ViPE's world frame, the frame of pose/<name>.npz (default: z-up, the mean camera up is +Z)",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> FusionSettings:
    """The ``--filter`` preset with every explicitly given setting on top."""

    given = {f.name: getattr(args, f.name) for f in dataclasses.fields(FusionSettings)}
    return fusion_settings(args.filter, **{name: value for name, value in given.items() if value is not None})


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = settings_from_args(args)
    if settings.frame_stride <= 0 or settings.pixel_stride <= 0:
        raise SystemExit("--frame-stride and --pixel-stride must be positive")

    artifacts = load_vipe_artifacts(args.output_dir, args.name)
    raster_hw = (int(artifacts.depth.shape[1]), int(artifacts.depth.shape[2]))
    rgb, input_frames, rgb_rule = open_rgb_source(
        raster_hw,
        video=args.video,
        frames=args.frames,
        input_raster=artifacts.input_raster,
        source_frame_ids=artifacts.source_frame_ids,
    )
    logger.info("[%s] colours: %s, on the %s depth raster", args.name, rgb_rule, list(raster_hw))
    count = len(artifacts.depth)
    if input_frames:
        if input_frames != count:
            logger.warning(
                "input has %d frames, the ViPE artifacts %d; fusing the first %d",
                input_frames,
                count,
                min(input_frames, count),
            )
        count = min(count, input_frames)

    def progress(done: int, total: int) -> None:
        if done % 100 == 0:
            logger.info("[%s] fused %d/%d", args.name, done, total)

    cloud = fuse_point_cloud(
        artifacts.depth[:count],
        artifacts.intrinsics[:count],
        artifacts.c2w[:count],
        rgb,
        settings,
        world_frame=args.world_frame,
        progress=progress,
    )
    output = args.output if args.output is not None else args.output_dir / "pointcloud" / f"{args.name}.ply"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_binary_point_cloud(output, cloud.points, cloud.colors)
    summary = {
        "name": args.name,
        "output": str(output),
        "input_hw": list(raster_hw),
        "frame": "vipe_world" if args.world_frame else "z_up",
        "rgb": rgb_rule,
        "filter": args.filter,
        **cloud.stats,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
