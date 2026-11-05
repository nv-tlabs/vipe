"""Shared point-cloud metric core (Oxford-Spires and 7-Scenes ``--metric points``).

Accuracy / completeness / Chamfer / precision / recall / F1 after one
Sim(3) Umeyama on pixel correspondences, voxel down-sampling and Open3D
point-to-point ICP.  The registration and metric primitives below are a port of
the third-party Apache-2.0 point-cloud evaluator that ships with the vendored
trajectory evaluator (licence: ``licenses/LICENSE.trajectory_eval``); changes:
only the functions used here are kept, inputs are always ``[N, 3]``, the
Umeyama solve refuses a zero-variance source, and ``apply_transform`` has a
switch for the variant without homogeneous division used by the Oxford
protocol.  Every NumPy / Open3D / SciPy call and its operand dtypes are kept as
in the reference implementation, because the records are reproduced bit for
bit.

``unproject_depth_maps`` and ``resize_nearest_rows_like_torch`` re-implement the
arithmetic of the Oxford reference pipeline (pinhole unprojection through the
closed-form inverse of a world-to-camera matrix; legacy ``nearest``
interpolation) with the same operations.

Optional dependencies: ``open3d`` (0.19.0 for the records) and ``opencv``
(``cv2``) are imported only when point metrics are requested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "--metric points needs open3d (the records used open3d==0.19.0)"
        ) from error
    return o3d


def cv2_module() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit("--metric points needs OpenCV (cv2)") from error
    return cv2


# --------------------------------------------------------------------------- registration


def apply_transform(
    points: np.ndarray, transformation: np.ndarray, *, homogeneous_divide: bool = True
) -> np.ndarray:
    """Apply a 4x4 transform to ``[N, 3]`` points.

    ``homogeneous_divide=True``: float64 homogeneous coordinates and division by
    the last coordinate (7-Scenes reference).  ``False``: homogeneous column in
    the input dtype and no division (Oxford reference).  For matrices whose last
    row is exactly ``[0, 0, 0, 1]`` both give identical values.
    """

    xyz_flat = points[..., :3].reshape(-1, 3)
    if homogeneous_divide:
        ones = np.ones((xyz_flat.shape[0], 1))
        xyz_homogeneous = np.hstack([xyz_flat, ones])
        xyz_transformed = xyz_homogeneous @ transformation.T
        return xyz_transformed[:, :3] / xyz_transformed[:, 3:4]
    ones = np.ones((xyz_flat.shape[0], 1), dtype=xyz_flat.dtype)
    xyz_homogeneous = np.hstack([xyz_flat, ones])
    return (xyz_homogeneous @ transformation.T)[:, :3]


def umeyama_registration(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """4x4 Sim(3) (``c R``, ``t``) mapping ``source`` onto corresponding ``target`` points."""

    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Source and target must have same shape, got {source_points.shape} "
            f"and {target_points.shape}"
        )
    source_xyz = source_points[..., :3].reshape(-1, 3).astype(np.float64)
    target_xyz = target_points[..., :3].reshape(-1, 3).astype(np.float64)
    X = source_xyz.T
    Y = target_xyz.T
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    X_centered = X - mu_x
    Y_centered = Y - mu_y
    var_x = np.square(X_centered).sum() / X.shape[1]
    if var_x < 1e-18:
        raise ValueError("Umeyama: source variance is zero")
    cov_xy = (Y_centered @ X_centered.T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[2, 2] = -1
    R = U @ S @ VH
    c = np.trace(np.diag(D) @ S) / var_x
    t = mu_y - c * R @ mu_x
    transformation = np.eye(4)
    transformation[:3, :3] = c * R
    transformation[:3, 3:4] = t
    return transformation


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Open3D voxel-grid filter (voxel centroids), returned as float32 ``[M, 3]``."""

    o3d = _open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_down.points).astype(np.float32)


def icp_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    icp_threshold: float,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Open3D point-to-point ICP from the identity; 4x4 transform source -> target."""

    o3d = _open3d()
    source_xyz = np.asarray(source_points[..., :3].reshape(-1, 3), dtype=np.float64)
    target_xyz = np.asarray(target_points[..., :3].reshape(-1, 3), dtype=np.float64)
    if source_xyz.shape[0] < 3 or target_xyz.shape[0] < 3:
        raise ValueError(
            f"ICP needs >=3 points, got source={source_xyz.shape[0]}, target={target_xyz.shape[0]}"
        )
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_xyz)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_xyz)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=tolerance, relative_rmse=tolerance, max_iteration=max_iterations
    )
    result = o3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=icp_threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria,
    )
    return np.asarray(result.transformation, dtype=np.float64)


# --------------------------------------------------------------------------- metrics


def nearest_distances(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Distance from every source point to its nearest target point (xyz, float64)."""

    from scipy.spatial import cKDTree

    source_xyz = np.asarray(source_points[..., :3].reshape(-1, 3), dtype=np.float64)
    target_xyz = np.asarray(target_points[..., :3].reshape(-1, 3), dtype=np.float64)
    distances, _ = cKDTree(target_xyz).query(source_xyz, workers=-1)
    return distances


def threshold_suffix(threshold: float) -> str:
    return f"{float(threshold):.4f}".rstrip("0").rstrip(".")


def evaluate_pointcloud(
    source_points: np.ndarray, target_points: np.ndarray, thresholds: Sequence[float]
) -> dict[str, float]:
    """Accuracy (source->target mean NN), completeness (target->source), Chamfer
    = their mean, and precision / recall / F1 in percent with strict ``<``."""

    thresholds = list(thresholds)
    source = source_points.reshape(-1, source_points.shape[-1])
    target = target_points.reshape(-1, target_points.shape[-1])
    dist_s2t = nearest_distances(source, target)
    dist_t2s = nearest_distances(target, source)
    accuracy = float(np.mean(dist_s2t))
    completeness = float(np.mean(dist_t2s))
    results = {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer": (accuracy + completeness) / 2.0,
    }
    for threshold in thresholds:
        precision = float(np.mean(dist_s2t < threshold) * 100.0)
        recall = float(np.mean(dist_t2s < threshold) * 100.0)
        f1 = float(2 * (precision * recall) / (precision + recall)) if precision + recall > 0 else 0.0
        suffix = threshold_suffix(threshold)
        results[f"precision_{suffix}"] = precision
        results[f"recall_{suffix}"] = recall
        results[f"f1_{suffix}"] = f1
    return results


# --------------------------------------------------------------------------- geometry


def unproject_depth_maps(
    depth: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """``[S, H, W]`` depth, ``[S, 3, 4]`` world-to-camera, ``[S, 3, 3]`` K -> ``[S, H, W, 3]``.

    Camera points are formed in float64 and stored as float32; the
    camera-to-world matrix is the closed-form inverse ``[R^T | -R^T t]`` of the
    given matrix (in its dtype, embedded in a float64 identity); world points
    keep the dtype of ``float32 @ float64`` (float64).
    """

    frames = []
    for index in range(depth.shape[0]):
        depth_map = depth[index]
        extrinsic = world_to_camera[index]
        intrinsic = intrinsics[index]
        height, width = depth_map.shape
        if intrinsic.shape != (3, 3) or intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0:
            raise ValueError("intrinsics must be 3x3 without skew")
        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        x_cam = (u - cu) * depth_map / fu
        y_cam = (v - cv) * depth_map / fv
        camera = np.stack((x_cam, y_cam, depth_map), axis=-1).astype(np.float32)
        rotation = extrinsic[None][:, :3, :3]
        translation = extrinsic[None][:, :3, 3:]
        rotation_t = np.transpose(rotation, (0, 2, 1))
        inverse = np.tile(np.eye(4), (1, 1, 1))
        inverse[:, :3, :3] = rotation_t
        inverse[:, :3, 3:] = -np.matmul(rotation_t, translation)
        frames.append(np.dot(camera, inverse[0][:3, :3].T) + inverse[0][:3, 3])
    return np.stack(frames, axis=0)


def resize_nearest_rows_like_torch(maps: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Legacy ``nearest`` resize of ``[S, H, W]`` float32 maps (PyTorch CPU semantics).

    Source index = ``min(floor(dst * (float32(in) / out)), in - 1)`` in float32;
    identity when the size is unchanged, ``dst >> 1`` when it doubles.
    """

    def indices(size_in: int, size_out: int) -> np.ndarray:
        if size_out == size_in:
            return np.arange(size_out, dtype=np.int64)
        if size_out == 2 * size_in:
            return np.arange(size_out, dtype=np.int64) >> 1
        scale = np.float32(np.float32(size_in) / np.float32(size_out))
        index = np.floor(np.arange(size_out, dtype=np.float32) * scale).astype(np.int64)
        return np.minimum(index, size_in - 1)

    maps = np.asarray(maps, dtype=np.float32)
    rows = indices(maps.shape[1], int(target_hw[0]))
    cols = indices(maps.shape[2], int(target_hw[1]))
    return np.ascontiguousarray(maps[:, rows][:, :, cols])


# --------------------------------------------------------------------------- depth cache


def find_depth_cache(
    prediction: Path, case: str, cache_dirs: Sequence[str], override: Path | None
) -> Path | None:
    """``<root>/<case>`` of a frontend depth cache (``depth.f32`` + ``mask.u8``).

    ``--depth-cache ROOT`` wins; otherwise the nearest ancestor of the prediction
    file holding ``frontend-cache/<dataset dir>/<case>`` (MetricTrail run layout).
    """

    if override is not None:
        path = Path(override) / case
        return path if (path / "depth.f32").is_file() else None
    for parent in Path(prediction).resolve().parents:
        for name in cache_dirs:
            path = parent / "frontend-cache" / name / case
            if (path / "depth.f32").is_file() and (path / "mask.u8").is_file():
                return path
    return None


def open_depth_cache(path: Path, count: int, input_hw: tuple[int, int]) -> tuple[np.memmap, np.memmap]:
    """Validated read-only memmaps of a complete frontend cache (raw depth, depth-model mask)."""

    contract = json.loads((path / "contract.json").read_text(encoding="utf-8"))
    success_path = path / "SUCCESS"
    success = json.loads(success_path.read_text(encoding="utf-8")) if success_path.is_file() else {}
    shape = (int(count), *map(int, input_hw))
    if (
        int(contract.get("count", -1)) != int(count)
        or list(contract.get("input_hw", [])) != list(input_hw)
        or success.get("status") != "success"
        or int(success.get("completed_frames", -1)) != int(count)
        or (path / "depth.f32").stat().st_size != int(np.prod(shape)) * 4
        or (path / "mask.u8").stat().st_size != int(np.prod(shape))
    ):
        raise ValueError(f"frontend depth cache is incomplete or does not match {shape}: {path}")
    depth = np.memmap(path / "depth.f32", dtype=np.float32, mode="r", shape=shape)
    mask = np.memmap(path / "mask.u8", dtype=np.uint8, mode="r", shape=shape)
    return depth, mask


def json_ready(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in values.items()
    }
