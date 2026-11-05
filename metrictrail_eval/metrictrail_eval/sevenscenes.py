"""7-Scenes (18 official test sequences, every 5th frame), Sim(3) ATE RMSE in metres.

Port of the trajectory part of the benchmark package's
``seven_scenes434/evaluate.py`` as called by the 7-Scenes K-matrix harness
(``_evaluate_seven_case``; the 0.0639 record of the METRIC_GLOBAL_BA arm):

* frames: raw ids ``0, 5, 10, ...`` of each clip (1000 raw frames, 500 for
  stairs) -> 200 / 100 evaluated frames per sequence;
* GT: ``<raw root>/<scene>/seq-NN/frame-XXXXXX.pose.txt`` loaded with
  ``np.loadtxt`` and cast to float32 (the historical dataset adapter
  serialised the GT poses as float32; that precision boundary is kept);
* prediction: ``frame_ids == 0..N-1``, ``source_frame_ids`` (when present)
  equal to the raw ids above, ``c2w`` float64 finite;
* metric: the shared hash-locked ``TrajectoryEvaluator(align=True,
  correct_scale=True)`` (``sim3_ate``; ``ate``, ``rpe_trans``, ``rpe_rot``);
  rotations are orthogonalised inside the evaluator (the GT R is not exactly
  SO(3));
* average: unweighted mean over the 18 sequences.

``--metric points`` (``evaluate_points_case``) ports the benchmark package's
own-K reconstruction evaluator at F1@25cm
(``seven_scenes434_robust128_f1at25cm/evaluate.py`` ``own_k_pointcloud_metrics``
on ``seven_scenes434_ownk_reconstruction``; the CD 0.0405 m / F1 97.60 record):

* predicted depth: the prediction's ``depth_m`` [N, 434, 574] if present,
  otherwise the run's frontend depth cache (``depth.f32`` + ``mask.u8``) with
  the tracking-time validity rule ``mask & finite & 0.2 <= d <= 500`` (others
  set to 0), which is how ``depth_m`` was written;
* each frame's depth is resized to the native 480x640 with
  ``cv2.INTER_NEAREST``; the prediction's own K (``intrinsics`` [N, 4] at
  434x574) is scaled by ``640/574`` (fx, cx) and ``480/434`` (fy, cy), no
  half-pixel shift; poses are the prediction's ``c2w`` (no inversion);
* GT: ``frame-XXXXXX.depth.proj.png`` (uint16 mm; 65535, > 10 m and < 1 mm
  -> invalid) back-projected with the dataset K (525, 525, 320, 240) and the
  float32 GT ``pose.txt``;
* pixels valid in both (depth > 1e-4, finite, finite pose) give the Umeyama
  Sim(3) correspondences (all of them, no subsampling); the aligned
  prediction (common pixels) and the full valid GT cloud are voxel
  down-sampled at 4/512 m, then Open3D point-to-point ICP (0.1 m, 20
  iterations, 1e-6) from the identity;
* metrics (``points.evaluate_pointcloud``): accuracy, completeness, Chamfer
  (their mean) in m; precision / recall / F1 in percent at 0.05 and 0.25 m
  (strict ``<``); headline Chamfer and ``f1_0.25``; unweighted mean over the
  18 sequences in canonical order.

The pose-pair AUC fields of that harness are not reproduced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import aligned as aligned_trajectory
from . import points, sim3_ate
from .predictions import find_prediction as _find_generic

NAME = "7scenes"
SCENE_SEQUENCES = (
    "chess/seq-03",
    "chess/seq-05",
    "fire/seq-03",
    "fire/seq-04",
    "heads/seq-01",
    "office/seq-02",
    "office/seq-06",
    "office/seq-07",
    "office/seq-09",
    "pumpkin/seq-01",
    "pumpkin/seq-07",
    "redkitchen/seq-03",
    "redkitchen/seq-04",
    "redkitchen/seq-06",
    "redkitchen/seq-12",
    "redkitchen/seq-14",
    "stairs/seq-01",
    "stairs/seq-04",
)
CASES = tuple(item.replace("/", "__") for item in SCENE_SEQUENCES)
SOURCE_BY_KEY = dict(zip(CASES, SCENE_SEQUENCES, strict=True))
STRIDE = 5
EXPECTED_RAW_FRAMES = {key: (500 if key.startswith("stairs__") else 1000) for key in CASES}
UNITS = "m"
PROTOCOLS = ("sim3_ate",)
DEFAULT_PROTOCOL = "sim3_ate"
LOGGER = logging.getLogger("metrictrail_eval.sevenscenes")

# --metric points
POINT_PROTOCOL = "sim3_voxel_icp_own_k"
POINT_THRESHOLDS_M = (0.05, 0.25)
POINT_HEADLINE = ("chamfer", "f1_0.25")
POINT_UNITS = {"distance": "m", "precision_recall_f1": "%"}
POINT_CACHE_DIRS = ("seven_scenes", "sevenscenes", "7scenes")
INPUT_HW = (434, 574)
NATIVE_HW = (480, 640)
NATIVE_INTRINSICS = np.asarray([525.0, 525.0, 320.0, 240.0], dtype=np.float64)
CACHE_DEPTH_RANGE_M = (0.2, 500.0)
VOXEL_SIZE_M = 4.0 / 512.0
ICP_THRESHOLD_M = 0.1


def find_prediction(pred: Path, stage: str, case: str, protocol: str) -> Path | None:
    return _find_generic(pred, stage, case)


def source_ids(case: str) -> np.ndarray:
    return np.arange(EXPECTED_RAW_FRAMES[case], dtype=np.int32)[::STRIDE]


def load_gt(raw_root: Path, case: str, ids: np.ndarray) -> tuple[np.ndarray, Path]:
    """seven_scenes434.evaluate.load_gt_pose_depth, pose part."""

    sequence_root = Path(raw_root).expanduser().resolve() / SOURCE_BY_KEY[case]
    poses: list[np.ndarray] = []
    for frame_id in ids.tolist():
        pose = np.loadtxt(sequence_root / f"frame-{int(frame_id):06d}.pose.txt").astype(np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"{case}/frame-{int(frame_id):06d}: invalid pose")
        poses.append(pose)
    return np.stack(poses), sequence_root


def evaluate_case(
    case: str, prediction: Path, gt_root: Path, *, protocol: str, stage: str
) -> dict[str, Any]:
    return _evaluate_case(case, prediction, gt_root, protocol=protocol, stage=stage, keep_alignment=False)[0]


def evaluate_case_aligned(
    case: str, prediction: Path, gt_root: Path, *, protocol: str, stage: str
) -> tuple[dict[str, Any], aligned_trajectory.AlignedTrajectory | None]:
    """``evaluate_case`` (same computation, same result) plus the aligned trajectory its ATE used."""

    return _evaluate_case(case, prediction, gt_root, protocol=protocol, stage=stage, keep_alignment=True)


def _evaluate_case(
    case: str, prediction: Path, gt_root: Path, *, protocol: str, stage: str, keep_alignment: bool
) -> tuple[dict[str, Any], aligned_trajectory.AlignedTrajectory | None]:
    if case not in SOURCE_BY_KEY:
        raise ValueError(f"unknown 7-Scenes sequence {case}")
    ids = source_ids(case)
    count = len(ids)
    with np.load(prediction, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int32)
        pred_c2w = np.asarray(archive["c2w"], dtype=np.float64)
        stored_source_ids = (
            np.asarray(archive["source_frame_ids"], dtype=np.int32)
            if "source_frame_ids" in archive.files
            else None
        )
    if (
        not np.array_equal(frame_ids, np.arange(count, dtype=np.int32))
        or (stored_source_ids is not None and not np.array_equal(stored_source_ids, ids))
        or pred_c2w.shape != (count, 4, 4)
        or not np.isfinite(pred_c2w).all()
    ):
        raise ValueError(f"7Scenes/{case}: prediction contract failed")
    gt_c2w, sequence_root = load_gt(gt_root, case, ids)
    raw = sim3_ate.trajectory_evaluator_class()(align=True, correct_scale=True).evaluate(
        sim3_ate.ArrayTrajectoryLoader(gt_c2w), sim3_ate.ArrayTrajectoryLoader(pred_c2w), LOGGER
    )
    transform = raw.pop("traj_transform", None)
    trajectory = (
        aligned_trajectory.from_shared_evaluator(gt_c2w, pred_c2w, None, transform)
        if keep_alignment else None
    )
    return {
        **{key: float(raw[key]) for key in ("ate", "rpe_trans", "rpe_rot")},
        "frames": int(count),
        "gt_dir": str(sequence_root),
    }, trajectory


def macro(values: dict[str, float]) -> float:
    return float(np.mean([values[case] for case in values]))


# --------------------------------------------------------------------------- points


def gt_depth_paths(raw_root: Path, case: str, ids: np.ndarray) -> list[Path]:
    sequence_root = Path(raw_root).expanduser().resolve() / SOURCE_BY_KEY[case]
    paths = [sequence_root / f"frame-{int(frame_id):06d}.depth.proj.png" for frame_id in ids.tolist()]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_gt_depth(path: Path) -> np.ndarray:
    """uint16 millimetres -> float32 metres; 65535, > 10 m and < 1 mm become 0."""

    cv2 = points.cv2_module()
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or tuple(depth.shape) != NATIVE_HW:
        raise ValueError(f"Invalid 7-Scenes projected depth: {path}")
    result = depth.astype(np.float32)
    result[result == 65535] = 0.0
    result *= 0.001
    result[~np.isfinite(result)] = 0.0
    result[result > 10.0] = 0.0
    result[result < 1e-3] = 0.0
    return result


def scale_intrinsics(intrinsics: np.ndarray, source_hw: tuple[int, int]) -> np.ndarray:
    """fx, cx *= W_native / W_in; fy, cy *= H_native / H_in (no half-pixel shift)."""

    value = np.asarray(intrinsics, dtype=np.float64)
    src_h, src_w = map(int, source_hw)
    dst_h, dst_w = NATIVE_HW
    scale = np.asarray([dst_w / src_w, dst_h / src_h, dst_w / src_w, dst_h / src_h], dtype=np.float64)
    result = value * scale
    if not np.isfinite(result).all() or np.any(result[..., :2] <= 0):
        raise ValueError("invalid scaled intrinsics")
    return result


def backproject(depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """World points (float32) of the ``mask`` pixels of a native-raster depth map."""

    if depth.shape != NATIVE_HW or mask.shape != NATIVE_HW:
        raise ValueError("native depth/mask raster mismatch")
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        return np.empty((0, 3), dtype=np.float32)
    vv, uu = np.nonzero(mask)
    if len(vv) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[vv, uu].astype(np.float64, copy=False)
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    camera = np.stack(((uu - cx) * z / fx, (vv - cy) * z / fy, z), axis=1)
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32, copy=False)


def load_point_prediction(
    case: str, prediction: Path, depth_cache: Path | None
) -> dict[str, Any]:
    """c2w [N,4,4] float64, own K [N,4] float64 at the input raster, depth [N,H,W] float32."""

    count = len(source_ids(case))
    with np.load(prediction, allow_pickle=False) as archive:
        files = set(archive.files)
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int32)
        c2w = np.asarray(archive["c2w"], dtype=np.float64)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
        input_hw = (
            tuple(np.asarray(archive["input_hw"], dtype=np.int32).tolist()) if "input_hw" in files else INPUT_HW
        )
        stored_source_ids = (
            np.asarray(archive["source_frame_ids"], dtype=np.int32) if "source_frame_ids" in files else None
        )
        depth = np.asarray(archive["depth_m"], dtype=np.float32) if "depth_m" in files else None
    if (
        not np.array_equal(frame_ids, np.arange(count, dtype=np.int32))
        or (stored_source_ids is not None and not np.array_equal(stored_source_ids, source_ids(case)))
        or c2w.shape != (count, 4, 4)
        or not np.isfinite(c2w).all()
        or intrinsics.shape != (count, 4)
        or not np.isfinite(intrinsics).all()
    ):
        raise ValueError(f"7Scenes/{case}: point prediction contract failed")
    if depth is not None:
        depth_source = "prediction depth_m"
    else:
        cache = points.find_depth_cache(prediction, case, POINT_CACHE_DIRS, depth_cache)
        if cache is None:
            raise FileNotFoundError(
                f"7Scenes/{case}: prediction has no depth_m and no frontend depth cache was found "
                "(use --depth-cache <root holding <case>/depth.f32>)"
            )
        raw, mask = points.open_depth_cache(cache, count, input_hw)
        raw = np.asarray(raw, dtype=np.float32)
        low, high = (np.float32(value) for value in CACHE_DEPTH_RANGE_M)
        valid = (np.asarray(mask) != 0) & np.isfinite(raw) & (raw >= low) & (raw <= high)
        depth = np.where(valid, raw, np.float32(0.0)).astype(np.float32, copy=False)
        depth_source = str(cache)
    if depth.shape != (count, *input_hw) or not np.isfinite(depth).all():
        raise ValueError(f"7Scenes/{case}: prediction depth shape {depth.shape} != {(count, *input_hw)}")
    return {"c2w": c2w, "intrinsics": intrinsics, "depth": depth, "input_hw": input_hw,
            "depth_source": depth_source}


def evaluate_points_case(
    case: str, prediction: Path, gt_root: Path, *, protocol: str, stage: str,
    depth_cache: Path | None = None,
) -> dict[str, Any]:
    if case not in SOURCE_BY_KEY:
        raise ValueError(f"unknown 7-Scenes sequence {case}")
    cv2 = points.cv2_module()
    ids = source_ids(case)
    pred = load_point_prediction(case, prediction, depth_cache)
    gt_c2w, sequence_root = load_gt(gt_root, case, ids)
    depth_paths = gt_depth_paths(gt_root, case, ids)
    pred_c2w = pred["c2w"]
    pred_k_native = scale_intrinsics(pred["intrinsics"], pred["input_hw"])
    pred_chunks: list[np.ndarray] = []
    gt_common_chunks: list[np.ndarray] = []
    gt_full_chunks: list[np.ndarray] = []
    for index in range(len(ids)):
        gt_depth = load_gt_depth(depth_paths[index])
        pred_depth = cv2.resize(
            pred["depth"][index], (NATIVE_HW[1], NATIVE_HW[0]), interpolation=cv2.INTER_NEAREST
        )
        gt_valid = (gt_depth > 1e-4) & np.isfinite(gt_depth)
        pred_valid = (pred_depth > 1e-4) & np.isfinite(pred_depth)
        if not np.isfinite(gt_c2w[index]).all():
            gt_valid[:] = False
        if not np.isfinite(pred_c2w[index]).all():
            pred_valid[:] = False
        common = gt_valid & pred_valid
        gt_full_chunks.append(backproject(gt_depth, NATIVE_INTRINSICS, gt_c2w[index], gt_valid))
        gt_common_chunks.append(backproject(gt_depth, NATIVE_INTRINSICS, gt_c2w[index], common))
        pred_chunks.append(backproject(pred_depth, pred_k_native[index], pred_c2w[index], common))
    pred_common = np.concatenate(pred_chunks)
    gt_common = np.concatenate(gt_common_chunks)
    gt_full = np.concatenate(gt_full_chunks)
    if len(pred_common) < 3 or len(gt_common) != len(pred_common) or len(gt_full) < 3:
        raise ValueError(f"7Scenes/{case}: insufficient reconstruction correspondences")
    sim3 = points.umeyama_registration(pred_common, gt_common)
    pred_sim3 = points.apply_transform(pred_common, sim3)
    pred_voxel = points.voxel_downsample(pred_sim3, VOXEL_SIZE_M)
    gt_voxel = points.voxel_downsample(gt_full, VOXEL_SIZE_M)
    icp = points.icp_registration(pred_voxel, gt_voxel, ICP_THRESHOLD_M)
    pred_eval = points.apply_transform(pred_voxel, icp)
    metrics = points.evaluate_pointcloud(pred_eval, gt_voxel, POINT_THRESHOLDS_M)
    return {
        **metrics,
        "frames": int(len(ids)),
        "sim3_correspondences": int(len(pred_common)),
        "full_gt_points": int(len(gt_full)),
        "pred_voxel_points": int(len(pred_voxel)),
        "gt_voxel_points": int(len(gt_voxel)),
        "prediction_K_native_fx_fy_cx_cy": pred_k_native[0].tolist(),
        "sim3_transform": sim3.tolist(),
        "icp_transform": icp.tolist(),
        "depth_source": pred["depth_source"],
        "gt_dir": str(sequence_root),
    }
