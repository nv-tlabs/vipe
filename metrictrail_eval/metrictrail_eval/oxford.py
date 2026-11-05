"""Oxford-Spires long, stride 1 (ten scenes, 378x518), Sim(3) ATE RMSE in metres.

Port of the benchmark package's ``oxford_spires_long518`` evaluation path
(``protocol.load_prepared_scene`` + ``evaluate.evaluate_scene`` /
``aggregate``) combined with the long-sequence pipeline's Oxford prediction
contract (``evaluation._oxford_prediction``):

* prediction: ``frame_ids == 0..N-1`` and, when present,
  ``source_processed_ordinals == 0..N-1`` (stride 1), ``source_timestamps``
  equal to the prepared ones, ``w2c == inv(c2w)``; N = the scene's full
  frame count (every stride-1 frame, no filling);
* GT: ``<root>/<scene>/sequence.npz`` ``c2w_gt`` (float64) after the scene's
  attestation chain (``prepare.SUCCESS.json`` / input contract / manifest /
  ``sequence.npz`` / ``rgb_timing.npz`` hashes) and the suite manifest;
* metric: the shared hash-locked ``TrajectoryEvaluator(align=True,
  correct_scale=True)`` (``sim3_ate``) on rows ``0..N-1`` (``ate``, ``rpe_trans``,
  ``rpe_rot``);
* average: unweighted mean over the ten scenes; no all-pairs AUC (disabled by
  the released ``oxford_long`` config).

The FP16/FP32 uncalibrated records (1.776 / 1.756) were scored with the
long-sequence repo's KITTI/Oxford trajectory script (``--gt-kind oxford_npz``; evo
called directly on ``sequence.npz`` ``c2w_gt``, rows ``0..min(N_pred, N_gt)-1``);
for full-length predictions that is the same computation, and the per-scene
values agree bit for bit (EVAL_PARITY.md).  The PNG frame inventory checks of
``load_prepared_scene`` are not repeated here (frames are not ground truth).

``--metric points`` (``evaluate_points_case``) ports the benchmark package's
exact Oxford dense-point replay (``evaluate_exact.py``: its
prediction-to-evaluation-grid step + the reference reconstruction evaluator's
``evaluate_reconstruction``; the CD 0.976 m / F1@4m 95.12 record,
``aggregate.py`` mean over ten scenes):

* metric frames: ``pc/<scene>/frame_ids.txt`` of the points GT root (every
  10th frame ``0, 10, 20, ...``);
* GT: projected full-TLS depth ``pc/<scene>/depth/XXXXXX.npy`` (native
  1080x1440; NaN/Inf and depths outside [0.1, 200] m -> 0), resized to
  388x518 with ``cv2.INTER_NEAREST`` and K scaled per axis with the
  pixel-centre convention (``c' = (c + 0.5) s - 0.5``), unprojected with the
  float32 inverse of ``processed_sequences/<scene>/poses_c2w.txt``; valid =
  depth > 0.1 m, alignment subset = valid & depth <= 80 m;
* prediction: the run's frontend depth cache (``depth.f32`` + ``mask.u8``,
  378x518) on the metric frames, ``mask & finite & d > 1e-4`` (others 0),
  nearest-resized to 388x518 (legacy PyTorch ``nearest`` indices), the
  prediction's sequence-shared K (``intrinsics`` row 0) with its rows scaled by
  518/518 and 388/378, unprojected with ``inv(c2w)`` of the prediction;
  valid = depth > 1e-4; no prediction depth cutoff;
* Umeyama Sim(3) on pixels valid in both and inside the GT alignment subset,
  applied to all common pixels; both clouds (prediction: common pixels; GT:
  every valid pixel) voxel down-sampled at 0.05 m; Open3D point-to-point ICP
  (0.5 m, 20 iterations, 1e-6) from the identity;
* metrics: accuracy, completeness, Chamfer in m; precision / recall / F1 in
  percent at 2 and 4 m (strict ``<``); headline Chamfer and ``f1_4``;
  unweighted mean over the ten scenes.

Building the points GT (projecting the merged TLS maps into every 10th
camera with near 0.1 m / far 200 m, a visibility filter of threshold 2.5 and
radius 20 px, and a zeroed 20 px border) is data preparation and is not part
of this package; ``DONE.json`` of each scene records those parameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import aligned as aligned_trajectory
from . import points, sim3_ate
from .common import sha256_file, text_sha256
from .predictions import find_prediction as _find_generic

NAME = "oxford"
DATASET = "Oxford-Spires"
# sha256 of the preparation profile id in the manifests / markers / sequence.npz.
PROFILE_SHA256 = "a397b59032e779c308e442968164725437dcb0c5a599cd880163d9228128683c"
TARGET_HW = (378, 518)
INPUT_CONTRACT_FILENAME = "oxford_spires_long518_input_contract.json"
EXPECTED_SCENE_FRAMES = {
    "bodleian-library-02": 3840,
    "christ-church-02": 3821,
    "christ-church-03": 3840,
    "christ-church-05": 3840,
    "keble-college-02": 3840,
    "keble-college-03": 3840,
    "keble-college-04": 3840,
    "keble-college-05": 3840,
    "observatory-quarter-01": 3840,
    "observatory-quarter-02": 3839,
}
CASES = tuple(EXPECTED_SCENE_FRAMES)
EXPECTED_TOTAL_FRAMES = sum(EXPECTED_SCENE_FRAMES.values())
UNITS = "m"
PROTOCOLS = ("sim3_ate",)
DEFAULT_PROTOCOL = "sim3_ate"
LOGGER = logging.getLogger("metrictrail_eval.oxford")

# --metric points
POINT_PROTOCOL = "sim3_voxel_icp_projected_tls"
POINT_THRESHOLDS_M = (2.0, 4.0)
POINT_HEADLINE = ("chamfer", "f1_4")
POINT_UNITS = {"distance": "m", "precision_recall_f1": "%"}
POINT_CACHE_DIRS = ("oxford",)
POINT_INTERVAL = 10
POINT_EVAL_WIDTH = 518
POINT_GT_DEPTH_RANGE_M = (0.1, 200.0)
POINT_ALIGNMENT_GT_MAX_M = 80.0
POINT_PRED_MIN_DEPTH_M = 1.0e-4
POINT_VOXEL_M = 0.05
POINT_ICP_THRESHOLD_M = 0.5
POINT_MIN_CORRESPONDENCES = 6


def find_prediction(pred: Path, stage: str, case: str, protocol: str) -> Path | None:
    return _find_generic(pred, stage, case)


def validate_suite(root: Path) -> None:
    manifest_path = root / "manifest.json"
    success_path = root / "prepare.SUCCESS.json"
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(f"Oxford long prepared suite is incomplete: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "success"
        or manifest.get("dataset") != DATASET
        or text_sha256(manifest.get("profile")) != PROFILE_SHA256
        or manifest.get("scenes") != list(CASES)
        or manifest.get("input_hw") != list(TARGET_HW)
        or manifest.get("total_input_frames") != EXPECTED_TOTAL_FRAMES
        or success.get("status") != "success"
        or text_sha256(success.get("profile")) != PROFILE_SHA256
        or success.get("scenes") != list(CASES)
        or success.get("total_input_frames") != EXPECTED_TOTAL_FRAMES
        or success.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError(f"Oxford long prepared suite contract drifted: {root}")


def load_gt(gt_root: Path, scene: str) -> dict[str, Any]:
    """oxford_spires_long518.protocol.load_prepared_scene, without the PNG checks."""

    prepared_root = Path(gt_root).expanduser().resolve()
    validate_suite(prepared_root)
    root = prepared_root / scene
    success_path = root / "prepare.SUCCESS.json"
    contract_path = root / INPUT_CONTRACT_FILENAME
    manifest_path = root / "manifest.json"
    sequence_path = root / "sequence.npz"
    timing_path = root / "rgb_timing.npz"
    intrinsics_path = root / "intrinsics.yaml"
    for path in (success_path, contract_path, manifest_path, sequence_path, timing_path, intrinsics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    success = json.loads(success_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    full_count = EXPECTED_SCENE_FRAMES[scene]
    identities = {
        "contract_sha256": sha256_file(contract_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sequence_npz_sha256": sha256_file(sequence_path),
        "rgb_timing_npz_sha256": sha256_file(timing_path),
        "intrinsics_yaml_sha256": sha256_file(intrinsics_path),
    }
    if (
        success.get("status") != "success"
        or text_sha256(success.get("profile")) != PROFILE_SHA256
        or success.get("scene") != scene
        or success.get("frame_count") != full_count
        or success.get("input_contract_sha256") != identities["contract_sha256"]
        or success.get("manifest_sha256") != identities["manifest_sha256"]
        or success.get("sequence_npz_sha256") != identities["sequence_npz_sha256"]
        or success.get("rgb_timing_npz_sha256") != identities["rgb_timing_npz_sha256"]
        or text_sha256(contract.get("profile")) != PROFILE_SHA256
        or contract.get("scene") != scene
        or contract.get("frame_count") != full_count
        or contract.get("input_hw") != list(TARGET_HW)
        or contract.get("manifest_sha256") != identities["manifest_sha256"]
        or contract.get("sequence_npz_sha256") != identities["sequence_npz_sha256"]
        or contract.get("rgb_timing_npz_sha256") != identities["rgb_timing_npz_sha256"]
        or contract.get("intrinsics_yaml_sha256") != identities["intrinsics_yaml_sha256"]
    ):
        raise ValueError(f"{scene}: Oxford long prepared attestation drifted")
    with np.load(sequence_path, allow_pickle=False) as source:
        frame_ids = np.asarray(source["frame_ids"], dtype=np.int32)
        c2w_gt = np.asarray(source["c2w_gt"], dtype=np.float64)
        timestamps = np.asarray(source["source_timestamps"], dtype=np.float64)
        ordinals = np.asarray(source["source_processed_ordinals"], dtype=np.int32)
        input_hw = tuple(np.asarray(source["input_hw"], dtype=np.int32).tolist())
        stored_scene = str(np.asarray(source["scene"]).item())
        stored_profile = str(np.asarray(source["profile"]).item())
    with np.load(timing_path, allow_pickle=False) as timing:
        timing_frame_ids = np.asarray(timing["frame_ids"], dtype=np.int32)
        timing_timestamps = np.asarray(timing["source_timestamps"], dtype=np.float64)
        timing_ordinals = np.asarray(timing["source_processed_ordinals"], dtype=np.int32)
    expected_ids = np.arange(full_count, dtype=np.int32)
    if (
        frame_ids.shape != (full_count,)
        or not np.array_equal(frame_ids, expected_ids)
        or not np.array_equal(ordinals, expected_ids)
        or contract.get("frame_ids") != expected_ids.tolist()
        or c2w_gt.shape != (full_count, 4, 4)
        or timestamps.shape != (full_count,)
        or input_hw != TARGET_HW
        or stored_scene != scene
        or text_sha256(stored_profile) != PROFILE_SHA256
        or not np.array_equal(timing_frame_ids, frame_ids)
        or not np.array_equal(timing_timestamps, timestamps)
        or not np.array_equal(timing_ordinals, ordinals)
        or not np.isfinite(c2w_gt).all()
        or not np.isfinite(timestamps).all()
    ):
        raise ValueError(f"{scene}: Oxford long prepared array contract drifted")
    return {
        "count": full_count,
        "frame_ids": frame_ids,
        "c2w_gt": c2w_gt,
        "source_timestamps": timestamps,
        "source_processed_ordinals": ordinals,
        "sequence_path": sequence_path,
    }


def load_prediction(path: Path, prepared: dict[str, Any], scene: str) -> np.ndarray:
    count = int(prepared["count"])
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        if not {"frame_ids", "c2w"}.issubset(files):
            raise ValueError(f"{scene}: prediction needs frame_ids and c2w")
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int32)
        pred_c2w = np.asarray(archive["c2w"], dtype=np.float64)
        pred_w2c = np.asarray(archive["w2c"], dtype=np.float64) if "w2c" in files else None
        ordinals = (
            np.asarray(archive["source_processed_ordinals"], dtype=np.int32)
            if "source_processed_ordinals" in files
            else None
        )
        timestamps = (
            np.asarray(archive["source_timestamps"], dtype=np.float64)
            if "source_timestamps" in files
            else None
        )
        input_hw = (
            tuple(np.asarray(archive["input_hw"], dtype=np.int32).tolist())
            if "input_hw" in files
            else TARGET_HW
        )
    expected = np.arange(count, dtype=np.int32)
    if (
        not np.array_equal(frame_ids, expected)
        or not np.array_equal(frame_ids, prepared["frame_ids"])
        or pred_c2w.shape != (count, 4, 4)
        or not np.isfinite(pred_c2w).all()
        or input_hw != TARGET_HW
        or (ordinals is not None and not np.array_equal(ordinals, expected))
        or (ordinals is not None and not np.array_equal(ordinals, prepared["source_processed_ordinals"]))
        or (timestamps is not None and not np.array_equal(timestamps, prepared["source_timestamps"]))
        or (
            pred_w2c is not None
            and (
                pred_w2c.shape != (count, 4, 4)
                or not np.isfinite(pred_w2c).all()
                or not np.allclose(pred_c2w @ pred_w2c, np.eye(4)[None], rtol=1.0e-5, atol=1.0e-6)
            )
        )
    ):
        raise ValueError(f"{scene}: Oxford prediction contract drifted")
    return pred_c2w


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
    if case not in EXPECTED_SCENE_FRAMES:
        raise ValueError(f"unknown Oxford scene {case}")
    prepared = load_gt(gt_root, case)
    pred_c2w = load_prediction(prediction, prepared, case)
    raw = sim3_ate.trajectory_evaluator_class()(align=True, correct_scale=True).evaluate(
        sim3_ate.ArrayTrajectoryLoader(prepared["c2w_gt"]),
        sim3_ate.ArrayTrajectoryLoader(pred_c2w),
        LOGGER,
    )
    transform = np.asarray(raw.pop("traj_transform"), dtype=np.float64)
    trajectory = (
        aligned_trajectory.from_shared_evaluator(prepared["c2w_gt"], pred_c2w, None, transform)
        if keep_alignment else None
    )
    return {
        **{key: float(raw[key]) for key in ("ate", "rpe_trans", "rpe_rot")},
        "frames": int(len(pred_c2w)),
        "alignment_scale": float(np.cbrt(np.linalg.det(transform[:3, :3]))),
        "gt_file": str(prepared["sequence_path"]),
    }, trajectory


def macro(values: dict[str, float]) -> float:
    return float(np.mean([values[case] for case in values]))


# --------------------------------------------------------------------------- points


def _matrix_rows(path: Path, columns: int) -> np.ndarray:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, got {rows.shape}")
    return rows


def resize_gt_depth_and_intrinsics(
    depth: np.ndarray, intrinsic: np.ndarray, output_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """``cv2.INTER_NEAREST`` resize to ``output_width``; K scaled with the pixel-centre convention."""

    src_h, src_w = depth.shape
    if output_width <= 0 or output_width == src_w:
        return np.asarray(depth, dtype=np.float32), np.asarray(intrinsic, dtype=np.float32)
    cv2 = points.cv2_module()
    dst_w = int(output_width)
    dst_h = max(1, int(round(src_h * dst_w / float(src_w))))
    resized = cv2.resize(depth, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    sx, sy = dst_w / float(src_w), dst_h / float(src_h)
    K = np.asarray(intrinsic, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return resized.astype(np.float32, copy=False), K.astype(np.float32)


def load_point_gt(points_root: Path, scene: str) -> dict[str, Any]:
    """Projected-TLS GT of the metric frames on the 388x518 evaluation grid."""

    root = Path(points_root).expanduser().resolve()
    processed = root / "processed_sequences" / scene
    gt_dir = root / "pc" / scene
    count = EXPECTED_SCENE_FRAMES[scene]
    poses_c2w = _matrix_rows(processed / "poses_c2w.txt", 16).reshape(-1, 4, 4)
    intrinsic_values = np.loadtxt(processed / "intrinsics.txt", dtype=np.float64).reshape(-1)
    if len(intrinsic_values) != 6:
        raise ValueError(f"{scene}: intrinsics.txt must contain fx fy cx cy W H")
    fx, fy, cx, cy, width, height = intrinsic_values
    done = json.loads((gt_dir / "DONE.json").read_text(encoding="utf-8"))
    frame_ids = np.atleast_1d(np.loadtxt(gt_dir / "frame_ids.txt", dtype=np.int64)).astype(np.int64)
    sampled_poses = _matrix_rows(gt_dir / "poses_c2w.txt", 16).reshape(-1, 4, 4)
    if (
        len(poses_c2w) != count
        or int(done["source_frame_count"]) != count
        or len(frame_ids) != len(sampled_poses)
        or len(frame_ids) == 0
        or np.any(frame_ids < 0)
        or np.any(frame_ids >= count)
        or not np.all(np.diff(frame_ids) == POINT_INTERVAL)
        or (int(width), int(height)) != (int(done["width"]), int(done["height"]))
    ):
        raise ValueError(f"{scene}: points GT inventory does not match the scene")
    np.testing.assert_allclose(
        sampled_poses, poses_c2w[frame_ids], rtol=1e-5, atol=1e-5,
        err_msg=f"{scene}: sampled GT poses do not match processed poses",
    )
    native_hw = (int(height), int(width))
    base_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    depths: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    low, high = POINT_GT_DEPTH_RANGE_M
    for frame_id in frame_ids.tolist():
        path = gt_dir / "depth" / f"{frame_id:06d}.npy"
        depth = np.load(path).astype(np.float32, copy=False)
        if depth.shape != native_hw:
            raise ValueError(f"{scene} frame {frame_id}: depth {depth.shape} != {native_hw}")
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth < low) | (depth > high)] = 0.0
        depth, K = resize_gt_depth_and_intrinsics(depth, base_k, POINT_EVAL_WIDTH)
        depths.append(depth)
        intrinsics.append(K)
    depths_array = np.stack(depths, axis=0)
    intrinsics_array = np.stack(intrinsics, axis=0)
    metric_w2c = np.linalg.inv(poses_c2w[frame_ids]).astype(np.float32)
    world = points.unproject_depth_maps(depths_array, metric_w2c[:, :3, :], intrinsics_array)
    alignment_max = min(POINT_ALIGNMENT_GT_MAX_M, high)
    return {
        "frame_ids": frame_ids,
        "points": world.astype(np.float32, copy=False),
        "valid": depths_array > low,
        "alignment": (depths_array > low) & (depths_array <= alignment_max),
        "hw": tuple(depths_array.shape[-2:]),
        "gt_dir": gt_dir,
    }


def load_point_prediction(
    scene: str, prediction: Path, frame_ids: np.ndarray, target_hw: tuple[int, int],
    depth_cache: Path | None,
) -> dict[str, Any]:
    count = EXPECTED_SCENE_FRAMES[scene]
    with np.load(prediction, allow_pickle=False) as archive:
        files = set(archive.files)
        ids = np.asarray(archive["frame_ids"], dtype=np.int32)
        c2w = np.asarray(archive["c2w"], dtype=np.float64)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
        input_hw = tuple(np.asarray(archive["input_hw"], dtype=np.int32).tolist()) if "input_hw" in files else TARGET_HW
    if (
        not np.array_equal(ids, np.arange(count, dtype=np.int32))
        or c2w.shape != (count, 4, 4)
        or not np.isfinite(c2w).all()
        or intrinsics.shape != (count, 4)
        or not np.array_equal(intrinsics, np.broadcast_to(intrinsics[:1], intrinsics.shape))
        or input_hw != TARGET_HW
    ):
        raise ValueError(f"{scene}: point prediction contract failed (full scene, shared K, {TARGET_HW})")
    cache = points.find_depth_cache(prediction, scene, POINT_CACHE_DIRS, depth_cache)
    if cache is None:
        raise FileNotFoundError(
            f"{scene}: no frontend depth cache found for {prediction} "
            "(use --depth-cache <root holding <scene>/depth.f32>)"
        )
    depth_memmap, mask_memmap = points.open_depth_cache(cache, count, TARGET_HW)
    depth = np.asarray(depth_memmap[frame_ids], dtype=np.float32)
    cached_mask = np.asarray(mask_memmap[frame_ids] != 0, dtype=bool)
    depth = np.where(cached_mask & np.isfinite(depth) & (depth > POINT_PRED_MIN_DEPTH_M), depth, 0.0)
    resized = points.resize_nearest_rows_like_torch(depth, target_hw).astype(np.float32, copy=False)
    fx, fy, cx, cy = (float(value) for value in intrinsics[0])
    base_k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    scaled_k = base_k.copy()
    scaled_k[0, :] *= target_hw[1] / float(TARGET_HW[1])
    scaled_k[1, :] *= target_hw[0] / float(TARGET_HW[0])
    k_stack = np.broadcast_to(scaled_k, (len(frame_ids), 3, 3)).copy()
    w2c = np.linalg.inv(np.asarray(c2w[frame_ids], dtype=np.float64))
    world = points.unproject_depth_maps(resized, w2c[:, :3, :], k_stack)
    return {
        "points": world,
        "valid": resized > POINT_PRED_MIN_DEPTH_M,
        "k_fx_fy_cx_cy": [fx, fy, cx, cy],
        "scaled_k": scaled_k.tolist(),
        "depth_cache": str(cache),
    }


def evaluate_points_case(
    case: str, prediction: Path, gt_root: Path, *, protocol: str, stage: str,
    depth_cache: Path | None = None,
) -> dict[str, Any]:
    if case not in EXPECTED_SCENE_FRAMES:
        raise ValueError(f"unknown Oxford scene {case}")
    gt = load_point_gt(gt_root, case)
    pred = load_point_prediction(case, prediction, gt["frame_ids"], gt["hw"], depth_cache)
    pred_pts, gt_pts = pred["points"], gt["points"]
    gt_mask, pred_mask = gt["valid"], pred["valid"]
    if pred_pts.shape != gt_pts.shape:
        raise ValueError(f"{case}: prediction/GT point grids differ: {pred_pts.shape} vs {gt_pts.shape}")
    common = gt_mask & pred_mask
    alignment_common = common & gt["alignment"] & pred_mask
    if int(alignment_common.sum()) < POINT_MIN_CORRESPONDENCES:
        raise ValueError(f"{case}: insufficient Umeyama correspondences")
    T_umeyama = points.umeyama_registration(
        pred_pts[alignment_common][:, :3], gt_pts[alignment_common][:, :3]
    )
    pred_after_umeyama = points.apply_transform(
        pred_pts[common][:, :3], T_umeyama, homogeneous_divide=False
    )
    gt_full = gt_pts[gt_mask][:, :3]
    pred_ds = points.voxel_downsample(pred_after_umeyama, POINT_VOXEL_M)
    gt_ds = points.voxel_downsample(gt_full, POINT_VOXEL_M)
    T_icp = points.icp_registration(pred_ds, gt_ds, POINT_ICP_THRESHOLD_M)
    pred_eval = points.apply_transform(pred_ds, T_icp, homogeneous_divide=False)
    metrics = points.evaluate_pointcloud(pred_eval, gt_ds, POINT_THRESHOLDS_M)
    return {
        **metrics,
        "metric_frames": int(len(gt["frame_ids"])),
        "num_pred": int(len(pred_eval)),
        "num_gt": int(len(gt_ds)),
        "num_correspondences": int(common.sum()),
        "num_alignment_correspondences": int(alignment_common.sum()),
        "valid_prediction_pixels": int(pred_mask.sum()),
        "prediction_K_fx_fy_cx_cy": pred["k_fx_fy_cx_cy"],
        "T_umeyama": T_umeyama.tolist(),
        "T_icp": T_icp.tolist(),
        "depth_source": pred["depth_cache"],
        "gt_dir": str(gt["gt_dir"]),
    }
