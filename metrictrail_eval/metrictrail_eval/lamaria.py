"""LaMaria (five training sequences, Aria cam0), Sim(3) ATE RMSE in metres.

Port of the long-sequence pipeline's LaMaria evaluation, i.e. the benchmark
package's LaMaria protocol module ``vtl_protocol.py`` (``lamaria_rgb_case``
timestamps + ``load_lamaria_evaluation_gt``) and the ``compute_pose_metrics``
of its LaMaria run script:

* frames: every row of ``asl_folder/<seq>/aria/cam0/data.csv`` (strictly
  increasing ns timestamps, fixed count per sequence);
* GT: the first N rows of ``ground_truth/pGT/<seq>.txt``
  (``t_ns tx ty tz qx qy qz qw``, imu/body pose) whose rounded timestamps must
  equal the cam0 timestamps, right-multiplied by the rigid ``cam0.T_b_s`` of
  ``pinhole_calibrations/<seq>.json`` (the only calibration field read; it is
  used for evaluation only);
* prediction: ``c2w`` (float64, finite), same shape as GT;
* metric: rotations of both trajectories projected to SO(3); closed-form
  Umeyama Sim(3) on camera centres; ``ate_rmse_m`` = RMSE of centre errors;
  delta-1 RPE on the aligned trajectory;
* average: unweighted mean over the five sequences.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from . import aligned as aligned_trajectory
from .predictions import find_prediction as _find_generic

NAME = "lamaria"
LAMARIA_COUNTS = {
    "R_01_easy": 2898,
    "R_04_medium": 5253,
    "R_06_medium": 7808,
    "R_07_medium": 8010,
    "R_08_hard": 12328,
}
CASES = tuple(LAMARIA_COUNTS)
UNITS = "m"
PROTOCOLS = ("umeyama",)
DEFAULT_PROTOCOL = "umeyama"


def find_prediction(pred: Path, stage: str, case: str, protocol: str) -> Path | None:
    return _find_generic(pred, stage, case)


def cam0_timestamps(data_root: Path, case: str) -> np.ndarray:
    """lamaria_rgb_case: the cam0 inventory from data.csv (timestamps only)."""

    expected = LAMARIA_COUNTS[case]
    root = Path(data_root).expanduser().resolve() / "training" / case
    csv_path = root / "asl_folder" / case / "aria" / "cam0" / "data.csv"
    rows: list[tuple[int, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError(f"invalid LaMARia cam0 CSV row: {row}")
            rows.append((int(row[0]), row[1]))
    if len(rows) != expected or any(rows[i][0] >= rows[i + 1][0] for i in range(len(rows) - 1)):
        raise ValueError(f"LaMARia/{case} RGB inventory differs")
    for timestamp, name in rows:
        if name != f"{timestamp}.png":
            raise ValueError(f"LaMARia/{case}: cam0 row {name} does not match {timestamp}")
    return np.asarray([row[0] for row in rows], dtype=np.int64)


def _make_pose(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    result[:3, 3] = translation
    return result


def load_gt(data_root: Path, case: str) -> tuple[np.ndarray, Path]:
    """vtl_protocol.load_lamaria_evaluation_gt."""

    selected_timestamps = cam0_timestamps(data_root, case)
    root = Path(data_root).expanduser().resolve() / "training" / case
    pgt_path = root / "ground_truth" / "pGT" / f"{case}.txt"
    calibration_path = root / "pinhole_calibrations" / f"{case}.json"
    pose_data = np.loadtxt(pgt_path, dtype=np.float64, ndmin=2)
    if pose_data.shape != (LAMARIA_COUNTS[case], 8) or not np.isfinite(pose_data).all():
        raise ValueError(f"invalid LaMARia pGT: {pgt_path}")
    timestamps = np.rint(pose_data[:, 0]).astype(np.int64)
    if not np.array_equal(timestamps[: len(selected_timestamps)], selected_timestamps):
        raise ValueError("LaMARia evaluation timestamps differ from RGB inventory")
    # Only the rigid cam0 extrinsic T_b_s is read from the calibration file.
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    transform = calibration["cam0"]["T_b_s"]
    imu_from_cam = _make_pose(
        np.asarray(transform["tvec"], dtype=np.float64),
        np.asarray(transform["qvec"], dtype=np.float64),
    )
    gt = np.stack(
        [
            _make_pose(row[1:4], row[4:8]) @ imu_from_cam
            for row in pose_data[: len(selected_timestamps)]
        ]
    )
    return gt, pgt_path


def _nearest_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return u @ correction @ vt


def _umeyama_sim3(
    prediction_xyz: np.ndarray, target_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    prediction_xyz = np.asarray(prediction_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if (
        prediction_xyz.shape != target_xyz.shape
        or prediction_xyz.ndim != 2
        or prediction_xyz.shape[1] != 3
    ):
        raise ValueError("Umeyama inputs must be matching [N,3] arrays")
    if len(prediction_xyz) < 3:
        raise ValueError("Umeyama alignment requires at least three camera centres")
    pred_mean = prediction_xyz.mean(axis=0)
    target_mean = target_xyz.mean(axis=0)
    pred_centered = prediction_xyz - pred_mean
    target_centered = target_xyz - target_mean
    covariance = target_centered.T @ pred_centered / len(prediction_xyz)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(pred_centered**2, axis=1))
    if variance <= 1e-12:
        raise ValueError("Predicted trajectory has degenerate translation variance")
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = target_mean - scale * (rotation @ pred_mean)
    return rotation, translation, scale


def compute_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    """The LaMaria run script's compute_pose_metrics (full-sequence Sim(3) ATE + delta-1 RPE)."""

    return _compute_pose_metrics(pred_c2w, gt_c2w)[0]


def _compute_pose_metrics(
    pred_c2w: np.ndarray, gt_c2w: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
    """compute_pose_metrics' computation; also returns the aligned prediction, the target and the Sim(3)."""

    prediction = np.asarray(pred_c2w, dtype=np.float64).copy()
    target = np.asarray(gt_c2w, dtype=np.float64).copy()
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (4, 4):
        raise ValueError(f"Pose arrays must match [N,4,4], got {prediction.shape}, {target.shape}")
    if len(prediction) < 3 or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Pose metrics require at least three finite poses")
    for index in range(len(prediction)):
        prediction[index, :3, :3] = _nearest_so3(prediction[index, :3, :3])
        target[index, :3, :3] = _nearest_so3(target[index, :3, :3])
    rotation, translation, scale = _umeyama_sim3(prediction[:, :3, 3], target[:, :3, 3])
    aligned = prediction.copy()
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rotation, prediction[:, :3, :3])
    aligned[:, :3, 3] = scale * (prediction[:, :3, 3] @ rotation.T) + translation
    position_error = np.linalg.norm(aligned[:, :3, 3] - target[:, :3, 3], axis=1)
    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    for index in range(len(aligned) - 1):
        gt_delta = np.linalg.inv(target[index]) @ target[index + 1]
        pred_delta = np.linalg.inv(aligned[index]) @ aligned[index + 1]
        error = np.linalg.inv(gt_delta) @ pred_delta
        rpe_translation.append(float(np.linalg.norm(error[:3, 3])))
        cosine = np.clip((np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rpe_rotation.append(float(np.degrees(np.arccos(cosine))))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(rpe_translation)))),
        "rpe_rotation_deg_rmse": float(np.sqrt(np.mean(np.square(rpe_rotation)))),
        "sim3_scale_pred_to_gt": scale,
    }, aligned, target, (rotation, translation, scale)


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
    if case not in LAMARIA_COUNTS:
        raise ValueError(f"unknown LaMaria sequence {case}")
    gt, pgt_path = load_gt(gt_root, case)
    with np.load(prediction, allow_pickle=False) as archive:
        pred = np.asarray(archive["c2w"], dtype=np.float64)
    if pred.ndim != 3 or pred.shape[1:] != (4, 4) or not np.isfinite(pred).all():
        raise ValueError(f"invalid prediction for evaluation: {prediction}")
    if pred.shape != gt.shape:
        raise ValueError("LaMARia prediction/GT counts differ")
    values, aligned_c2w, target, (rotation, translation, scale) = _compute_pose_metrics(pred, gt)
    trajectory = (
        aligned_trajectory.from_umeyama(aligned_c2w, target, rotation, translation, scale, gt, "umeyama")
        if keep_alignment else None
    )
    return {
        "ate": values["ate_rmse_m"],
        "rpe_trans": values["rpe_translation_rmse_m"],
        "rpe_rot": values["rpe_rotation_deg_rmse"],
        "alignment_scale": values["sim3_scale_pred_to_gt"],
        "frames": int(len(pred)),
        "gt_file": str(pgt_path),
    }, trajectory


def macro(values: dict[str, float]) -> float:
    return float(np.mean([values[case] for case in values]))
