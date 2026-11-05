"""Closed-form Umeyama Sim(3) ATE/RPE (no evo).

Verbatim port of ``compute_pose_metrics`` and its helpers from the benchmark
package's ``bench/metrics.py`` (its body is ``compute_pose_metrics_aligned``,
which also returns the aligned trajectory for plotting; the arithmetic is
unchanged).  This is the evaluator that wrote the
2026-08-21 KITTI GT-K mono/stereo safe-loop-closure records
(``local_sim3_ate_rmse_m`` / ``safe_lc_sim3_ate_rmse_m``); it is used only by
``--protocol umeyama`` of the KITTI scorer.

Differences to the evo path: every rotation (prediction and GT) is projected to
the nearest SO(3) and the bottom row is reset to (0, 0, 0, 1) before
alignment; the Umeyama solve is written out with NumPy.
"""

from __future__ import annotations

import math

import numpy as np


def to_homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Poses must be [S,3,4] or [S,4,4], got {poses.shape}")
    if poses.shape[-2:] == (4, 4):
        return poses.copy()
    result = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    result[:, :3, :4] = poses
    return result


def nearest_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    return u @ correction @ vt


def sanitize_c2w(poses: np.ndarray) -> np.ndarray:
    poses = to_homogeneous(poses)
    if not np.isfinite(poses).all():
        raise ValueError("Camera poses contain non-finite values")
    for index in range(len(poses)):
        poses[index, :3, :3] = nearest_so3(poses[index, :3, :3])
        poses[index, 3] = (0.0, 0.0, 0.0, 1.0)
    return poses


def umeyama_sim3(
    prediction_xyz: np.ndarray, target_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares proper Sim(3) mapping ``prediction_xyz`` to ``target_xyz``."""
    prediction_xyz = np.asarray(prediction_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if (
        prediction_xyz.shape != target_xyz.shape
        or prediction_xyz.ndim != 2
        or prediction_xyz.shape[1] != 3
    ):
        raise ValueError(
            f"Umeyama inputs must both be [N,3], got {prediction_xyz.shape}, {target_xyz.shape}"
        )
    if len(prediction_xyz) < 3:
        raise ValueError("Umeyama Sim(3) requires at least three camera centers")
    pred_mean = prediction_xyz.mean(axis=0)
    target_mean = target_xyz.mean(axis=0)
    pred_centered = prediction_xyz - pred_mean
    target_centered = target_xyz - target_mean
    variance = float(np.mean(np.sum(pred_centered**2, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Degenerate predicted camera trajectory")
    covariance = target_centered.T @ pred_centered / len(prediction_xyz)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = target_mean - scale * (rotation @ pred_mean)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Invalid Umeyama scale {scale}")
    return rotation, translation, scale


def align_c2w_sim3(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    prediction = sanitize_c2w(prediction)
    target = sanitize_c2w(target)
    if prediction.shape != target.shape:
        raise ValueError(f"Pose shape mismatch: pred={prediction.shape}, gt={target.shape}")
    rotation, translation, scale = umeyama_sim3(prediction[:, :3, 3], target[:, :3, 3])
    aligned = prediction.copy()
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rotation, prediction[:, :3, :3])
    aligned[:, :3, 3] = (
        scale * np.einsum("ij,nj->ni", rotation, prediction[:, :3, 3]) + translation
    )
    return aligned, {"rotation": rotation, "translation": translation, "scale": scale}


def _rmse(values: np.ndarray) -> float:
    return (
        float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))
        if len(values)
        else math.nan
    )


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rpe_errors(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation_errors = []
    rotation_errors = []
    for index in range(len(prediction) - 1):
        pred_relative = np.linalg.inv(prediction[index]) @ prediction[index + 1]
        target_relative = np.linalg.inv(target[index]) @ target[index + 1]
        error = np.linalg.inv(target_relative) @ pred_relative
        translation_errors.append(np.linalg.norm(error[:3, 3]))
        rotation_errors.append(_rotation_angle_degrees(error[:3, :3]))
    return np.asarray(translation_errors), np.asarray(rotation_errors)


def compute_pose_metrics(
    prediction_c2w: np.ndarray,
    target_c2w: np.ndarray,
) -> dict[str, float | str]:
    """Compute one-Sim3 ATE and consecutive-frame translation/rotation RPE."""
    return compute_pose_metrics_aligned(prediction_c2w, target_c2w)[0]


def compute_pose_metrics_aligned(
    prediction_c2w: np.ndarray,
    target_c2w: np.ndarray,
) -> tuple[dict[str, float | str], np.ndarray | None, np.ndarray, dict[str, np.ndarray | float] | None]:
    """``compute_pose_metrics`` plus the aligned prediction, the sanitised target and the transform."""
    prediction = sanitize_c2w(prediction_c2w)
    target = sanitize_c2w(target_c2w)
    if prediction.shape != target.shape:
        raise ValueError(f"Pose shape mismatch: pred={prediction.shape}, gt={target.shape}")
    if len(prediction) < 3:
        return {
            "pose_status": "unavailable_too_few_frames",
            "ate_rmse": math.nan,
            "rpe_translation_rmse": math.nan,
            "rpe_rotation_deg_rmse": math.nan,
        }, None, target, None
    aligned, transform = align_c2w_sim3(prediction, target)
    position_error = np.linalg.norm(aligned[:, :3, 3] - target[:, :3, 3], axis=1)
    rpe_translation, rpe_rotation = _rpe_errors(aligned, target)
    return {
        "ate_rmse": _rmse(position_error),
        "rpe_translation_rmse": _rmse(rpe_translation),
        "rpe_rotation_deg_rmse": _rmse(rpe_rotation),
    }, aligned, target, transform
