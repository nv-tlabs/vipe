"""KITTI odometry 00-10, Sim(3) ATE RMSE in metres.

Protocol ``evo`` (default) is a port of the long-sequence repo's KITTI/Oxford
trajectory script, which produced the KITTI uncalibrated FP32/FP16 records
(13.245 / 9.013 and 13.883 / 9.635):

* prediction pose key: the first of ``c2w``, ``predicted_c2w``,
  ``c2w_cam0_baseline``, ``c2w_cam0`` present in the NPZ (float64);
  ``frame_ids`` is not used;
* GT: ``c2w_gt`` (float64) of the prepared 280x504 sequence NPZ;
* frame matching: by row, ``n = min(len(pred), len(gt))``, timestamps
  ``0..n-1``;
* alignment: evo ``PoseTrajectory3D.align(correct_scale=True)`` (Umeyama
  Sim(3) on positions) after ``sync.associate_trajectories``;
* metric: evo APE translation part, RMSE;
* average: unweighted mean over the evaluated sequences.

This is the configuration of the shared ``TrajectoryEvaluator(align=True,
correct_scale=True)`` (``sim3_ate``); the result is also computed through that
hash-locked evaluator and reported as ``ate_trajectory_eval`` (cross-check).

Protocol ``umeyama`` reproduces the 2026-08-21 KITTI GT-K mono/stereo safe
loop-closure records: predictions are ``sequences/<NN>/predictions.npz``
(``c2w_cam0_baseline`` for ``--stage local``, ``c2w_cam0`` for ``--stage LC``)
scored with ``sim3_umeyama.compute_pose_metrics`` against
``c2w_gt[:len(pred)]`` (field ``sim3.ate_rmse`` of those records).

GT root: either a flat ``<root>/<NN>.npz`` directory (the MetricTrail
``data/KITTI/prepared_280x504`` layout) or the historical
``<root>/<NN>/attempt-*/inputs/<NN>_<profile>.npz`` tree (one fixed file name
per sequence, matched by the SHA-256 of its ``<profile>.npz`` suffix), where
the last attempt in sorted order is used (as the original script did).
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import aligned as aligned_trajectory
from . import sim3_ate, sim3_umeyama
from .common import text_sha256
from .predictions import find_prediction as _find_generic

NAME = "kitti"
CASES = tuple(f"{index:02d}" for index in range(11))
UNITS = "m"
PROTOCOLS = ("evo", "umeyama")
DEFAULT_PROTOCOL = "evo"
PRED_KEYS = ("c2w", "predicted_c2w", "c2w_cam0_baseline", "c2w_cam0")
UMEYAMA_KEYS = {"local": "c2w_cam0_baseline", "LC": "c2w_cam0"}
# sha256 of the file-name suffix after "<NN>_" in the historical GT tree.
HISTORICAL_GT_SUFFIX_SHA256 = "d162a50d2d605001a28902ab8ca5c127a612f981142d9c64460d4b2e7a2a3bcc"
LOGGER = logging.getLogger("metrictrail_eval.kitti")


def gt_path(gt_root: Path, case: str) -> Path:
    flat = Path(gt_root) / f"{case}.npz"
    if flat.is_file():
        return flat
    prefix = f"{case}_"
    hits = sorted(
        path
        for path in glob.glob(str(Path(gt_root) / case / "attempt-*" / "inputs" / f"{prefix}*.npz"))
        if text_sha256(Path(path).name[len(prefix):]) == HISTORICAL_GT_SUFFIX_SHA256
    )
    if not hits:
        raise FileNotFoundError(f"KITTI {case}: no GT NPZ under {gt_root}")
    return Path(hits[-1])


def load_gt(gt_root: Path, case: str) -> tuple[np.ndarray, Path]:
    path = gt_path(gt_root, case)
    with np.load(path, allow_pickle=False) as archive:
        gt = np.asarray(archive["c2w_gt"], dtype=np.float64)
    return gt, path


def find_prediction(pred: Path, stage: str, case: str, protocol: str) -> Path | None:
    if protocol == "umeyama":
        path = Path(pred) / "sequences" / case / "predictions.npz"
        return path if path.is_file() else None
    return _find_generic(pred, stage, case)


def load_c2w(path: Path) -> tuple[np.ndarray, str]:
    """The original script's load_c2w: first known pose key, float64."""

    with np.load(path, allow_pickle=False) as archive:
        key = next(k for k in PRED_KEYS if k in archive.files)
        c2w = np.asarray(archive[key], dtype=np.float64)
    return c2w, key


def ate_evo(pred_c2w: np.ndarray, gt_c2w: np.ndarray, stamps: np.ndarray) -> float:
    """The original script's ate(): evo Sim(3) align + APE translation RMSE."""

    return _ate_evo(pred_c2w, gt_c2w, stamps)[0]


def _ate_evo(pred_c2w: np.ndarray, gt_c2w: np.ndarray, stamps: np.ndarray) -> tuple[float, Any, Any, Any]:
    """ate_evo's computation; also returns the evo trajectories the APE used and the alignment."""

    from evo.core import metrics, sync
    from evo.core.trajectory import PoseTrajectory3D

    def to_traj(c2w: np.ndarray) -> PoseTrajectory3D:
        return PoseTrajectory3D(
            poses_se3=[np.asarray(m, dtype=np.float64) for m in c2w],
            timestamps=np.asarray(stamps, dtype=np.float64),
        )

    ref = to_traj(gt_c2w)
    est = to_traj(pred_c2w)
    ref, est = sync.associate_trajectories(ref, est)
    alignment = est.align(ref, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    return float(ape.get_statistic(metrics.StatisticsType.rmse)), ref, est, alignment


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
    gt, gt_file = load_gt(gt_root, case)
    if protocol == "umeyama":
        key = UMEYAMA_KEYS[stage]
        with np.load(prediction, allow_pickle=False) as archive:
            pred = np.asarray(archive[key], dtype=np.float64)
        target = gt[: len(pred)]
        values, aligned_c2w, sanitized_target, transform = sim3_umeyama.compute_pose_metrics_aligned(pred, target)
        trajectory = None
        if keep_alignment and aligned_c2w is not None:
            trajectory = aligned_trajectory.from_umeyama(
                aligned_c2w, sanitized_target, transform["rotation"], transform["translation"],
                float(transform["scale"]), gt, "umeyama",
            )
        return {
            "ate": float(values["ate_rmse"]),
            "rpe_trans": float(values["rpe_translation_rmse"]),
            "rpe_rot_deg": float(values["rpe_rotation_deg_rmse"]),
            "frames": int(len(pred)),
            "pose_key": key,
            "gt_file": str(gt_file),
        }, trajectory
    pred, key = load_c2w(prediction)
    n = min(len(pred), len(gt))
    value, ref, est, alignment = _ate_evo(pred[:n], gt[:n], np.arange(n, dtype=np.float64))
    cross = sim3_ate.trajectory_evaluator_class()(align=True, correct_scale=True).evaluate(
        sim3_ate.ArrayTrajectoryLoader(gt[:n]),
        sim3_ate.ArrayTrajectoryLoader(pred[:n]),
        LOGGER,
    )
    trajectory = aligned_trajectory.from_evo(ref, est, alignment, gt, "evo") if keep_alignment else None
    return {
        "ate": value,
        "ate_trajectory_eval": float(cross["ate"]),
        "frames": int(n),
        "pose_key": key,
        "gt_file": str(gt_file),
    }, trajectory


def macro(values: dict[str, float]) -> float:
    return float(np.mean([values[case] for case in values]))
