"""Wild-SLAM MoCap (ten sequences, every frame), Sim(3) ATE RMSE in centimetres.

Port of the Wild-SLAM experiment's ``evaluate_wild_slam.py`` (the 2.70 cm
uncalibrated / 1.65 cm calibrated records), whose primary metric mirrors the
third-party Wild-SLAM evaluation utility ``vendored/wild_slam_eval.py``
(vendored verbatim, Apache-2.0, ``licenses/LICENSE.wild_slam_eval``):

* GT: ``<root>/<seq>/sequence.npz`` ``c2w_gt`` (float64) with rows where
  ``gt_valid`` is false set to NaN; the manifest must be a successful
  preparation of the fixed native 720x1280 undistorted-frames profile
  (checked by the SHA-256 of its profile id);
* prediction: ``frame_ids == 0..N-1`` (no dropping, no filling), ``c2w``
  finite with homogeneous rows ``[0,0,0,1]`` (atol 1e-6), proper and
  near-orthonormal rotations; ``w2c == inv(c2w)`` when present;
* primary metric (``wild_slam_eval.align_full_traj`` lines 52-75 +
  ``traj_eval_and_plot`` APE lines 78-91): skip frames whose GT pose sums to
  NaN/Inf, evo ``PoseTrajectory3D`` with timestamps = frame index,
  ``sync.associate_trajectories``, ``align(correct_scale=True)``, APE
  translation part RMSE; reported x100 in cm (``ate``);
* cross-check: the shared hash-locked ``TrajectoryEvaluator`` (``sim3_ate``);
  the two must agree within 1e-9 m or the sequence fails;
* average: unweighted mean of the per-sequence cm values (the dataset's
  reference summary script: ``np.nanmean``; tables print 2 decimals).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import aligned as aligned_trajectory
from . import sim3_ate
from .common import text_sha256
from .predictions import find_prediction as _find_generic

NAME = "wild_slam"
# sha256 of the preparation profile id in manifest.json.
PROFILE_SHA256 = "2fe4040086c39d9f11124e7d46cd0e4a76756cb0186668a67241fecda24029c2"
CASES = (
    "ANYmal1",
    "ANYmal2",
    "ball",
    "crowd",
    "person_tracking",
    "racket",
    "stones",
    "table_tracking1",
    "table_tracking2",
    "umbrella",
)
SUCCESS_MARKER = "SUCCESS"
UNITS = "cm"
PROTOCOLS = ("evo",)
DEFAULT_PROTOCOL = "evo"
CROSSCHECK_ATOL_M = 1e-9
HOMOGENEOUS_ATOL = 1e-6
ROTATION_ORTHONORMAL_ATOL = 5e-2
ROTATION_MIN_DET = 0.5
LOGGER = logging.getLogger("metrictrail_eval.wild_slam")


def find_prediction(pred: Path, stage: str, case: str, protocol: str) -> Path | None:
    return _find_generic(pred, stage, case)


def validate_c2w(values: Any, count: int | None = None) -> np.ndarray:
    c2w = np.asarray(values, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w must be [N,4,4], got {c2w.shape}")
    if count is not None and c2w.shape[0] != count:
        raise ValueError(f"c2w has {c2w.shape[0]} poses, expected {count}")
    if not np.isfinite(c2w).all():
        raise ValueError("c2w contains non-finite values")
    if not np.allclose(c2w[:, 3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=HOMOGENEOUS_ATOL):
        raise ValueError("c2w homogeneous rows are not [0,0,0,1]")
    rotations = c2w[:, :3, :3]
    determinants = np.linalg.det(rotations)
    if np.any(determinants < ROTATION_MIN_DET):
        raise ValueError(f"c2w rotations are not proper (min det {determinants.min():.6g})")
    gram = np.einsum("nij,nkj->nik", rotations, rotations) - np.eye(3)
    if float(np.abs(gram).max()) > ROTATION_ORTHONORMAL_ATOL:
        raise ValueError("c2w rotations are not orthonormal")
    return c2w


def load_gt(inputs_root: Path, seq: str) -> dict[str, Any]:
    root = Path(inputs_root) / seq
    manifest_path = root / "manifest.json"
    npz_path = root / "sequence.npz"
    if not manifest_path.is_file() or not npz_path.is_file() or not (root / SUCCESS_MARKER).is_file():
        raise FileNotFoundError(f"{seq}: prepared inputs incomplete under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "success"
        or text_sha256(manifest.get("profile")) != PROFILE_SHA256
        or manifest.get("sequence") != seq
    ):
        raise ValueError(
            f"{seq}: manifest is not a successful preparation of the expected profile "
            f"(got {manifest.get('profile')!r})"
        )
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("frame_ids", "c2w_gt", "gt_valid")}
    count = int(len(arrays["frame_ids"]))
    if (
        not np.array_equal(arrays["frame_ids"], np.arange(count, dtype=np.int32))
        or arrays["c2w_gt"].shape != (count, 4, 4)
        or arrays["gt_valid"].shape != (count,)
        or int(manifest["num_frames"]) != count
    ):
        raise ValueError(f"{seq}: sequence.npz is inconsistent")
    return {"count": count, "npz_path": npz_path, **arrays}


def full_traj_ate(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, Any]:
    """Mirror of wild_slam_eval.align_full_traj + traj_eval_and_plot (APE part, no plotting)."""

    return _full_traj_ate(pred_c2w, gt_c2w)[0]


def _full_traj_ate(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> tuple[dict[str, Any], Any, Any, Any]:
    """full_traj_ate's computation; also returns the evo trajectories the APE used and the alignment."""

    from evo.core import metrics, sync
    from evo.core.trajectory import PoseTrajectory3D

    timestamps: list[float] = []
    traj_ref: list[np.ndarray] = []
    traj_est: list[np.ndarray] = []
    for i in range(len(gt_c2w)):  # wild_slam_eval.py:57-64
        val = gt_c2w[i].sum()
        if np.isnan(val) or np.isinf(val):
            continue
        traj_est.append(pred_c2w[i])
        traj_ref.append(gt_c2w[i])
        timestamps.append(float(i))
    traj_est_t = PoseTrajectory3D(poses_se3=traj_est, timestamps=timestamps)  # wild_slam_eval.py:68
    traj_ref_t = PoseTrajectory3D(poses_se3=traj_ref, timestamps=timestamps)  # wild_slam_eval.py:69
    traj_ref_t, traj_est_t = sync.associate_trajectories(traj_ref_t, traj_est_t)  # :73
    _r_a, _t_a, s = traj_est_t.align(traj_ref_t, correct_scale=True)  # wild_slam_eval.py:74
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)  # wild_slam_eval.py:89
    ape_metric.process_data((traj_ref_t, traj_est_t))  # wild_slam_eval.py:88,90
    ape_statistics = ape_metric.get_all_statistics()  # wild_slam_eval.py:91
    return {
        "ate_rmse_m": float(ape_statistics["rmse"]),
        "alignment_scale": float(s),
        "num_pairs": int(len(timestamps)),
    }, traj_ref_t, traj_est_t, (_r_a, _t_a, s)


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
    if case not in CASES:
        raise ValueError(f"unknown Wild-SLAM sequence {case}")
    prepared = load_gt(gt_root, case)
    count = prepared["count"]
    gt_c2w = np.asarray(prepared["c2w_gt"], dtype=np.float64)
    gt_valid = np.asarray(prepared["gt_valid"], dtype=bool)
    gt_c2w = np.where(gt_valid[:, None, None], gt_c2w, np.nan)
    with np.load(prediction, allow_pickle=False) as archive:
        files = set(archive.files)
        if not {"frame_ids", "c2w"}.issubset(files):
            raise ValueError(f"{prediction}: prediction needs frame_ids and c2w")
        pred_c2w = validate_c2w(archive["c2w"], count)
        ids = np.asarray(archive["frame_ids"])
        if ids.shape != (count,) or not np.array_equal(ids.astype(np.int64), np.arange(count)):
            raise ValueError(f"{prediction}: frame_ids must equal arange({count})")
        for key in ("scene", "sequence"):
            if key in files and str(np.asarray(archive[key]).item()) != case:
                raise ValueError(f"{prediction}: {key} label is not {case!r}")
        if "w2c" in files:
            w2c = np.asarray(archive["w2c"], dtype=np.float64)
            if not np.allclose(w2c, np.linalg.inv(pred_c2w), rtol=1e-5, atol=1e-6):
                raise ValueError(f"{prediction}: w2c != inv(c2w)")
    primary, traj_ref, traj_est, alignment = _full_traj_ate(pred_c2w, gt_c2w)
    raw = sim3_ate.trajectory_evaluator_class()(align=True, correct_scale=True).evaluate(
        sim3_ate.ArrayTrajectoryLoader(gt_c2w), sim3_ate.ArrayTrajectoryLoader(pred_c2w), LOGGER
    )
    diff = abs(primary["ate_rmse_m"] - float(raw["ate"]))
    if diff > CROSSCHECK_ATOL_M:
        raise ValueError(f"{case}: primary evo ATE and shared-evaluator ATE differ by {diff} m")
    if not math.isfinite(primary["ate_rmse_m"]):
        raise ValueError(f"{case}: non-finite ATE")
    return {
        "ate": primary["ate_rmse_m"] * 1e2,
        "ate_2dp": f"{primary['ate_rmse_m'] * 1e2:.2f}",
        "ate_m": primary["ate_rmse_m"],
        "ate_trajectory_eval_cm": float(raw["ate"]) * 1e2,
        "crosscheck_abs_diff_m": diff,
        "rpe_trans_m": float(raw["rpe_trans"]),
        "rpe_rot": float(raw["rpe_rot"]),
        "alignment_scale": primary["alignment_scale"],
        "frames": int(count),
        "valid_gt_frames": int(gt_valid.sum()),
        "gt_file": str(prepared["npz_path"]),
    }, (aligned_trajectory.from_evo(traj_ref, traj_est, alignment, gt_c2w, "evo") if keep_alignment else None)


def macro(values: dict[str, float]) -> float:
    return float(np.mean([values[case] for case in values]))
