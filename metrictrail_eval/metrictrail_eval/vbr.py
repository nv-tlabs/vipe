"""VBR (seven train sequences, 280x504), Sim(3) ATE RMSE in metres.

Port of the benchmark package's ``vbr280/evaluate.py`` (``evaluate_sequence``
+ ``aggregate_results``), the evaluator the long-sequence pipeline called for
every VBR record (e.g. the focal-refinement run 16.357 local / 10.882 LC):

* prediction: neutral NPZ with integer ``frame_ids`` and ``c2w`` (or the
  alias ``predicted_c2w``); finite, homogeneous, non-singular; ``w2c`` if present
  must equal ``inv(c2w)``; rows sorted by ``frame_ids``; at least 3 poses,
  strictly increasing ids;
* GT: ``<root>/<seq>/sequence.npz`` ``c2w_gt`` with contiguous ordinal
  ``frame_ids`` ``0..N-1`` (N fixed per sequence) and its manifest/marker
  attestation;
* frame matching: exact ordinal id lookup, no interpolation or filling;
* metric: the shared hash-locked ``TrajectoryEvaluator(align=True,
  correct_scale=True)`` (``sim3_ate``) with the full GT and the prediction's ids
  (``ate``, ``rpe_trans``, ``rpe_rot``);
* average: unweighted mean over the seven sequences.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import aligned as aligned_trajectory
from . import sim3_ate
from .common import sha256_file, text_sha256
from .predictions import find_prediction as _find_generic

NAME = "vbr"
# sha256 of the preparation profile id in manifest.json / sequence.npz.
PROFILE_SHA256 = "164320c7921b7c99edd8a1e2c181a0142db0d7e0504547f1533650a15f62034f"
TARGET_HW = (280, 504)
CASES = (
    "campus_train0",
    "campus_train1",
    "ciampino_train1",
    "colosseo_train0",
    "diag_train0",
    "pincio_train0",
    "spagna_train0",
)
EXPECTED_SEQUENCE_FRAMES = {
    "campus_train0": 12_042,
    "campus_train1": 11_671,
    "ciampino_train1": 18_846,
    "colosseo_train0": 8_815,
    "diag_train0": 10_021,
    "pincio_train0": 11_142,
    "spagna_train0": 14_141,
}
UNITS = "m"
PROTOCOLS = ("sim3_ate",)
DEFAULT_PROTOCOL = "sim3_ate"
LOGGER = logging.getLogger("metrictrail_eval.vbr")


def find_prediction(pred: Path, stage: str, case: str, protocol: str) -> Path | None:
    return _find_generic(pred, stage, case)


def _validate_ids(frame_ids: np.ndarray, count: int, *, source: str) -> np.ndarray:
    raw = np.asarray(frame_ids)
    if raw.ndim != 1 or len(raw) != count or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{source} frame_ids must be integer [N], got {raw.dtype} {raw.shape}")
    ids = raw.astype(np.int64, copy=False)
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{source} frame_ids must be unique non-negative ordinals")
    return ids


def _validate_c2w(poses: np.ndarray, count: int, *, source: str) -> np.ndarray:
    result = np.asarray(poses, dtype=np.float64)
    if result.shape != (count, 4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{source} C2W poses must be finite [N,4,4], got {result.shape}")
    expected_bottom = np.broadcast_to(np.asarray([0.0, 0.0, 0.0, 1.0]), (count, 4))
    if not np.allclose(result[:, 3], expected_bottom, rtol=0.0, atol=1e-7):
        raise ValueError(f"{source} C2W poses have non-homogeneous bottom rows")
    determinants = np.linalg.det(result[:, :3, :3])
    if np.any(np.abs(determinants) <= 1e-12):
        raise ValueError(f"{source} C2W poses contain singular rotations")
    return result


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """vbr280.evaluate.load_neutral_npz + load_prediction (neutral NPZ branch)."""

    with np.load(path, allow_pickle=False) as archive:
        frame_ids = _validate_ids(archive["frame_ids"], len(archive["frame_ids"]), source="NPZ")
        pose_key = "c2w" if "c2w" in archive.files else "predicted_c2w"
        poses = _validate_c2w(archive[pose_key], len(frame_ids), source=f"NPZ {pose_key}")
        if "c2w" in archive.files and "predicted_c2w" in archive.files:
            alias = _validate_c2w(
                archive["predicted_c2w"], len(frame_ids), source="NPZ predicted_c2w"
            )
            if not np.allclose(poses, alias, rtol=1e-7, atol=1e-9):
                raise ValueError("NPZ c2w and predicted_c2w disagree")
        if "w2c" in archive.files:
            w2c = _validate_c2w(archive["w2c"], len(frame_ids), source="NPZ w2c")
            if not np.allclose(poses, np.linalg.inv(w2c), rtol=1e-5, atol=1e-7):
                raise ValueError("NPZ c2w and inverse(w2c) disagree")
    order = np.argsort(frame_ids, kind="stable")
    poses, frame_ids = poses[order], frame_ids[order]
    # load_prediction re-validates after normalisation and sorts again.
    frame_ids = _validate_ids(frame_ids, len(frame_ids), source="npz")
    poses = _validate_c2w(poses, len(frame_ids), source="npz")
    order = np.argsort(frame_ids, kind="stable")
    return poses[order], frame_ids[order]


def load_gt(gt_root: Path, sequence: str) -> dict[str, Any]:
    """vbr280.protocol.load_prepared_sequence(require_frames=False), GT part."""

    sequence_dir = Path(gt_root).expanduser().resolve() / sequence
    manifest_path = sequence_dir / "manifest.json"
    archive_path = sequence_dir / "sequence.npz"
    intrinsics_yaml_path = sequence_dir / "intrinsics.yaml"
    success_path = sequence_dir / ".success"
    for path in (manifest_path, archive_path, intrinsics_yaml_path, success_path):
        if not path.is_file():
            raise FileNotFoundError(f"VBR {sequence}: missing {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(success_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "success"
        or manifest.get("dataset") != "vbr"
        or manifest.get("sequence") != sequence
        or text_sha256(manifest.get("profile")) != PROFILE_SHA256
    ):
        raise ValueError(f"Invalid VBR sequence manifest: {manifest_path}")
    if (
        marker.get("status") != "success"
        or marker.get("sequence") != sequence
        or marker.get("manifest_sha256") != sha256_file(manifest_path)
        or marker.get("metadata_sha256") != sha256_file(archive_path)
    ):
        raise ValueError(f"Invalid or stale VBR sequence marker: {success_path}")
    if manifest.get("intrinsics_yaml", {}).get("sha256") != sha256_file(intrinsics_yaml_path):
        raise ValueError(f"Invalid or stale VBR intrinsics YAML: {intrinsics_yaml_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"])
        c2w_gt = np.asarray(archive["c2w_gt"], dtype=np.float64)
        stored_sequence = str(np.asarray(archive["sequence"]).item())
        stored_profile = str(np.asarray(archive["profile"]).item())
        input_hw = tuple(int(item) for item in archive["input_hw"])
    count = int(manifest["frame_count"])
    if stored_sequence != sequence or text_sha256(stored_profile) != PROFILE_SHA256:
        raise ValueError(f"VBR {sequence} metadata labels disagree")
    if c2w_gt.shape != (count, 4, 4) or not np.isfinite(c2w_gt).all():
        raise ValueError(f"VBR {sequence} GT is invalid: {c2w_gt.shape}")
    if input_hw != TARGET_HW:
        raise ValueError(f"VBR {sequence} prepared raster is not HxW={TARGET_HW}")
    return {"frame_ids": frame_ids, "c2w_gt": c2w_gt, "archive_path": archive_path}


def _matched_gt(metadata: dict[str, Any], frame_ids: np.ndarray, sequence: str) -> np.ndarray:
    gt_ids = _validate_ids(
        np.asarray(metadata["frame_ids"]), len(metadata["frame_ids"]), source="prepared VBR GT"
    )
    c2w_gt = _validate_c2w(np.asarray(metadata["c2w_gt"]), len(gt_ids), source="prepared VBR GT")
    expected_count = int(EXPECTED_SEQUENCE_FRAMES[sequence])
    if len(gt_ids) != expected_count or not np.array_equal(
        gt_ids, np.arange(expected_count, dtype=np.int64)
    ):
        raise ValueError(
            f"prepared VBR {sequence} must expose contiguous ordinal GT IDs 0..{expected_count - 1}"
        )
    if np.any(frame_ids >= expected_count):
        raise ValueError("prediction contains out-of-range VBR ordinal IDs")
    positions = np.searchsorted(gt_ids, frame_ids)
    if np.any(positions >= len(gt_ids)) or not np.array_equal(gt_ids[positions], frame_ids):
        raise ValueError("prediction frame_ids do not exactly match prepared VBR ordinal GT IDs")
    return c2w_gt


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
    if case not in EXPECTED_SEQUENCE_FRAMES:
        raise ValueError(f"unknown VBR sequence {case}")
    metadata = load_gt(gt_root, case)
    predicted_c2w, frame_ids = load_prediction(prediction)
    if len(frame_ids) < 3:
        raise ValueError(f"VBR/{case} produced only {len(frame_ids)} poses; Sim(3) needs >=3")
    if not np.all(np.diff(frame_ids) > 0):
        raise ValueError(f"VBR/{case} frame IDs are not strictly increasing")
    full_gt = _matched_gt(metadata, frame_ids, case)
    raw = sim3_ate.trajectory_evaluator_class()(align=True, correct_scale=True).evaluate(
        sim3_ate.ArrayTrajectoryLoader(full_gt, np.arange(len(full_gt), dtype=np.int64)),
        sim3_ate.ArrayTrajectoryLoader(predicted_c2w, frame_ids),
        LOGGER,
    )
    transform = np.asarray(raw.pop("traj_transform"), dtype=np.float64)
    metrics = {key: float(raw[key]) for key in ("ate", "rpe_trans", "rpe_rot")}
    full = bool(np.array_equal(frame_ids, np.arange(len(full_gt), dtype=np.int64)))
    trajectory = (
        aligned_trajectory.from_shared_evaluator(full_gt, predicted_c2w, frame_ids, transform)
        if keep_alignment else None
    )
    return {
        **metrics,
        "frames": int(len(frame_ids)),
        "gt_frames": int(len(full_gt)),
        "coverage": "full_sequence" if full else "sparse_exact_ordinal_subset",
        "alignment_scale": float(np.cbrt(np.linalg.det(transform[:3, :3]))),
        "gt_file": str(metadata["archive_path"]),
    }, trajectory


def macro(values: dict[str, float]) -> float:
    return float(np.mean([values[case] for case in values]))
