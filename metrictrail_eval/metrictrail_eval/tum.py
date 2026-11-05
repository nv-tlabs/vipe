"""TUM RGB-D fr1 (nine scenes, stride 1), Sim(3) ATE RMSE in metres.

Port of the TUM path of the short-sequence pipeline
(``evaluation._evaluate_tum518_case``), which calls the benchmark package's
``tum518.protocol.load_prepared_sequence`` and
``tum518.evaluate``'s trajectory-metric function:

* GT: ``<root>/<scene>/sequence.npz`` ``c2w_gt`` cast to float64 (rows of
  frames without an associated GT pose are NaN), after the manifest / marker /
  input-contract attestation of the prepared scene;
* prediction: ``frame_ids == 0..N-1`` equal to the prepared ``frame_ids``,
  ``c2w`` finite [N,4,4] (``w2c == inv(c2w)`` when present);
* metric: the shared hash-locked ``TrajectoryEvaluator(align=True,
  correct_scale=True)`` (``sim3_ate``) with the full GT and the prediction's
  ``frame_ids``; GT rows with NaN/Inf are dropped inside the evaluator, then
  one Sim(3) per scene (``ate``, ``rpe_trans``, ``rpe_rot``);
* average: ``sum(values) / n`` over the scenes in the order
  desk, desk2, rpy, 360, xyz, plant, floor, room, teddy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import aligned as aligned_trajectory
from . import sim3_ate
from .common import sha256_file, text_sha256
from .predictions import find_prediction as _find_generic

NAME = "tum"
CASES = ("360", "desk", "desk2", "floor", "plant", "room", "rpy", "teddy", "xyz")
MACRO_ORDER = ("desk", "desk2", "rpy", "360", "xyz", "plant", "floor", "room", "teddy")
# sha256 of the preparation profile id (manifest.json / sequence.npz) per raster.
PROFILE_SHA256 = {
    (388, 518): "445e5614cd7af9de537c856e2ca628f786546ca4aaa162952adc648a6c9302db",
    (378, 518): "23b4b120de7c419a332257acb537a4174a5d69752c9c138ec7ff5b9ef99f30b7",
}
INPUT_CONTRACT_FILENAME = "tum518_input_contract.json"
UNITS = "m"
PROTOCOLS = ("sim3_ate",)
DEFAULT_PROTOCOL = "sim3_ate"
LOGGER = logging.getLogger("metrictrail_eval.tum")


def find_prediction(pred: Path, stage: str, case: str, protocol: str) -> Path | None:
    return _find_generic(pred, stage, case)


def load_gt(gt_root: Path, scene: str) -> dict[str, Any]:
    """tum518.protocol.load_prepared_directory(require_frames=False), GT part."""

    sequence_dir = Path(gt_root).expanduser().resolve() / scene
    manifest_path = sequence_dir / "manifest.json"
    archive_path = sequence_dir / "sequence.npz"
    marker_path = sequence_dir / ".success"
    contract_path = sequence_dir / INPUT_CONTRACT_FILENAME
    for path in (manifest_path, archive_path, marker_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete prepared TUM scene: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    profile = manifest.get("profile")
    if (
        manifest.get("status") != "success"
        or text_sha256(profile) not in PROFILE_SHA256.values()
        or manifest.get("scene") != scene
        or marker.get("manifest_sha256") != sha256_file(manifest_path)
        or marker.get("metadata_sha256") != sha256_file(archive_path)
        or marker.get("input_contract_sha256") != sha256_file(contract_path)
    ):
        raise ValueError(f"Stale/invalid prepared TUM scene: {sequence_dir}")
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in ("frame_ids", "c2w_gt", "gt_valid", "profile", "scene", "input_hw")}
    count = int(manifest["frame_count"])
    frame_ids = np.asarray(metadata["frame_ids"])
    input_hw = tuple(int(value) for value in np.asarray(metadata["input_hw"]))
    if frame_ids.shape != (count,) or metadata["c2w_gt"].shape != (count, 4, 4):
        raise ValueError(f"Invalid prepared metadata shapes: {archive_path}")
    if not np.issubdtype(frame_ids.dtype, np.integer) or not np.array_equal(
        frame_ids.astype(np.int64), np.arange(count, dtype=np.int64)
    ):
        raise ValueError(f"Prepared TUM frame IDs are not contiguous: {archive_path}")
    if (
        str(np.asarray(metadata["profile"]).item()) != profile
        or str(np.asarray(metadata["scene"]).item()) != scene
        or PROFILE_SHA256.get(input_hw) != text_sha256(profile)
    ):
        raise ValueError(f"Prepared TUM metadata profile/scene/raster changed: {archive_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_contract = {
        "schema_version": 1,
        "dataset": "TUM-RGBD-Freiburg1-9",
        "profile": profile,
        "scene": scene,
        "input_hw": list(input_hw),
        "frame_count": int(len(frame_ids)),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "sequence_npz_sha256": sha256_file(archive_path),
        "output_image_inventory_sha256": manifest.get("output", {}).get("image_inventory_sha256"),
        "frame_ids_sha256_i64le": hashlib.sha256(
            np.asarray(frame_ids, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    if contract != expected_contract:
        raise ValueError(f"Prepared TUM input contract changed: {contract_path}")
    return {
        "count": count,
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "c2w_gt": np.asarray(metadata["c2w_gt"], dtype=np.float64),
        "gt_valid": np.asarray(metadata["gt_valid"], dtype=bool),
        "archive_path": archive_path,
        "profile": profile,
    }


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
        raise ValueError(f"unknown TUM scene {case}")
    prepared = load_gt(gt_root, case)
    count = prepared["count"]
    with np.load(prediction, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int32)
        c2w = np.asarray(archive["c2w"], dtype=np.float64)
        w2c = np.asarray(archive["w2c"], dtype=np.float64) if "w2c" in archive.files else None
    if (
        not np.array_equal(frame_ids, np.arange(count, dtype=np.int32))
        or not np.array_equal(prepared["frame_ids"], frame_ids)
        or c2w.shape != (count, 4, 4)
        or not np.isfinite(c2w).all()
        or (
            w2c is not None
            and (
                w2c.shape != (count, 4, 4)
                or not np.isfinite(w2c).all()
                or not np.allclose(c2w @ w2c, np.eye(4)[None], rtol=1.0e-5, atol=1.0e-6)
            )
        )
    ):
        raise ValueError(f"{case}: TUM prediction contract drifted")
    raw = sim3_ate.trajectory_evaluator_class()(align=True, correct_scale=True).evaluate(
        sim3_ate.ArrayTrajectoryLoader(prepared["c2w_gt"], np.arange(count)),
        sim3_ate.ArrayTrajectoryLoader(c2w, frame_ids),
        LOGGER,
    )
    transform = np.asarray(raw.pop("traj_transform"), dtype=np.float64)
    trajectory = (
        aligned_trajectory.from_shared_evaluator(prepared["c2w_gt"], c2w, frame_ids, transform)
        if keep_alignment else None
    )
    return {
        **{key: float(raw[key]) for key in ("ate", "rpe_trans", "rpe_rot")},
        "frames": int(count),
        "valid_gt_frames": int(prepared["gt_valid"].sum()),
        "alignment_scale": float(np.cbrt(np.linalg.det(transform[:3, :3]))),
        "gt_profile": prepared["profile"],
        "gt_file": str(prepared["archive_path"]),
    }, trajectory


def macro(values: dict[str, float]) -> float:
    order = [case for case in MACRO_ORDER if case in values]
    return sum(values[case] for case in order) / len(order)
