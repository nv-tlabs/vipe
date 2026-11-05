# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU correction of saved ViPE camera poses using external frame pairs."""

from __future__ import annotations

import io
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from . import loop_optimizer as optimizer
from .loop_files import (
    MAX_APPLIED_EDGES,
    _identity,
    _json_object,
    atomic_json,
    load_candidates,
)

LC_IMPLEMENTATION = "external_candidates_pypose_identity_se3_pgo_v1"
SCHEMA = "metrictrail_external_candidates_vipe_v1"
OUTPUT_DIR = "pose_lc"
OUTPUT_SUFFIXES = (".npz", ".json", ".candidates.json", ".identity_edges.json")


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("name must be a single filename component")
    return name


def output_paths(output_dir: Path, name: str) -> dict[str, Path]:
    name = validate_name(name)
    root = Path(output_dir).expanduser().resolve() / OUTPUT_DIR
    return dict(
        zip(
            ("npz", "json", "candidates", "edges"),
            (root / f"{name}{s}" for s in OUTPUT_SUFFIXES),
        )
    )


def load_trajectory(output_dir: Path, name: str) -> dict[str, Any]:
    name = validate_name(name)
    path = (Path(output_dir) / "pose" / f"{name}.npz").expanduser().resolve()
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if "data" not in arrays or "inds" not in arrays:
        raise ValueError("ViPE pose NPZ requires data and inds arrays")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are not accepted")
    frames = arrays["inds"]
    if (
        frames.ndim != 1
        or frames.dtype.kind not in "iu"
        or not len(frames)
        or np.any(frames < 0)
        or np.any(frames > np.iinfo(np.int64).max)
    ):
        raise ValueError("inds must be nonempty nonnegative integers")
    frames = frames.astype(np.int64)
    if np.any(frames[1:] <= frames[:-1]):
        raise ValueError("inds must be unique and strictly increasing")
    stored = arrays["data"]
    if stored.dtype.kind != "f":
        raise ValueError("pose data must contain floating-point matrices")
    poses = np.asarray(stored, dtype=np.float64)
    if poses.shape != (len(frames), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("pose data must be finite with shape [len(inds),4,4]")
    rotations = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1.0e-7)
        or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), rtol=0, atol=2.0e-3)
        or not np.allclose(np.linalg.det(rotations), 1.0, rtol=0, atol=2.0e-3)
    ):
        raise ValueError("pose data must contain proper SE(3) transforms")
    record_path = Path(output_dir) / "vipe" / f"{name}_metrictrail.json"
    if not record_path.is_file():
        raise ValueError("a successful MetricTrail run record is required")
    record_bytes = record_path.read_bytes()
    record = _json_object(record_bytes)
    record_identity = _identity(record_path, record_bytes)
    identity = _identity(path, data)
    if record.get("status") != "success" or record.get("name", record.get("case")) != name:
        raise ValueError("run record must attest to a successful trajectory with the selected name")
    if "tracking_status" not in record:
        raise ValueError("run record requires tracking_status")
    if "num_frames" in record and record["num_frames"] != len(frames):
        raise ValueError("run record frame count differs from the pose NPZ")
    if "pipeline" in record:
        dpvo = (record["pipeline"].get("slam") or {}).get("dpvo") or {}
        local_only = dpvo.get("METRIC_GLOBAL_BA") is False and dpvo.get("LOOP_CLOSURE") is False
    else:
        local_only = record.get("global_ba") is False and record.get("loop_closure") is False
    if not local_only:
        raise ValueError("loop closure requires a local-only trajectory")
    return {
        "path": path,
        "identity": identity,
        "arrays": arrays,
        "frame_ids": frames,
        "c2w": poses,
        "record": record,
        "record_identity": record_identity,
    }


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            fields: dict[str, Any] = arrays
            np.savez(handle, **fields)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_on_vipe_output(
    output_dir: Path,
    name: str,
    *,
    candidates: Path,
    preset: str = "kitti",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Read one saved pose NPZ and write its corrected pose_lc artifacts."""

    if preset not in {"kitti", "vbr"}:
        raise ValueError("preset must be kitti or vbr")
    paths = output_paths(output_dir, name)
    baseline = load_trajectory(output_dir, name)
    manifest, candidate_identity, found = load_candidates(candidates, baseline)
    protected = {baseline["path"], Path(candidate_identity["path"])}
    if baseline["record_identity"] is not None:
        protected.add(Path(baseline["record_identity"]["path"]))
    if any(path.resolve() in protected for path in paths.values()):
        raise ValueError("loop output must not overwrite an input file")
    if any(path.exists() for path in paths.values()) and not overwrite:
        raise FileExistsError("loop output already exists; use --overwrite or a new output directory")
    record = baseline["record"] or {}
    if record.get("tracking_status", "tracked") != "tracked":
        if overwrite:
            for path in paths.values():
                path.unlink(missing_ok=True)
        return {
            "schema": SCHEMA,
            "status": "skipped",
            "name": name,
            "reason": str(record["tracking_status"]),
        }

    settings = optimizer.graph_settings(preset)
    applied = [row for row in found[:MAX_APPLIED_EDGES] if row["score"] >= settings.loop_score_threshold]
    positions = {int(frame): index for index, frame in enumerate(baseline["frame_ids"])}
    pairs = [(positions[row["src_frame"]], positions[row["dst_frame"]]) for row in applied]
    started = time.perf_counter()
    corrected, solve = optimizer.optimize_identity_graph(baseline["c2w"], pairs, settings)
    arrays = dict(baseline["arrays"])
    arrays["data"] = corrected.astype(arrays["data"].dtype) if applied else arrays["data"]
    if not np.isfinite(arrays["data"]).all():
        raise RuntimeError("CPU loop closure returned invalid poses")
    # Publish the report after all output files have been written.
    paths["json"].unlink(missing_ok=True)
    _atomic_npz(paths["npz"], arrays)
    atomic_json(paths["candidates"], {**manifest, "candidates": found})
    atomic_json(
        paths["edges"],
        {
            "schema": "metrictrail_identity_loop_edges_v1",
            "edges": [
                {
                    **row,
                    "src_pos": src,
                    "dst_pos": dst,
                    "measurement": np.eye(4).tolist(),
                }
                for row, (src, dst) in zip(applied, pairs)
            ],
        },
    )
    result = {
        "schema": SCHEMA,
        "status": "success",
        "name": name,
        "lc_implementation": LC_IMPLEMENTATION,
        "num_frames": len(baseline["frame_ids"]),
        "supplied_candidates": len(found),
        "applied_loop_edges": len(applied),
        "contract": {
            "baseline": baseline["identity"],
            "run_record": baseline["record_identity"],
            "candidate_file": candidate_identity,
            "preset": preset,
            "settings": asdict(settings),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pypose": solve.get("runtime", {}).get("pypose"),
            "torch": solve.get("runtime", {}).get("torch"),
            "implementation": LC_IMPLEMENTATION,
        },
        "optimization": solve,
        "runtime_seconds": time.perf_counter() - started,
        "algorithm": {
            "retrieval": "external_candidate_file",
            "loop_measurement": "identity_SE3",
            "pose_graph": "pypose_SE3_LM_scipy_sparse_normal_equations",
            "first_pose_fixed": True,
            "anchor_representation": "float32_inverse_pose_tangent_for_nonempty_loops",
            "geometric_verification": "external_only_not_performed_by_this_command",
        },
        "outputs": {key: {"path": str(path)} for key, path in paths.items() if key != "json"},
    }
    atomic_json(paths["json"], result)
    return result
