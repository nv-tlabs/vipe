# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, immutable SharedK artifacts for comparisons using one frozen calibration.

The caller chooses one path per prepared video. This is not an automatically
invalidated model cache: choose a new path when changing calibration code or
weights. Original calibration records, including fallbacks and timings, remain
unchanged; the separate execution record measures this invocation only.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vipe.config.base_schema import config_to_primitive
from vipe.streams.base import VideoStream

from .config import SharedKCalibrationConfig
from .engine import calibration_slam_config
from .processor import (
    apply_calibration_determinism,
    canonical_calibration_frame,
    require_rgb_only,
    select_shared_k,
)

ARTIFACT_SCHEMA = "sharedk_frozen_calibration_v1"


def stream_identity(stream: VideoStream, normalization: str) -> dict[str, Any]:
    """Hash every prepared RGB frame without retaining the video in memory.

    Args:
        stream: Replayable, RGB-only calibration stream, before tracking subsampling.
        normalization: SharedK RGB normalization policy.
    """
    require_rgb_only(stream)
    image_hw = tuple(int(size) for size in stream.frame_size())
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", *image_hw))
    count, last = 0, -1
    # The iterator must not be iterated again: some ViPE streams restart in __iter__.
    iterator = iter(stream)
    while True:
        try:
            frame = next(iterator)
        except StopIteration:
            break
        frame_id, canonical = canonical_calibration_frame(frame, normalization)
        if frame_id <= last:
            raise ValueError("calibration cache requires strictly increasing source frame IDs")
        if tuple(canonical.shape[:2]) != image_hw:
            raise ValueError("calibration cache frame raster differs from the declared raster")
        rgb = canonical.detach().cpu().contiguous().numpy()
        if not np.isfinite(rgb).all():
            raise ValueError("calibration cache requires finite RGB")
        # Fixed-width little-endian metadata and float32 bytes make the hash unambiguous.
        digest.update(struct.pack("<q", frame_id))
        digest.update(rgb.astype("<f4", copy=False).tobytes() if normalization == "stream" else rgb.tobytes())
        last = frame_id
        count += 1
    if count == 0 or count != len(stream):
        raise ValueError(f"calibration cache expected {len(stream)} frames but read {count}")
    return {"rgb_sha256": digest.hexdigest(), "frames": count, "input_hw": list(image_hw)}


def _json_bytes(value: Any) -> bytes:
    """Serialize a JSON artifact deterministically.

    Args:
        value: JSON-compatible calibration artifact or identity.
    """
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _publish_once(path: Path, content: bytes) -> None:
    """Publish a complete artifact atomically, refusing to replace any existing path.

    Args:
        path: Destination on a filesystem supporting atomic hard links.
        content: Complete serialized artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # link is exclusive; a concurrent creator cannot overwrite the frozen K.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def frozen_calibration(
    stream: VideoStream,
    config: SharedKCalibrationConfig,
    path: Path,
    calibrate: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create or validate a frozen calibration, returning its record and current execution cost.

    Args:
        stream: Replayable, prepared RGB-only stream used by calibration.
        config: Complete SharedK settings; resolved SLAM defaults are also bound.
        path: Explicit immutable artifact path, one per prepared video.
        calibrate: Fresh calibration callback, never invoked for an existing artifact.
    """
    started = time.perf_counter()
    identity = {
        **stream_identity(stream, config.rgb_normalization),
        "calibration_config": config.model_dump(mode="json", exclude={"slam"}),
        "resolved_slam": config_to_primitive(calibration_slam_config(config.slam)),
    }
    # JSON also normalizes tuples in resolved configuration for exact readback comparison.
    identity = json.loads(_json_bytes(identity))
    reused = path.exists()
    if reused:
        content = path.read_bytes()
        artifact = json.loads(content)
        if not isinstance(artifact, dict) or artifact.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("invalid frozen SharedK artifact schema")
        if artifact.get("identity") != identity:
            raise ValueError("frozen SharedK artifact RGB, frame IDs/count, raster or calibration config mismatch")
        record = artifact.get("calibration")
        if not isinstance(record, dict):
            raise ValueError("frozen SharedK artifact lacks a calibration record")
        if artifact.get("calibration_sha256") != hashlib.sha256(_json_bytes(record)).hexdigest():
            raise ValueError("frozen SharedK calibration record checksum mismatch")
    else:
        record = calibrate()
    # Reuse all ordinary GT-free stability/fallback gates; a file is not proof of validity.
    selected = select_shared_k(record, allow_unstable=config.allow_unstable)
    if record.get("input_hw") != identity["input_hw"] or record.get("input_stream_frames") != identity["frames"]:
        raise ValueError("frozen SharedK calibration record raster/count mismatch")
    record["bound_K_fx_fy_cx_cy"] = selected.tolist()
    if reused:
        # Fresh calibration applies this policy too, including when tracking uses seed=null.
        apply_calibration_determinism(seed=config.seed, disable_tf32=config.disable_tf32)
    else:
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "identity": identity,
            "calibration": record,
            "calibration_sha256": hashlib.sha256(_json_bytes(record)).hexdigest(),
        }
        content = _json_bytes(artifact)
        _publish_once(path, content)
    execution = {
        "mode": "reused" if reused else "created",
        "artifact_path": str(path.resolve()),
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
        "current_run_wall_seconds": time.perf_counter() - started,
        "original_calibration_wall_seconds": (record.get("timing") or {}).get("end_to_end_wall_seconds"),
    }
    return record, execution
