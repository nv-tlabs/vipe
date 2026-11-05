# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loop candidate validation and atomic JSON output."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CANDIDATE_SCHEMA = "metrictrail_loop_candidates_v1"
MAX_CANDIDATES = 1000
MAX_APPLIED_EDGES = 300
MAX_JSON_BYTES = 4 * 1024 * 1024


def _identity(path: Path, data: bytes | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    data = path.read_bytes() if data is None else data
    return {"path": str(path), "bytes": len(data)}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json_object(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON input exceeds the 4 MiB file limit")
    value = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def load_candidates(
    path: Path | None, baseline: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if path is None:
        raise ValueError("an external candidate JSON file is required")
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("candidate input must be a JSON file")
    data = path.read_bytes()
    identity = _identity(path, data)
    manifest = _json_object(data)
    if set(manifest) - {"schema", "candidates", "producer"}:
        raise ValueError("unknown candidate manifest fields")
    if manifest.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError(f"candidate schema must be {CANDIDATE_SCHEMA}")
    if "producer" in manifest and (not isinstance(manifest["producer"], str) or len(manifest["producer"]) > 1024):
        raise ValueError("producer must be an informational string of at most 1024 characters")
    values = manifest.get("candidates")
    if not isinstance(values, list) or len(values) > MAX_CANDIDATES:
        raise ValueError(f"candidates must be a list containing at most {MAX_CANDIDATES} rows")
    positions = {int(frame): pos for pos, frame in enumerate(baseline["frame_ids"])}
    normalized, seen = [], set()
    for row in values:
        if not isinstance(row, dict) or set(row) != {"src_frame", "dst_frame", "score"}:
            raise ValueError("each candidate needs exactly src_frame, dst_frame, and score")
        src, dst, score = row["src_frame"], row["dst_frame"], row["score"]
        if type(src) is not int or type(dst) is not int:
            raise ValueError("candidate frame identifiers must be integers, not booleans or positions")
        if src <= dst:
            raise ValueError("candidate src_frame must be later than dst_frame")
        if src not in positions or dst not in positions:
            raise ValueError("candidate frame identifier is absent from baseline frame_ids")
        if type(score) not in (float, int) or not 0 <= score <= 1:
            raise ValueError("candidate score must be finite and within [0,1]")
        if not math.isfinite(float(score)):
            raise ValueError("candidate score must be finite and within [0,1]")
        if (src, dst) in seen:
            raise ValueError("duplicate loop pair")
        seen.add((src, dst))
        normalized.append({"src_frame": src, "dst_frame": dst, "score": float(score)})
    normalized.sort(key=lambda row: (-row["score"], row["src_frame"], row["dst_frame"]))
    return manifest, identity, normalized


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, resolved)
