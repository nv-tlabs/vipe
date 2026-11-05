# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViPE-format artifacts of a stereo run, in the layout step 2's MetricTrail pipeline writes.

- ``pose/<n>.npz``: ``data`` (N,4,4) float64 OpenCV camera-to-world of the left
  camera, ``inds``, ``keyframe_ids``, ``runtime``, ``source_frame_ids``, plus
  ``stereo_baseline_m``;
- ``intrinsics/<n>.npz`` (``data`` (N,4) ``[fx,fy,cx,cy]``, ``inds``) and
  ``intrinsics/<n>_camera.txt``;
- ``vipe/<n>_info.pkl`` (``ba_residual``; ViPE's ``glob_artifacts`` finds runs by it);
- ``vipe/<n>_metrictrail.json``: the run record;
- ``vipe/<n>_slam_map.pt`` with ``save_slam_map``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vipe.metrictrail.depth_producer import jsonable
from vipe.metrictrail.slam import DPVOSLAMOutput
from vipe.utils import io
from vipe.utils.cameras import CameraType


def record_path(artifact_path: io.ArtifactPath) -> Path:
    """``vipe/<name>_metrictrail.json`` (the path step 2's ``pipeline.record_path`` returns)."""

    from vipe.metrictrail.pipeline import record_path as pipeline_record_path

    return pipeline_record_path(artifact_path)


def save_stereo_artifacts(
    output: str | Path,
    name: str,
    final: DPVOSLAMOutput,
    k4: np.ndarray,
    record: dict[str, Any],
    *,
    save_slam_map: bool = False,
) -> io.ArtifactPath:
    artifact_path = io.ArtifactPath(Path(output), name)
    count = len(final.c2w)
    inds = np.arange(count)
    keyframe_ids = (
        np.asarray(final.slam_map.dense_disp_frame_inds, dtype=np.int64)
        if final.slam_map is not None
        else np.zeros(0, dtype=np.int64)
    )
    artifact_path.pose_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        artifact_path.pose_path,
        data=final.c2w,
        inds=inds,
        keyframe_ids=keyframe_ids,
        runtime=np.float64(final.record.get("runtime_seconds", 0.0)),
        source_frame_ids=final.source_frame_ids,
        stereo_baseline_m=np.float64(final.record.get("stereo_baseline_m", 0.0)),
    )

    artifact_path.intrinsics_path.parent.mkdir(exist_ok=True, parents=True)
    k4_rows = np.repeat(np.asarray(k4, dtype=np.float32).reshape(1, 4), count, axis=0)
    np.savez(artifact_path.intrinsics_path, data=k4_rows, inds=inds)
    with artifact_path.camera_type_path.open("w") as handle:
        for frame_idx in range(count):
            handle.write(f"{frame_idx}: {CameraType.PINHOLE.name}\n")

    artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
    with artifact_path.meta_info_path.open("wb") as handle:
        pickle.dump({"ba_residual": final.ba_residual}, handle)
    if save_slam_map and final.slam_map is not None:
        artifact_path.slam_map_path.parent.mkdir(exist_ok=True, parents=True)
        final.slam_map.save(artifact_path.slam_map_path)
    path = record_path(artifact_path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(jsonable(record), indent=2, sort_keys=True, allow_nan=True) + "\n")
    temporary.replace(path)
    return artifact_path
