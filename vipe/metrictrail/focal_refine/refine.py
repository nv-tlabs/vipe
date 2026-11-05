# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The focal-refinement flow of the MetricTrail pipeline (stage B).

route (GT-free gates on stage A) -> one stage-B attempt at the coarse matching
resolution -> stage B's own accept gate. The caller rebinds the focal and
re-tracks only when the outcome is accepted; a refused or failed attempt leaves
stage A's focal and trajectory in place. There is no second (upsampled)
attempt. Skipped entirely when stage A's DPVO never initialized.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .routing import route
from .stage_b import run_stage_b, settings_from_config

ROMA_PYTHON_ENV = "VIPE_METRICTRAIL_ROMA_PYTHON"
ROMA_CHECKPOINTS_ENV = "VIPE_METRICTRAIL_ROMA_CHECKPOINTS"


def roma_interpreter(configured: str | None) -> str:
    """Config value, else ``$VIPE_METRICTRAIL_ROMA_PYTHON``, else the running interpreter (not resolved)."""

    return configured or os.environ.get(ROMA_PYTHON_ENV) or sys.executable


def roma_checkpoints(configured: str | None) -> Path:
    """Config value, else ``$VIPE_METRICTRAIL_ROMA_CHECKPOINTS``, else ``$TORCH_HOME/hub/metrictrail/roma``."""

    if configured:
        return Path(configured).expanduser()
    if os.environ.get(ROMA_CHECKPOINTS_ENV):
        return Path(os.environ[ROMA_CHECKPOINTS_ENV]).expanduser()
    import torch

    return Path(torch.hub.get_dir()) / "metrictrail" / "roma"


def rebind_focal(k4: np.ndarray, focal: float) -> np.ndarray:
    """Stage A's K with the refined focal (fx = fy = focal), float32."""

    rebound = np.asarray(k4, dtype=np.float32).copy()
    rebound[0] = rebound[1] = focal
    return rebound


def run_focal_refinement(
    config: Any,
    *,
    stage_a_c2w: np.ndarray,
    stage_a_k4: np.ndarray,
    calibration_record: Mapping[str, Any],
    tracking_status: str,
    frames: Callable[[], Iterable[np.ndarray]],
    image_hw: tuple[int, int],
    device: Any,
    workdir: Path,
    stage_b: Callable[..., dict[str, Any]] = run_stage_b,
) -> dict[str, Any]:
    """Route, then (if routed in) one stage-B attempt. Returns the stage-B record.

    ``config`` is the pipeline's ``focal_refine`` section; ``frames()`` yields the
    sequence as CHW float32 frames in [0, 1].
    ``record["accepted"]`` and ``record["focal"]`` carry the decision.
    """

    record: dict[str, Any] = {"schema": "metrictrail_vipe_focal_refine_v1", "enabled": True, "entered": False}
    if tracking_status != "tracked":
        # Nothing to route on: DPVO never initialized.
        record.update(skipped=tracking_status, accepted=False)
        return record
    settings = settings_from_config(config)
    decision = route(stage_a_c2w, calibration_record, settings)
    record["routing"] = decision.record()
    if bool(config.route_automatically) and not decision.enter:
        record.update(accepted=False, skipped="routed_out")
        return record

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stage_a_npz = workdir / "stage_a.npz"
    np.savez(stage_a_npz, c2w=np.asarray(stage_a_c2w, dtype=np.float64))
    outcome = stage_b(
        settings,
        float(np.asarray(stage_a_k4, dtype=np.float32)[0]),
        stage_a_npz,
        frames(),
        image_hw,
        device,
        workdir / "work",
        seed=int(config.seed),
        interpreter=roma_interpreter(config.roma_python),
        roma_repo=Path(config.roma_repo).expanduser() if config.roma_repo else None,
        roma_checkpoints=roma_checkpoints(config.roma_checkpoints),
    )
    record.update(outcome)
    record["entered"] = True
    record["routing"] = decision.record()
    record["accepted"] = bool(outcome.get("accepted"))
    record["stage_a_focal"] = float(np.asarray(stage_a_k4, dtype=np.float32)[0])
    record["selection_uses_GT"] = False
    return record
