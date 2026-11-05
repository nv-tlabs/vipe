# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GT-free decision rules of MetricTrail's RGB-only shared-K calibration.

Pure numpy, no ViPE runtime dependency: the keyframe budgets, the
milestone/final backend step counts, the K sanity checks and the final-7 /
final-24 convergence gate. Every function here only looks at intrinsics the
calibration itself produced; nothing reads a sensor K or ground truth.

The standard-library part (schema name, budgets, milestones, terminal policy)
is defined in :mod:`.policy`, so that the config schema can validate it without
numpy, and is re-exported here under the same names.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

# The pure-Python rules are defined in ``policy.py`` (part of the numpy-free import
# closure of ``vipe.config``) and re-exported here unchanged.
from .policy import (  # noqa: F401  (re-exported)
    FINAL_STEPS,
    MAX_SAFE_ACCEPTED_KEYFRAMES,
    MILESTONE_STEPS,
    MODES,
    OFFICIAL_MILESTONES,
    PIPELINE_BUDGETS,
    SCHEMA,
    TERMINAL_MIN_FLOOR,
    ModeSpec,
    check_terminal_policy,
    permits_terminal_exhaustion_endpoint,
)


def validate_k4(value: Any, image_hw: tuple[int, int]) -> np.ndarray:
    """``[fx, fy, cx, cy]`` as float64; raises unless finite, positive focal, sane centre."""

    k4 = np.asarray(value, dtype=np.float64)
    height, width = image_hw
    if (
        k4.shape != (4,)
        or not np.isfinite(k4).all()
        or np.any(k4[:2] <= 0.0)
        or not -0.25 * width <= k4[2] <= 1.25 * width
        or not -0.25 * height <= k4[3] <= 1.25 * height
    ):
        raise ValueError(f"invalid [fx,fy,cx,cy] for {image_hw}: {k4}")
    return k4


def plausible_k4(value: Any, image_hw: tuple[int, int]) -> bool:
    try:
        fx, fy, cx, cy = validate_k4(value, image_hw)
    except ValueError:
        return False
    height, width = image_hw
    return bool(
        0.1 * width <= fx <= 5.0 * width
        and 0.1 * height <= fy <= 5.0 * height
        and 0.5 <= fx / fy <= 2.0
        and -0.25 * width <= cx <= 1.25 * width
        and -0.25 * height <= cy <= 1.25 * height
    )


def focal_relative_change(previous: Any, current: Any) -> float:
    old = np.asarray(previous, dtype=np.float64)
    new = np.asarray(current, dtype=np.float64)
    if old.shape != (4,) or new.shape != (4,):
        raise ValueError("focal endpoints must be K4 vectors")
    old_f = math.sqrt(float(old[0] * old[1]))
    new_f = math.sqrt(float(new[0] * new[1]))
    if not math.isfinite(old_f) or not math.isfinite(new_f) or new_f <= 0.0:
        raise ValueError("focal endpoints must be finite and positive")
    return abs(new_f - old_f) / new_f


def internal_stability(before: Any, after7: Any, after24: Any) -> dict[str, Any]:
    before_to_7 = focal_relative_change(before, after7)
    seven_to_24 = focal_relative_change(after7, after24)
    return {
        "before_to_after7_relative": before_to_7,
        "after7_to_after24_relative": seven_to_24,
        "stable_at_0_5_percent": seven_to_24 <= 0.005,
        "stable_at_1_percent": seven_to_24 <= 0.01,
        "stable_at_2_percent": seven_to_24 <= 0.02,
        "criterion": "fresh final-7 endpoint versus fresh final-24 endpoint",
    }


def convergence_from_final_audit(
    *,
    stability: Mapping[str, Any],
    k_plausible: bool,
    final7_event: Mapping[str, Any],
    final24_event: Mapping[str, Any],
    minimum_effective_focal_updates: int = 2,
) -> dict[str, Any]:
    """Make a GT-free convergence decision from the final audit.

    The numerical criterion is the final-7 to final-24 1% endpoint check. The
    extra evidence gate prevents two rejected/no-op backend calls from being
    mislabeled as convergence merely because their endpoints are identical.
    """

    if minimum_effective_focal_updates < 1:
        raise ValueError("minimum effective focal updates must be positive")
    events = (final7_event, final24_event)
    effective = [
        bool(event.get("focal_changed"))
        and bool(event.get("full_retained_keyframe_relatch_verified"))
        and int(event.get("moge2_metric_relatch_calls", 0)) > 0
        for event in events
    ]
    count = sum(effective)
    endpoint_stable = bool(stability.get("stable_at_1_percent"))
    enough_updates = count >= minimum_effective_focal_updates
    converged = bool(k_plausible and endpoint_stable and enough_updates)
    if not k_plausible:
        reason = "implausible_final_K"
    elif not endpoint_stable:
        reason = "final7_to_final24_endpoint_change_exceeds_1_percent"
    elif not enough_updates:
        reason = "fewer_than_two_effective_focal_updates"
    else:
        reason = "two_effective_focal_updates_and_1_percent_endpoint_stability"
    return {
        "converged_without_GT": converged,
        "reason": reason,
        "endpoint_stability_criterion": ("fresh final-7 endpoint versus fresh final-24 endpoint <= 1 percent"),
        "minimum_effective_focal_updates": minimum_effective_focal_updates,
        "effective_focal_updates": count,
        "final7_effective_focal_update": effective[0],
        "final24_effective_focal_update": effective[1],
        "K_plausible_without_GT": bool(k_plausible),
        "GT_used_for_decision": False,
    }


def jsonable(value: Any) -> Any:
    """Convert a calibration record to JSON-safe builtins (non-finite floats become None)."""

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
