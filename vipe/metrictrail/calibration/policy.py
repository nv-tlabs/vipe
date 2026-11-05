# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python rules of the SharedK calibration: budgets, milestones, terminal policy.

This module is part of the import closure of ``vipe.config`` (ViPE's
``PipelineConfig`` union registers ``MetricTrailPipelineConfig``, whose SharedK
schema validates its terminal policy here). ViPE's Docs workflow imports that
closure in an environment with only hydra, omegaconf and pydantic installed, so
this module must stay standard-library only: no numpy, no torch
(``tests/metrictrail/test_config_import_closure.py``).

The numpy decision rules (K checks, convergence gate) live in :mod:`.protocol`,
which re-exports every name defined here.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA = "sharedk_calibration_v1"

#: ViPE's default graph buffer holds 1024 keyframes; a calibration that would add
#: the 1024th is refused instead of overflowing it.
MAX_SAFE_ACCEPTED_KEYFRAMES = 1023

#: Backend steps of the convergence audit: a fresh 7-step pass, then a fresh
#: 24-step pass (ViPE's own end-of-sequence schedule).
FINAL_STEPS = (7, 24)

#: Backend steps run when the accepted-keyframe count lands on a milestone.
MILESTONE_STEPS = 5

#: Accepted-keyframe counts at which the backend runs during the sweep
#: (ViPE's ``frontend_backend_iters``).
OFFICIAL_MILESTONES = (16, 64, 256)

#: Smallest admissible terminal minimum: the first milestone, i.e. the first
#: point at which the focal has been optimised at all.
TERMINAL_MIN_FLOOR = OFFICIAL_MILESTONES[0]


@dataclass(frozen=True)
class ModeSpec:
    name: str
    target_accepted_keyframes: int | None
    consume_all_raw_frames: bool
    early_convergence_checkpoints: tuple[int, ...] = ()


MODES: dict[str, ModeSpec] = {
    "fast32": ModeSpec("fast32", 32, False),
    "fast64": ModeSpec("fast64", 64, False),
    "robust128": ModeSpec("robust128", 128, False),
    # ``robust256`` is a maximum budget, not a requirement to consume all 256
    # retained keyframes. It runs the final-7/final-24 audit at 128 KFs and
    # exits there only when that audit has both the 1% endpoint stability and
    # evidence of two real focal updates. Otherwise it continues to the hard
    # 256-KF endpoint and audits once more.
    "robust256": ModeSpec("robust256", 256, False, (128,)),
    "all": ModeSpec("all", None, True),
}

#: Budgets a MetricTrail pipeline may select (the others exist for the protocol only).
PIPELINE_BUDGETS = ("fast64", "robust128", "robust256")


def permits_terminal_exhaustion_endpoint(*, mode: str, accepted: int) -> bool:
    """Return whether a short sequence may finalize below its nominal target.

    Only ``robust256`` is a maximum-budget protocol. Once its first audited
    budget (128 retained keyframes) has been reached, exhausting the RGB
    sequence is a legitimate terminal endpoint: the caller must still run the
    full final-7/final-24 audit and must report a failed convergence gate as an
    unconverged fallback. Fixed-budget modes remain fail-closed.
    """

    if mode not in MODES:
        raise ValueError(f"unknown calibration mode: {mode}")
    if not isinstance(accepted, int) or isinstance(accepted, bool) or accepted < 0:
        raise ValueError("accepted keyframes must be a non-negative integer")
    spec = MODES[mode]
    if mode != "robust256" or spec.target_accepted_keyframes is None:
        return False
    if not spec.early_convergence_checkpoints:
        return False
    minimum_audited_budget = min(spec.early_convergence_checkpoints)
    return minimum_audited_budget <= accepted < spec.target_accepted_keyframes


def check_terminal_policy(
    *,
    mode: str,
    terminal_min_accepted_keyframes: int | None,
    fallback_to_initializer: bool,
) -> None:
    """The pipeline-level terminal-policy rules of the SharedK calibration config.

    A terminal minimum must be at least the first milestone (16) and below the
    budget's keyframe target; the initializer fallback needs a terminal minimum.
    """

    if mode not in MODES:
        raise ValueError(f"unknown calibration mode: {mode}")
    target = MODES[mode].target_accepted_keyframes
    if terminal_min_accepted_keyframes is not None:
        minimum = terminal_min_accepted_keyframes
        if not isinstance(minimum, int) or isinstance(minimum, bool):
            raise ValueError("terminal minimum must be an integer")
        if target is None or minimum < TERMINAL_MIN_FLOOR or minimum >= target:
            raise ValueError(
                "terminal_min_accepted_keyframes must be at least the first audited ViPE "
                f"milestone ({TERMINAL_MIN_FLOOR}) and below the requested budget"
            )
    if fallback_to_initializer and terminal_min_accepted_keyframes is None:
        raise ValueError("the initializer fallback requires an explicit terminal keyframe minimum")
