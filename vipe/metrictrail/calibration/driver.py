# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared-focal calibration sweep, independent of the SLAM implementation.

The sweep feeds RGB frames to a keyframe SLAM that optimises one shared focal
with the principal point fixed, runs the backend at the milestones, audits
convergence GT-free with a fresh final-7 / final-24 pass, and selects the
optimised K, or falls back to the initializer K under the terminal policy.

The SLAM is reached only through the small :class:`SharedFocalEngine` protocol:
:mod:`vipe.metrictrail.calibration.engine` implements it with ViPE's own
``SLAMSystem`` / ``GraphBuffer`` / bundle adjustment, and the CPU unit tests
implement it with a scripted fake.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator, Protocol

import numpy as np
import torch

from .protocol import (
    FINAL_STEPS,
    MAX_SAFE_ACCEPTED_KEYFRAMES,
    MILESTONE_STEPS,
    MODES,
    OFFICIAL_MILESTONES,
    convergence_from_final_audit,
    internal_stability,
    permits_terminal_exhaustion_endpoint,
    plausible_k4,
    validate_k4,
)

OPTIMIZED_K_SOURCE = "vipe_final24_optimized"


class SequenceExhaustedBeforeTarget(RuntimeError):
    def __init__(self, *, accepted: int, raw: int, target: int) -> None:
        super().__init__(f"sequence exhausted at {accepted} accepted KFs/{raw} raw frames; target={target}")
        self.accepted = accepted
        self.raw = raw
        self.target = target


class AcceptedKeyframeLimitExceeded(RuntimeError):
    pass


class SharedFocalEngine(Protocol):
    """What the sweep needs from a keyframe SLAM with a shared, optimisable focal."""

    #: Resolution (H, W) the SLAM runs at internally.
    internal_hw: tuple[int, int]

    def candidate(self, frame_index: int, frame: Any) -> bool:
        """Prepare ``frame`` (features, tracks) and return the motion-filter verdict."""
        ...

    def add_keyframe(self, frame_index: int) -> None:
        """Insert the frame prepared by the last :meth:`candidate` call as a keyframe."""
        ...

    def run_frontend(self) -> None: ...

    def run_backend(self, steps: int, *, update_depth: bool) -> None: ...

    def accepted_keyframes(self) -> int: ...

    def recovered_k(self) -> np.ndarray:
        """Current shared K at the input resolution, ``[fx, fy, cx, cy]``."""
        ...

    def depth_calls(self) -> int:
        """Number of K-conditioned metric-depth inferences so far (relatch audit)."""
        ...

    def begin_benchmark(self) -> None: ...

    def end_benchmark(self) -> dict[str, float]:
        """``{"gpu_event_seconds", "peak_allocated_bytes", "peak_reserved_bytes"}``."""
        ...

    def depth_diagnostics(self) -> dict[str, Any]: ...

    def backend_contract(self) -> dict[str, Any]: ...


def _with_last_flag(frames: Iterable[tuple[int, Any]]) -> Iterator[tuple[int, Any, bool]]:
    """Yield ``(source_frame_id, frame, is_last)`` with a one-frame lookahead."""

    iterator = iter(frames)
    try:
        current = next(iterator)
    except StopIteration:
        return
    for following in iterator:
        yield current[0], current[1], False
        current = following
    yield current[0], current[1], True


def run_shared_focal_protocol(
    engine: SharedFocalEngine,
    frames: Iterable[tuple[int, Any]],
    initial_k: Any,
    *,
    image_hw: tuple[int, int],
    mode: str,
    terminal_min_accepted_keyframes: int | None = None,
    fallback_to_initial_k_on_terminal_failure: bool = False,
    initializer_source: str = "moge2_free_fov_component_median",
) -> dict[str, Any]:
    """Run the calibration sweep on ``frames`` = iterable of ``(source_frame_id, frame)``.

    Every frame must already carry ``initial_k`` (the engine binds the shared K
    from the first keyframe). ``initializer_source`` names the initializer that
    produced ``initial_k``; the fallback K is recorded as ``<source>_fallback``.
    Returns the calibration record.
    """

    spec = MODES[mode]
    target = spec.target_accepted_keyframes
    if terminal_min_accepted_keyframes is not None:
        if (
            not isinstance(terminal_min_accepted_keyframes, int)
            or isinstance(terminal_min_accepted_keyframes, bool)
            or terminal_min_accepted_keyframes < 1
        ):
            raise ValueError("terminal minimum must be a positive integer")
        if target is None or terminal_min_accepted_keyframes >= target:
            raise ValueError("terminal minimum must be below the fixed KF target")
    if fallback_to_initial_k_on_terminal_failure and terminal_min_accepted_keyframes is None:
        raise ValueError("initializer fallback requires a terminal KF minimum")
    initial_k = validate_k4(initial_k, image_hw).astype(np.float32)

    def recovered_k() -> np.ndarray:
        return validate_k4(engine.recovered_k(), image_hw).astype(np.float32)

    def run_backend_audited(steps: int, *, update_depth: bool = True) -> dict[str, Any]:
        before_k = recovered_k()
        before_calls = int(engine.depth_calls())
        backend_started = time.perf_counter()
        engine.run_backend(steps, update_depth=update_depth)
        elapsed = time.perf_counter() - backend_started
        after_k = recovered_k()
        after_calls = int(engine.depth_calls())
        relatch_calls = after_calls - before_calls
        focal_changed = not np.allclose(before_k[:2], after_k[:2], rtol=1.0e-5, atol=1.0e-8)
        retained = int(engine.accepted_keyframes())
        # MODEL_METRIC_DEPTH must be recomputed for every retained KF after a
        # focal update. Fail rather than silently accepting stale depth.
        if focal_changed and relatch_calls < retained:
            raise RuntimeError(
                "focal changed without a full retained-keyframe metric-depth relatch: "
                f"calls={relatch_calls}, retained={retained}"
            )
        return {
            "K_before_px": before_k.tolist(),
            "K_after_px": after_k.tolist(),
            "wall_seconds": elapsed,
            "focal_changed": focal_changed,
            "moge2_metric_relatch_calls": relatch_calls,
            "full_retained_keyframe_relatch_verified": bool(not focal_changed or relatch_calls >= retained),
        }

    def run_final_audit(*, checkpoint: str, raw_consumed: int) -> dict[str, Any]:
        """Run the final-7/final-24 convergence audit."""

        accepted = int(engine.accepted_keyframes())
        before_final = recovered_k()
        audit_started = time.perf_counter()
        final7 = run_backend_audited(FINAL_STEPS[0])
        after7 = recovered_k()
        # ViPE's backend ignores update_depth=False whenever a depth model is
        # present, so this remains a second fresh graph/depth pass.
        final24 = run_backend_audited(FINAL_STEPS[1], update_depth=False)
        after24 = recovered_k()
        stability = internal_stability(before_final, after7, after24)
        plausible = plausible_k4(after24, image_hw)
        convergence = convergence_from_final_audit(
            stability=stability,
            k_plausible=plausible,
            final7_event=final7,
            final24_event=final24,
        )
        return {
            "checkpoint": checkpoint,
            "accepted_keyframes": accepted,
            "raw_frames_consumed": raw_consumed,
            "K_before_final_px": before_final.tolist(),
            "K_after_final7_px": after7.tolist(),
            "K_after_final24_px": after24.tolist(),
            "K_plausible_without_GT": plausible,
            "internal_stability": stability,
            "convergence": convergence,
            "final_backend_events": {"final7": final7, "final24": final24},
            "wall_seconds": time.perf_counter() - audit_started,
        }

    milestone_events: list[dict[str, Any]] = []
    convergence_audits: list[dict[str, Any]] = []
    checked_convergence_counts: set[int] = set()
    selected_final_audit: dict[str, Any] | None = None
    raw_consumed = 0
    stopped_source_id: int | None = None
    termination = "unknown"
    terminal_exhaustion = False
    initializer_fallback_reason: str | None = None
    last_source_id: int | None = None
    # Components (including the network weights) are fully resident before the
    # benchmark interval, so peak memory and elapsed time measure calibration
    # inference and optimization only, not model loading.
    engine.begin_benchmark()
    started = time.perf_counter()
    with torch.no_grad():
        for frame_index, (source_id, frame, is_last) in enumerate(_with_last_flag(frames)):
            last_source_id = int(source_id)
            is_keyframe = bool(engine.candidate(frame_index, frame)) or is_last
            if is_keyframe:
                if int(engine.accepted_keyframes()) >= MAX_SAFE_ACCEPTED_KEYFRAMES:
                    raise AcceptedKeyframeLimitExceeded("fresh calibration would exceed the reviewed 1023-KF limit")
                engine.add_keyframe(frame_index)
            engine.run_frontend()
            accepted = int(engine.accepted_keyframes())
            raw_consumed = frame_index + 1
            # If the frontend replaces the second-newest KF and n_frames remains
            # exactly on a milestone, a later accepted candidate may trigger that
            # milestone again (ViPE's own semantics; the released endpoints were
            # produced this way).
            if accepted in OFFICIAL_MILESTONES and is_keyframe:
                event = run_backend_audited(MILESTONE_STEPS)
                milestone_events.append(
                    {
                        "accepted_keyframes": accepted,
                        "raw_frames_consumed": raw_consumed,
                        "source_frame_id": int(source_id),
                        "backend_steps": MILESTONE_STEPS,
                        **event,
                    }
                )
            if accepted in spec.early_convergence_checkpoints and accepted not in checked_convergence_counts:
                # Check each retained-KF budget once. A replacement can leave
                # n_frames unchanged; repeatedly auditing that same nominal
                # budget would make the stopping point input-order dependent.
                checked_convergence_counts.add(accepted)
                audit = run_final_audit(
                    checkpoint=f"early_{accepted}_accepted_keyframes",
                    raw_consumed=raw_consumed,
                )
                convergence_audits.append(audit)
                if audit["convergence"]["converged_without_GT"]:
                    selected_final_audit = audit
                    stopped_source_id = int(source_id)
                    termination = "GT_free_internal_convergence_before_maximum"
                    break
            if target is not None and accepted == target:
                stopped_source_id = int(source_id)
                termination = "accepted_keyframe_target_reached"
                break
            if target is not None and accepted > target:
                raise RuntimeError(f"SLAM skipped accepted-KF target {target}: {accepted}")
        else:
            termination = "raw_sequence_exhausted"
        accepted = int(engine.accepted_keyframes())
        if target is not None and accepted < target and selected_final_audit is None:
            adaptive_terminal = terminal_min_accepted_keyframes is not None
            if permits_terminal_exhaustion_endpoint(mode=mode, accepted=accepted):
                # A sequence can exhaust its RGB inventory before retaining 256
                # KFs. robust256 is explicitly a maximum budget, so audit the
                # actual terminal inventory. Fixed-budget modes do not get this.
                terminal_exhaustion = True
                termination = "terminal_exhaustion_before_maximum"
                if raw_consumed:
                    stopped_source_id = last_source_id
            elif adaptive_terminal:
                assert terminal_min_accepted_keyframes is not None
                terminal_exhaustion = True
                if raw_consumed:
                    stopped_source_id = last_source_id
                if accepted >= terminal_min_accepted_keyframes:
                    # Short/low-motion video: the fixed target is a maximum in
                    # this explicitly opted-in policy. Audit the terminal
                    # optimized K, then retain it only if the GT-free
                    # final-7/final-24 convergence gate passes.
                    termination = "adaptive_terminal_exhaustion_before_target"
                elif fallback_to_initial_k_on_terminal_failure:
                    # Too little parallax for a defensible focal optimization.
                    # Do not pretend the BA estimate converged: use the RGB-only
                    # initializer K instead.
                    termination = "terminal_exhaustion_below_safe_kf_minimum"
                    initializer_fallback_reason = "accepted_keyframes_below_terminal_minimum"
                else:
                    raise SequenceExhaustedBeforeTarget(accepted=accepted, raw=raw_consumed, target=target)
            else:
                raise SequenceExhaustedBeforeTarget(accepted=accepted, raw=raw_consumed, target=target)
        if selected_final_audit is None:
            if initializer_fallback_reason is not None:
                selected_final_audit = _initializer_fallback_audit(
                    initial_k,
                    image_hw,
                    accepted=accepted,
                    raw_consumed=raw_consumed,
                    reason=initializer_fallback_reason,
                )
                convergence_audits.append(selected_final_audit)
            # If the early audit happened on the final raw frame and no state
            # changed afterwards, reuse that audit. Repeating final-7/final-24
            # on identical data could manufacture a convergence pass merely by
            # spending more optimizer iterations.
            latest_audit = convergence_audits[-1] if convergence_audits else None
            if initializer_fallback_reason is not None:
                pass
            elif (
                terminal_exhaustion
                and latest_audit is not None
                and int(latest_audit["accepted_keyframes"]) == accepted
                and int(latest_audit["raw_frames_consumed"]) == raw_consumed
            ):
                selected_final_audit = latest_audit
            else:
                selected_final_audit = run_final_audit(
                    checkpoint=(
                        "terminal_sequence_exhaustion" if terminal_exhaustion else "maximum_target_or_sequence_end"
                    ),
                    raw_consumed=raw_consumed,
                )
                convergence_audits.append(selected_final_audit)
    benchmark = engine.end_benchmark()
    wall_seconds = time.perf_counter() - started
    assert selected_final_audit is not None
    before_final = np.asarray(selected_final_audit["K_before_final_px"], dtype=np.float32)
    after7 = np.asarray(selected_final_audit["K_after_final7_px"], dtype=np.float32)
    after24 = np.asarray(selected_final_audit["K_after_final24_px"], dtype=np.float32)
    stability = selected_final_audit["internal_stability"]
    final_events = selected_final_audit["final_backend_events"]
    final7 = final_events["final7"]
    final24 = final_events["final24"]
    converged_without_gt = bool(selected_final_audit["convergence"]["converged_without_GT"])
    if terminal_exhaustion and not converged_without_gt and fallback_to_initial_k_on_terminal_failure:
        initializer_fallback_reason = initializer_fallback_reason or "terminal_optimized_k_failed_gt_free_convergence"
    selected_k = initial_k if initializer_fallback_reason else after24
    selected_k_source = f"{initializer_source}_fallback" if initializer_fallback_reason else OPTIMIZED_K_SOURCE
    fallback_unconverged = bool(
        (mode == "robust256" and not converged_without_gt) or initializer_fallback_reason is not None
    )
    if initializer_fallback_reason is not None:
        endpoint_disposition = "moge2_initializer_fallback_gt_free"
    elif terminal_exhaustion:
        endpoint_disposition = (
            "terminal_exhaustion_converged" if converged_without_gt else "terminal_exhaustion_fallback_unconverged"
        )
    elif mode == "robust256" and accepted == target:
        endpoint_disposition = (
            "maximum_budget_converged" if converged_without_gt else "maximum_budget_fallback_unconverged"
        )
    elif termination == "GT_free_internal_convergence_before_maximum":
        endpoint_disposition = "early_converged"
    else:
        endpoint_disposition = "fixed_endpoint"
    return {
        "mode": mode,
        "requested_target_accepted_keyframes": target,
        "termination": termination,
        "terminal_exhaustion": terminal_exhaustion,
        "fallback_unconverged": fallback_unconverged,
        "endpoint_disposition": endpoint_disposition,
        "raw_frames_consumed": raw_consumed,
        "accepted_keyframes": int(engine.accepted_keyframes()),
        "stopped_source_frame_id": stopped_source_id,
        "internal_hw": [int(item) for item in engine.internal_hw],
        "initial_K4_px": initial_k.tolist(),
        "K_before_final_px": before_final.tolist(),
        "K_after_final7_px": after7.tolist(),
        "K_after_final24_px": after24.tolist(),
        "optimized_candidate_K_fx_fy_cx_cy": after24.tolist(),
        "calibrated_K_fx_fy_cx_cy": selected_k.tolist(),
        "selected_K_source": selected_k_source,
        "initializer_fallback_reason": initializer_fallback_reason,
        "K_plausible_without_GT": plausible_k4(selected_k, image_hw),
        "internal_stability": stability,
        "convergence": selected_final_audit["convergence"],
        "convergence_audits": convergence_audits,
        "milestone_backend_events": milestone_events,
        "final_backend_events": final_events,
        "timing": {
            "vipe_calibration_wall_seconds": wall_seconds,
            "vipe_calibration_gpu_event_seconds": float(benchmark.get("gpu_event_seconds", 0.0)),
            "final7_wall_seconds": float(final7["wall_seconds"]),
            "final24_wall_seconds": float(final24["wall_seconds"]),
            "all_convergence_audits_wall_seconds": sum(float(audit["wall_seconds"]) for audit in convergence_audits),
        },
        "benchmark_memory": {
            "vipe_peak_allocated_bytes": int(benchmark.get("peak_allocated_bytes", 0)),
            "vipe_peak_reserved_bytes": int(benchmark.get("peak_reserved_bytes", 0)),
        },
        "backend_contract": {
            **engine.backend_contract(),
            "shared_focal_optimized": True,
            "principal_point_fixed": True,
            "milestone_steps": MILESTONE_STEPS,
            "final_steps": list(FINAL_STEPS),
            "inner_gauss_newton_iterations_per_outer_step": 16,
            "final24_removed": False,
            "second_final_pass_update_depth_argument": False,
            "effective_depth_relatch_with_current_vipe_backend": True,
            "early_convergence_checkpoints": list(spec.early_convergence_checkpoints),
            "early_convergence_uses_GT": False,
            "early_convergence_requires_effective_focal_updates": 2,
            "terminal_min_accepted_keyframes": terminal_min_accepted_keyframes,
            "terminal_failure_fallback": initializer_source if fallback_to_initial_k_on_terminal_failure else "reject",
            "terminal_fallback_uses_GT": False,
        },
        "moge2_depth_adapter": engine.depth_diagnostics(),
    }


def _initializer_fallback_audit(
    initial_k: np.ndarray,
    image_hw: tuple[int, int],
    *,
    accepted: int,
    raw_consumed: int,
    reason: str,
) -> dict[str, Any]:
    no_event = {
        "executed": False,
        "wall_seconds": 0.0,
        "focal_changed": False,
        "moge2_metric_relatch_calls": 0,
        "full_retained_keyframe_relatch_verified": False,
        "reason": reason,
    }
    plausible = plausible_k4(initial_k, image_hw)
    return {
        "checkpoint": "moge2_initializer_fallback_without_ba_audit",
        "accepted_keyframes": accepted,
        "raw_frames_consumed": raw_consumed,
        "K_before_final_px": initial_k.tolist(),
        "K_after_final7_px": initial_k.tolist(),
        "K_after_final24_px": initial_k.tolist(),
        "K_plausible_without_GT": plausible,
        "internal_stability": {
            "before_to_after7_relative": None,
            "after7_to_after24_relative": None,
            "stable_at_0_5_percent": False,
            "stable_at_1_percent": False,
            "stable_at_2_percent": False,
            "criterion": "not evaluated: insufficient accepted geometry",
        },
        "convergence": {
            "converged_without_GT": False,
            "reason": reason,
            "endpoint_stability_criterion": "not evaluated",
            "minimum_effective_focal_updates": 2,
            "effective_focal_updates": 0,
            "final7_effective_focal_update": False,
            "final24_effective_focal_update": False,
            "K_plausible_without_GT": plausible,
            "GT_used_for_decision": False,
        },
        "final_backend_events": {"final7": dict(no_event), "final24": dict(no_event)},
        "wall_seconds": 0.0,
    }
