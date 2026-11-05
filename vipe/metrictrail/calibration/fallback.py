# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-the-wild calibration fallback (``calibration.fallback: provider_focal``).

With ``provider_focal`` a SharedK run that does not converge, runs out of frames,
keeps too few keyframes or raises inside the calibration sweep does not stop the
pipeline: the depth provider's own focal (the median of the initializer frames)
with the principal point at the image centre is bound instead, and the record
states the status and the reason. ``none`` keeps the benchmark behaviour. Nothing
is retried, and no GT is read.

Pure Python (no numpy/torch), so the rules can be imported anywhere.
"""

from __future__ import annotations

from typing import Any, Mapping

#: ``calibration.fallback`` values.
CALIBRATION_FALLBACKS = ("none", "provider_focal")

#: ``calibration_fallback.reason`` values.
FALLBACK_NOT_CONVERGED = "not_converged"
FALLBACK_RAN_OUT_OF_FRAMES = "ran_out_of_frames"
FALLBACK_TOO_FEW_KEYFRAMES = "too_few_keyframes"
FALLBACK_VIPE_RAISED = "vipe_stage_raised"
FALLBACK_REASONS = (
    FALLBACK_NOT_CONVERGED,
    FALLBACK_RAN_OUT_OF_FRAMES,
    FALLBACK_TOO_FEW_KEYFRAMES,
    FALLBACK_VIPE_RAISED,
)

#: ``selected_K_source`` of the provider-focal fallback.
PROVIDER_FOCAL_FALLBACK = "provider_focal_centred_principal_point_fallback"
#: ``selection_status`` / ``calibration_status`` of the provider-focal fallback.
PROVIDER_FOCAL_STATUS = "fallback_provider_focal"


def provider_focal_k4_values(initial_k: Any, image_hw: tuple[int, int]) -> list[float]:
    """The provider's own focal with the principal point at the image centre: ``[f, f, W/2, H/2]``."""

    focal = float(list(initial_k)[0])
    height, width = int(image_hw[0]), int(image_hw[1])
    return [focal, focal, width / 2.0, height / 2.0]


def is_broken_installation(error: BaseException) -> bool:
    """A GPU or toolchain failure is never an in-the-wild property of the video.

    ``calibration.fallback: provider_focal`` must not turn it into a fallback: e.g. a ``vipe_ext``
    built without code for the GPU fails with "CUDA error: ... unsupported toolchain" on the first
    kernel, and a silent provider-focal K would hide the broken installation.
    """

    if type(error).__name__ in ("AcceleratorError", "OutOfMemoryError"):
        return True
    text = str(error)
    return text.startswith("CUDA error") or any(
        marker in text for marker in ("no kernel image is available", "unsupported toolchain", "CUDA driver version")
    )


def vipe_error_reason(
    error: BaseException, *, terminal_min_accepted_keyframes: int | None
) -> tuple[str, dict[str, Any]]:
    """Fallback reason for an exception raised by the calibration sweep."""

    detail: dict[str, Any] = {
        "exception_type": type(error).__name__,
        "exception_message": str(error)[:1000],
    }
    accepted = getattr(error, "accepted", None)
    if type(error).__name__ == "SequenceExhaustedBeforeTarget" and isinstance(accepted, int):
        detail.update(
            {
                "accepted_keyframes": accepted,
                "raw_frames_consumed": getattr(error, "raw", None),
                "target_accepted_keyframes": getattr(error, "target", None),
                "terminal_min_accepted_keyframes": terminal_min_accepted_keyframes,
            }
        )
        if terminal_min_accepted_keyframes is not None and accepted < terminal_min_accepted_keyframes:
            return FALLBACK_TOO_FEW_KEYFRAMES, detail
        return FALLBACK_RAN_OUT_OF_FRAMES, detail
    return FALLBACK_VIPE_RAISED, detail


def unconverged_reason(calibrated: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Fallback reason for a completed sweep that is not stable for downstream use."""

    detail = {
        "termination": calibrated.get("termination"),
        "endpoint_disposition": calibrated.get("endpoint_disposition"),
        "accepted_keyframes": calibrated.get("accepted_keyframes"),
        "raw_frames_consumed": calibrated.get("raw_frames_consumed"),
        "convergence_reason": (calibrated.get("convergence") or {}).get("reason"),
        "vipe_selected_K_fx_fy_cx_cy": calibrated.get("calibrated_K_fx_fy_cx_cy"),
        "vipe_selected_K_source": calibrated.get("selected_K_source"),
    }
    if calibrated.get("initializer_fallback_reason") == "accepted_keyframes_below_terminal_minimum":
        return FALLBACK_TOO_FEW_KEYFRAMES, detail
    if calibrated.get("terminal_exhaustion"):
        return FALLBACK_RAN_OUT_OF_FRAMES, detail
    return FALLBACK_NOT_CONVERGED, detail


def check_fallback_policy(*, fallback: str, fallback_to_initializer: bool) -> None:
    """``provider_focal`` replaces ``fallback_to_initializer``: both bind the provider's initializer focal.

    ``allow_unstable`` is not refused here: it combines with ``provider_focal``
    and keeps the optimised K of a sweep that reached its keyframe budget
    without passing the convergence gate (the benchmark policy), see :mod:`.processor`.
    """

    if fallback not in CALIBRATION_FALLBACKS:
        raise ValueError(f"calibration fallback must be one of {CALIBRATION_FALLBACKS}, got {fallback!r}")
    if fallback == "provider_focal" and fallback_to_initializer:
        raise ValueError(
            "calibration fallback provider_focal replaces fallback_to_initializer (both bind the depth provider's "
            "initializer focal); set it to false"
        )
