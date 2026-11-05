# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration of the SharedK calibration init stage.

A separate schema: it does not extend or modify ViPE's ``DefaultInitConfig``.
The defaults are the in-the-wild policy: robust128 budget, terminal minimum 16,
provider-focal fallback (no unstable optimised K, no initializer fallback), FP32 MoGe-2,
seed 1234, TF32 off. The benchmark policy is the default plus ``allow_unstable: true``:
the provider-focal fallback binds the same K as the MoGe-2 initializer fallback
(``fallback_to_initializer``). ``fallback: none`` keeps the benchmark switches
(``allow_unstable`` / ``fallback_to_initializer``) selectable on their own.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from vipe.config.base_schema import BaseConfigSchema, Field
from vipe.config.slam import SLAMConfig

from .fallback import check_fallback_policy

# ``.policy``, not ``.protocol``: this schema is in ViPE's config import closure, which
# must not need numpy (ViPE's Docs workflow installs only hydra/omegaconf/pydantic).
from .policy import check_terminal_policy


class SharedKCalibrationConfig(BaseConfigSchema):
    """RGB-only shared-intrinsics calibration on ViPE's keyframe SLAM (MetricTrail init stage)."""

    budget: Literal["fast64", "robust128", "robust256"] = Field(
        default="robust128",
        description="Accepted-keyframe budget of the calibration sweep (robust256 may stop early at 128).",
    )
    initializer_frames: int = Field(
        default=32,
        ge=1,
        description="Number of leading frames whose free-FoV focal predictions give the initial shared K (median).",
    )
    terminal_min_accepted_keyframes: int | None = Field(
        default=16,
        description="When the video ends before the budget, accept the terminal optimised K if at least this many "
        "keyframes were retained (and it passes the convergence gate). Null fails such videos instead.",
    )
    fallback_to_initializer: bool = Field(
        default=False,
        description="Use the depth provider's own (initializer) K when the video ends below the terminal minimum or "
        "the terminal optimised K fails the convergence gate. Requires terminal_min_accepted_keyframes. Only with "
        "fallback none: provider_focal already binds that K in those cases.",
    )
    allow_unstable: bool = Field(
        default=False,
        description="Accept an optimised K that reached the budget but did not pass the GT-free convergence gate "
        "(benchmark policy). With provider_focal only a sweep that reached its keyframe budget keeps such a K; every "
        "other unconverged sweep still binds the provider focal.",
    )
    fallback: Literal["none", "provider_focal"] = Field(
        default="provider_focal",
        description="provider_focal (in-the-wild policy): when SharedK does not converge, runs out of frames, keeps "
        "too few keyframes or raises, bind the depth provider's own focal (initializer median) with the principal "
        "point at the image centre and record the reason (allow_unstable keeps a budget-reached optimised K). "
        "none: such a run fails or follows allow_unstable / fallback_to_initializer.",
    )
    depth_model: str = Field(
        default="moge2-l",
        description="Depth provider used during calibration (ViPE depth registry name; moge2* uses MetricTrail's "
        "exact MoGe-2 call). It must predict its own focal and be re-inferred per focal (MODEL_METRIC_DEPTH).",
    )
    depth_use_fp16: bool = Field(
        default=False,
        description="Autocast the calibration depth provider to FP16. MetricTrail calibrates in FP32.",
    )
    rgb_normalization: Literal["metrictrail", "stream"] = Field(
        default="metrictrail",
        description="metrictrail re-quantises each frame to uint8 and divides by 255 on the device (8-bit-exact "
        "inputs, independent of the stream's float path); stream uses the stream's float RGB as is.",
    )
    seed: int = Field(default=1234, description="Python/numpy/torch seed set before calibration.")
    disable_tf32: bool = Field(
        default=True,
        description="Disable TF32 matmul/cuDNN and use highest float32 matmul precision (process-wide).",
    )
    slam: SLAMConfig | None = Field(
        default=None,
        description="ViPE SLAM config for the calibration sweep. Null composes configs/slam/default.yaml. The SharedK "
        "overrides (no ViPE keyframe-depth prior, focal optimisation, fused BA, milestones 16/64/256, 24 final "
        "iterations) are always applied on top.",
    )

    @model_validator(mode="after")
    def validate_terminal_policy(self) -> SharedKCalibrationConfig:
        check_terminal_policy(
            mode=self.budget,
            terminal_min_accepted_keyframes=self.terminal_min_accepted_keyframes,
            fallback_to_initializer=self.fallback_to_initializer,
        )
        check_fallback_policy(fallback=self.fallback, fallback_to_initializer=self.fallback_to_initializer)
        return self
