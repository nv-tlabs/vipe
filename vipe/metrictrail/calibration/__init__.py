# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RGB-only shared-K calibration (MetricTrail init stage) on ViPE's keyframe SLAM.

- :mod:`.policy`: schema name, budgets, milestones and terminal policy (standard
  library only; imported by the config schema, see ``policy.py``).
- :mod:`.protocol`: K checks and the GT-free convergence gate (numpy only); it also
  re-exports :mod:`.policy`.
- :mod:`.initializer`: initial K from the depth provider's own focal.
- :mod:`.driver`: the calibration sweep against a small engine protocol.
- :mod:`.engine`: that engine on ViPE's ``SLAMSystem`` / ``GraphBuffer`` / BA.
- :mod:`.processor`: :class:`SharedKIntrinsicsProcessor`, the ``StreamProcessor``.
- :mod:`.config`: :class:`SharedKCalibrationConfig`.

The names below are resolved lazily so that the policy and the numpy-only protocol
can be used without importing the ViPE runtime.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "MODES": "policy",
    "SCHEMA": "policy",
    "SharedKCalibrationConfig": "config",
    "run_shared_focal_protocol": "driver",
    "SequenceExhaustedBeforeTarget": "driver",
    "AcceptedKeyframeLimitExceeded": "driver",
    "ViPESharedFocalEngine": "engine",
    "calibration_slam_config": "engine",
    "calibrate_shared_k": "processor",
    "select_shared_k": "processor",
    "SharedKIntrinsicsProcessor": "processor",
    "GTFreeViolation": "processor",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
