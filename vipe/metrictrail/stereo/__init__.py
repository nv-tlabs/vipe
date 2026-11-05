# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MetricTrail stereo runner (optional input; outside the ViPE pipeline registry).

KITTI-style stereo tracking on ViPE streams: a rectified left/right pair, a known
pinhole K and a metric baseline -> DINO-fused stereo DPVO with per-frame MoGe-2
metric depth (FP16, overlapped producer, depth_queue 4) -> ViPE-format artifacts.

    python -m vipe.metrictrail.stereo --left L --right R --intrinsics FX FY CX CY --baseline B --output DIR

- :mod:`.config`: :class:`StereoRunnerConfig` (its defaults are also in ``stereo.yaml``)
  and :func:`resolve_stereo_dpvo_config`.
- :mod:`.runner`: :class:`StereoDPVOSystem`, the stereo tracker.
- :mod:`.features`: the symmetric DINOv3 ``feature_only`` patch for the right image.
- :mod:`.transport`: :class:`StereoSlamProxy` (right frame + baseline per DPVO call)
  and :class:`RightFrames`.
- :mod:`.artifacts`: the ViPE artifact writer; :mod:`.cli`: the command line.
- :mod:`.dpvo_stereo`: the vendored stereo DPVO tree (``VENDORED.md``); its CUDA
  extensions are built by ``python -m vipe.metrictrail.build --stereo``.

Nothing here is imported by the MetricTrail pipeline; submodules load on demand.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "StereoRunnerConfig": ".config",
    "StereoDPVOConfig": ".config",
    "load_stereo_config": ".config",
    "resolve_stereo_dpvo_config": ".config",
    "StereoDPVOSystem": ".runner",
    "StereoSlamProxy": ".transport",
    "RightFrames": ".transport",
    "install_stereo_patch": ".features",
    "save_stereo_artifacts": ".artifacts",
    "run_stereo": ".cli",
    "main": ".cli",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value
