# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Third-party modules the vendored DPVO imports that are not ViPE dependencies.

The vendored tree (:mod:`vipe.metrictrail.dpvo`) imports ``yacs`` and ``numba``
through this module (marked ``# [vipe]`` import rewrites, see
``dpvo/VENDORED.md``). Each name is imported on first access, so ViPE itself
(and its type check) never needs them, and a missing package fails with an
install hint instead of a bare ``ModuleNotFoundError`` deep inside DPVO.

- ``CfgNode``: ``yacs.config.CfgNode`` (DPVO's config, ``dpvo/config.py``).
- ``numba``: DPVO's jitted edge reduction (``edge_utils.py``,
  imported by ``patchgraph.py``).
- ``Viewer``: DPVO's optional ``dpviewer`` (never used: MetricTrail runs DPVO
  with ``viz=False``).
"""

from __future__ import annotations

import importlib
from typing import Any

#: pip requirement of every module served here.
REQUIREMENTS = {
    "CfgNode": "yacs",
    "numba": "numba",
    "Viewer": "dpviewer",
}
INSTALL_HINT = "pip install yacs numba"


def _require(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"MetricTrail's DPVO tracker needs the '{module}' package, which is not a ViPE dependency ({INSTALL_HINT})"
        ) from error


def missing_requirements() -> list[str]:
    """pip names of the DPVO runtime requirements that cannot be imported here (``dpviewer`` excluded)."""

    missing = []
    for name in ("yacs", "numba"):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def __getattr__(name: str) -> Any:
    if name == "CfgNode":
        return _require("yacs.config").CfgNode
    if name == "numba":
        return _require(name)
    if name == "Viewer":
        return _require("dpviewer").Viewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
