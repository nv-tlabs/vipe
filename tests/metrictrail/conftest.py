# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test setup for the MetricTrail package.

``vipe.ext`` imports the compiled ``vipe_ext`` module and, when it is missing,
JIT-compiles it with nvcc. These tests never execute a ViPE kernel, so on a
machine where the kernels could not run anyway (no built ``vipe_ext`` and no
CUDA device) a stub module is registered instead: every submodule and kernel
attribute resolves to a stub that raises if it is ever called.

The stub is process-wide (``sys.modules``) and pytest collects this directory
before ViPE's own ``tests/test_*.py``, so it is never installed where it could
replace a usable ``vipe_ext``: with a built ``vipe_ext`` (GPU CI) or with a CUDA
device (where ViPE's JIT fallback can build the real one) ViPE's import path is
left untouched. ``VIPE_METRICTRAIL_VIPE_EXT_STUB=0`` disables the stub and ``=1``
forces it (still never over an importable ``vipe_ext``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

STUB_ENV = "VIPE_METRICTRAIL_VIPE_EXT_STUB"


def _stub_submodule(name: str) -> types.ModuleType:
    module = types.ModuleType(f"vipe_ext.{name}")

    def _missing(attribute: str):
        if attribute.startswith("__"):
            raise AttributeError(attribute)

        def _unavailable(*args, **kwargs):
            raise RuntimeError(f"vipe_ext.{name}.{attribute} is a CPU test stub; build vipe_ext to run kernels")

        return _unavailable

    module.__dict__["__getattr__"] = _missing
    return module


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def should_install_vipe_ext_stub(*, vipe_ext_importable: bool, cuda_available: bool, setting: str | None) -> bool:
    """Only where ViPE's kernels could not run anyway, unless forced on or off by ``setting``."""

    if vipe_ext_importable:
        return False
    if setting == "0":
        return False
    if setting == "1":
        return True
    return not cuda_available


def _install_vipe_ext_stub() -> None:
    importable = "vipe_ext" in sys.modules or importlib.util.find_spec("vipe_ext") is not None
    setting = os.environ.get(STUB_ENV)
    if not should_install_vipe_ext_stub(
        vipe_ext_importable=importable,
        cuda_available=False if importable or setting in ("0", "1") else _cuda_available(),
        setting=setting,
    ):
        return
    root = types.ModuleType("vipe_ext")
    root.__dict__["METRICTRAIL_CPU_TEST_STUB"] = True
    submodules: dict[str, types.ModuleType] = {}

    def _submodule(name: str) -> types.ModuleType:
        if name.startswith("__"):
            raise AttributeError(name)
        return submodules.setdefault(name, _stub_submodule(name))

    # Every extension submodule ``vipe.ext`` binds (``vipe_ext.<name>_ext``) resolves to a stub.
    root.__dict__["__getattr__"] = _submodule
    sys.modules["vipe_ext"] = root


_install_vipe_ext_stub()
