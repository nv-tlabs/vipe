# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import (or, on request, JIT-compile) one compiled DPVO module of a MetricTrail extension package."""

from __future__ import annotations

import importlib
import os
from types import ModuleType

from vipe.metrictrail.build import EXTENSION_NAMES, EXTENSIONS, TREES, eigen_include_dir, extension_kwargs

JIT_ENV = "VIPE_METRICTRAIL_DPVO_JIT"


class DPVOExtensionNotBuilt(ImportError):
    """The DPVO CUDA extensions of a MetricTrail tree are not built."""


def _jit(tree: str, name: str) -> ModuleType:
    from torch.utils.cpp_extension import load

    spec = next(spec for spec in EXTENSIONS if spec.name == name)
    kwargs = extension_kwargs(tree, spec, eigen_include_dir() if spec.needs_eigen else None)
    return load(
        name=f"vipe_metrictrail_{tree}_{name}",
        sources=kwargs["sources"],
        extra_include_paths=kwargs["include_dirs"],
        extra_cflags=kwargs["extra_compile_args"]["cxx"],
        extra_cuda_cflags=kwargs["extra_compile_args"]["nvcc"],
        verbose=True,
    )


def load_extension(tree: str, package: str, name: str) -> ModuleType:
    if name.startswith("__") or name not in EXTENSION_NAMES:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    if TREES.get(tree) != package:
        raise ValueError(f"{package} is not the extension package of tree {tree!r}")
    try:
        return importlib.import_module(f"{package}.{name}")
    except ImportError as error:
        if os.environ.get(JIT_ENV, "0") == "1":
            return _jit(tree, name)
        stereo = " --stereo" if tree == "dpvo_stereo" else ""
        raise DPVOExtensionNotBuilt(
            f"{package}.{name} is not built; run `python -m vipe.metrictrail.build{stereo}` "
            f"(or set {JIT_ENV}=1 to JIT-compile it): {error}"
        ) from error
