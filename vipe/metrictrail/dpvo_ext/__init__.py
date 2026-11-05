# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DPVO's CUDA extensions for MetricTrail: ``cuda_corr``, ``cuda_ba``, ``lietorch_backends``.

Built from ``vipe/metrictrail/csrc/dpvo`` by ``python -m vipe.metrictrail.build``
into this package (never shipped in the repository). The vendored DPVO imports
them as ``from vipe.metrictrail.dpvo_ext import cuda_ba`` (see
``vipe/metrictrail/dpvo/VENDORED.md``). With ``VIPE_METRICTRAIL_DPVO_JIT=1`` a
missing module is JIT-compiled with the same sources and flags instead
(``torch.utils.cpp_extension.load``; cached under ``TORCH_EXTENSIONS_DIR``).
"""

from __future__ import annotations

from typing import Any

from vipe.metrictrail.dpvo_ext_loader import load_extension

TREE = "dpvo"


def __getattr__(name: str) -> Any:
    module = load_extension(TREE, __name__, name)
    globals()[name] = module
    return module
