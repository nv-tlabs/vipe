# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional stereo DPVO CUDA extensions (``cuda_corr``, ``cuda_ba``, ``lietorch_backends``).

Built from ``vipe/metrictrail/csrc/dpvo_stereo`` only on request
(``python -m vipe.metrictrail.build --stereo``). The stereo tree keeps its own
``fastba`` kernels (stereo factors), so it is a separate package and is never
merged with :mod:`vipe.metrictrail.dpvo_ext`. The MetricTrail pipeline does not
import it; the stereo runner (:mod:`vipe.metrictrail.stereo`, vendored tree
``stereo/dpvo_stereo``) does.
"""

from __future__ import annotations

from typing import Any

from vipe.metrictrail.dpvo_ext_loader import load_extension

TREE = "dpvo_stereo"


def __getattr__(name: str) -> Any:
    module = load_extension(TREE, __name__, name)
    globals()[name] = module
    return module
