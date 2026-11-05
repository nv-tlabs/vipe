# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Age gate of the final re-measurement (M1; REMEASURE_MIN_AGE; EFFECTIVE_CHANGES §12.1).

Pure tensor arithmetic (no CUDA extension), used ONLY from DPVO._remeasure_final -> DPVO.refresh_factors
(the terminate-time re-measured global BA).  The tracking-time update() never calls refresh_factors, so the
flag cannot touch it.  The default (REMEASURE_MIN_AGE 0) leaves every value byte-identical to the ungated
re-measurement.

Hypothesis: the DPVO update operator is only reliable inside its correlation radius around the current
reprojection; re-measuring the young in-window edges trades floor for plant / 360 / room (EFFECTIVE_CHANGES
§12.1), so the gate is exposed but off in the shipped TUM preset.

The per-edge delta gate that used to live here (REMEASURE_DELTA_GATE, delta_gate_mask / apply_delta_gate) was
tested and rejected (arms g1 / g2 / g4, §12.1) and is not part of the consolidated backend; CHANGELOG.md
records where it lives.
"""

from __future__ import annotations

import torch


def age_mask(ii: torch.Tensor, jj: torch.Tensor, min_age: int) -> torch.Tensor:
    """bool (E,) True where the edge is ELIGIBLE for re-measurement: jj - ii >= min_age (keyframe-index age of
    the target frame relative to the patch's source frame).  min_age <= 0 -> every edge eligible."""
    if min_age is None or int(min_age) <= 0:
        return torch.ones(ii.shape, dtype=torch.bool, device=ii.device)
    return (jj - ii) >= int(min_age)
