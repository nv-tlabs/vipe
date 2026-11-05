# SPDX-FileCopyrightText: Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
# SPDX-License-Identifier: MIT
#
# Source: https://github.com/rusty1s/pytorch_scatter (2.1.2).
# See THIRD_PARTY_LICENSES.md.

# Modified for MetricTrail integration.
# The following Apache-2.0 terms apply to NVIDIA modifications only.
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

"""The ``torch_scatter`` functions the vendored DPVO calls, in plain PyTorch.

DPVO imports ``torch_scatter`` (``blocks.py``, ``net.py``, ``ba.py``); the
vendored copy imports this module instead (``# [vipe]`` import rewrites, see
``dpvo/VENDORED.md``). ``broadcast``, ``scatter_sum`` and ``scatter_softmax``
are the Python code of ``torch_scatter`` 2.1.2 unchanged. Its compiled
``scatter_max`` is replaced by ``Tensor.scatter_reduce(..., "amax",
include_self=False)`` on a zero-initialised output: a maximum is exact in any
order, and segments without an element stay 0 in both (``scatter_softmax``
only gathers non-empty segments). No ViPE extension is needed.
"""

from __future__ import annotations

from typing import Optional

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def _output_size(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: Optional[int]) -> list[int]:
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return size


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        out = torch.zeros(_output_size(src, index, dim, dim_size), dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_max_values(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """``torch_scatter.scatter_max(...)[0]``: per-segment maximum, 0 for empty segments."""

    index = broadcast(index, src, dim)
    out = torch.zeros(_output_size(src, index, dim, dim_size), dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(dim, index, src, reduce="amax", include_self=False)


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_softmax` can only be computed over tensors with floating point data types.")

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max_values(src, index, dim=dim, dim_size=dim_size)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = scatter_sum(recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)
