# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # (B, S, H, D) -> (B, H, S, D)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias)

    return out.transpose(1, 2)


def scaled_index_add(
    input: torch.Tensor,  # [B, M, D]
    index: torch.Tensor,  # [Bi] - int64
    source: torch.Tensor,  # [Bi, M, D]
    scaling: Optional[torch.Tensor] = None,  # [D]
    alpha: float = 1.0,
) -> torch.Tensor:
    if scaling is not None:
        source = scaling * source
    return torch.index_add(input, dim=0, source=source, index=index, alpha=alpha)


def index_select_cat(sources: Sequence[torch.Tensor], indices: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([s[i.long()].flatten() for s, i in zip(sources, indices)], dim=0)
