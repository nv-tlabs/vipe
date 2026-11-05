# SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
# SPDX-License-Identifier: MIT
#
# Source: https://github.com/princeton-vl/DPVO
# See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

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

import torch
from vipe.metrictrail.dpvo_ext import cuda_ba  # [vipe] was: import cuda_ba

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(
    poses,
    patches,
    intrinsics,
    target,
    weight,
    lmbda,
    ii,
    jj,
    kk,
    t0,
    t1,
    M,
    iterations,
    eff_impl=False,
    *,
    depth_prior=None,
    depth_prior_weight=None,
    depth_prior_lambda=0.0,
    depth_prior_huber=0.3,
):
    """Run fixed-lag BA with an optional robust log-inverse-depth prior."""

    if depth_prior is None:
        depth_prior = torch.empty(0, dtype=torch.float32, device=patches.device)
    if depth_prior_weight is None:
        depth_prior_weight = torch.empty(
            0, dtype=torch.float32, device=patches.device
        )
    depth_prior = depth_prior.to(
        device=patches.device, dtype=torch.float32
    ).contiguous()
    depth_prior_weight = depth_prior_weight.to(
        device=patches.device, dtype=torch.float32
    ).contiguous()
    if depth_prior.numel() != depth_prior_weight.numel():
        raise ValueError(
            "depth_prior and depth_prior_weight must have equal flattened size"
        )
    patch_count = patches.numel() // (
        patches.shape[-3] * patches.shape[-2] * patches.shape[-1]
    )
    if depth_prior.numel() not in (0, patch_count):
        raise ValueError(
            "depth_prior must be empty or contain one value per flattened "
            f"patch ({patch_count}), got {depth_prior.numel()}"
        )
    if float(depth_prior_lambda) > 0 and depth_prior.numel() == 0:
        raise ValueError("a positive depth_prior_lambda requires depth priors")
    if float(depth_prior_lambda) < 0:
        raise ValueError("depth_prior_lambda must be non-negative")
    if float(depth_prior_huber) <= 0:
        raise ValueError("depth_prior_huber must be positive")

    return cuda_ba.forward(
        poses.data,
        patches,
        intrinsics,
        target,
        weight,
        lmbda,
        depth_prior,
        depth_prior_weight,
        float(depth_prior_lambda),
        float(depth_prior_huber),
        ii,
        jj,
        kk,
        M,
        t0,
        t1,
        iterations,
        eff_impl,
    )
