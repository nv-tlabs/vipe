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

import math

import torch
from vipe.metrictrail.dpvo_stereo_ext import cuda_ba  # [vipe] was: import cuda_ba

neighbors = cuda_ba.neighbors


def _prepare_stereo_edges(edge_is_stereo, edge_count, device, stereo_baseline_m):
    """Validate and normalize the optional per-edge stereo metadata."""

    stereo_baseline_m = float(stereo_baseline_m)
    if not math.isfinite(stereo_baseline_m) or stereo_baseline_m < 0.0:
        raise ValueError("stereo_baseline_m must be finite and non-negative")

    if edge_is_stereo is None:
        edge_is_stereo = torch.empty(0, dtype=torch.bool, device=device)
    else:
        edge_is_stereo = torch.as_tensor(edge_is_stereo)
        if edge_is_stereo.dtype != torch.bool:
            raise TypeError("edge_is_stereo must have boolean dtype")
        if edge_is_stereo.ndim != 1:
            raise ValueError("edge_is_stereo must be a one-dimensional tensor")
        if edge_is_stereo.numel() != edge_count:
            raise ValueError(
                "edge_is_stereo must contain one flag per edge "
                f"({edge_count}), got {edge_is_stereo.numel()}"
            )
        edge_is_stereo = edge_is_stereo.to(device=device).contiguous()
        # Avoid a Python-side CUDA synchronization on every BA call. The C++
        # boundary performs the same active-edge check for CUDA tensors.
        if (
            edge_is_stereo.device.type != "cuda"
            and edge_is_stereo.any().item()
            and stereo_baseline_m <= 0.0
        ):
            raise ValueError(
                "stereo_baseline_m must be positive when any stereo edge is enabled"
            )

    return edge_is_stereo, stereo_baseline_m


def reproject(
    poses,
    patches,
    intrinsics,
    ii,
    jj,
    kk,
    *,
    edge_is_stereo=None,
    stereo_baseline_m=0.0,
):
    """Reproject patches, optionally using a fixed left-to-right stereo rig edge."""

    edge_is_stereo, stereo_baseline_m = _prepare_stereo_edges(
        edge_is_stereo, ii.numel(), patches.device, stereo_baseline_m
    )
    return cuda_ba.reproject(
        poses,
        patches,
        intrinsics,
        ii,
        jj,
        kk,
        edge_is_stereo,
        stereo_baseline_m,
    )

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
    edge_is_stereo=None,
    stereo_baseline_m=0.0,
):
    """Run fixed-lag BA with optional metric-depth and stereo factors.

    Stereo edges use the fixed left-to-right transform ``[-baseline, 0, 0]``
    and contribute only to the inverse-depth normal equations. Camera poses are
    therefore never allowed to absorb the physical rig baseline.
    """

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

    edge_is_stereo, stereo_baseline_m = _prepare_stereo_edges(
        edge_is_stereo, ii.numel(), patches.device, stereo_baseline_m
    )

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
        edge_is_stereo,
        stereo_baseline_m,
    )
