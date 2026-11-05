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

"""Utilities for injecting metric z-depth into DPVO's patch graph."""

from __future__ import annotations

import torch


def prepare_metric_disparity(
    depth: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    image_hw: tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | bool]]:
    """Convert full-resolution z-depth to DPVO's exact 1::4 grid.

    DPVO training subsamples inverse depth with ``[..., 1::4, 1::4]``.
    Reusing that convention is more precise than an arbitrary interpolation and
    keeps the supplied disparity grid aligned with ``Patchifier`` coordinates.
    Invalid pixels receive the frame median only for initialization; their
    returned validity is false, so they never create a BA prior factor.
    """

    height, width = (int(image_hw[0]), int(image_hw[1]))
    if height % 4 or width % 4:
        raise ValueError(f"DPVO metric-depth input must be divisible by 4: {image_hw}")
    if not (0.0 < float(min_depth) < float(max_depth)):
        raise ValueError(f"Invalid metric depth range: {min_depth}, {max_depth}")

    depth = torch.as_tensor(depth, dtype=torch.float32)
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    if depth.shape != (height, width):
        raise ValueError(
            f"Metric depth shape {tuple(depth.shape)} does not match image {image_hw}"
        )

    if mask is None:
        valid = torch.ones_like(depth, dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask, device=depth.device)
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.shape != depth.shape:
            raise ValueError(
                f"Metric mask shape {tuple(mask.shape)} does not match depth {tuple(depth.shape)}"
            )
        valid = mask.bool()

    valid &= torch.isfinite(depth)
    valid &= depth >= float(min_depth)
    valid &= depth <= float(max_depth)

    disparity = torch.zeros_like(depth)
    disparity[valid] = depth[valid].reciprocal()
    has_valid_full_resolution = bool(valid.any())
    if has_valid_full_resolution:
        fallback = disparity[valid].median()
    else:
        # Keep the utility output finite; DPVO rejects this diagnostic before
        # patchification, so the fallback can never enter the graph as 1 m.
        fallback = disparity.new_tensor(1.0)
    disparity = torch.where(valid, disparity, fallback)

    disparity_grid = disparity[1::4, 1::4].contiguous()[None, None]
    valid_grid = valid[1::4, 1::4].contiguous()[None, None]
    has_valid_depth = bool(valid_grid.any())
    expected = (height // 4, width // 4)
    if disparity_grid.shape[-2:] != expected:
        raise RuntimeError(
            f"Unexpected DPVO disparity grid {tuple(disparity_grid.shape[-2:])}, expected {expected}"
        )
    diagnostics = {
        "has_valid_depth": has_valid_depth,
        "has_valid_full_resolution": has_valid_full_resolution,
        "valid_pixel_fraction": float(valid.float().mean().item()),
        "fallback_disparity": float(fallback.item()),
    }
    return disparity_grid, valid_grid, diagnostics


def patch_metric_priors(
    patches: torch.Tensor,
    valid_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read center inverse-depth priors and validity for sampled DPVO patches."""

    if patches.ndim != 5 or patches.shape[0] != 1 or patches.shape[2] != 3:
        raise ValueError(f"Invalid DPVO patch tensor: {tuple(patches.shape)}")
    if valid_grid.ndim != 4 or valid_grid.shape[:2] != (1, 1):
        raise ValueError(f"Invalid metric valid grid: {tuple(valid_grid.shape)}")
    center = patches.shape[-1] // 2
    x = patches[0, :, 0, center, center].round().long()
    y = patches[0, :, 1, center, center].round().long()
    height, width = valid_grid.shape[-2:]
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = x.clamp(0, width - 1)
    safe_y = y.clamp(0, height - 1)
    priors = patches[0, :, 2, center, center].float()
    weights = valid_grid[0, 0, safe_y, safe_x].float() * in_bounds.float()
    weights *= torch.isfinite(priors).float() * (priors > 0).float()
    return priors.contiguous(), weights.contiguous()
