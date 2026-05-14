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

import torch


def align_inv_depth_to_depth(
    source_inv_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    quantile_masking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply affine transformation to align source inverse depth to target depth.

    Args:
        source_inv_depth: Inverse depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
        scale: Scaling factor.
        bias: Bias term.
    """
    target_inv_depth = 1.0 / target_depth
    source_mask = source_inv_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    if quantile_masking:
        outlier_quantiles = torch.tensor([0.1, 0.9], device=source_inv_depth.device)
        source_data_low, source_data_high = torch.quantile(source_inv_depth[source_mask], outlier_quantiles)
        target_data_low, target_data_high = torch.quantile(target_inv_depth[target_mask], outlier_quantiles)
        source_mask = (source_inv_depth > source_data_low) & (source_inv_depth < source_data_high)
        target_mask = (target_inv_depth > target_data_low) & (target_inv_depth < target_data_high)

    mask = torch.logical_and(source_mask, target_mask)

    source_data = source_inv_depth[mask].view(-1, 1)
    target_data = target_inv_depth[mask].view(-1, 1)

    ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
    source_data_h = torch.cat([source_data, ones], dim=1)
    transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

    scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
    aligned_inv_depth = source_inv_depth * scale + bias
    aligned_depth = torch.clamp(aligned_inv_depth.reciprocal(), min=1e-4)

    return aligned_depth, scale, bias


def align_depth_to_depth(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    quantile_masking: bool = True,
    bias: bool = True,
) -> torch.Tensor:
    """
    Apply affine transformation to align source depth to target depth.

    Args:
        source_inv_depth: Depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
    """
    source_mask = source_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    if quantile_masking:
        outlier_quantiles = torch.tensor([0.1, 0.9], device=source_depth.device)

        source_data_low, source_data_high = torch.quantile(source_depth[source_mask], outlier_quantiles)
        target_data_low, target_data_high = torch.quantile(target_depth[target_mask], outlier_quantiles)
        source_mask = (source_depth > source_data_low) & (source_depth < source_data_high)
        target_mask = (target_depth > target_data_low) & (target_depth < target_data_high)

    mask = torch.logical_and(source_mask, target_mask)

    source_data = source_depth[mask].view(-1, 1)
    target_data = target_depth[mask].view(-1, 1)

    if not bias:
        ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
        source_data_h = torch.cat([source_data, ones], dim=1)
        transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

        scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
        aligned_depth = source_depth * scale + bias
        aligned_depth = torch.clamp(aligned_depth, min=1e-4)

    else:
        scale = torch.median(target_data / source_data)
        aligned_depth = source_depth * scale
        aligned_depth = torch.clamp(aligned_depth, min=1e-4)

    return aligned_depth
