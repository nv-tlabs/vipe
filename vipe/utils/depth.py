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

from typing import Literal, Optional

import torch
import torch.nn.functional as F

from vipe.ext.lietorch import SE3


def get_pixel_uv(
    batch: int,
    height: int,
    width: int,
    device: torch.device | str,
    center: bool = False,
):
    """
    Get pixel center coordinates for a given batch, height and width.

    Args:
        batch (int): Batch size.
        height (int): Height of the image.
        width (int): Width of the image.
        device (torch.device | str): Device to put the tensor.
        center (bool): If True, the pixel coordinates are at the center of the pixel.

    Returns:
        torch.Tensor: Pixel coordinates with shape (batch, height, width, 2).
    """
    offset = 0.5 if center else 0.0
    uu, vv = torch.meshgrid(
        torch.arange(width, device=device) + offset,
        torch.arange(height, device=device) + offset,
        indexing="xy",
    )
    return torch.stack([uu, vv], dim=-1).expand(batch, -1, -1, -1)


def normal_weight_from_xyz(xyz: torch.Tensor, robust: bool = True) -> torch.Tensor:
    """
    Compute normal from XYZ in camera space.

    Args:
        xyz (torch.Tensor): XYZ with shape ([B], H, W, 3).

    Returns:
        normal_weight (torch.Tensor): Concatenation of normal and weight of shape ([B], H, W, 4)
    """
    batch_dim = xyz.dim() == 4
    if not batch_dim:
        xyz = xyz.unsqueeze(0)

    from vipe.ext import _C

    assert xyz.size(0) == 1, "Batch size must be 1."
    if robust:
        normal_weight = _C.compute_normal_weight_robust(xyz[0])
    else:
        normal_weight = _C.compute_normal_weight(xyz[0])
    normal_weight = normal_weight.unsqueeze(0)

    return normal_weight if batch_dim else normal_weight.squeeze(0)


def get_camera_rays(height: int, width: int, intrinsics: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Unproject a depth image to a point cloud using pinhole camera model.

    Args:
        height (int): Height of the depth image.
        width (int): Width of the depth image.
        intrinsics (torch.Tensor): Intrinsics matrix with shape ([B], 4) fx, fy, cx, cy.
        normalize (bool): If False, depth is z, otherwise depth is distance to camera center.

    Returns:
        torch.Tensor: XYZ with shape ([B], H, W, 3).
    """
    batch_dim = intrinsics.dim() == 2
    if not batch_dim:
        intrinsics = intrinsics.unsqueeze(0)

    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    device = intrinsics.device
    b = intrinsics.shape[0]

    uu, vv = get_pixel_uv(b, height, width, device).unbind(dim=-1)
    uu = (uu - cx.view(b, 1, 1)) / fx.view(b, 1, 1)
    vv = (vv - cy.view(b, 1, 1)) / fy.view(b, 1, 1)

    xyz = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)

    if normalize:
        xyz = xyz / torch.linalg.norm(xyz, dim=-1, keepdim=True)

    return xyz if batch_dim else xyz.squeeze(0)


def get_camera_rays_panorama(height: int, width: int) -> torch.Tensor:
    v, u = torch.meshgrid(torch.linspace(0.0, 1.0, height), torch.linspace(0.0, 1.0, width), indexing="ij")
    v = v * torch.pi
    u = (u - 0.5) * 2 * torch.pi
    y = -torch.cos(v)
    x = torch.sin(u) * torch.sin(v)
    z = torch.cos(u) * torch.sin(v)
    return torch.stack([x, y, z], dim=-1)


def bilinear_splatting_inplace(
    data: torch.Tensor,
    uv: torch.Tensor,
    out_data: torch.Tensor,
    out_weight: torch.Tensor,
):
    H, W, V = out_data.shape
    assert out_weight.shape == (H, W)
    assert data.shape[-1] == V

    # UV is image-space coordinates [0, W] x [0, H]
    u, v = uv.unbind(dim=-1)
    x0 = torch.floor(u + 0.5).long()
    x1 = x0 + 1
    y0 = torch.floor(v + 0.5).long()
    y1 = y0 + 1
    in_mask = torch.where((x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H))[0]
    x0, x1, y0, y1 = x0[in_mask], x1[in_mask], y0[in_mask], y1[in_mask]
    data, u, v = data[in_mask], u[in_mask], v[in_mask]
    wx, wy = u - x0.float(), v - y0.float()

    flat_out = out_data.view(-1, V)
    flat_weight = out_weight.view(-1)

    def splat(x_idx, y_idx, w):
        idx = y_idx * W + x_idx  # flatten 2D to 1D
        flat_out.index_add_(0, idx, data * w.unsqueeze(-1))
        flat_weight.index_add_(0, idx, w)

    splat(x0, y0, (1 - wx) * (1 - wy))
    splat(x0, y1, (1 - wx) * wy)
    splat(x1, y0, wx * (1 - wy))
    splat(x1, y1, wx * wy)


def bilinear_splatting(
    data: torch.Tensor,
    uv: torch.Tensor,
    weight: torch.Tensor | None = None,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
):
    """
    Forward warp using bilinear splatting.

    Args:
        data (torch.Tensor): Input data with shape ([B], H, W, C).
        uv (torch.Tensor): UV coordinates with shape ([B], H, W, 2).
            Note that this is the pixel center coordinates, e.g. 0.5, 0.5 is the center of the top-left pixel.
        weight (torch.Tensor): Weight with shape ([B], H, W).
        target_height (int): Target height.
        target_width (int): Target width.

    Returns:
        warped_data(torch.Tensor): Warped data with shape ([B], target_height, target_width, C).
        warped_weight(torch.Tensor): Warped weight with shape ([B], target_height, target_width).

    TODO: Merge with flow.forward_warp where a kNN strategy is used.
    """
    batch_dim = data.dim() == 4
    if not batch_dim:
        data = data.unsqueeze(0)
        uv = uv.unsqueeze(0)
        if weight is not None:
            weight = weight.unsqueeze(0)

    if target_height is None:
        target_height = uv.shape[1]

    if target_width is None:
        target_width = uv.shape[2]

    b, h, w, c = data.shape

    if weight is None:
        weight = torch.ones(size=(b, h, w)).to(data)

    trans_pos_offset = uv + 0.5
    trans_pos_floor = torch.floor(trans_pos_offset).long()  # (b, h, w, 2)
    trans_pos_ceil = torch.ceil(trans_pos_offset).long()  # (b, h, w, 2)

    def clamp_and_stack(src: torch.Tensor):
        return torch.stack(
            [
                torch.clamp(src[..., 0], min=0, max=target_width + 1),
                torch.clamp(src[..., 1], min=0, max=target_height + 1),
            ],
            dim=-1,
        )

    trans_pos_offset = clamp_and_stack(trans_pos_offset)
    trans_pos_floor = clamp_and_stack(trans_pos_floor)
    trans_pos_ceil = clamp_and_stack(trans_pos_ceil)  # (b, h, w, 2)

    weight_nw = (
        (1 - (trans_pos_offset[..., 1] - trans_pos_floor[..., 1]))
        * (1 - (trans_pos_offset[..., 0] - trans_pos_floor[..., 0]))
        * weight
    )
    weight_sw = (
        (1 - (trans_pos_ceil[..., 1] - trans_pos_offset[..., 1]))
        * (1 - (trans_pos_offset[..., 0] - trans_pos_floor[..., 0]))
        * weight
    )
    weight_ne = (
        (1 - (trans_pos_offset[..., 1] - trans_pos_floor[..., 1]))
        * (1 - (trans_pos_ceil[..., 0] - trans_pos_offset[..., 0]))
        * weight
    )
    weight_se = (
        (1 - (trans_pos_ceil[..., 1] - trans_pos_offset[..., 1]))
        * (1 - (trans_pos_ceil[..., 0] - trans_pos_offset[..., 0]))
        * weight
    )

    warped_frame = torch.zeros(size=(b, target_height + 2, target_width + 2, c), dtype=data.dtype).to(data)
    warped_weights = torch.zeros(size=(b, target_height + 2, target_width + 2)).to(data)
    batch_indices = torch.arange(b)[:, None, None].to(data.device)

    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[..., 1], trans_pos_floor[..., 0]),
        data * weight_nw.unsqueeze(-1),
        accumulate=True,
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[..., 1], trans_pos_floor[..., 0]),
        data * weight_sw.unsqueeze(-1),
        accumulate=True,
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[..., 1], trans_pos_ceil[..., 0]),
        data * weight_ne.unsqueeze(-1),
        accumulate=True,
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[..., 1], trans_pos_ceil[..., 0]),
        data * weight_se.unsqueeze(-1),
        accumulate=True,
    )

    warped_weights.index_put_(
        (batch_indices, trans_pos_floor[..., 1], trans_pos_floor[..., 0]),
        weight_nw,
        accumulate=True,
    )
    warped_weights.index_put_(
        (batch_indices, trans_pos_ceil[..., 1], trans_pos_floor[..., 0]),
        weight_sw,
        accumulate=True,
    )
    warped_weights.index_put_(
        (batch_indices, trans_pos_floor[..., 1], trans_pos_ceil[..., 0]),
        weight_ne,
        accumulate=True,
    )
    warped_weights.index_put_(
        (batch_indices, trans_pos_ceil[..., 1], trans_pos_ceil[..., 0]),
        weight_se,
        accumulate=True,
    )

    warped_frame = warped_frame[:, 1:-1, 1:-1]
    warped_weights = warped_weights[:, 1:-1, 1:-1]

    warped_frame = torch.where(
        (warped_weights > 0).unsqueeze(-1).repeat(1, 1, 1, c),
        warped_frame / warped_weights.unsqueeze(-1),
        torch.zeros_like(warped_frame),
    )

    if not batch_dim:
        warped_frame = warped_frame.squeeze(0)
        warped_weights = warped_weights.squeeze(0)

    return warped_frame, warped_weights


def reproject(
    frame1: torch.Tensor,
    depth1: torch.Tensor,
    pose1: SE3,
    intrinsic1: torch.Tensor,
    pose2: SE3,
    intrinsic2: torch.Tensor,
    normal1: torch.Tensor | None = None,
    filtering: Literal["normal", "none"] = "normal",
    height2: Optional[int] = None,
    width2: Optional[int] = None,
    mask1: torch.Tensor | None = None,
):
    """
    Reproject a frame to another frame using depth and normal information.

    Args:
        frame1 (torch.Tensor): Frame with shape ([B], H, W, C).
        depth1 (torch.Tensor): Depth with shape ([B], H, W).
        intrinsic1 (torch.Tensor): Intrinsics matrix with shape ([B], 4).
        intrinsic2 (torch.Tensor): Intrinsics matrix with shape ([B], 4).
        normal1 (torch.Tensor): Normal with shape ([B], H, W, 3).
        filtering (str): Filtering method, either "normal" or "none".
        height2 (int): Height of the target frame.
        width2 (int): Width of the target frame.
        mask1 (torch.Tensor): Mask with shape ([B], H, W).

    Returns:
        torch.Tensor: Reprojected frame with shape ([B], H, W, C).
        torch.Tensor: Weight with shape ([B], H, W).
    """
    batch_dim = frame1.dim() == 4
    if not batch_dim:
        frame1 = frame1.unsqueeze(0)
        depth1 = depth1.unsqueeze(0)
        intrinsic1 = intrinsic1.unsqueeze(0)
        intrinsic2 = intrinsic2.unsqueeze(0)
        pose1 = pose1[None]
        pose2 = pose2[None]
        if normal1 is not None:
            normal1 = normal1.unsqueeze(0)
        if mask1 is not None:
            mask1 = mask1.unsqueeze(0)

    b, h, w, c = frame1.shape
    rel_pose = (pose2.inv() * pose1).matrix()

    cam_rays1 = get_camera_rays(h, w, intrinsic1)
    xyz1 = cam_rays1 * depth1.unsqueeze(-1)
    xyz2 = torch.einsum("bij,bhwj->bhwi", rel_pose[:, :3, :3], xyz1) + rel_pose[:, None, None, :3, 3]
    depth2 = xyz2[..., 2]
    render_mask = torch.logical_and(depth2 > 1e-6, depth1 > 1e-6)

    # Backface culling
    if filtering == "normal":
        if normal1 is None:
            normal_weight = normal_weight_from_xyz(xyz1)
            normal1 = normal_weight[..., :3]
            normal_mask = normal_weight[..., 3] > 0.0
        else:
            normal_mask = torch.ones_like(render_mask)

        normal2 = torch.einsum("bij,bhwj->bhwi", rel_pose[:, :3, :3], normal1)
        normal_mask = torch.logical_and(normal_mask, torch.sum(normal2 * xyz2, dim=-1) < 0)
        render_mask = torch.logical_and(render_mask, normal_mask)

    sat_depth = torch.clamp(depth2, min=0, max=1000)
    log_depth = torch.log(1 + sat_depth)
    depth_weight = torch.exp(log_depth / (log_depth.max() + 1e-7) * 50)
    flow_weight = render_mask.float() / depth_weight
    if mask1 is not None:
        flow_weight = flow_weight * mask1.float()

    uv2 = xyz2[..., :2] / (depth2.unsqueeze(-1) + 1e-6)
    uv2[..., 0] = uv2[..., 0] * intrinsic2[:, 0] + intrinsic2[:, 2]
    uv2[..., 1] = uv2[..., 1] * intrinsic2[:, 1] + intrinsic2[:, 3]
    res, weight = bilinear_splatting(frame1, uv2, flow_weight, height2, width2)
    if not batch_dim:
        res = res.squeeze(0)
        weight = weight.squeeze(0)
    return res, weight


def reliable_depth_mask_range(
    depth: torch.Tensor,
    window_size: int = 5,
    ratio_thresh: float = 0.1,
    eps: float = 1e-6,
):
    """
    Generate a mask for reliable depth pixels based on local consistency.

    A pixel is reliable if the variation in its surrounding depth values
    (measured as the ratio of the local range to the local mean) is low.

    Args:
        depth (torch.Tensor): Depth image of shape (H, W).
        window_size (int): Size of the local neighborhood (must be odd).
        ratio_thresh (float): Maximum allowed variation ratio.
        eps (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Boolean mask (H, W) where True indicates a reliable depth pixel.
    """
    assert window_size % 2 == 1, "Window size must be odd."

    # Prepare the tensor for 2D pooling.
    depth_unsq = depth.unsqueeze(0).unsqueeze(0)

    # Compute local max, min, and mean values using pooling.
    local_max = F.max_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_min = -F.max_pool2d(-depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_mean = F.avg_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)

    # Calculate the ratio of the local range to the local mean.
    ratio = (local_max - local_min) / (local_mean + eps)
    ratio = ratio.squeeze(0).squeeze(0)  # Restore original shape (H, W)

    # Mark pixels as reliable if their local variation is below the threshold.
    reliable_mask = (ratio < ratio_thresh) & (depth > 0)
    return reliable_mask
