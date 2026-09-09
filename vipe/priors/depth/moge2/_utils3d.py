# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Portions of this file are adapted from utils3d
# (https://github.com/EasternJournalist/utils3d), MIT License, at commit
# 3fab839f0be9931dac7c8488eb0e1600c236e183 -- the commit MoGe pins in its
# pyproject.toml. The PyPI `utils3d` release has a different namespace layout,
# so depending on it instead would be a silent-breakage risk.
#
# Why this file exists: upstream MoGe references seven `utils3d` symbols, but the
# code that reached the other three was pruned along with the rest of the
# non-inference paths (see VENDORED.md), so only these four remain. Vendoring
# them keeps `utils3d` -- which upstream pins to a git commit -- out of the
# dependency set entirely.

from types import SimpleNamespace
from typing import Tuple, Union

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Vendored functions                                                          #
# --------------------------------------------------------------------------- #


def intrinsics_from_focal_center(
    fx: Union[float, Tensor],
    fy: Union[float, Tensor],
    cx: Union[float, Tensor],
    cy: Union[float, Tensor],
) -> Tensor:
    """OpenCV intrinsics matrix from focal lengths and principal point.

    Returns `[..., 3, 3]`. Upstream gets its argument handling from the
    `@totensor(_others=torch.float32)` / `@batched(_others=0)` decorators; the
    broadcast below reproduces that for the scalar-or-`[B]` arguments MoGe-2
    passes (`fx`/`fy` per batch element, `cx`/`cy` as 0-dim tensors).
    """
    tensors = [v for v in (fx, fy, cx, cy) if isinstance(v, Tensor)]
    dtype = tensors[0].dtype if tensors else torch.float32
    device = tensors[0].device if tensors else None

    fx, fy, cx, cy = (torch.as_tensor(v, dtype=dtype, device=device) for v in (fx, fy, cx, cy))
    fx, fy, cx, cy = torch.broadcast_tensors(fx, fy, cx, cy)

    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    return torch.stack(
        [
            fx, zeros, cx,
            zeros, fy, cy,
            zeros, zeros, ones,
        ],
        dim=-1,
    ).unflatten(-1, (3, 3))


def uv_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 0.0,
    left: float = 0.0,
    bottom: float = 1.0,
    right: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tensor:
    """Pixel-center UV coordinate map, `(height, width, 2)`, `(0,0)` at top-left."""
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.linspace(left + 0.5 / width, right - 0.5 / width, width, dtype=dtype, device=device)
    v = torch.linspace(top + 0.5 / height, bottom - 0.5 / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=2)


def unproject_cv(uv: Tensor, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor = None) -> Tensor:
    """Unproject `[..., N, 2]` UV + `[..., N]` depth to `[..., N, 3]` points (OpenCV convention)."""
    intrinsics = torch.cat(
        [
            torch.cat(
                [intrinsics, torch.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype, device=intrinsics.device)],
                dim=-1,
            ),
            torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device).expand(
                *intrinsics.shape[:-2], 1, 4
            ),
        ],
        dim=-2,
    )
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = torch.cat([uv, torch.ones((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1) * depth[..., None]
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1)
    points = points @ torch.linalg.inv(transform).mT
    return points[..., :3]


def depth_map_to_point_map(depth: Tensor, intrinsics: Tensor, extrinsics: Tensor = None) -> Tensor:
    """`[..., H, W]` depth -> `[..., H, W, 3]` camera-space points."""
    height, width = depth.shape[-2:]
    uv = uv_map(height, width, dtype=depth.dtype, device=depth.device)
    return unproject_cv(
        uv,
        depth,
        intrinsics=intrinsics[..., None, :, :],
        extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None,
    )


# --------------------------------------------------------------------------- #
# Namespace: MoGe calls these as `utils3d.pt.*`                               #
# --------------------------------------------------------------------------- #

pt = SimpleNamespace(
    intrinsics_from_focal_center=intrinsics_from_focal_center,
    uv_map=uv_map,
    unproject_cv=unproject_cv,
    depth_map_to_point_map=depth_map_to_point_map,
)
