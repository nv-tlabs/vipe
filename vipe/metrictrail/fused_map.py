# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense keyframe map for ViPE's depth fusion on DPVO (``post.depth_align_model``).

ViPE's ``AdaptiveDepthProcessor`` prompts its depth model with the SLAM map
projected into every frame. ViPE's own SLAM exports BA-refined dense depth of its
keyframes (1/8 resolution) as that map. DPVO only refines its sparse patches
(96 per keyframe), so its patch map can leave a frame with too few projected
points, above all once ViPE's dynamic masks remove the moving objects' patches;
ViPE's prompted depth model then fails ("There are not enough known points").

With depth fusion on, every keyframe therefore also contributes its dense depth:
the depth provider's metric depth (MoGe-2, the depth DPVO tracked with), sampled
every ``stride`` pixels with the dynamic and invalid pixels left out, and scaled
to the BA-refined patch depths of that keyframe (the median ratio over its
patches). The patch points stay in the map unchanged. Without depth fusion
nothing here runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

#: Sampling stride of the dense depth, in input pixels (ViPE's SLAM map is at 1/8 resolution).
DENSE_STRIDE = 8
#: Fewest patches with a provider depth needed to scale a keyframe on its own.
MIN_SCALE_SAMPLES = 8


class KeyframeDepthKeeper:
    """Every frame's provider depth (0 = no depth) and RGB, sampled every ``stride`` pixels.

    Filled frame by frame while tracking (any frame can become a keyframe); read for
    the keyframes DPVO keeps when the map is built.
    """

    def __init__(self, count: int, image_hw: tuple[int, int], device: torch.device, stride: int = DENSE_STRIDE):
        self.stride = int(stride)
        self.offset = self.stride // 2
        rows = torch.arange(self.offset, int(image_hw[0]), self.stride)
        cols = torch.arange(self.offset, int(image_hw[1]), self.stride)
        self.grid_hw = (int(rows.numel()), int(cols.numel()))
        self.depth = torch.zeros((int(count), *self.grid_hw), dtype=torch.float32, device=device)
        self.rgb = torch.zeros((int(count), 3, *self.grid_hw), dtype=torch.uint8, device=device)
        self.kept = torch.zeros(int(count), dtype=torch.bool, device=device)

    def keep(self, index: int, depth: torch.Tensor, valid: torch.Tensor, rgb_chw: torch.Tensor) -> None:
        """Store frame ``index``: ``depth`` (H,W) metres where ``valid`` (H,W), ``rgb_chw`` (3,H,W) in [0,1]."""

        o, s = self.offset, self.stride
        sampled = torch.where(valid, depth, torch.zeros_like(depth))[o::s, o::s]
        self.depth[index] = torch.nan_to_num(sampled.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.rgb[index] = rgb_chw[:, o::s, o::s].mul(255.0).round().clamp(0.0, 255.0).to(torch.uint8)
        self.kept[index] = True

    def pixel_grid(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """(u, v) input-pixel coordinates (Hs,Ws) of the samples."""

        rows = torch.arange(self.grid_hw[0], device=device, dtype=torch.float32) * self.stride + self.offset
        cols = torch.arange(self.grid_hw[1], device=device, dtype=torch.float32) * self.stride + self.offset
        v, u = torch.meshgrid(rows, cols, indexing="ij")
        return u, v


@dataclass
class FusedKeyframePoints:
    xyz: torch.Tensor  # (n,G,3) world
    rgb: torch.Tensor  # (n,G,3) in [0,1]
    valid: torch.Tensor  # (n,G)
    record: dict[str, Any]


def fused_keyframe_points(
    keeper: KeyframeDepthKeeper,
    ordinals: np.ndarray,
    w2c: torch.Tensor,
    k4: np.ndarray,
    patch_world: torch.Tensor,
    patch_valid: torch.Tensor,
) -> FusedKeyframePoints:
    """Dense world points of keyframes ``ordinals`` (frame indices), scaled to their refined patches.

    ``w2c`` (n,4,4) world-to-camera of the keyframes, ``k4`` the input-raster pinhole
    ``fx, fy, cx, cy``, ``patch_world`` (n,M,3) the keyframes' refined patch points
    and ``patch_valid`` (n,M) the ones in the map (static, finite). A keyframe's scale
    is the median of patch depth / provider depth over its patches that land on a
    sample with provider depth; with fewer than ``MIN_SCALE_SAMPLES`` of them, the
    median over all keyframes' patches (1 if there are none).
    """

    device = keeper.depth.device
    index = torch.as_tensor(np.asarray(ordinals, dtype=np.int64), device=device)
    if not bool(keeper.kept[index].all()):
        raise RuntimeError("a DPVO keyframe has no kept provider depth")
    depth = keeper.depth[index]  # (n,Hs,Ws)
    rgb = keeper.rgb[index].float().div(255.0)  # (n,3,Hs,Ws)
    w2c = w2c.to(device=device, dtype=torch.float32)
    fx, fy, cx, cy = (float(value) for value in np.asarray(k4, dtype=np.float64).reshape(4))
    n, (hs, ws) = int(index.numel()), keeper.grid_hw

    # Scale per keyframe: refined patch depth over the provider depth at the patch's sample.
    world = patch_world.to(device=device, dtype=torch.float32)
    camera = torch.einsum("nij,nmj->nmi", w2c[:, :3, :3], world) + w2c[:, None, :3, 3]
    z = camera[..., 2]
    in_front = patch_valid.to(device) & torch.isfinite(z) & (z > 0)
    safe_z = torch.where(in_front, z, torch.ones_like(z))
    col = torch.floor((fx * camera[..., 0] / safe_z + cx) / keeper.stride).long()
    row = torch.floor((fy * camera[..., 1] / safe_z + cy) / keeper.stride).long()
    inside = in_front & (col >= 0) & (col < ws) & (row >= 0) & (row < hs)
    sampled = depth[torch.arange(n, device=device)[:, None], row.clamp(0, hs - 1), col.clamp(0, ws - 1)]
    usable = inside & (sampled > 0)
    ratio = torch.where(usable, z / torch.where(usable, sampled, torch.ones_like(sampled)), torch.nan)
    counts = usable.sum(dim=1)
    global_scale = float(torch.nanmedian(ratio).item()) if bool(usable.any()) else 1.0
    per_keyframe = torch.nanmedian(ratio, dim=1).values
    fallback = (counts < MIN_SCALE_SAMPLES) | ~torch.isfinite(per_keyframe)
    scale = torch.where(fallback, torch.full_like(per_keyframe, global_scale), per_keyframe)

    # Dense points: scaled provider depth through the pinhole, then to world.
    u, v = keeper.pixel_grid(device)
    d = depth * scale[:, None, None]
    points = torch.stack(((u - cx) / fx * d, (v - cy) / fy * d, d), dim=-1).reshape(n, hs * ws, 3)
    c2w_rot = w2c[:, :3, :3].transpose(1, 2)
    xyz = torch.einsum("nij,nmj->nmi", c2w_rot, points - w2c[:, None, :3, 3])
    valid = (depth > 0).reshape(n, hs * ws) & torch.isfinite(xyz).all(dim=-1)
    finite = scale[torch.isfinite(scale)]
    record = {
        "enabled": True,
        "rule": "provider metric depth every stride px (dynamic/invalid px left out), scaled per keyframe to the "
        "median refined-patch / provider depth ratio, plus DPVO's patch points",
        "stride_px": keeper.stride,
        "grid_hw": [hs, ws],
        "keyframes": n,
        "dense_points": int(valid.sum().item()),
        "scale_median": float(finite.median().item()) if finite.numel() else None,
        "scale_p10_p90": (
            [float(torch.quantile(finite, 0.1).item()), float(torch.quantile(finite, 0.9).item())]
            if finite.numel()
            else None
        ),
        "scale_fallback_keyframes": int(fallback.sum().item()),
        "global_scale": global_scale,
    }
    return FusedKeyframePoints(xyz=xyz, rgb=rgb.permute(0, 2, 3, 1).reshape(n, hs * ws, 3), valid=valid, record=record)
