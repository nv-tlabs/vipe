# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense keyframe map for ViPE's depth fusion: provider depth scaled to the refined patches."""

from __future__ import annotations

import numpy as np
import torch

from vipe.metrictrail.fused_map import DENSE_STRIDE, MIN_SCALE_SAMPLES, KeyframeDepthKeeper, fused_keyframe_points

H, W = 64, 96
K4 = np.array([80.0, 80.0, 48.0, 32.0])


def _w2c(tx: float) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = -tx  # camera centre at x = tx, looking along +z
    return pose


def _world_points(w2c: torch.Tensor, u: torch.Tensor, v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    fx, fy, cx, cy = K4
    camera = torch.stack(((u - cx) / fx * z, (v - cy) / fy * z, z), dim=-1)
    return (camera - w2c[:3, 3]) @ w2c[:3, :3]  # R^T (X - t)


def test_dense_points_take_the_scale_of_the_refined_patches():
    keeper = KeyframeDepthKeeper(count=3, image_hw=(H, W), device=torch.device("cpu"))
    true_depth = torch.full((H, W), 4.0)
    true_depth[:, W // 2 :] = 6.0  # a step: two planes
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[: H // 4] = False  # e.g. dynamic pixels: no depth, not in the map
    rgb = torch.rand(3, H, W)
    provider_scale = {0: 0.5, 2: 0.8}  # the provider is off by a per-frame scale
    for index in (0, 1, 2):
        keeper.keep(index, true_depth * provider_scale.get(index, 1.0), valid, rgb)
    ordinals = np.array([0, 2])
    w2c = torch.stack((_w2c(0.0), _w2c(1.5)))

    # refined patches: true geometry at random static pixels (one wild outlier per keyframe)
    generator = torch.Generator().manual_seed(0)
    u = torch.randint(0, W, (2, 40), generator=generator).float()
    v = torch.randint(H // 4, H, (2, 40), generator=generator).float()
    z = true_depth[v.long(), u.long()]
    z[:, 0] *= 10.0
    patches = torch.stack([_world_points(w2c[k], u[k], v[k], z[k]) for k in range(2)])
    patch_valid = torch.ones(2, 40, dtype=torch.bool)

    fused = fused_keyframe_points(keeper, ordinals, w2c, K4, patches, patch_valid)

    assert fused.record["scale_fallback_keyframes"] == 0
    assert np.allclose(sorted(fused.record["scale_p10_p90"]), [1 / 0.8, 1 / 0.5], rtol=0.15)
    assert fused.xyz.shape == (2, keeper.grid_hw[0] * keeper.grid_hw[1], 3)
    gu, gv = keeper.pixel_grid(torch.device("cpu"))
    for k in range(2):
        expected = _world_points(w2c[k], gu.flatten(), gv.flatten(), true_depth[gv.long(), gu.long()].flatten())
        keep = fused.valid[k]
        assert keep.sum() > 0 and not keep[: keeper.grid_hw[1]].any()  # the masked top rows are left out
        assert torch.allclose(fused.xyz[k][keep], expected[keep], atol=1e-3)


def test_too_few_patches_fall_back_to_the_global_scale():
    keeper = KeyframeDepthKeeper(count=2, image_hw=(H, W), device=torch.device("cpu"))
    depth = torch.full((H, W), 2.0)
    for index in (0, 1):
        keeper.keep(index, depth, torch.ones(H, W, dtype=torch.bool), torch.zeros(3, H, W))
    w2c = torch.stack((_w2c(0.0), _w2c(0.5)))
    count = MIN_SCALE_SAMPLES + 4
    u = torch.full((2, count), 40.0)
    v = torch.full((2, count), 30.0)
    z = torch.full((2, count), 3.0)  # patches say 1.5x the provider depth
    patches = torch.stack([_world_points(w2c[k], u[k], v[k], z[k]) for k in range(2)])
    patch_valid = torch.ones(2, count, dtype=torch.bool)
    patch_valid[1, 2:] = False  # keyframe 1 has 2 usable patches only
    fused = fused_keyframe_points(keeper, np.array([0, 1]), w2c, K4, patches, patch_valid)
    assert fused.record["scale_fallback_keyframes"] == 1
    assert abs(fused.record["global_scale"] - 1.5) < 1e-5
    assert torch.allclose(fused.xyz[1][fused.valid[1]][:, 2], torch.full((int(fused.valid[1].sum()),), 3.0))
    assert DENSE_STRIDE == 8
