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

"""Culled-frame interpolation (CULLED_INTERP; EFFECTIVE_CHANGES §6, fix F2) and the robust per-frame
log-scale offset shared by the scale latent and the keyframe_prior sidecar.
Pure torch + lietorch, importable on a CPU-only node.

Motivated by the TUM fr1 burst trace (EFFECTIVE_CHANGES §6): the trajectory breaks at the first kept
keyframe after a run of culled frames.  The companion per-frame prior gate (PRIOR_GATE_LOG, fix F1) was
tested and rejected (§6) and is not part of this backend; CHANGELOG.md records where it lives.

F2  CULLED_INTERP  (room 177-222: culled frames keep the cull-time delta relative to their kept
    predecessor A forever; the next kept keyframe B is later corrected by BA, so the trajectory snaps
    at B).  At terminate(), for each maximal run of culled frames a < t < b between kept keyframes A and
    B, the cull-time relative pose dPn (recorded when t = b-1 was culled: P_b = dPn * P_{b-1}) chains
    the run to T_B^chain = dPn * P_{b-1}^chain; the discrepancy D = P_B^final * (T_B^chain)^-1 is spread
    as P_t <- Exp(alpha_t Log D) * P_t with alpha_t = (t - a) / (b - a).  Keyframe poses are unchanged;
    with D = I the culled poses are unchanged.
"""

from __future__ import annotations

import math

import torch

from .lietorch import SE3


def robust_log_scale_offsets(
    log_ratio: torch.Tensor,
    valid: torch.Tensor,
    *,
    huber: float,
    min_patches: int,
    patches_per_frame: int,
    iterations: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Huber location of ``log_ratio`` per row (iteratively reweighted; provenance in CHANGELOG.md).

    Returns (m [T], count [T] long, w [T] = count / M where count >= min_patches else 0).
    """

    if log_ratio.ndim != 2 or valid.shape != log_ratio.shape:
        raise ValueError(
            f"log_ratio {tuple(log_ratio.shape)} and valid {tuple(valid.shape)} must be [T, M]"
        )
    if not float(huber) > 0.0:
        raise ValueError("huber must be positive")
    valid = valid & torch.isfinite(log_ratio)
    count = valid.sum(dim=1)
    values = torch.where(valid, log_ratio, torch.zeros_like(log_ratio))
    nan_filled = torch.where(
        valid, log_ratio, torch.full_like(log_ratio, float("nan"))
    )
    m = torch.nanmedian(nan_filled, dim=1).values
    m = torch.nan_to_num(m, nan=0.0)
    for _ in range(int(iterations)):
        deviation = (values - m[:, None]).abs()
        robust = torch.where(
            deviation <= float(huber),
            torch.ones_like(deviation),
            float(huber) / deviation.clamp_min(1e-6),
        )
        robust = robust * valid.to(robust.dtype)
        denominator = robust.sum(dim=1)
        candidate = (robust * values).sum(dim=1) / denominator.clamp_min(1e-12)
        m = torch.where(denominator > 0, candidate, m)
    enough = count >= int(min_patches)
    w = torch.where(
        enough,
        count.to(log_ratio.dtype) / float(patches_per_frame),
        torch.zeros_like(m),
    )
    return m, count, w


def _rot_deg_from_quat(q: torch.Tensor) -> float:
    w = float(q[3].abs().clamp(max=1.0).item())
    return math.degrees(2.0 * math.acos(w))


def culled_interp(
    poses: torch.Tensor,
    kept: torch.Tensor,
    next_rel: dict,
) -> tuple[torch.Tensor, dict]:
    """F2: spread the BA correction of each kept keyframe over the culled run preceding it.

    Args:
        poses: [F, 7] world-to-camera SE3 data (tx ty tz qx qy qz qw) of EVERY input frame as
            produced by DPVO.get_pose (kept keyframes: final BA pose; culled: cull-time delta chain).
        kept: [F] bool, True for frames whose pose comes from the keyframe buffer.
        next_rel: {t: (t_next, dPn[7])} recorded when frame t was culled, with
            P_{t_next}(cull time) = dPn * P_t(cull time) and t_next the next buffered frame.
    Returns:
        (new_poses [F, 7], stats).  Kept poses are returned unchanged.  A run is skipped (left as the
        old chain) when its last culled frame has no next_rel entry pointing at the kept keyframe b.
    """

    if poses.ndim != 2 or poses.shape[1] != 7 or kept.shape != (poses.shape[0],):
        raise ValueError("poses must be [F, 7] and kept [F]")
    out = poses.clone()
    kept_idx = torch.nonzero(kept).flatten().tolist()
    stats = {
        "runs": 0,
        "runs_corrected": 0,
        "runs_skipped": 0,
        "culled_frames_corrected": 0,
        "D_trans_max_m": 0.0,
        "D_rot_max_deg": 0.0,
        "runs_detail": [],  # [a, b, |t_D| m, rot_D deg]
    }
    for a, b in zip(kept_idx[:-1], kept_idx[1:]):
        if b - a < 2:
            continue
        stats["runs"] += 1
        rel = next_rel.get(b - 1)
        if rel is None or int(rel[0]) != b:
            stats["runs_skipped"] += 1
            continue
        dPn = SE3(torch.as_tensor(rel[1]).to(poses).reshape(1, 7))
        P_last = SE3(poses[b - 1].reshape(1, 7))
        T_chain = dPn * P_last
        P_B = SE3(poses[b].reshape(1, 7))
        D = P_B * T_chain.inv()
        xi = D.log()  # [1, 6]
        idx = torch.arange(a + 1, b, device=poses.device, dtype=poses.dtype)
        alpha = (idx - float(a)) / float(b - a)
        corr = SE3.exp(alpha[:, None] * xi)
        out[a + 1 : b] = (corr * SE3(poses[a + 1 : b])).data
        t_norm = float(D.data[0, :3].norm().item())
        r_deg = _rot_deg_from_quat(D.data[0, 3:])
        stats["runs_corrected"] += 1
        stats["culled_frames_corrected"] += b - a - 1
        stats["D_trans_max_m"] = max(stats["D_trans_max_m"], t_norm)
        stats["D_rot_max_deg"] = max(stats["D_rot_max_deg"], r_deg)
        stats["runs_detail"].append([int(a), int(b), round(t_norm, 5), round(r_deg, 4)])
    return out, stats
