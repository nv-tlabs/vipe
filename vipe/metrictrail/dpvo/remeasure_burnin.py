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

"""Cold-start helpers of the final re-measurement (M1: REMEASURE_BURNIN / REMEASURE_WARM_ACTIVE /
REMEASURE_CARRY_STATE; EFFECTIVE_CHANGES §12.3) plus the pass schedule and the terminate-time keyframe_prior
sidecar summary.

Pure tensor arithmetic (no CUDA extension), used ONLY from DPVO._remeasure_final and the terminate-time sidecar
diagnostics.  The tracking-time update() never reaches this module.  Defaults leave every value byte-identical to
the plain re-measurement (cold GRU, state zeroed between passes).

Hypothesis: every M1 pass starts the update operator from net = 0 on already-converged poses; the cold GRU moves the
targets ~6 px and drops the inactive weights to ~1/3 for the first 1-3 refreshes, and those undamped 2-GN global BA
calls (plant at full confidence) are what regresses plant / floor pass 2.  Burn-in (refresh without BA), warm-starting
the active rows from pg.net and carrying the state across passes remove the cold start without touching the operator.

Tested and rejected (EFFECTIVE_CHANGES §12.3), hence not part of this module: REMEASURE_BA_ITERS (GN iterations per
M1 global BA call, arm f_iters), REMEASURE_SCHEDULE (free-form pass lengths) and GBA_FINAL_ITERS (extra stale-target
global BA calls, control arm f_ctrl).  The pass lengths are therefore fixed to the schedule below.
"""

from __future__ import annotations

import torch

AGE_CLASSES = ("age_le13", "age_14_30", "age_gt30")


def pass_schedule(mode: str) -> tuple[int, ...]:
    """Iteration counts of the M1 passes: (8,) for GBA_REMEASURE_FINAL 'one', (8, 24) for 'two'
    (the ViPE run(7) + run(24) analogue; GRU state handling between the passes is governed by
    REMEASURE_CARRY_STATE)."""
    return (8,) if str(mode) == "one" else (8, 24)


def warm_start_active(net: torch.Tensor, pg_net: torch.Tensor, n_inac: int) -> int:
    """Copy the tracking-time GRU state of the active edges (pg.net, rows in pg.ii order) into the active slice
    net[:, n_inac:] of the M1 state (rows = cat(ii_inac, ii)).  Returns the number of rows written (0 when the shapes
    do not line up; the state then stays as it was)."""
    n_act = int(net.shape[1]) - int(n_inac)
    if pg_net is None or pg_net.ndim != 3 or int(pg_net.shape[1]) != n_act or int(pg_net.shape[2]) != int(net.shape[2]):
        return 0
    if n_act > 0:
        net[:, n_inac:] = pg_net.to(net.dtype)
    return n_act


def age_class_ids(ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    """0: |jj - ii| <= 13, 1: 14..30, 2: > 30 (keyframe-index age; sign-free so reverse edges bucket by baseline)."""
    age = (jj - ii).abs()
    return (age > 13).long() + (age > 30).long()


def mean_weight_by_age_class(weight: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor, sel: torch.Tensor | None = None) -> dict:
    """weight (1, E, C); sel (E,) bool = edges to include (the refreshed ones).  Returns
    {age_le13: mean|None, age_14_30: ..., age_gt30: ..., counts: {...}} with the mean taken over edges and channels."""
    cls = age_class_ids(ii, jj)
    if sel is None:
        sel = torch.ones_like(cls, dtype=torch.bool)
    w = weight.float().mean(dim=-1).reshape(-1)  # (E,) mean over channels
    out: dict = {"counts": {}}
    for k, name in enumerate(AGE_CLASSES):
        m = sel & (cls == k)
        cnt = int(m.sum().item())
        out["counts"][name] = cnt
        out[name] = float(w[m].mean().item()) if cnt else None
    return out


def oob_count(centres: torch.Tensor, wd4: int, ht4: int) -> torch.Tensor:
    """centres (..., 2) = (x, y) patch-centre reprojections in target-grid (1/4-res) pixels.  Returns a 0-dim long
    tensor counting centres outside [0, wd4) x [0, ht4) (non-finite counts as outside).  No host sync."""
    x = centres[..., 0].float()
    y = centres[..., 1].float()
    inside = (x >= 0) & (x < float(wd4)) & (y >= 0) & (y < float(ht4))
    return (~inside).sum()


def move_stats(t_new: torch.Tensor, t_old: torch.Tensor, n_first: int) -> tuple[float | None, float | None]:
    """median / p90 of ||t_new - t_old|| over the first n_first edges (the inactive slice); None when empty."""
    d = (t_new[:, :n_first] - t_old[:, :n_first]).norm(dim=-1).float().reshape(-1)
    d = torch.nan_to_num(d, nan=1e3, posinf=1e3).clamp(max=1e3)
    if d.numel() == 0:
        return None, None
    return float(d.median().item()), float(torch.quantile(d, 0.9).item())


def keyframe_scale_summary(m: torch.Tensor, count: torch.Tensor, parallax: torch.Tensor | None, tstamps, *,
                           min_patches: int = 16, warmup_frames: int = 12, parallax_thresh_px: float = 2.0) -> dict:
    """Per-keyframe robust log-scale offset m_t (BA patch inverse depth vs MoGe prior) and translation-only parallax_t.
    m, count: (n,); parallax: (n-1,) for frames 1..n-1 or None; tstamps: n input-frame indices.  Summary fields:
    std(m_t), first/last-quartile means (by keyframe index), frac(parallax < 2 px), warm-up excursion =
    mean m_t over keyframes 0..warmup_frames-1 minus the median m_t."""
    n = int(m.numel())
    ok = count >= int(min_patches)
    m_list = m.detach().float().cpu().tolist()
    ok_list = ok.detach().cpu().tolist()
    valid_m = [v for v, o in zip(m_list, ok_list) if o]
    out: dict = {
        "tstamps": [int(t) for t in list(tstamps)[:n]],
        "m_t": [float(v) if o else None for v, o in zip(m_list, ok_list)],
        "count_t": [int(c) for c in count.detach().cpu().tolist()],
        "n_keyframes": n,
        "n_with_prior": len(valid_m),
        "min_patches": int(min_patches),
    }
    if valid_m:
        mv = torch.tensor(valid_m)
        med = float(mv.median().item())
        q = max(1, n // 4)
        first = [v for v, o in zip(m_list[:q], ok_list[:q]) if o]
        last = [v for v, o in zip(m_list[n - q:], ok_list[n - q:]) if o]
        warm = [v for v, o in zip(m_list[:warmup_frames], ok_list[:warmup_frames]) if o]
        out.update({
            "m_median": med,
            "m_mean": float(mv.mean().item()),
            "m_std": float(mv.std(unbiased=False).item()) if mv.numel() > 1 else 0.0,
            "m_first_quartile_mean": (sum(first) / len(first)) if first else None,
            "m_last_quartile_mean": (sum(last) / len(last)) if last else None,
            "warmup_excursion": ((sum(warm) / len(warm)) - med) if warm else None,
            "warmup_frames": int(warmup_frames),
        })
    else:
        out.update({"m_median": None, "m_mean": None, "m_std": None, "m_first_quartile_mean": None,
                    "m_last_quartile_mean": None, "warmup_excursion": None, "warmup_frames": int(warmup_frames)})
    if parallax is not None and parallax.numel() > 0:
        p = parallax.detach().float().cpu()
        out.update({
            "parallax_t": [None] + [float(v) for v in p.tolist()],
            "parallax_median_px": float(p.median().item()),
            "frac_parallax_lt2px": float((p < float(parallax_thresh_px)).float().mean().item()),
        })
    else:
        out.update({"parallax_t": None, "parallax_median_px": None, "frac_parallax_lt2px": None})
    return out
