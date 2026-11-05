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

"""Pure-tensor helpers behind the two graph-structure flags (EFFECTIVE_CHANGES §12.3).

  LOOP_REVERSE_EDGES  -> loop_pairs / reverse_loop_edges (reverse-index construction: the new frame's
                         patches observed in the old frame of an accepted pair) and the per-edge tag constants;
  KEYFRAME_MAX_CULL   -> HoleGuard (culled-run / path-rotation counters for keyframe()).

No CUDA extension or lietorch import: the CPU unit test (ROOT/tests/test_graph_guards.py) loads this file by
path.  Rotation angles come from quaternion dot products (angle(qa * qb^-1) = 2 acos |<qa, qb>|), so no SE3
algebra is needed here.  Nothing in this module reads ground truth.
"""

from __future__ import annotations

import math

import torch

# Per-edge tag (pg.tag / pg.tag_inac, aligned with pg.ii / pg.ii_inac).
TAG_REGULAR = 0     # frontend forward/backward edges
TAG_LOOP_FWD = 1    # patchgraph.edges_loop: old patches (frame i) observed in the new frame j
TAG_LOOP_REV = 2    # reverse of an accepted edges_loop pair: new patches (frame j) observed in old frame i
TAG_NAMES = {TAG_REGULAR: "regular", TAG_LOOP_FWD: "forward_loop", TAG_LOOP_REV: "reverse_loop"}

# Age classes (|jj - ii| in kept-keyframe indices) for the weight-by-age diagnostic.
AGE_BINS = ((0, 2, "age_le2"), (3, 12, "age_3_12"), (13, 30, "age_13_30"), (31, 10**9, "age_gt30"))


def loop_pairs(ii_frames: torch.Tensor, jj_frames: torch.Tensor) -> torch.Tensor:
    """Unique (i, j) keyframe pairs of one edges_loop batch given the per-edge source frame (ix[kk]) and
    target frame; edges_loop emits M consecutive rows per pair, so callers pass every M-th row."""
    if ii_frames.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=ii_frames.device)
    return torch.unique(torch.stack((ii_frames.long(), jj_frames.long()), dim=1), dim=0)


def reverse_loop_edges(pairs: torch.Tensor, n: int, mem: int, M: int, margin: int):
    """Reverse-index construction.  For every accepted forward pair (i old, j new) whose old frame's matching
    features stay resident in the fmap ring for the whole active lifetime of the new edge, i.e.
    (n - 1 - i) < mem - margin, emit frame j's M patches observed in frame i:
        kk_r = j * M + arange(M)      (patch ids; append_factors derives ii = ix[kk] = j)
        jj_r = i
    Returns (kk_r, jj_r, n_pairs_used, n_pairs_skipped_nonresident)."""
    device = pairs.device
    if pairs.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, 0, 0
    i = pairs[:, 0].long()
    j = pairs[:, 1].long()
    ok = (int(n) - 1 - i) < (int(mem) - int(margin))
    i, j = i[ok], j[ok]
    ar = torch.arange(int(M), device=device, dtype=torch.long)
    kk_r = (j[:, None] * int(M) + ar[None, :]).reshape(-1)
    jj_r = i[:, None].expand(-1, int(M)).reshape(-1)
    return kk_r, jj_r, int(ok.sum().item()), int((~ok).sum().item())


def weight_mean_by_tag_and_age(weight: torch.Tensor, tag: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> dict:
    """Mean of the per-edge weight (mean over its 2 components) per tag and per (tag, age class).
    weight: (1, E, 2); tag / ii / jj: (E,).  Returns plain floats / ints (JSON-safe)."""
    out = {}
    if weight.numel() == 0:
        return out
    w = weight[0].float().mean(dim=-1)
    age = (jj.long() - ii.long()).abs()
    for t, name in TAG_NAMES.items():
        m = tag == t
        c = int(m.sum().item())
        out[name] = {"edges": c, "mean_weight": float(w[m].mean().item()) if c else None}
        for lo, hi, aname in AGE_BINS:
            ma = m & (age >= lo) & (age <= hi)
            ca = int(ma.sum().item())
            out[name][aname] = {"edges": ca, "mean_weight": float(w[ma].mean().item()) if ca else None}
    return out


def quat_angle_deg(q: torch.Tensor, q_ref: torch.Tensor | None = None) -> float:
    """Rotation angle in degrees of quaternion q (x y z w), or of q * q_ref^-1 when q_ref is given."""
    # float64: acos loses ~1e-4 deg per call in float32 near |d| = 1 (small increments).
    q = q.detach().double().reshape(-1)[-4:]
    q = q / q.norm().clamp_min(1e-12)
    if q_ref is None:
        d = q[3].abs()
    else:
        q_ref = q_ref.detach().double().reshape(-1)[-4:]
        q_ref = q_ref / q_ref.norm().clamp_min(1e-12)
        d = (q * q_ref).sum().abs()
    d = float(d.clamp(max=1.0).item())
    return math.degrees(2.0 * math.acos(d))


class HoleGuard:
    """Keyframe-hole guard state (proposal #4).  keyframe() calls should_keep() before its flow test,
    on_cull(dP) when it culls the candidate (dP = cumulative relative pose from the last kept keyframe,
    SE3 data [tx ty tz qx qy qz qw]) and on_keep(tstamp, forced) when the candidate is kept.
    With max_cull <= 0 every method is a no-op and should_keep() is always False."""

    def __init__(self, max_cull: int, rot_deg: float):
        self.max_cull = int(max_cull)
        self.rot_deg = float(rot_deg)
        self.enabled = self.max_cull > 0
        self.run = 0                    # candidates culled since the last kept keyframe
        self.rot_path_deg = 0.0         # summed |rotation increment| between consecutive culled candidates
        self.last_q = None              # quaternion of the previous culled candidate's cumulative dP
        # diagnostics
        self.keeps = 0                  # candidates kept ONLY because of the guard
        self.keep_frames = []           # input-frame indices (tstamps_) of those keeps
        self.keep_runs = []             # [culled run length, path rotation deg] at each guard keep
        self.run_max = 0                # longest culled run seen (any cause)
        self.rot_path_max_deg = 0.0
        self.candidates = 0             # keyframe() decisions taken
        self.culls = 0

    def should_keep(self) -> bool:
        return self.enabled and self.run >= self.max_cull and self.rot_path_deg >= self.rot_deg

    def on_cull(self, dP_data: torch.Tensor) -> None:
        if not self.enabled:
            return
        q = dP_data.detach().double().reshape(-1)[3:7].clone()
        inc = quat_angle_deg(q) if self.last_q is None else quat_angle_deg(q, self.last_q)
        self.run += 1
        self.culls += 1
        self.candidates += 1
        self.rot_path_deg += inc
        self.last_q = q
        self.run_max = max(self.run_max, self.run)
        self.rot_path_max_deg = max(self.rot_path_max_deg, self.rot_path_deg)

    def on_keep(self, tstamp: int, forced: bool) -> None:
        if not self.enabled:
            return
        self.candidates += 1
        if forced:
            self.keeps += 1
            self.keep_frames.append(int(tstamp))
            self.keep_runs.append([int(self.run), round(float(self.rot_path_deg), 3)])
        self.run = 0
        self.rot_path_deg = 0.0
        self.last_q = None

    def diagnostics(self) -> dict:
        return {
            "keyframe_max_cull": self.max_cull,
            "keyframe_max_cull_rot_deg": self.rot_deg,
            "keyframe_guard_keeps": int(self.keeps),
            "keyframe_guard_keep_frames": list(self.keep_frames),
            "keyframe_guard_keep_runs": [list(r) for r in self.keep_runs],
            "keyframe_candidates": int(self.candidates),
            "keyframe_culls": int(self.culls),
            "keyframe_cull_run_max": int(self.run_max),
            "keyframe_cull_rot_path_max_deg": float(self.rot_path_max_deg),
        }
