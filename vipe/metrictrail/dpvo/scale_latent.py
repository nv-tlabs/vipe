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

"""GBA-time per-keyframe log-scale latent for the MoGe prior (GBA_SCALE_LATENT; EFFECTIVE_CHANGES §12.2 / §12.3).

Pure tensor arithmetic, no CUDA extension import, so it is unit-testable on the CPU.  dpvo.py calls
``gba_scale_latent`` from ``__run_global_BA`` only (never from the tracking-time local BA) and only when
``GBA_SCALE_LATENT`` is on; with the flag off no numeric path of the backend is touched.

Model: for keyframe t let m_t be the Huber location of log(d_BA) - log(d_MoGe) over its valid patches
(dpvo.burst_fixes.robust_log_scale_offsets, computed against the RAW MoGe prior every time, so nothing
compounds) and w_t its data weight (count / M, 0 below the min-patch count).  The latent s (one scalar per
keyframe) minimises

    sum_t w_t (s_t - m_t)^2 + lambda_smooth * sum_t (s_{t+1} - s_t)^2 + lambda_anchor * sum_t s_t^2

over the keyframes whose data term is not gated (parallax below GBA_SCALE_MIN_PARALLAX_PX or |m_t| above the
clamp -> w_t = 0, they are only carried by the smoothness term), then the w-weighted mean is removed
(sum_t w_t s_t = 0, so the Sim(3) gauge cannot move: the metric scale of the whole trajectory is still
pinned by the raw MoGe prior) and s is clamped to [-clamp, clamp].  The global BA then sees the prior
d_MoGe * exp(s_t) for every patch of keyframe t.

``solve_smooth_scale`` is a verbatim port of the earlier per-frame scale-smoothing solver (provenance in
CHANGELOG.md; tests/test_scale_latent.py checks the port against the original when that file is available).
"""

from __future__ import annotations

import torch


def solve_smooth_scale(
    m: torch.Tensor,
    w: torch.Tensor,
    *,
    lambda_smooth: float,
    lambda_anchor: float,
    s_boundary: float | None = None,
) -> torch.Tensor:
    """Solve the 1-D smoothing problem of the module docstring.

    Args:
        m: [T] per-frame robust offsets.
        w: [T] per-frame data weights (0 disables the data term of a frame).
        lambda_smooth: weight of the first-difference penalty.
        lambda_anchor: weight of the zero (metric) anchor; must be positive so
            the system is SPD even when every w_t is 0.
        s_boundary: frozen s of the frame preceding the window, coupled to
            s_0 through the smoothness term; None at the sequence start.

    Returns:
        s: [T] in float64.
    """

    if m.ndim != 1 or w.shape != m.shape:
        raise ValueError("m and w must be one-dimensional and equally long")
    if not float(lambda_anchor) > 0.0:
        raise ValueError("lambda_anchor must be positive")
    if float(lambda_smooth) < 0.0:
        raise ValueError("lambda_smooth must be non-negative")
    count = int(m.numel())
    if count == 0:
        return m.new_zeros(0, dtype=torch.float64)
    m64 = m.to(torch.float64)
    w64 = w.to(torch.float64)
    A = torch.diag(w64 + float(lambda_anchor))
    if count > 1:
        index = torch.arange(count - 1, device=m.device)
        A[index, index] += float(lambda_smooth)
        A[index + 1, index + 1] += float(lambda_smooth)
        A[index, index + 1] -= float(lambda_smooth)
        A[index + 1, index] -= float(lambda_smooth)
    b = w64 * m64
    if s_boundary is not None:
        A[0, 0] += float(lambda_smooth)
        b[0] += float(lambda_smooth) * float(s_boundary)
    return torch.linalg.solve(A, b)


def gba_scale_latent(
    m: torch.Tensor,
    w: torch.Tensor,
    parallax_px: torch.Tensor,
    *,
    clamp: float,
    min_parallax_px: float,
    lambda_smooth: float,
    lambda_anchor: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """Zero-mean, parallax-gated, smoothed, clamped per-keyframe log-scale latent.

    Args:
        m: [T] robust per-keyframe log(d_BA) - log(d_prior) (inverse depths; robust_log_scale_offsets).
        w: [T] data weights from robust_log_scale_offsets (already 0 below the min-patch count).
        parallax_px: [T] per-keyframe mean translation-only flow into the predecessor keyframe (px).
        clamp: |s| bound and |m| gate (log units).
        min_parallax_px: keyframes below it get w = 0 (carried by the smoothness term only).
        lambda_smooth, lambda_anchor: solve_smooth_scale weights.

    Returns:
        (s [T] float32 on m.device, stats dict with the gate fractions and the post-clamp weighted mean).
    """

    if m.ndim != 1 or w.shape != m.shape or parallax_px.shape != m.shape:
        raise ValueError("m, w and parallax_px must be one-dimensional and equally long")
    clamp = float(clamp)
    if not clamp > 0.0:
        raise ValueError("clamp must be positive")
    T = int(m.numel())
    m = torch.nan_to_num(m.to(torch.float32), nan=0.0, posinf=clamp * 10.0, neginf=-clamp * 10.0)
    w = torch.nan_to_num(w.to(torch.float32), nan=0.0)
    parallax = torch.nan_to_num(parallax_px.to(torch.float32), nan=0.0, posinf=1e3)
    has_data = w > 0                                     # >= min patches
    gate_parallax = parallax < float(min_parallax_px)    # too little parallax: BA depth not informative
    gate_clamp = m.abs() > clamp                         # outlier frame offset: do not chase it
    keep = has_data & ~gate_parallax & ~gate_clamp
    w_eff = torch.where(keep, w, torch.zeros_like(w))
    if T == 0:
        s = m.new_zeros(0)
        wsum = 0.0
    else:
        s = solve_smooth_scale(m, w_eff, lambda_smooth=float(lambda_smooth), lambda_anchor=float(lambda_anchor))
        wsum = float(w_eff.sum().item())
        if wsum > 0.0:
            s = s - (w_eff.to(s.dtype) * s).sum() / w_eff.to(s.dtype).sum()
        else:
            s = torch.zeros_like(s)
        s = s.clamp(-clamp, clamp)
    s = s.to(torch.float32)
    n_data = int(has_data.sum().item())
    wmean = float((w_eff * s).sum().item() / wsum) if wsum > 0.0 else 0.0
    stats = {
        "n_keyframes": T,
        "n_with_data": n_data,                                                   # >= min patches
        "n_gated_parallax": int((has_data & gate_parallax).sum().item()),        # among frames with data
        "n_gated_clamp": int((has_data & ~gate_parallax & gate_clamp).sum().item()),
        "n_gated_min_patches": int(T - n_data),
        "n_used": int(keep.sum().item()),
        "n_clamped": int((s.abs() >= clamp - 1e-7).sum().item()),
        "w_sum": wsum,
        "weighted_mean_s": wmean,                    # 0 by construction up to the clamp / float32
        "std_s": float(s.std(unbiased=False).item()) if T > 1 else 0.0,
        "max_abs_s": float(s.abs().max().item()) if T > 0 else 0.0,
        "used": keep,
    }
    return s, stats
