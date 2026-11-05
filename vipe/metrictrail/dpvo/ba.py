# SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
# SPDX-License-Identifier: MIT
#
# Source: https://github.com/princeton-vl/DPVO
# See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

# Modified for MetricTrail integration.
# The following Apache-2.0 terms apply to NVIDIA modifications only.
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

import torch


def _scatter_sum(src, index, *, dim, dim_size):
    """Import the optional CUDA scatter dependency only when BA is executed."""
    from vipe.metrictrail.dpvo_scatter import scatter_sum  # [vipe] was: from torch_scatter import scatter_sum

    return scatter_sum(src, index, dim=dim, dim_size=dim_size)


def _as_patch_values(values, *, name, batch, patch_count, device, dtype):
    """Normalize one-value-per-patch inputs to ``[batch, patch_count]``."""
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, device=device, dtype=dtype)
    else:
        values = values.to(device=device, dtype=dtype)

    if values.ndim == 1:
        if values.numel() == patch_count:
            values = values.unsqueeze(0).expand(batch, -1)
        elif values.numel() == batch * patch_count:
            values = values.reshape(batch, patch_count)
        else:
            raise ValueError(
                f"{name} must contain one value per patch; expected "
                f"{patch_count} or {batch * patch_count} values, got "
                f"{values.numel()}"
            )
    elif values.ndim == 2:
        if values.shape == (1, patch_count):
            values = values.expand(batch, -1)
        elif values.shape != (batch, patch_count):
            raise ValueError(
                f"{name} must have shape [{batch}, {patch_count}] (or "
                f"[1, {patch_count}]), got {list(values.shape)}"
            )
    else:
        raise ValueError(
            f"{name} must be a 1D or 2D one-value-per-patch tensor, got "
            f"{values.ndim} dimensions"
        )

    return values


def _metric_unary_terms(
    current_disps,
    depth_prior,
    depth_prior_weight,
    kx,
    depth_prior_lambda,
    depth_prior_huber,
):
    """Build robust metric inverse-depth unary normal-equation terms.

    The residual follows the inference CUDA implementation exactly::

        r = log(d_prior) - log(d)
        robust = min(1, huber_delta / abs(r))
        weight = prior_lambda * confidence * robust
        C += weight / d**2
        rhs += weight * r / d

    ``kx`` maps the compact Schur depth variables back to their (possibly
    non-contiguous) patch indices. Invalid or non-positive priors, current
    disparities, and confidence values contribute zero rather than NaNs.

    Args:
        current_disps: Current patch-centre inverse depths, ``[B, N]`` or
            ``[N]``.
        depth_prior: Metric inverse-depth prior with one value per patch.
        depth_prior_weight: Non-negative prior confidence per patch.
        kx: 1D integer tensor selecting the compact depth variables.
        depth_prior_lambda: Non-negative scalar unary strength.
        depth_prior_huber: Positive scalar Huber transition in log-depth.

    Returns:
        A pair ``(C, rhs)`` with shape ``[B, len(kx), 1, 1]`` suitable for
        direct addition to DPVO's depth Hessian diagonal and right-hand side.
    """
    if not isinstance(current_disps, torch.Tensor):
        current_disps = torch.as_tensor(current_disps)
    if not current_disps.is_floating_point():
        raise TypeError("current_disps must have a floating-point dtype")
    if current_disps.ndim == 1:
        current_disps = current_disps.unsqueeze(0)
    if current_disps.ndim != 2:
        raise ValueError(
            "current_disps must have shape [B, N] or [N], got "
            f"{list(current_disps.shape)}"
        )

    batch, patch_count = current_disps.shape
    if not isinstance(kx, torch.Tensor):
        kx = torch.as_tensor(kx, device=current_disps.device)
    else:
        kx = kx.to(device=current_disps.device)
    if kx.ndim != 1:
        raise ValueError(f"kx must be 1D, got shape {list(kx.shape)}")
    if kx.dtype == torch.bool or kx.dtype.is_floating_point:
        raise TypeError("kx must have an integer dtype")
    kx = kx.to(dtype=torch.long)
    if kx.numel() and bool(((kx < 0) | (kx >= patch_count)).any().item()):
        raise ValueError(
            f"kx contains an index outside [0, {patch_count}): "
            f"{kx.detach().cpu().tolist()}"
        )

    prior = _as_patch_values(
        depth_prior,
        name="depth_prior",
        batch=batch,
        patch_count=patch_count,
        device=current_disps.device,
        dtype=current_disps.dtype,
    )
    confidence = _as_patch_values(
        depth_prior_weight,
        name="depth_prior_weight",
        batch=batch,
        patch_count=patch_count,
        device=current_disps.device,
        dtype=current_disps.dtype,
    )

    prior_lambda = torch.as_tensor(
        depth_prior_lambda, device=current_disps.device, dtype=current_disps.dtype
    )
    huber_delta = torch.as_tensor(
        depth_prior_huber, device=current_disps.device, dtype=current_disps.dtype
    )
    if prior_lambda.numel() != 1:
        raise ValueError("depth_prior_lambda must be a scalar")
    if huber_delta.numel() != 1:
        raise ValueError("depth_prior_huber must be a scalar")
    if not bool(torch.isfinite(prior_lambda.detach()).item()):
        raise ValueError("depth_prior_lambda must be finite")
    if not bool(torch.isfinite(huber_delta.detach()).item()):
        raise ValueError("depth_prior_huber must be finite")
    if bool((prior_lambda.detach() < 0).item()):
        raise ValueError("depth_prior_lambda must be non-negative")
    if bool((huber_delta.detach() <= 0).item()):
        raise ValueError("depth_prior_huber must be positive")

    current = current_disps.index_select(1, kx)
    prior = prior.index_select(1, kx)
    confidence = confidence.index_select(1, kx)

    valid = (
        torch.isfinite(current)
        & torch.isfinite(prior)
        & torch.isfinite(confidence)
        & (current > 0)
        & (prior > 0)
        & (confidence > 0)
    )
    safe_current = torch.where(valid, current, torch.ones_like(current))
    safe_prior = torch.where(valid, prior, torch.ones_like(prior))
    safe_confidence = torch.where(valid, confidence, torch.zeros_like(confidence))

    residual = safe_prior.log() - safe_current.log()
    abs_residual = residual.abs()
    robust = torch.where(
        abs_residual <= huber_delta,
        torch.ones_like(abs_residual),
        huber_delta / abs_residual.clamp_min(1e-6),
    )
    weight = prior_lambda * safe_confidence * robust
    jacobian = safe_current.clamp_min(1e-4).reciprocal()

    C = weight * jacobian.square()
    rhs = weight * residual * jacobian
    return C[..., None, None], rhs[..., None, None]

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b, strict=False):
        # don't crash training if cholesky decomp fails
        U, info = torch.linalg.cholesky_ex(H)

        if torch.any(info):
            if strict:
                failed = torch.nonzero(info, as_tuple=False).detach().cpu().tolist()
                raise RuntimeError(
                    "differentiable BA Cholesky factorization failed for "
                    f"batch entries {failed}"
                )
            ctx.failed = True
            return torch.zeros_like(b)

        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz, None

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return _scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return _scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + _scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(_scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

def block_matmul(A, B):
    """ block matrix multiply """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)
    return torch.matmul(A, B).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)

def block_solve(A, B, ep=1.0, lm=1e-4, *, strict_cholesky=False):
    """ block matrix solve """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)

    A = A + (ep + lm * A) * torch.eye(n1*p1, device=A.device)

    X = CholeskySolver.apply(A, B, strict_cholesky)
    return X.reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_show(A):
    import matplotlib.pyplot as plt
    b, n1, m1, p1, q1 = A.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()

def BA(
    poses,
    patches,
    intrinsics,
    targets,
    weights,
    lmbda,
    ii,
    jj,
    kk,
    bounds,
    ep=100.0,
    PRINT=False,
    fixedp=1,
    structure_only=False,
    *,
    depth_prior=None,
    depth_prior_weight=None,
    depth_prior_lambda=0.0,
    depth_prior_huber=0.3,
    strict_cholesky=False,
):
    """Bundle adjustment with an optional differentiable metric-depth unary.

    The keyword-only metric arguments default to a disabled unary, preserving
    the original DPVO training behaviour and all existing call sites.
    """

    from . import projective_ops as pops

    b = 1
    n = max(ii.max().item(), jj.max().item()) + 1

    coords, v, (Ji, Jj, Jz) = \
        pops.transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    p = coords.shape[3]
    r = targets - coords[...,p//2,p//2,:]

    v *= (r.norm(dim=-1) < 250).float()

    in_bounds = \
        (coords[...,p//2,p//2,0] > bounds[0]) & \
        (coords[...,p//2,p//2,1] > bounds[1]) & \
        (coords[...,p//2,p//2,0] < bounds[2]) & \
        (coords[...,p//2,p//2,1] < bounds[3])

    v *= in_bounds.float()

    if PRINT:
        print((r * v[...,None]).norm(dim=-1).mean().item())

    r = (v[...,None] * r).unsqueeze(dim=-1)    
    weights = (v[...,None] * weights).unsqueeze(dim=-1)

    wJiT = (weights * Ji).transpose(2,3)
    wJjT = (weights * Jj).transpose(2,3)
    wJzT = (weights * Jz).transpose(2,3)

    Bii = torch.matmul(wJiT, Ji)
    Bij = torch.matmul(wJiT, Jj)
    Bji = torch.matmul(wJjT, Ji)
    Bjj = torch.matmul(wJjT, Jj)

    Eik = torch.matmul(wJiT, Jz)
    Ejk = torch.matmul(wJjT, Jz)

    vi = torch.matmul(wJiT, r)
    vj = torch.matmul(wJjT, r)

    # fix first pose
    ii = ii.clone()
    jj = jj.clone()

    n = n - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)

    B = safe_scatter_add_mat(Bii, ii, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bij, ii, jj, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bji, jj, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj, jj, n, n).view(b, n, n, 6, 6)

    E = safe_scatter_add_mat(Eik, ii, kk, n, m).view(b, n, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj, kk, n, m).view(b, n, m, 6, 1) 

    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk, m)

    v = safe_scatter_add_vec(vi, ii, n).view(b, n, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj, n).view(b, n, 1, 6, 1)

    w = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk, m)

    metric_requested = (
        depth_prior is not None
        or depth_prior_weight is not None
        or (
            isinstance(depth_prior_lambda, torch.Tensor)
            and depth_prior_lambda.numel() == 1
            and bool((depth_prior_lambda.detach() != 0).item())
        )
        or (
            not isinstance(depth_prior_lambda, torch.Tensor)
            and depth_prior_lambda != 0
        )
    )
    if metric_requested:
        if depth_prior is None or depth_prior_weight is None:
            raise ValueError(
                "depth_prior and depth_prior_weight are both required when "
                "the metric unary is enabled"
            )
        current_disps = patches[:, :, 2, p // 2, p // 2]
        metric_C, metric_w = _metric_unary_terms(
            current_disps,
            depth_prior,
            depth_prior_weight,
            kx,
            depth_prior_lambda,
            depth_prior_huber,
        )
        C = C + metric_C
        w = w + metric_w

    if isinstance(lmbda, torch.Tensor):
        lmbda = lmbda.reshape(*C.shape)
        
    Q = 1.0 / (C + lmbda)
    
    ### solve w/ schur complement ###
    EQ = E * Q[:,None]

    if structure_only or n == 0:
        dZ = (Q * w).view(b, -1, 1, 1)

    else:
        S = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))
        dX = block_solve(
            S,
            y,
            ep=ep,
            lm=1e-4,
            strict_cholesky=strict_cholesky,
        )

        dZ = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    x, y, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x, y, disps], dim=2)

    if not structure_only and n > 0:
        poses = pose_retr(poses, dX, fixedp + torch.arange(n))

    return poses, patches
