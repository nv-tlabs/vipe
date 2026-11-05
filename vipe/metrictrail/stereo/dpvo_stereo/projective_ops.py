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

import math

import torch
import torch.nn.functional as F

from .lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)


def iproj(patches, intrinsics):
    """ inverse projection """
    x, y, d = patches.unbind(dim=2)
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    i = torch.ones_like(d)
    xn = (x - cx) / fx
    yn = (y - cy) / fy

    X = torch.stack([xn, yn, i, d], dim=-1)
    return X


def proj(X, intrinsics, depth=False):
    """ projection """

    X, Y, Z, W = X.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    # d = 0.01 * torch.ones_like(Z)
    # d[Z > 0.01] = 1.0 / Z[Z > 0.01]
    # d = torch.ones_like(Z)
    # d[Z.abs() > 0.1] = 1.0 / Z[Z.abs() > 0.1]

    d = 1.0 / Z.clamp(min=0.1)
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy

    if depth:
        return torch.stack([x, y, d], dim=-1)

    return torch.stack([x, y], dim=-1)


def _stereo_mask(edge_is_stereo, edge_count, device):
    """Normalize an optional per-edge stereo selector."""
    if edge_is_stereo is None:
        return None
    mask = torch.as_tensor(edge_is_stereo, device=device)
    if mask.ndim != 1 or mask.numel() != edge_count:
        raise ValueError(
            "edge_is_stereo must contain exactly one boolean per edge; "
            f"expected {edge_count}, got shape {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError("edge_is_stereo must have boolean dtype")
    return mask


def transform(
    poses,
    patches,
    intrinsics,
    ii,
    jj,
    kk,
    depth=False,
    valid=False,
    jacobian=False,
    tonly=False,
    *,
    edge_is_stereo=None,
    stereo_baseline_m=0.0,
):
    """Project left-camera patches into temporal-left or stereo-right targets.

    Stereo edges use a fixed rectified ``T_right_left`` translation of
    ``[-baseline, 0, 0]``.  They constrain inverse depth but not the trajectory
    pose, so their pose Jacobians are explicitly zeroed.
    """

    # backproject
    X0 = iproj(patches[:,kk], intrinsics[:,ii])

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()
    stereo = _stereo_mask(edge_is_stereo, len(ii), Gij.device)
    if stereo is not None:
        baseline = float(stereo_baseline_m)
        if not math.isfinite(baseline) or baseline <= 0.0:
            # Explicit all-false masks remain a supported mono-equivalent API.
            # Only this exceptional compatibility path needs a device sync.
            if bool(stereo.any()):
                raise ValueError(
                    "stereo_baseline_m must be finite and positive when stereo "
                    "edges are present"
                )
        else:
            fixed_rig = torch.as_tensor(
                [-baseline, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                device=Gij.device,
                dtype=Gij.data.dtype,
            )
            Gij.data[:, stereo] = fixed_rig

    if tonly:
        Gij[...,3:] = torch.as_tensor([0,0,0,1], device=Gij.device)

    X1 = Gij[:,:,None,None] * X0

    # project
    x1 = proj(X1, intrinsics[:,jj], depth)


    if jacobian:
        p = X1.shape[2]
        X, Y, Z, H = X1[...,p//2,p//2,:].unbind(dim=-1)
        o = torch.zeros_like(H)
        i = torch.zeros_like(H)

        fx, fy, cx, cy = intrinsics[:,jj].unbind(dim=-1)

        d = torch.zeros_like(Z)
        d[Z.abs() > 0.2] = 1.0 / Z[Z.abs() > 0.2]

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                H,  o,  o,  o,  Z, -Y,
                o,  H,  o, -Z,  o,  X,
                o,  o,  H,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(1, len(ii), 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                H,  o,  o,  o,  Z, -Y,  X,
                o,  H,  o, -Z,  o,  X,  Y,
                o,  o,  H,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o,
            ], dim=-1).view(1, len(ii), 4, 7)
        
        Jp = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
        ], dim=-1).view(1, len(ii), 2, 4)

        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None].adjT(Jj)
        
        Jz = torch.matmul(Jp, Gij.matrix()[...,:,3:])

        if stereo is not None:
            Ji[:, stereo] = 0
            Jj[:, stereo] = 0

        return x1, (Z > 0.2).float(), (Ji, Jj, Jz)

    if valid:
        return x1, (X1[...,2] > 0.2).float()
        
    return x1

def point_cloud(poses, patches, intrinsics, ix):
    """ generate point cloud from patches """
    return poses[:,ix,None,None].inv() * iproj(patches, intrinsics[:,ix])


def flow_mag(poses, patches, intrinsics, ii, jj, kk, beta=0.3):
    """ projective transform """

    coords0 = transform(poses, patches, intrinsics, ii, ii, kk)
    coords1, val = transform(poses, patches, intrinsics, ii, jj, kk, tonly=False, valid=True)
    coords2 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)

    flow1 = (coords1 - coords0).norm(dim=-1)
    flow2 = (coords2 - coords0).norm(dim=-1)

    return beta * flow1 + (1-beta) * flow2, (val > 0.5)
