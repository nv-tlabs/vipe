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

import torch

from vipe.ext import slam_ext
from vipe.ext.lietorch import SE3, Sim3
from vipe.utils.cameras import BaseCameraModel, CameraType


def iproj_disp(
    disps: torch.Tensor,
    disps_uv: torch.Tensor | None,
    intrinsics: torch.Tensor,
    camera_type: CameraType,
    compute_jz: bool = False,
    compute_jf: bool = False,
):
    """
    Pinhole camera inverse projection using disparities.

    Args:
        disps: (N, ...) tensor of disparities
        disps_uv: (N, ..., 2) tensor of uv coordinates
            if None, will use meshgrid to generate uv coordinates
        intrinsics: (N, 4+D) tensor of intrinsics
        camera_type: CameraType to be used
        compute_jz: bool to compute jacobian of the disps
        compute_jf: bool to compute jacobian of the intrinsics

    Returns:
        pts: (N, ..., 4) tensor of homogeneous points
        Jz: (N, ..., 4) tensor of jacobian of the disps
        Jf: (N, ..., 4, 1+D) tensor of jacobian of the intrinsics (focal + distortion)
    """
    if disps_uv is None:
        assert disps.dim() == 3, "When disps_uv is not provided, disps must be of shape (N, H, W)"
        ht, wd = disps.shape[1:]
        y, x = torch.meshgrid(
            torch.arange(ht).to(disps.device).float(),
            torch.arange(wd).to(disps.device).float(),
            indexing="ij",
        )
    else:
        x, y = disps_uv.unbind(dim=-1)

    camera_model = camera_type.build_camera_model(intrinsics)
    pts, Jz, Jf = camera_model.iproj_disp(disps, x, y, compute_jz=compute_jz, compute_jf=compute_jf)
    return pts, Jz, Jf


def proj_points(
    ps: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_type: CameraType,
    compute_jp: bool = False,
    compute_jf: bool = False,
):
    """
    Pinhole camera projection using dense disparities.

    Args:
        ps: (N, ..., 4) tensor of homogeneous points
        intrinsics: (N, 4+D) tensor of intrinsics
        camera_type: CameraType to be used
        compute_jp: bool to compute jacobian of the homogeneous points
        compute_jf: bool to compute jacobian of the intrinsics

    Returns:
        coords: (N, ..., 2) tensor of coordinates
        Jp: (N, ..., 2, 4) tensor of jacobian of the homogeneous points
        Jf: (N, ..., 2, 1+D) tensor of jacobian of the focal + distortion
    """
    camera_model = camera_type.build_camera_model(intrinsics)
    coords, Jp, Jf = camera_model.proj_points(ps, compute_jp=compute_jp, compute_jf=compute_jf)
    return coords, Jp, Jf


def actp(Gij: SE3, X0: torch.Tensor, compute_jp: bool = False):
    """
    action on point cloud
    Args:
        Gij: (N, ) SE3 of poses
        X0: (N, ..., 4) tensor of homogeneous points
        compute_jp: bool to compute jacobian of the homogeneous points

    Returns:
        X1: (N, ..., 4) tensor of homogeneous points
        Jp: (N, ..., 4, 6) tensor of jacobian of the homogeneous points
    """

    extra_dim_shapes = X0.shape[1:-1]
    n_extra_dim = len(extra_dim_shapes)  # Dim of "..."
    X1: torch.Tensor = Gij.view((-1,) + (1,) * n_extra_dim) * X0  # type: ignore

    Jp = None
    if compute_jp:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        N = d.shape[0]

        if isinstance(Gij, SE3):
            Jp = torch.stack(
                [
                    # fmt: off
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    # fmt: on
                ],
                dim=-1,
            ).view(N, *extra_dim_shapes, 4, 6)

        elif isinstance(Gij, Sim3):
            Jp = torch.stack(
                [
                    # fmt: off
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    X,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    Y,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    Z,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    # fmt: on
                ],
                dim=-1,
            ).view(N, *extra_dim_shapes, 4, 7)

    return X1, Jp


def iproj_i_proj_j_disp(
    poses: SE3,
    disps: torch.Tensor,
    disps_uv: torch.Tensor | None,
    intrinsics: torch.Tensor,
    camera_type: CameraType,
    rig: SE3,
    pi: torch.Tensor,
    pj: torch.Tensor,
    qi: torch.Tensor,
    qj: torch.Tensor,
    di: torch.Tensor | None,
    jacobian_p_d: bool,
    jacobian_f: bool,
    jacobian_r: bool,
):
    """
    Compute proj[rig_qj.inv() * pose_j * pose_i.inv() * rig_qi * iproj(disp_i, intr_qi), intr_qj]

    Args:
        poses: (N, ) SE3 of inverse poses
        disps: (NV/M, ...) tensor of disparities
        disps_uv: (NV/M, ..., 2) tensor of uv coordinates (if None, disps must be dense_disps)
        intrinsics: (Q, 4+D) tensor of intrinsics
        rig: (Q, ) SE3 of rig c2w poses
        pi: (M,) tensor of indices for pose_i indexing in N space
        pj: (M,) tensor of indices for pose_j indexing in N space
        qi: (M,) tensor of indices for intr_qi indexing in Q space
        qj: (M,) tensor of indices for intr_qj indexing in Q space
        di: (M,) tensor of indices for dense_disp_i indexing in NV space
            if None, disps/disps_uv must have leading dimension of M (i.e. n_terms).
        jacobian_p_d: bool to compute jacobian of the dense_disps
        jacobian_f: bool to compute jacobian of the focal/distortion
        jacobian_r: bool to compute jacobian of the rig

    Returns:
        x1: (M, ..., 2) tensor of coordinates
        valid: (M, ..., 1) tensor of valid points
        (Ji, Jj, Jz): tuple of jacobian of the poses (M, ..., 2, 6), (M, ..., 2, 6), (M, ..., 2, 1)
        (Jfi, Jfj): tuple of jacobian of the intrinsics (M, ..., 2, 1+D), (M, ..., 2, 1+D)
        (Jri, Jrj): tuple of jacobian of the rig    (M, ..., 2, 6), (M, ..., 2, 6)
    """
    jacobian_p_d = jacobian_p_d or jacobian_f or jacobian_r

    # Convert leading dimension from M to NV.
    if di is not None:
        disps = disps[di]
        if disps_uv is not None:
            disps_uv = disps_uv[di]

    # inverse project (pinhole)
    # X0 (n_terms, ..., 4)
    # Jz = d(iproj_z)/dz = (n_terms, ..., 4)
    # Jfi = d(iproj_z)/dfi = (n_terms, ..., 4, 1+D)
    X0, Jz, Jfi = iproj_disp(
        disps,
        disps_uv,
        intrinsics[qi],
        camera_type,
        compute_jz=jacobian_p_d,
        compute_jf=jacobian_f,
    )

    # transform
    Gij = poses[pj] * poses[pi].inv()
    T = rig[qj].inv() * Gij * rig[qi]  # type: ignore
    assert T is not None
    # X1 (n_terms, ..., 4), Ja = d(T*iproj_z)/dT = (n_terms, ..., 4, 6)
    X1, Ja = actp(T, X0, compute_jp=jacobian_p_d)

    # project (pinhole),
    # Jp = d(proj)/d(T*iproj_z) = (n_terms, ..., 2, 4)
    # Jfj = d(proj)/dfj = (n_terms, ..., 2, 1+D)
    x1, Jp, Jfj = proj_points(X1, intrinsics[qj], camera_type, compute_jp=jacobian_p_d, compute_jf=jacobian_f)

    # exclude points too close to camera
    valid = ((X1[..., 2] > BaseCameraModel.MIN_DEPTH) & (X0[..., 2] > BaseCameraModel.MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    # (Below we use the formula that T.adjT(Ji) = Ji @ Adj(T))
    n_dot_dim = len(X0.shape) - 2
    dot_indexing = (slice(None),) + (None,) * n_dot_dim
    expanded_indexing = (slice(None),) + (None,) * (n_dot_dim + 1)

    if jacobian_p_d:
        # Ja now becomes d(T*iproj_z)/dGj
        Ja = rig[qj].inv()[expanded_indexing].adjT(Ja)
        # dproj/dGj = Jp * Ja
        Jj = torch.matmul(Jp, Ja)  # type: ignore
        # dproj/dGi = dproj/dT * Adj_(Rjinv Gj) * -Adj_iinv = -Jj @ Adj(Gij)
        Ji = -Gij[expanded_indexing].adjT(Jj)  # type: ignore

        # d(T*iproj_z)/dz = d(T*iproj_z)/d(iproj_z) * d(iproj_z)/dz
        Jz = T[dot_indexing] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))  # type: ignore
    else:
        Ji = Jj = None

    if jacobian_f:
        # d(proj)/dfi = d(proj)/d(T*iproj_z) * d(T*iproj_z)/d(iproj_z) * d(iproj_z)/dfi
        Jfi = T[expanded_indexing] * Jfi.transpose(-1, -2)
        Jfi = torch.matmul(Jp, Jfi.transpose(-1, -2))  # type: ignore
    else:
        Jfi = Jfj = None

    if jacobian_r:
        # Rig should behave opposite to the poses
        Jri, Jrj = -Ji, -Jj  # type: ignore
    else:
        Jri = Jrj = None

    return x1, valid, (Ji, Jj, Jz), (Jfi, Jfj), (Jri, Jrj)


def frame_distance_dense_disp(
    poses: SE3,
    dense_disps: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_type: CameraType,
    rig: SE3,
    pi: torch.Tensor,
    pj: torch.Tensor,
    qi: torch.Tensor,
    qj: torch.Tensor,
    di: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Compute the 'frame_distance' metric for the provided edges.
        frame_distance = beta * proj_T_ij + (1 - beta) * proj_t_ij
    where proj_T_ij is the sum of induced motion flows between i and j
    and proj_t_ij is the sum of induced motion flows between i and j assuming that T only has translational component.

    Args:
        poses: (N, ) SE3 of inverse poses
        dense_disps: (NV, H, W) tensor of disparities
        intrinsics: (Q, 4+D) tensor of intrinsics
        camera_type: CameraType to be used
        rig: (Q, ) SE3 of rig c2w poses
        pi: (M,) tensor of indices for pose_i indexing in N space
        pj: (M,) tensor of indices for pose_j indexing in N space
        qi: (M,) tensor of indices for intr_qi indexing in Q space
        qj: (M,) tensor of indices for intr_qj indexing in Q space
        di: (M,) tensor of indices for dense_disp_i indexing in NV space

    Returns:
        distance: (M,) tensor of frame distances
    """
    pinhole_intrinsics = camera_type.build_camera_model(intrinsics).pinhole().intrinsics

    # Expand pose into NQ space
    poses = (rig.inv().view((1, -1)) * poses.view((-1, 1))).view((-1,))  # type: ignore
    num_views = rig.shape[0]
    pi = pi * num_views + qi
    pj = pj * num_views + qj

    return slam_ext.frame_distance(
        poses.data,
        dense_disps,
        pinhole_intrinsics,
        pi,
        pj,
        qi,
        qj,
        di,
        beta,
    )
