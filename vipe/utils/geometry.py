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

"""Utility functions dealing e.g. with generic geometric sampling or transformations."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

from pycg.isometry import Isometry, Quaternion

from vipe.ext.lietorch import SE3, SO3, LieGroupParameter, Sim3


def uniformly_sample_aabb(mins: torch.Tensor, maxes: torch.Tensor, spacing: float) -> torch.Tensor:
    """Return uniformly spaced 3d point coordinates within the bounding cube, defined by its min/max coords.

    Args:
        mins: the (x, y, z) coordinates corresponding to the minimums of the box.
        maxes: the (x, y, z) coordinates corresponding to the maximums of the box.
        spacing: the equal spacing of sampled points, used for all three axes.

    Returns:
        3d points uniformly filling the bounding cube, shape (num_points, 3[x, y, z]).
            Where: num_points = (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / (spacing ** 3)
    """
    xmin, ymin, zmin = mins.tolist()
    xmax, ymax, zmax = maxes.tolist()
    x_steps = (xmax - xmin) / spacing
    y_steps = (ymax - ymin) / spacing
    z_steps = (zmax - zmin) / spacing
    xs = torch.linspace(xmin, xmax, int(x_steps))
    ys = torch.linspace(ymin, ymax, int(y_steps))
    zs = torch.linspace(zmin, zmax, int(z_steps))
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="xy"), dim=0)
    assert isinstance(grid, torch.Tensor), "Assertion for mypy"
    return grid.T.reshape(-1, 3).to(dtype=torch.float32)


def project_points_to_pinhole(
    xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_size: tuple[int, int],
    return_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        xyz: 3D points in the local coordinate system, shape (num_points, 3[x, y, z]).
        return_depth: if the depth values should be returned as well.

    Returns:
        2D points in the panorama image, shape (num_points, 2[u, v, (d)]).
        Here u and v is within [0, 1], and d is the actual depth value.
    """
    xyz = xyz.clone()

    xyz[..., :2] /= xyz[..., 2:3]
    u = intrinsics[0] * xyz[..., 0] + intrinsics[2]
    v = intrinsics[1] * xyz[..., 1] + intrinsics[3]

    u /= frame_size[1]
    v /= frame_size[0]
    d = xyz[..., 2]

    valid_mask = (d > 0) & (u > 0) & (u < 1) & (v > 0) & (v < 1)
    u[~valid_mask] = 0.0
    v[~valid_mask] = 0.0
    d[~valid_mask] = 0.0

    if return_depth:
        return torch.stack((u, v, d), dim=-1), valid_mask
    return torch.stack((u, v), dim=-1), valid_mask


def project_points_to_panorama(
    xyz: torch.Tensor,
    return_depth: bool = False,
) -> torch.Tensor:
    """
    Camera convention (with rotation = I, up of panorama is outward, Y is inward):
       -----
      (  Z  )
     (   |   )
    (    Y-X  )
     (       )
      (     )
       -<|>-
         |
    [boundary of image]

    Args:
        xyz: 3D points in the local coordinate system, shape (num_points, 3[x, y, z]).
        return_depth: if the depth values should be returned as well.

    Returns:
        2D points in the panorama image, shape (num_points, 2[u, v, (d)]).
        Here u and v is within [0, 1], and d is the actual depth value.
    """
    depth = torch.linalg.norm(xyz, dim=-1)
    u = torch.atan2(xyz[..., 0], xyz[..., 2])  # [-pi, pi]
    v = torch.acos(-xyz[..., 1] / depth)  # [0, pi]
    u = u / (2 * torch.pi) + 0.5  # [0, 1]
    v = v / torch.pi  # [0, 1]
    if return_depth:
        return torch.stack((u, v, depth), dim=-1)
    return torch.stack((u, v), dim=-1)


def se3_matrix_inverse(se3: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """Compute the inverse of rigid transformations given as SE3 matrices

    Args:
        se3: single / batch of SE3 transformation matrices [bs, 4, 4] or [4,4]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single / batch of SE3 matrices [bs, 4, 4] or [4,4]
    """

    # Convert numpy array to torch tensor
    if isinstance(se3, np.ndarray):
        se3 = torch.from_numpy(se3)

    # batch dimensions unconditionally
    se3 = se3.reshape((-1, 4, 4))  # (N,4,4)

    ret = torch.eye(4, dtype=se3.dtype, device=se3.device).reshape(1, 4, 4).repeat((len(se3), 1, 1))
    ret[:, :3, :3] = (Rt := se3[:, :3, :3].transpose(1, 2))
    ret[:, :3, 3:] = -Rt @ se3[:, :3, 3:]

    # unbatch dimensions conditionally
    if unbatch:
        ret = ret.squeeze()

    return ret  # (N,4,4) or (4,4)


def se3_matrix_to_tquat(se3: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """
    Converts a singe / batch of SE3 matrices (4x4) into a single / batch [t,q]
    7d transformation representations consisting of [translation, normalized_quaternion] parts

    Args:
        se3: single / batch of SE3 transformation matrices [bs, 4, 4] or [4,4]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single/ batch of 7D quaternion representation [translation, unit_quaternion]  [bs, 7] or [7]
    """

    # Convert numpy array to torch tensor
    if isinstance(se3, np.ndarray):
        se3 = torch.from_numpy(se3)

    # batch dimensions unconditionally
    se3 = se3.reshape((-1, 4, 4))  # (N,4,4)

    ret = torch.empty((len(se3), 7), dtype=se3.dtype, device=se3.device)
    if len(se3):
        ret[:, :3] = se3[:, :3, 3]
        ret[:, 3:] = so3_matrix_to_quat(se3[:, :3, :3], unbatch=False)

    if unbatch:  # unbatch dimensions conditionally
        ret = ret.squeeze()

    return ret  # (N,7) or (7,)


def so3_matrix_to_quat(R: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """
    Converts a singe / batch of SO3 rotation matrices (3x3) to unit quaternion representation.

    Args:
        R: single / batch of SO3 rotation matrices [bs, 3, 3] or [3,3]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single / batch of unit quaternions (XYZW convention)  [bs, 4] or [4]
    """

    # Convert numpy array to torch tensor
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)

    R = R.reshape((-1, 3, 3))  # batch dimensions unconditionally
    num_rotations, D1, D2 = R.shape
    assert (D1, D2) == (3, 3), "so3_matrix_to_quat: Input has to be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
    quat = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)

    decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
    choices = decision_matrix.argmax(dim=1)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
    quat[ind, j] = R[ind, j, i] + R[ind, i, j]
    quat[ind, k] = R[ind, k, i] + R[ind, i, k]
    quat[ind, 3] = R[ind, k, j] - R[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
    quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
    quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]

    if unbatch:  # unbatch dimensions conditionally
        quat = quat.squeeze()

    return quat  # (N,4) or (4,)


def quat_to_so3_matrix(quat: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """
    Converts a singe / batch of quaternions (4) to SO3 representation.

    Args:
        quat: single / batch of quaternions (XYZW convention) [bs, 4] or [4]]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single / batch of SO3 matrices [bs, 3, 3] or [3,3]
    """

    # Convert numpy array to torch tensor
    if isinstance(quat, np.ndarray):
        quat = torch.from_numpy(quat)

    quat = quat.reshape((-1, 4))  # batch dimensions unconditionally
    num_quats, _ = quat.shape

    x, y, z, w = torch.unbind(quat, -1)
    x_2 = x * x
    y_2 = y * y
    z_2 = z * z
    xy = x * y
    xz = x * z
    xw = x * w
    yz = y * z
    yw = y * w
    zw = z * w

    R = torch.stack(
        (
            1 - 2 * (y_2 + z_2),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (x_2 + z_2),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (x_2 + y_2),
        ),
        -1,
    ).reshape(num_quats, 3, 3)

    if unbatch:  # unbatch dimensions conditionally
        R = R.squeeze()

    return R  # (N,3,3) or (3,3)


def tquat_to_se3_matrix(tquat: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """
    Converts a single / batch of [t,q] 7d transformation representations consisting of
    [translation, normalized_quaternion] parts into a single / batch of N SE3 matrices (4x4)

    Args:
        quat: single/ batch of 7D quaternion representation [translation, unit_quaternion]  [bs, 7] or [7]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single / batch of SE3 matrices [bs, 4, 4] or [4,4]
    """

    # Convert numpy array to torch tensor
    if isinstance(tquat, np.ndarray):
        tquat = torch.from_numpy(tquat)

    # batch dimensions unconditionally
    tquat = tquat.reshape((-1, 7))  # (N,7)

    ret = torch.eye(4, dtype=tquat.dtype, device=tquat.device).reshape(1, 4, 4).repeat((len(tquat), 1, 1))
    ret[:, :3, :3] = quat_to_so3_matrix(tquat[:, 3:], unbatch=False)
    ret[:, :3, 3] = tquat[:, :3]

    # unbatch dimensions conditionally
    if unbatch:
        ret = ret.squeeze()

    return ret  # (N,4,4) or (4,4)


def se3_matrix_to_se3(T: torch.Tensor | np.ndarray, unbatch=True, reduced=False) -> SE3:
    """Converts a single / batch of rigid transformations represented as 4x4 / 3x4 (reduced) matrices

    ⎡ R  t ⎤
    ⎣ 0  1 ⎦

    to SE3 Lie group elements (unbatches conditionally)

    Args:
        T: single/ batch of SE3 transformation matrices [bs, D, 4] or [D, 4]
        unbatch: if the single example should be unbatched (first dimension removed) or not
        reduced: D = 3 if True ("reduced"), D = 4 if False ("not reduced")

    Returns:
        single / batch of SE3 Lie group elements [] / [bs]
    """

    # Convert numpy array to torch tensor
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T)

    T = T.reshape((-1, 4, 4)) if not reduced else T.reshape((-1, 3, 4))  # batch dimensions unconditionally

    vec = torch.hstack((T[:, :3, 3], so3_matrix_to_quat(T[:, :3, :3], unbatch=False)))

    if unbatch:  # unbatch dimensions conditionally
        vec = vec.squeeze()

    return SE3.InitFromVec(vec)


@torch.no_grad()
def se3_to_isometry(se3: SE3) -> list[Isometry] | Isometry:
    """Converts a single SE3 Lie group element to a list of Isometry objects.

    Args:
        se3: single / batch of SE3 Lie group elements [] / [bs]

    Returns:
        list of Isometry objects
    """

    tquat: torch.Tensor = se3.vec().cpu().detach()

    batch_dim = tquat.dim() > 1
    tquat = tquat if batch_dim else tquat[None]

    isometries = [
        Isometry(
            t=tquat_np[:3],
            q=Quaternion(tquat_np[3:][[3, 0, 1, 2]]),
        )
        for tquat_np in tquat.numpy()
    ]

    return isometries if batch_dim else isometries[0]


def so3_to_se3(so3: SO3) -> SE3:
    """Converts a single SO3 Lie group element to a SE3 Lie group element.

    Args:
        so3: single / batch of SO3 Lie group elements [] / [bs]

    Returns:
        single / batch of SE3 Lie group elements [] / [bs]
    """
    t_component = torch.zeros(so3.shape + (3,), device=so3.device, dtype=so3.dtype)
    return SE3.InitFromVec(torch.cat((t_component, so3.vec()), dim=-1))


def se3_to_so3(se3: SE3) -> SO3:
    """Converts a single SE3 Lie group element to a SO3 Lie group element.

    Args:
        se3: single / batch of SE3 Lie group elements [] / [bs]

    Returns:
        single / batch of SO3 Lie group elements [] / [bs]
    """
    return SO3.InitFromVec(se3.vec()[..., 3:])


def so3_average(so3: SE3, dim: int) -> SE3:
    """Computes the Chordal L2 average of the SO3 matrices, where the Isometry distance
    is defined as squared of chordal distance (Frobenius form).
    For other distance like geodesic distance or other norms like L1-norm, no closed-form is provided.

    Ref: Rotation Averaging, Hartley et al. 2013
    """
    q: torch.Tensor = so3.vec()
    q_mean_mat = (q.unsqueeze(-1) * q.unsqueeze(-2)).mean(dim=dim)
    q_mean = torch.linalg.eigh(q_mean_mat).eigenvectors[..., -1]
    return SE3.InitFromVec(q_mean)


def se3_average(se3: SE3, dim: int) -> SE3:
    """Computes the average of the transformations"""
    t: torch.Tensor = se3.translation()[..., :-1]
    t_mean = t.mean(dim=dim)
    q_mean = so3_average(se3_to_so3(se3), dim=dim)
    se3_mean = SE3.InitFromVec(torch.cat((t_mean, q_mean.vec()), dim=-1))
    return se3_mean


@dataclass(slots=True, kw_only=True)
class ScaledTransform:
    """
    Dataclass for storing the result of a similarity transform computation.
    Single batch only.
    """

    rotation: SE3
    translation: torch.Tensor
    scale: float = 1.0

    def apply_se3(self, se3: SE3) -> SE3:
        """Applies the similarity transform to an SE3 transformation."""
        se3_vec = se3.vec()
        target_t, target_quat = se3_vec[:, :3], SE3.InitFromVec(se3_vec[:, 3:])
        target_t = self.scale * self.rotation[None].act(target_t) + self.translation[None]
        target_quat: SE3 = self.rotation[None] * target_quat
        return SE3.InitFromVec(torch.cat((target_t, target_quat.vec()), dim=-1))

    def apply_points(self, points: torch.Tensor) -> torch.Tensor:
        """Applies the similarity transform to a set of 3D poinqts."""
        return self.scale * self.rotation.act(points) + self.translation[None]

    def inv(self) -> "ScaledTransform":
        """Returns the inverse of the similarity transform."""
        inv_rotation = self.rotation.inv()
        inv_scale = 1.0 / self.scale
        inv_translation = -inv_scale * inv_rotation.act(self.translation)
        return ScaledTransform(rotation=inv_rotation, translation=inv_translation, scale=inv_scale)

    def to_sim3(self) -> Sim3:
        """
        Converts the similarity transform to a Sim3 transformation.
        [sR t]
        [0  1]
        """
        device, dtype = self.translation.device, self.translation.dtype
        # trans + rxso3
        tquats = torch.cat(
            (
                self.translation,
                self.rotation.data,
                torch.tensor([self.scale], device=device, dtype=dtype),
            ),
            dim=-1,
        )
        return Sim3(tquats)

    @classmethod
    def from_sim3(self, sim3: Sim3):
        """
        Converts a Sim3 transformation to a similarity transform.
        """
        assert sim3.data.ndim == 1
        return ScaledTransform(
            rotation=SE3.InitFromVec(sim3.data[3:7]),
            translation=sim3.data[:3],
            scale=sim3.data[7].item(),
        )


@dataclass(slots=True, kw_only=True)
class IncompleteRigScaledTransform:
    transform: ScaledTransform
    right_translation: torch.Tensor

    def apply_se3(self, se3: SE3) -> SE3:
        sim3 = Sim3(se3)
        sim3_prior = self.transform.to_sim3()[None] * sim3
        sim3_post = Sim3.Identity(1).to(sim3.device)
        sim3_post.data[:, :3] = self.right_translation
        sim3_final = sim3_prior * sim3_post
        return SE3(sim3_final.data[..., :7])

    def apply_points(self, points: torch.Tensor) -> torch.Tensor:
        return self.transform.apply_points(points + self.right_translation[None])


def align_trajectories(source_traj: SE3, target_traj: SE3, scale: bool = True) -> ScaledTransform:
    return align_points(
        source_traj.translation()[:, :3],
        target_traj.translation()[:, :3],
        scale=scale,
    )


def align_trajectories_rigid(source_traj: SE3, target_traj: SE3, scale: bool = True) -> IncompleteRigScaledTransform:
    """
    Align the trajectories by minimizing using coordinate decent:

    min_{T, R} sum_i || trans(T * source_traj_i * R) - trans(target_traj_i)||^2
        where T is Sim3/SE3 (depending on whether scale is True), and R is SE3.

    As we extract only the translational component, the rotation part of R is not observable/ambiguous.
    Hence we return an IncopleteRigScaledTransform, containing only the translational part of R.
    Note that if you want to compare the output of the results using the RPE metric,
    you should still compare against the gt trajectory of the rig that the traj is estimated from!

    In practice, this converges pretty well:
        Non-aligned (3.8052258, 4.232766068866788)
        GT-aligned (0.0, 0.0)
        T-only (0.7890058, 0.8018749676508908)
        T and R (this algorithm) (1.8263513e-06, 1.9895173635449574e-06)
    """
    T: Sim3 = Sim3.Identity(1).to(source_traj.device)
    R: Sim3 = Sim3.Identity(1).to(source_traj.device)
    source_sim3 = Sim3(source_traj)
    target_trans = target_traj.translation()[:, :3]

    for _ in range(10):
        source_trans = source_sim3 * R
        T = align_points(
            source_trans.translation()[:, :3],
            target_trans,
            scale=scale,
        ).to_sim3()
        t_mat = (T[None] * source_sim3).matrix()
        lhs = t_mat[:, :3, :3].reshape(-1, 3)
        rhs = (target_trans - t_mat[:, :3, 3]).reshape(-1)
        tr = torch.linalg.lstsq(lhs, rhs[:, None]).solution
        R.data[:, :3] = tr[:, 0]

    return IncompleteRigScaledTransform(transform=ScaledTransform.from_sim3(T), right_translation=tr[:, 0])


def align_trajectories_full(source_traj: SE3, target_traj: SE3, n_iters: int = 200) -> SE3:
    """
    Align the trajectories by minimizing (ClusterVO-style):
    [In cases where this is not fully accurate, consider using align_trajectories_rigid to obtain an initialization]

    min_{T, R} sum_i log ||T * source_traj_i * R - target_traj_i||
        where T is SE3, and R is SE3.
    Returns T * source_traj_i * R.
    """
    r_se3, t_se3 = SE3.Identity(1).to(source_traj.device), SE3.Identity(1).to(source_traj.device)
    r_param = LieGroupParameter(r_se3)
    t_param = LieGroupParameter(t_se3)
    optimizer = optim.SGD([r_param, t_param], lr=0.01)

    for _ in range(n_iters):
        optimizer.zero_grad()

        log_diff = ((t_param.retr() * source_traj * r_param.retr()).inv() * target_traj).log()
        e = torch.sum(log_diff**2, dim=-1).mean()
        e.backward()
        optimizer.step()

    source_aligned = (t_param.retr() * source_traj * r_param.retr()).detach()
    return source_aligned


def align_points(source_pts: torch.Tensor, target_pts: torch.Tensor, scale: bool = True) -> ScaledTransform:
    """
    Computes the best transform that could be applied to the source points to align them with the target points.
    When scale is True, use umeyama algorithm, otherwise use procrustes.

    Args:
        source_pts: The source points to be transformed (N, 3)
        target_pts: The target points to be aligned with (N, 3)
        scale: Whether to compute the scale or not.

    Returns:
        The computed similarity transform.
    """
    assert source_pts.shape == target_pts.shape, "Source and target points must have the same shape."
    assert source_pts.size(1) == 3, "Points must be 3D."

    mu1 = source_pts.mean(dim=0)
    mu2 = target_pts.mean(dim=0)

    x1 = source_pts - mu1[None]
    x2 = target_pts - mu2[None]

    k = x1.T @ x2
    U, _, V = torch.svd(k)

    Z = torch.eye(3, device=U.device)
    Z[2, 2] = torch.sign(torch.linalg.det(U @ V.T))

    R = V @ Z @ U.T

    if scale:
        var1 = (x1**2).sum()
        s = (torch.trace(R @ k) / var1).item()

    else:
        s = 1.0

    t = mu2 - s * mu1 @ R.T

    return ScaledTransform(
        rotation=SE3.InitFromVec(so3_matrix_to_quat(R)),
        translation=t,
        scale=s,
    )


def depth_abs_relative_difference(
    output: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> float:
    """
    Implements the AbsRel metric for depth estimation (output should be pre-scaled if needed).
    """
    output, target = output.flatten(), target.flatten()
    assert output.shape == target.shape, "Output and target must have the same shape."
    if valid_mask is not None:
        valid_mask = valid_mask.flatten()
        assert output.shape == valid_mask.shape, "Output and valid mask must have the same shape."

    abs_relative_diff = torch.abs(output - target) / target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum().item()
    else:
        n = output.shape[0]
    abs_relative_diff = torch.sum(abs_relative_diff) / n
    return abs_relative_diff.item()


def depth_rmse_log(output: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> float:
    """
    Implements the RMSE-Log metric for depth estimation (output should be pre-scaled if needed).
    This computes log before rmse, hence takes less account into far away points.
    """
    output, target = output.flatten(), target.flatten()
    assert output.shape == target.shape, "Output and target must have the same shape."
    if valid_mask is not None:
        valid_mask = valid_mask.flatten()
        assert output.shape == valid_mask.shape, "Output and valid mask must have the same shape."
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum().item()
    else:
        n = output.shape[0]
    mse = torch.sum(torch.pow(diff, 2)) / n
    return torch.sqrt(mse).item()


def depth_delta1_accuracy(output: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> float:
    """
    Implements the Delta1 metric for depth estimation (output should be pre-scaled if needed).
    Delta1 is the percentage of inliers where max(z_pd/z_gt, z_gt/z_pd) < 1.25.
    """
    output, target = output.flatten(), target.flatten()
    assert output.shape == target.shape, "Output and target must have the same shape."
    if valid_mask is not None:
        valid_mask = valid_mask.flatten()
        assert output.shape == valid_mask.shape, "Output and valid mask must have the same shape."
    max_d1_d2 = torch.max(output / target, target / output).cpu()
    bit_mat = (max_d1_d2 < 1.25).float()
    if valid_mask is not None:
        bit_mat[~valid_mask.cpu()] = 0
        n = valid_mask.sum().item()
    else:
        n = output.shape[0]
    delta_acc = torch.sum(bit_mat) / n
    return delta_acc.item()
