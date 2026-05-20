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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange

from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType

from ..maths import geom
from ..maths.matrix import SparseBlockMatrix, SparseDenseBlockMatrix, SparseMDiagonalBlockMatrix
from ..maths.vector import SparseBlockVector
from .kernel import RobustKernel


class TermEvalReturn(ABC):
    @abstractmethod
    def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix: ...

    @abstractmethod
    def nwjtr(self, group_name: str) -> SparseBlockVector: ...

    @abstractmethod
    def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor): ...

    @abstractmethod
    def residual(self) -> torch.Tensor: ...

    def apply_robust_kernel(self, kernel: RobustKernel):
        raise NotImplementedError


@dataclass(kw_only=True)
class ConcreteTermEvalReturn(TermEvalReturn):
    J: dict[str, SparseBlockMatrix]  # group_name -> (n_occ, res_dim, manifold_dim)
    w: torch.Tensor  # (n_terms, res_dim, )
    r: torch.Tensor  # (n_terms, res_dim, )

    # n_occ = number of occurrences of this group_name in all the terms.
    # i.e. the number of blocks in the sparse Jacobian matrix with size n_terms x n_vars

    def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix:
        wJ = self.J[group_name_col].scale_w_left(self.w)
        try:
            return self.J[group_name_row].tmult_mat(wJ).coalesce()
        except NotImplementedError:
            return wJ.tmult_mat(self.J[group_name_row]).transpose().coalesce()

    def nwjtr(self, group_name: str) -> SparseBlockVector:
        return self.J[group_name].tmult_vec(-self.w * self.r).coalesce()

    def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor):
        j_group = self.J[group_name]
        keep_mask = torch.isin(j_group.j_inds, col_inds, invert=True)
        self.J[group_name] = j_group.subset(keep_mask)

    def apply_robust_kernel(self, kernel: RobustKernel):
        robust_weight = kernel.apply(self.r)
        self.w = self.w * robust_weight

    def residual(self) -> torch.Tensor:
        return torch.sum(self.r * self.r * self.w, dim=1)


class SolverTerm(ABC):
    @abstractmethod
    def forward(
        self,
        variables: dict[str, Any],
        jacobian: bool = True,
        active_group_names: set[str] | None = None,
    ) -> TermEvalReturn: ...

    @abstractmethod
    def group_names(self) -> set[str]: ...

    def update(self, solver):
        # Default implementation do nothing.
        pass


class DenseDepthFlowTerm(SolverTerm):
    """
    E(pose_pi, pose_pj, dense_disp_di, intr_qi, intr_qj) = \
        proj(rig_j.inv() * pose_j * pose_i.inv() * rig_i, dense_disp_di) - target_[ij di]

        Pose is the world2cam transform.
        Rig is the cam2world(central cam) transform.
        target_[ij di] is the target projected location.
    res_dim = H*W*2
    """

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        rig_i_inds: torch.Tensor,
        rig_j_inds: torch.Tensor,
        dense_disp_i_inds: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        intrinsics: torch.Tensor | None,
        intrinsics_factor: float,
        rig: SE3 | None,
        image_size: tuple[int, int],
        camera_type: CameraType,
    ) -> None:
        super().__init__()

        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert rig_i_inds.shape == (self.n_terms,)
        assert rig_j_inds.shape == (self.n_terms,)
        assert dense_disp_i_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.rig_i_inds = rig_i_inds
        self.rig_j_inds = rig_j_inds
        self.dense_disp_i_inds = dense_disp_i_inds
        self.image_size = image_size
        self.camera_type = camera_type

        n_pixels = image_size[0] * image_size[1]

        self.target = target.reshape(self.n_terms, n_pixels, 2)  # (n_terms, H*W, 2)
        self.weight = weight.reshape(self.n_terms, n_pixels, 2)  # (n_terms, H*W, 2)
        self.intrinsics = intrinsics.reshape(-1, 4) if intrinsics is not None else None  # (Q, 4)
        self.intrinsics_factor = intrinsics_factor
        self.rig = rig

    def group_names(self) -> set[str]:
        names = {"pose", "dense_disp"}
        if self.intrinsics is None:
            names.add("intrinsics")
        if self.rig is None:
            names.add("rig")
        return names

    def forward(
        self,
        variables: dict[str, Any],
        jacobian: bool = True,
        active_group_names: set[str] | None = None,
    ) -> TermEvalReturn:
        """
        variables contain:
            - pose: (n_var, ) SE3 of poses
            - dense_disp: (n_var, H*W) tensor of disparities
            - intrinsics: (Q, 4) tensor of intrinsics (optional)

        # TODO: To accelerate, you can return a PrecomputedTermEvalReturn with kernels from Droid-SLAM.
        """
        pose, dense_disp = variables["pose"], variables["dense_disp"]
        if active_group_names is None:
            active_group_names = self.group_names()

        if self.intrinsics is None:
            intrinsics = variables["intrinsics"]
        else:
            intrinsics = self.intrinsics
        optimize_intrinsics = self.intrinsics is None and "intrinsics" in active_group_names
        if self.rig is None:
            rig = variables["rig"]
        else:
            rig = self.rig
        optimize_rig = self.rig is None and "rig" in active_group_names
        optimize_pose_disp = "pose" in active_group_names or "dense_disp" in active_group_names

        assert isinstance(pose, SE3) and isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape[1] == self.image_size[0] * self.image_size[1]
        assert intrinsics.shape[0] == rig.shape[0]

        camera_model_cls = self.camera_type.camera_model_cls()

        coords, valid, (Ji, Jj, Jz), (Jfi, Jfj), (Jri, Jrj) = geom.iproj_i_proj_j_disp(
            pose,
            dense_disp.view(-1, self.image_size[0], self.image_size[1]),
            None,
            (camera_model_cls(intrinsics).scaled(1.0 / self.intrinsics_factor).intrinsics),
            self.camera_type,
            rig,
            self.pose_i_inds,
            self.pose_j_inds,
            self.rig_i_inds,
            self.rig_j_inds,
            self.dense_disp_i_inds,
            jacobian_p_d=jacobian and optimize_pose_disp,
            jacobian_f=jacobian and optimize_intrinsics,
            jacobian_r=jacobian and optimize_rig,
        )
        coords = rearrange(coords, "n h w c -> n (h w) c", c=2)
        weight = rearrange(valid, "n h w 1 -> n (h w) 1") * self.weight  # (n_terms, H*W, 2)
        weight = rearrange(weight, "n hw c -> n (hw c)", c=2)

        J_dict: dict[str, SparseBlockMatrix] = {}
        if jacobian:
            term_inds = torch.arange(self.n_terms).to(pose.device)
            if "pose" in active_group_names:
                assert Ji is not None and Jj is not None
                Ji = rearrange(Ji, "n h w c d -> n (h w c) d", c=2, d=6)
                Jj = rearrange(Jj, "n h w c d -> n (h w c) d", c=2, d=6)
            if "dense_disp" in active_group_names:
                assert Jz is not None
                Jz = rearrange(Jz, "n h w c d -> n (h w) (c d)", c=2, d=1)

            if "pose" in active_group_names:
                J_dict["pose"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([Ji, Jj], dim=0),
                )
            if "dense_disp" in active_group_names:
                J_dict["dense_disp"] = SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.dense_disp_i_inds,
                    data=Jz,
                )
            if optimize_intrinsics:
                assert Jfi is not None and Jfj is not None
                Jfi = rearrange(Jfi, "n h w c d -> n (h w c) d", c=2)
                Jfj = rearrange(Jfj, "n h w c d -> n (h w c) d", c=2)
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=camera_model_cls.J_scale(
                        1.0 / self.intrinsics_factor,
                        torch.cat([Jfi, Jfj], dim=0),
                    ),
                )
            if optimize_rig:
                assert Jri is not None and Jrj is not None
                Jri = rearrange(Jri, "n h w c d -> n (h w c) d", c=2, d=6)
                Jrj = rearrange(Jrj, "n h w c d -> n (h w c) d", c=2, d=6)
                J_dict["rig"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=torch.cat([Jri, Jrj], dim=0),
                )

        return ConcreteTermEvalReturn(
            J=J_dict,
            w=weight,
            r=rearrange(coords - self.target, "n hw c -> n (hw c)", c=2),
        )


class DispSensRegularizationTerm(SolverTerm):
    """
    E(dense_disp_i) = dense_disp_i - dense_disps_sens_i
    res_dim = H*W
    """

    @dataclass(kw_only=True)
    class ThisTermEvalReturn(TermEvalReturn):
        alpha: float
        i_inds: torch.Tensor
        disps_sens_res: torch.Tensor

        def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix:
            assert group_name_row == group_name_col == "dense_disp"
            return SparseMDiagonalBlockMatrix(
                i_inds=self.i_inds,
                j_inds=self.i_inds,
                data=torch.full_like(self.disps_sens_res, self.alpha).unsqueeze(-1),
            )

        def nwjtr(self, group_name: str) -> SparseBlockVector:
            assert group_name == "dense_disp"
            return SparseBlockVector(inds=self.i_inds, data=-self.alpha * self.disps_sens_res)

        def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor):
            assert group_name == "dense_disp"
            keep_mask = torch.isin(self.i_inds, col_inds, invert=True)
            self.i_inds = self.i_inds[keep_mask]
            self.disps_sens_res = self.disps_sens_res[keep_mask]

        def residual(self) -> torch.Tensor:
            return self.alpha * (self.disps_sens_res**2).sum(dim=1)

    def __init__(self, i_inds: torch.Tensor, alpha: float, disps_sens: torch.Tensor) -> None:
        super().__init__()

        self.i_inds = i_inds
        self.alpha = alpha
        self.disps_sens = disps_sens

    def group_names(self) -> set[str]:
        return {"dense_disp"}

    def forward(
        self,
        variables: dict[str, Any],
        jacobian: bool = True,
        active_group_names: set[str] | None = None,
    ) -> TermEvalReturn:
        """
        variables contain:
            - dense_disp: (n_var, H*W) tensor of disparities
        """
        dense_disp = variables["dense_disp"]

        assert isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape == self.disps_sens.shape

        return self.ThisTermEvalReturn(
            alpha=self.alpha,
            i_inds=self.i_inds,
            disps_sens_res=dense_disp[self.i_inds] - self.disps_sens[self.i_inds],
        )


class TracksFlowTerm(SolverTerm):
    """
    E (pose_pi, pose_pj, tracks_di, intr_qi, intr_qj) = \
        proj(rig_j.inv() * pose_j * pose_i.inv() * rig_i, tracks_di) - target_[ij di]

        Pose is the world2cam transform.
        Rig is the cam2world(central cam) transform.
        target_[ij di] is the target projected location.
    res_dim = n_tracks*2
    """

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        rig_i_inds: torch.Tensor,
        rig_j_inds: torch.Tensor,
        tracks_i_inds: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        tracks_uv: torch.Tensor,
        intrinsics: torch.Tensor | None,
        rig: SE3,
        camera_type: CameraType,
    ) -> None:
        super().__init__()

        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert rig_i_inds.shape == (self.n_terms,)
        assert rig_j_inds.shape == (self.n_terms,)
        assert tracks_i_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.rig_i_inds = rig_i_inds
        self.rig_j_inds = rig_j_inds
        self.tracks_i_inds = tracks_i_inds
        self.camera_type = camera_type

        self.target = target.reshape(self.n_terms, -1, 2)  # (n_terms, n_tracks, 2)
        self.weight = weight.reshape(self.n_terms, -1, 2)  # (n_terms, n_tracks, 2)
        self.tracks_uv = tracks_uv

        self.n_tracks = self.target.shape[1]
        assert self.target.shape[1] == self.n_tracks
        assert self.weight.shape[1] == self.n_tracks
        assert self.tracks_uv.shape[1] == self.n_tracks

        self.intrinsics = intrinsics.reshape(-1, 4) if intrinsics is not None else None
        self.rig = rig

    def group_names(self) -> set[str]:
        names = {"pose", "tracks_disp"}
        if self.intrinsics is None:
            names.add("intrinsics")
        return names

    def forward(
        self,
        variables: dict[str, Any],
        jacobian: bool = True,
        active_group_names: set[str] | None = None,
    ) -> TermEvalReturn:
        """
        variables contain:
            - pose: (n_var, ) SE3 of poses
            - tracks_disp: (n_var, n_tracks) tensor of disparities
            - intrinsics: (Q, 4) tensor of intrinsics (optional)
        """
        pose, tracks_disp = variables["pose"], variables["tracks_disp"]
        if active_group_names is None:
            active_group_names = self.group_names()

        if self.intrinsics is None:
            intrinsics = variables["intrinsics"]
        else:
            intrinsics = self.intrinsics
        optimize_intrinsics = self.intrinsics is None and "intrinsics" in active_group_names
        optimize_pose_disp = "pose" in active_group_names or "tracks_disp" in active_group_names

        assert isinstance(pose, SE3) and isinstance(tracks_disp, torch.Tensor)
        assert tracks_disp.shape[1] == self.n_tracks
        assert intrinsics.shape[0] == self.rig.shape[0]

        coords, valid, (Ji, Jj, Jz), (Jfi, Jfj), _ = geom.iproj_i_proj_j_disp(
            pose,
            tracks_disp,
            self.tracks_uv,
            intrinsics,
            self.camera_type,
            self.rig,
            self.pose_i_inds,
            self.pose_j_inds,
            self.rig_i_inds,
            self.rig_j_inds,
            self.tracks_i_inds,
            jacobian_p_d=jacobian and optimize_pose_disp,
            jacobian_f=jacobian and optimize_intrinsics,
            jacobian_r=False,
        )
        weight = self.weight * valid

        J_dict: dict[str, SparseBlockMatrix] = {}
        if jacobian:
            term_inds = torch.arange(self.n_terms).to(pose.device)
            if "pose" in active_group_names:
                assert Ji is not None and Jj is not None
                Ji = rearrange(Ji, "n t c d -> n (t c) d", c=2, d=6)
                Jj = rearrange(Jj, "n t c d -> n (t c) d", c=2, d=6)
            if "tracks_disp" in active_group_names:
                assert Jz is not None
                Jz = rearrange(Jz, "n t c d -> n (t) (c d)", c=2, d=1)

            if "pose" in active_group_names:
                J_dict["pose"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([Ji, Jj], dim=0),
                )
            if "tracks_disp" in active_group_names:
                J_dict["tracks_disp"] = SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.tracks_i_inds,
                    data=Jz,
                )
            if optimize_intrinsics:
                assert Jfi is not None and Jfj is not None
                Jfi = rearrange(Jfi, "n t c d -> n (t c) d", c=2)
                Jfj = rearrange(Jfj, "n t c d -> n (t c) d", c=2)
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=torch.cat([Jfi, Jfj], dim=0),
                )

        return ConcreteTermEvalReturn(
            J=J_dict,
            w=weight.view(self.n_terms, -1),
            r=rearrange(coords - self.target, "n t c -> n (t c)", c=2),
        )
