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

from collections import defaultdict
from dataclasses import dataclass

import torch
from einops import rearrange

from vipe.ext.scatter import scatter_add

from .vector import RavelMapping, SparseBlockVector, SparseNullVector, SparseVectorDict, SparseVectorSubview


@dataclass(kw_only=True)
class SparseBlockMatrix:
    i_inds: torch.Tensor
    j_inds: torch.Tensor

    def tmult_vec(self, vec: torch.Tensor | SparseBlockVector) -> SparseBlockVector:
        raise NotImplementedError

    def tmult_mat(self, mat: "SparseBlockMatrix") -> "SparseBlockMatrix":
        raise NotImplementedError

    def scale_w_left(self, vec: torch.Tensor) -> "SparseBlockMatrix":
        raise NotImplementedError

    def coalesce(self) -> "SparseBlockMatrix":
        raise NotImplementedError

    def transpose(self) -> "SparseBlockMatrix":
        raise NotImplementedError

    def subset(self, inds: torch.Tensor) -> "SparseBlockMatrix":
        raise NotImplementedError

    def has_inverse(self) -> bool:
        return False

    def inverse(self) -> "SparseBlockMatrix":
        raise NotImplementedError

    def __add__(self, other: "SparseBlockMatrix") -> "SparseBlockMatrix":
        raise NotImplementedError

    def __sub__(self, other: "SparseBlockMatrix") -> "SparseBlockMatrix":
        raise NotImplementedError

    def apply_damping_assume_coalesced(self, damping: SparseBlockVector | float, ep: float) -> None:
        raise NotImplementedError

    def _tmult_mat_elements(self, other_i_inds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        other_mapping: defaultdict = defaultdict(list)
        for i, v in enumerate(other_i_inds.cpu().numpy()):
            other_mapping[v].append(i)

        self_inds, other_inds = [], []
        for i, v in enumerate(self.i_inds.cpu().numpy()):
            if v in other_mapping:
                for j in other_mapping[v]:
                    self_inds.append(i)
                    other_inds.append(j)

        return (
            torch.tensor(self_inds).to(other_i_inds.device),
            torch.tensor(other_inds).to(other_i_inds.device),
        )

    def ravel(
        self,
        row_mapping: RavelMapping,
        row_start_inds: int,
        col_mapping: RavelMapping,
        col_start_inds: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


@dataclass(kw_only=True)
class SparseNullMatrix(SparseBlockMatrix):
    def __init__(self):
        super().__init__(i_inds=torch.tensor([]), j_inds=torch.tensor([]))

    def __add__(self, other: SparseBlockMatrix) -> SparseBlockMatrix:
        return other

    def transpose(self) -> SparseBlockMatrix:
        return self

    def tmult_mat(self, mat: SparseBlockMatrix) -> SparseBlockMatrix:
        return self

    def tmult_vec(self, vec: torch.Tensor | SparseBlockVector) -> SparseBlockVector:
        return SparseNullVector()

    def coalesce(self) -> SparseBlockMatrix:
        return self


@dataclass(kw_only=True)
class SparseDenseBlockMatrix(SparseBlockMatrix):
    # (n_blocks, n_block_rows, n_block_cols)
    data: torch.Tensor

    def __post_init__(self):
        assert self.i_inds.shape == self.j_inds.shape
        assert self.i_inds.shape[0] == self.data.shape[0]
        assert len(self.data.shape) == 3

    def tmult_vec(self, vec: torch.Tensor | SparseBlockVector) -> SparseBlockVector:
        if isinstance(vec, SparseBlockVector):
            self_inds, vec_inds = self._tmult_mat_elements(vec.inds)
            j_inds, data = self.j_inds[self_inds], self.data[self_inds]
            vec_data = vec.data[vec_inds]

        else:
            # assume that vec's inds is just arange of unique(i_inds), e.g. arange(n_terms)
            j_inds, data = self.j_inds, self.data
            vec_data = vec[self.i_inds]

        return SparseBlockVector(
            inds=j_inds,
            data=torch.einsum("bij,bi->bj", data, vec_data),
        )

    def tmult_mat(self, mat: SparseBlockMatrix) -> SparseBlockMatrix:
        if not isinstance(mat, SparseDenseBlockMatrix):
            raise NotImplementedError
        self_inds, mat_inds = self._tmult_mat_elements(mat.i_inds)

        new_data = torch.einsum("bij,bik->bjk", self.data[self_inds], mat.data[mat_inds])
        return SparseDenseBlockMatrix(
            i_inds=self.j_inds[self_inds],
            j_inds=mat.j_inds[mat_inds],
            data=new_data,
        )

    def scale_w_left(self, vec: torch.Tensor) -> SparseBlockMatrix:
        return SparseDenseBlockMatrix(
            i_inds=self.i_inds,
            j_inds=self.j_inds,
            data=vec[self.i_inds].unsqueeze(-1) * self.data,
        )

    def transpose(self):
        return SparseDenseBlockMatrix(
            i_inds=self.j_inds,
            j_inds=self.i_inds,
            data=self.data.transpose(1, 2),
        )

    def subset(self, inds: torch.Tensor):
        return SparseDenseBlockMatrix(
            i_inds=self.i_inds[inds],
            j_inds=self.j_inds[inds],
            data=self.data[inds],
        )

    def coalesce(self):
        ij_inds = torch.stack([self.i_inds, self.j_inds], dim=0)
        ij_inds, inverse = torch.unique(ij_inds, return_inverse=True, dim=1)
        data = scatter_add(self.data, inverse, dim=0)
        return SparseDenseBlockMatrix(i_inds=ij_inds[0], j_inds=ij_inds[1], data=data)

    def apply_damping_assume_coalesced(self, damping: SparseBlockVector | float, ep: float) -> None:
        assert self.data.shape[1] == self.data.shape[2]

        diag_mask = self.i_inds == self.j_inds
        if isinstance(damping, float):
            identity = torch.eye(self.data.shape[1], device=self.data.device).unsqueeze(0)
            self.data[diag_mask] += (ep + damping * self.data[diag_mask]) * identity

        else:
            assert isinstance(damping, SparseBlockVector)
            diag_i_inds = self.i_inds[diag_mask]
            assert torch.all(diag_i_inds[1:] > diag_i_inds[:-1]), "Assuming sorted"
            damping_data_inds = torch.searchsorted(diag_i_inds, damping.inds)
            self.data[torch.where(diag_mask)[0][damping_data_inds]] += ep + torch.diag_embed(damping.data)

    def __sub__(self, other: SparseBlockMatrix) -> SparseBlockMatrix:
        assert isinstance(other, SparseDenseBlockMatrix)
        return SparseDenseBlockMatrix(
            i_inds=torch.cat([self.i_inds, other.i_inds]),
            j_inds=torch.cat([self.j_inds, other.j_inds]),
            data=torch.cat([self.data, -other.data]),
        ).coalesce()

    def __add__(self, other: SparseBlockMatrix) -> SparseBlockMatrix:
        if isinstance(other, SparseDenseBlockMatrix):
            return SparseDenseBlockMatrix(
                i_inds=torch.cat([self.i_inds, other.i_inds]),
                j_inds=torch.cat([self.j_inds, other.j_inds]),
                data=torch.cat([self.data, other.data]),
            ).coalesce()
        elif isinstance(other, SparseNullMatrix):
            return self
        else:
            raise NotImplementedError

    def ravel(
        self,
        row_mapping: RavelMapping,
        row_start_inds: int,
        col_mapping: RavelMapping,
        col_start_inds: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_n_rows, data_n_cols = self.data.shape[1], self.data.shape[2]
        data_row_inds, data_col_inds = torch.meshgrid(
            torch.arange(data_n_rows, device=self.data.device),
            torch.arange(data_n_cols, device=self.data.device),
            indexing="ij",
        )
        data_row_inds = data_row_inds.reshape(1, -1)
        data_col_inds = data_col_inds.reshape(1, -1)
        full_row_inds = ((row_mapping.mapping[self.i_inds] * data_n_rows).reshape(-1, 1) + data_row_inds).reshape(-1)
        full_col_inds = ((col_mapping.mapping[self.j_inds] * data_n_cols).reshape(-1, 1) + data_col_inds).reshape(-1)
        return (
            row_start_inds + full_row_inds,
            col_start_inds + full_col_inds,
            self.data.reshape(-1),
        )


@dataclass(kw_only=True)
class SparseMDiagonalBlockMatrix(SparseBlockMatrix):
    # (n_blocks, N, n_diags)
    #  +-----+
    #  |x    |
    #  |  x  |  N  (diag-0)
    #  |    x|
    #  +-----+
    #  |x    |
    #  | ... |  N  (diag-1)
    data: torch.Tensor

    def __post_init__(self):
        assert self.i_inds.shape == self.j_inds.shape
        assert self.i_inds.shape[0] == self.data.shape[0]
        assert len(self.data.shape) == 3

    def tmult_vec(self, vec: torch.Tensor | SparseBlockVector) -> SparseBlockVector:
        if isinstance(vec, SparseBlockVector):
            self_inds, vec_inds = self._tmult_mat_elements(vec.inds)
            j_inds, data = self.j_inds[self_inds], self.data[self_inds]
            vec_data = vec.data[vec_inds]

        else:
            # assume that vec's inds is just arange of unique(i_inds), e.g. arange(n_terms)
            j_inds, data = self.j_inds, self.data
            vec_data = vec[self.i_inds]

        return SparseBlockVector(
            inds=j_inds,
            data=(data.view(data.shape[0], -1) * vec_data).view_as(data).sum(dim=-1),
        )

    def tmult_mat(self, mat: SparseBlockMatrix) -> SparseBlockMatrix:
        self_inds, mat_inds = self._tmult_mat_elements(mat.i_inds)

        if isinstance(mat, SparseMDiagonalBlockMatrix):
            new_data = torch.sum(self.data[self_inds] * mat.data[mat_inds], dim=-1)
            return SparseMDiagonalBlockMatrix(
                i_inds=self.j_inds[self_inds],
                j_inds=mat.j_inds[mat_inds],
                data=new_data.unsqueeze(-1),
            )

        else:
            assert isinstance(mat, SparseDenseBlockMatrix)
            # (n_matrix_elements, N, n_diags, 1) * (n_matrix_elements, (N, n_diags), block_cols)
            new_data = (
                self.data[self_inds].unsqueeze(-1)
                * rearrange(mat.data[mat_inds], "n (r d) c -> n r d c", d=self.data.shape[-1])
            ).sum(-2)
            return SparseDenseBlockMatrix(
                i_inds=self.j_inds[self_inds],
                j_inds=mat.j_inds[mat_inds],
                data=new_data,
            )

    def scale_w_left(self, vec: torch.Tensor) -> SparseBlockMatrix:
        return SparseMDiagonalBlockMatrix(
            i_inds=self.i_inds,
            j_inds=self.j_inds,
            data=vec[self.i_inds].view_as(self.data) * self.data,
        )

    def coalesce(self):
        ij_inds = torch.stack([self.i_inds, self.j_inds], dim=0)
        ij_inds, inverse = torch.unique(ij_inds, return_inverse=True, dim=1)
        data = scatter_add(self.data, inverse, dim=0)
        return SparseMDiagonalBlockMatrix(i_inds=ij_inds[0], j_inds=ij_inds[1], data=data)

    def subset(self, inds: torch.Tensor) -> SparseBlockMatrix:
        return SparseMDiagonalBlockMatrix(
            i_inds=self.i_inds[inds],
            j_inds=self.j_inds[inds],
            data=self.data[inds],
        )

    def transpose(self) -> SparseBlockMatrix:
        assert self.data.shape[-1] == 1
        return self

    def has_inverse(self) -> bool:
        return self.data.shape[-1] == 1

    def inverse(self) -> SparseBlockMatrix:
        assert self.has_inverse(), "Only implemented for diagonal matrices"
        return SparseMDiagonalBlockMatrix(
            i_inds=self.i_inds,
            j_inds=self.j_inds,
            data=self.data.reciprocal(),
        )

    def __add__(self, other: SparseBlockMatrix) -> SparseBlockMatrix:
        assert isinstance(other, SparseMDiagonalBlockMatrix)
        return SparseMDiagonalBlockMatrix(
            i_inds=torch.cat([self.i_inds, other.i_inds]),
            j_inds=torch.cat([self.j_inds, other.j_inds]),
            data=torch.cat([self.data, other.data]),
        ).coalesce()

    def __sub__(self, other: SparseBlockMatrix) -> SparseBlockMatrix:
        assert isinstance(other, SparseMDiagonalBlockMatrix)
        return SparseMDiagonalBlockMatrix(
            i_inds=torch.cat([self.i_inds, other.i_inds]),
            j_inds=torch.cat([self.j_inds, other.j_inds]),
            data=torch.cat([self.data, -other.data]),
        ).coalesce()

    def apply_damping_assume_coalesced(self, damping: SparseBlockVector | float, ep: float) -> None:
        assert self.data.shape[-1] == 1

        diag_mask = self.i_inds == self.j_inds
        if isinstance(damping, float):
            self.data[diag_mask] += ep + damping * self.data[diag_mask]

        else:
            assert isinstance(damping, SparseBlockVector)
            diag_i_inds = self.i_inds[diag_mask]
            assert torch.all(diag_i_inds[1:] > diag_i_inds[:-1]), "Assuming sorted"
            damping_data_inds = torch.searchsorted(diag_i_inds, damping.inds)
            self.data[torch.where(diag_mask)[0][damping_data_inds]] += ep + damping.data.unsqueeze(-1)

    def ravel(
        self,
        row_mapping: RavelMapping,
        row_start_inds: int,
        col_mapping: RavelMapping,
        col_start_inds: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_size, n_diags = self.data.shape[1], self.data.shape[2]
        data_row_inds = (
            torch.arange(n_diags * n_size, device=self.data.device).reshape(n_diags, n_size).T.reshape(1, -1)
        )
        data_col_inds = torch.arange(n_size, device=self.data.device).reshape(-1, 1).repeat(1, n_diags).reshape(1, -1)
        full_row_inds = ((row_mapping.mapping[self.i_inds] * n_size * n_diags).reshape(-1, 1) + data_row_inds).reshape(
            -1
        )
        full_col_inds = ((col_mapping.mapping[self.j_inds] * n_size).reshape(-1, 1) + data_col_inds).reshape(-1)
        return (
            row_start_inds + full_row_inds,
            col_start_inds + full_col_inds,
            self.data.reshape(-1),
        )


SparseBlockMatrixDict = dict[tuple[str, str], SparseBlockMatrix]


@dataclass
class SparseMatrixSubview:
    """
    Represents a matrix of sparse matrices.
    If the matrix is unspecified, it is assumed to be symmetric.
    """

    matrices: SparseBlockMatrixDict
    row_group_names: list[str]
    col_group_names: list[str]

    def get(self, row_group_name: str, col_group_name: str) -> SparseBlockMatrix:
        if (row_group_name, col_group_name) in self.matrices:
            return self.matrices[(row_group_name, col_group_name)]
        elif (col_group_name, row_group_name) in self.matrices:
            return self.matrices[(col_group_name, row_group_name)].transpose()
        else:
            return SparseNullMatrix()

    def __matmul__(self, other: "SparseMatrixSubview") -> "SparseMatrixSubview":
        matrices: SparseBlockMatrixDict = {}
        for self_row_name in self.row_group_names:
            for other_col_name in other.col_group_names:
                target_key = (self_row_name, other_col_name)
                matrices[target_key] = SparseNullMatrix()

                # Dense multiplication because we could not assume symmetry.
                for self_col_name in self.col_group_names:
                    self_mat = self.get(self_row_name, self_col_name)
                    other_mat = other.get(self_col_name, other_col_name)
                    try:
                        res_mat = self_mat.transpose().tmult_mat(other_mat)
                    except NotImplementedError:
                        res_mat = (other_mat.tmult_mat(self_mat.transpose())).transpose()

                    matrices[target_key] += res_mat.coalesce()

        return SparseMatrixSubview(
            matrices=matrices,
            row_group_names=self.row_group_names,
            col_group_names=other.col_group_names,
        )

    def has_inverse(self) -> bool:
        if self.row_group_names != self.col_group_names:
            return False
        for row_name in self.row_group_names:
            for col_name in self.col_group_names:
                if row_name != col_name and not isinstance(self.get(row_name, col_name), SparseNullMatrix):
                    return False
                if row_name == col_name and not self.get(row_name, col_name).has_inverse():
                    return False
        return True

    def inverse(self) -> "SparseMatrixSubview":
        assert self.has_inverse()
        matrices = {}
        for group_name in self.row_group_names:
            group_pair_name = (group_name, group_name)
            matrices[group_pair_name] = self.get(group_name, group_name).inverse()
        return SparseMatrixSubview(
            matrices=matrices,
            row_group_names=self.row_group_names,
            col_group_names=self.col_group_names,
        )

    def transpose(self) -> "SparseMatrixSubview":
        matrices = {}
        for row_name in self.row_group_names:
            for col_name in self.col_group_names:
                matrices[(col_name, row_name)] = self.get(row_name, col_name).transpose()
        return SparseMatrixSubview(
            matrices=matrices,
            row_group_names=self.col_group_names,
            col_group_names=self.row_group_names,
        )

    def __mul__(self, other: SparseVectorSubview) -> SparseVectorSubview:
        assert self.col_group_names == other.group_names

        vectors: SparseVectorDict = {group_name: SparseNullVector() for group_name in self.row_group_names}
        for row_name in self.row_group_names:
            for col_name in self.col_group_names:
                vectors[row_name] += (
                    self.get(row_name, col_name).transpose().tmult_vec(other.vectors[col_name]).coalesce()
                )

        return SparseVectorSubview(vectors=vectors, group_names=self.row_group_names)

    def __sub__(self, other: "SparseMatrixSubview") -> "SparseMatrixSubview":
        assert self.row_group_names == other.row_group_names
        assert self.col_group_names == other.col_group_names

        matrices: SparseBlockMatrixDict = {}
        for row_name in self.row_group_names:
            for col_name in self.col_group_names:
                matrices[(row_name, col_name)] = self.get(row_name, col_name) - other.get(row_name, col_name)

        return SparseMatrixSubview(
            matrices=matrices,
            row_group_names=self.row_group_names,
            col_group_names=self.col_group_names,
        )

    def ravel(self, ravel_mapping: list[RavelMapping]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(ravel_mapping) == len(self.row_group_names) == len(self.col_group_names)

        pis, pjs, pds = [], [], []
        start_indices = torch.cumsum(torch.tensor([0] + [r.n_variables for r in ravel_mapping]), dim=0)[:-1]

        for row_idx in range(len(self.row_group_names)):
            for col_idx in range(len(self.col_group_names)):
                matrix = self.get(self.row_group_names[row_idx], self.col_group_names[col_idx])
                pi, pj, pd = matrix.ravel(
                    ravel_mapping[row_idx],
                    int(start_indices[row_idx].item()),
                    ravel_mapping[col_idx],
                    int(start_indices[col_idx].item()),
                )
                pis.append(pi)
                pjs.append(pj)
                pds.append(pd)

        return torch.cat(pis), torch.cat(pjs), torch.cat(pds)

    @property
    def F(self):
        assert len(self.row_group_names) == 1
        assert len(self.col_group_names) == 1
        return self.get(self.row_group_names[0], self.col_group_names[0])
