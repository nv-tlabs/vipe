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

from dataclasses import dataclass

import torch

from vipe.ext.scatter import scatter_add


def _segment_sum_sorted(data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Sum consecutive, already-sorted segments without CUDA atomics."""
    if data.numel() == 0:
        return data.new_empty((lengths.numel(), *data.shape[1:]))
    return torch.segment_reduce(data, "sum", lengths=lengths)


@dataclass(kw_only=True)
class RavelMapping:
    mapping: torch.Tensor  # From i/j_inds to ravel index
    back_mapping: torch.Tensor  # From ravel index to i/j_inds
    n_variables: int
    n_blocks: int


@dataclass(kw_only=True)
class SparseBlockVector:
    inds: torch.Tensor
    data: torch.Tensor

    def __post_init__(self):
        assert self.inds.shape[0] == self.data.shape[0]
        assert len(self.data.shape) == 2

    def element_shape(self) -> int:
        return self.data.shape[1]

    def coalesce(self):
        if not torch.are_deterministic_algorithms_enabled():
            inds, inverse = torch.unique(self.inds, return_inverse=True)
            data = scatter_add(self.data, inverse, dim=0)
            return SparseBlockVector(inds=inds, data=data)

        order = torch.argsort(self.inds, stable=True)
        sorted_inds = self.inds[order]
        inds, lengths = torch.unique_consecutive(sorted_inds, return_counts=True)
        data = _segment_sum_sorted(self.data[order], lengths)
        return SparseBlockVector(inds=inds, data=data)

    def __sub__(self, other: "SparseBlockVector") -> "SparseBlockVector":
        if isinstance(other, SparseNullVector):
            return self
        return SparseBlockVector(
            inds=torch.cat([self.inds, other.inds]),
            data=torch.cat([self.data, -other.data]),
        ).coalesce()

    def __add__(self, other: "SparseBlockVector") -> "SparseBlockVector":
        if isinstance(other, SparseNullVector):
            return self
        return SparseBlockVector(
            inds=torch.cat([self.inds, other.inds]),
            data=torch.cat([self.data, other.data]),
        ).coalesce()


@dataclass(kw_only=True)
class SparseNullVector(SparseBlockVector):
    def __init__(self):
        super().__init__(inds=torch.tensor([]), data=torch.tensor([]).reshape(0, 1))

    def __add__(self, other: SparseBlockVector) -> SparseBlockVector:
        return other

    def coalesce(self):
        return self


SparseVectorDict = dict[str, SparseBlockVector]


@dataclass
class SparseVectorSubview:
    vectors: SparseVectorDict
    group_names: list[str]

    def get_dict(self) -> SparseVectorDict:
        return {name: self.vectors[name] for name in self.group_names}

    def get_ravel_mapping(self) -> list[RavelMapping]:
        # Will return for each group the mapping from its indices to a ravel index.
        mapping = []
        for name in self.group_names:
            unique_indices = torch.unique(vec_inds := self.vectors[name].inds)
            mapping_tensor = torch.full(
                (int(vec_inds.max().item()) + 1,),
                -1,
                dtype=torch.long,
                device=vec_inds.device,
            )
            mapping_tensor[unique_indices] = torch.arange(unique_indices.shape[0], device=vec_inds.device)
            n_blocks = unique_indices.shape[0]
            n_variables = n_blocks * self.vectors[name].element_shape()
            mapping.append(
                RavelMapping(
                    mapping=mapping_tensor,
                    back_mapping=unique_indices,
                    n_variables=n_variables,
                    n_blocks=n_blocks,
                )
            )
        return mapping

    def ravel(self, ravel_mapping: list[RavelMapping]) -> torch.Tensor:
        assert len(ravel_mapping) == len(self.group_names)

        data = []
        deterministic = torch.are_deterministic_algorithms_enabled()
        for group_idx in range(len(self.group_names)):
            mapping = ravel_mapping[group_idx]
            vector = self.vectors[self.group_names[group_idx]]
            if deterministic:
                vector = vector.coalesce()
            element_shape = vector.element_shape()
            full_inds = mapping.mapping[vector.inds].reshape(-1, 1) * element_shape + torch.arange(
                element_shape, device=vector.inds.device
            ).reshape(1, -1)
            if deterministic:
                group_data = vector.data.new_zeros(mapping.n_variables)
                group_data.index_copy_(0, full_inds.reshape(-1), vector.data.reshape(-1))
                data.append(group_data)
            else:
                data.append(
                    scatter_add(
                        vector.data.reshape(-1),
                        full_inds.reshape(-1),
                        dim_size=mapping.n_variables,
                    )
                )
        return torch.cat(data)

    def __sub__(self, other: "SparseVectorSubview") -> "SparseVectorSubview":
        assert self.group_names == other.group_names
        return SparseVectorSubview(
            vectors={name: self.vectors[name] - other.vectors[name] for name in self.group_names},
            group_names=self.group_names,
        )

    def unravel(self, res: torch.Tensor, ravel_mapping: list[RavelMapping]) -> "SparseVectorSubview":
        vectors = {}

        start_indices = torch.cumsum(torch.tensor([0] + [r.n_variables for r in ravel_mapping]), dim=0)
        for group_idx in range(len(ravel_mapping)):
            mapping = ravel_mapping[group_idx]
            vector = SparseBlockVector(
                inds=mapping.back_mapping,
                data=res[start_indices[group_idx] : start_indices[group_idx + 1]].reshape(mapping.n_blocks, -1),
            )
            vectors[self.group_names[group_idx]] = vector

        return SparseVectorSubview(vectors=vectors, group_names=self.group_names)

    @property
    def F(self):
        assert len(self.group_names) == 1
        return self.vectors[self.group_names[0]]
