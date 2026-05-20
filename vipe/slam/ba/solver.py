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

import logging
from collections import defaultdict
from typing import Any

import torch

from ..maths.matrix import SparseBlockMatrixDict, SparseMatrixSubview, SparseNullMatrix
from ..maths.retractor import BaseRetractor
from ..maths.vector import SparseBlockVector, SparseNullVector, SparseVectorDict, SparseVectorSubview
from .kernel import RobustKernel
from .terms import SolverTerm

logger = logging.getLogger(__name__)


def solve_scipy(pi: torch.Tensor, pj: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    lhs_np = lhs.cpu().numpy()
    rhs_np = rhs.cpu().numpy()
    lhs_sparse = coo_matrix((lhs_np, (pi.cpu().numpy(), pj.cpu().numpy())))
    # Convert to CSR format for efficient spsolve
    lhs_sparse = lhs_sparse.tocsr()

    x = spsolve(lhs_sparse, rhs_np)

    return torch.tensor(x, device=pi.device).float()


class Solver:
    def __init__(
        self,
        compute_energy: bool = False,
    ) -> None:
        """
        If the corresponding JTJ of this group is very sparse, it is faster to solve
        the linear system first with this group being marginalized, and then recover
        the state separately.
        """
        self.terms: list[SolverTerm] = []
        self.kernels: list[RobustKernel | None] = []
        self.compute_energy = compute_energy

        self.group_fixed_inds: dict[str, torch.Tensor | None] = {}
        self.group_damping: dict[str, SparseBlockVector | float] = {}
        self.group_ep: dict[str, float] = {}
        self.group_retractor: dict[str, BaseRetractor] = defaultdict(BaseRetractor)
        self.group_marginalized: dict[str, bool] = defaultdict(lambda: False)

    def _warn_if_no_terms(self, group_name: str):
        all_group_names = set.union(*[t.group_names() for t in self.terms])
        if group_name not in all_group_names:
            logger.warning(f"Group {group_name} is not used in any terms. This may be a mistake.")

    def add_term(self, term: SolverTerm, kernel: RobustKernel | None = None):
        self.terms.append(term)
        self.kernels.append(kernel)

    def set_fixed(self, group_name: str, fixed_inds: torch.Tensor | None = None):
        # None means everything is fixed
        self._warn_if_no_terms(group_name)
        self.group_fixed_inds[group_name] = fixed_inds

    def set_marginilized(self, group_name: str, marginalized: bool = True):
        self._warn_if_no_terms(group_name)
        self.group_marginalized[group_name] = marginalized

    def set_retractor(self, group_name: str, retractor: BaseRetractor):
        self._warn_if_no_terms(group_name)
        self.group_retractor[group_name] = retractor

    def set_damping(self, group_name: str, damping: SparseBlockVector | float, ep: float):
        """
        Set the damping factor.
        If this is a Tensor, it should be of shape (n_vars, n_vars)
            LHS += diag(damping) + ep * I.
        If this is a float, it will be added as
            LHS += diag(LHS) * damping + ep * I
        """
        self._warn_if_no_terms(group_name)
        self.group_damping[group_name] = damping
        self.group_ep[group_name] = ep

    def _solve(self, lhs: SparseMatrixSubview, rhs: SparseVectorSubview) -> SparseVectorSubview:
        assert lhs.row_group_names == lhs.col_group_names == rhs.group_names

        if lhs.has_inverse():
            return lhs.inverse() * rhs

        ravel_mappings = rhs.get_ravel_mapping()
        pi, pj, lhs_data = lhs.ravel(ravel_mappings)
        rhs_data = rhs.ravel(ravel_mappings)

        # print("Begin solution...")
        x_data = solve_scipy(pi, pj, lhs_data, rhs_data)
        # print("End solution...")

        return rhs.unravel(x_data, ravel_mappings)

    def run_inplace(self, variables: dict[str, Any]) -> float:
        lhs: SparseBlockMatrixDict = defaultdict(SparseNullMatrix)
        rhs: SparseVectorDict = defaultdict(SparseNullVector)

        fully_fixed_groups = {t for t, inds in self.group_fixed_inds.items() if inds is None}

        energy: torch.Tensor | None = None
        for term, kernel in zip(self.terms, self.kernels):
            # Compute the newest term formulation
            term.update(self)
            active_group_names = term.group_names().difference(fully_fixed_groups)
            term_return = term.forward(
                variables,
                jacobian=bool(active_group_names),
                active_group_names=active_group_names,
            )
            term_group_names = list(active_group_names)

            if kernel is not None:
                term_return.apply_robust_kernel(kernel)

            if self.compute_energy:
                cur_energy = term_return.residual().sum()
                energy = cur_energy if energy is None else energy + cur_energy

            for group_name, fixed_inds in self.group_fixed_inds.items():
                if group_name in term_group_names and fixed_inds is not None and fixed_inds.numel() > 0:
                    term_return.remove_jcol_inds(group_name, fixed_inds)

            # Compute RHS
            for group_name in term_group_names:
                rhs[group_name] += term_return.nwjtr(group_name)

            # Compute only upper triangular part of the LHS
            for group_i in range(len(term_group_names)):
                for group_j in range(group_i, len(term_group_names)):
                    group_name_i = term_group_names[group_i]
                    group_name_j = term_group_names[group_j]
                    if group_name_i in term_group_names and group_name_j in term_group_names:
                        jtwj = term_return.jtwj(group_name_i, group_name_j)
                        lhs[(group_name_i, group_name_j)] += jtwj

        all_group_names = list(rhs.keys())
        marginalized_group_names = [
            group_name
            for group_name, marginalized in self.group_marginalized.items()
            if marginalized and group_name in all_group_names
        ]
        regular_group_names = list(set(all_group_names).difference(marginalized_group_names))

        for group_name in all_group_names:
            damping = self.group_damping.get(group_name, 0.0)
            ep = self.group_ep.get(group_name, 0.0)
            lhs[(group_name, group_name)].apply_damping_assume_coalesced(damping, ep)

        # Build matrices
        lhs_h = SparseMatrixSubview(lhs, regular_group_names, regular_group_names)
        rhs_v = SparseVectorSubview(rhs, regular_group_names)

        if len(marginalized_group_names) > 0:
            lhs_e = SparseMatrixSubview(lhs, regular_group_names, marginalized_group_names)
            lhs_c = SparseMatrixSubview(lhs, marginalized_group_names, marginalized_group_names)
            rhs_w = SparseVectorSubview(rhs, marginalized_group_names)

            # Apply Schur's formula
            h_cinv = lhs_e @ lhs_c.inverse()
            lhs_reg = lhs_h - h_cinv @ lhs_e.transpose()
            rhs_reg = rhs_v - h_cinv * rhs_w

            x_reg: SparseVectorSubview = self._solve(lhs_reg, rhs_reg)

            rhs_marg = rhs_w - lhs_e.transpose() * x_reg
            x_marg: SparseVectorSubview = self._solve(lhs_c, rhs_marg)

            x_dict = x_reg.get_dict() | x_marg.get_dict()

        else:
            x_dict = self._solve(lhs_h, rhs_v).get_dict()

        for group_name in all_group_names:
            self.group_retractor[group_name].oplus(
                variables[group_name],
                x_dict[group_name].inds,
                x_dict[group_name].data,
            )

        return energy.item() if energy is not None else 0.0
