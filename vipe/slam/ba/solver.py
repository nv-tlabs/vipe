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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from ..maths.matrix import RelativeBlockDamping, SparseBlockMatrixDict, SparseMatrixSubview, SparseNullMatrix
from ..maths.retractor import BaseRetractor
from ..maths.vector import SparseBlockVector, SparseNullVector, SparseVectorDict, SparseVectorSubview
from .kernel import RobustKernel
from .terms import FrozenTermObjective, SolverTerm

logger = logging.getLogger(__name__)

_GROUP_ORDER = {
    "pose": 0,
    "dense_disp": 1,
    "tracks_disp": 2,
    "intrinsics": 3,
    "rig": 4,
}


def _ordered_group_names(group_names) -> list[str]:
    return sorted(group_names, key=lambda name: (_GROUP_ORDER.get(name, len(_GROUP_ORDER)), name))


@dataclass(frozen=True)
class AdaptiveStepResult:
    pre_energy: float
    post_energy: float
    accepted: bool
    attempts: int


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
        self.group_damping: dict[str, SparseBlockVector | RelativeBlockDamping | float] = {}
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

    def set_damping(
        self,
        group_name: str,
        damping: SparseBlockVector | RelativeBlockDamping | float,
        ep: float,
    ):
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

    def _linearize_and_solve(
        self,
        variables: dict[str, Any],
        *,
        freeze_objective: bool,
    ) -> tuple[dict[str, SparseBlockVector], torch.Tensor | None, list[FrozenTermObjective]]:
        lhs: SparseBlockMatrixDict = defaultdict(SparseNullMatrix)
        rhs: SparseVectorDict = defaultdict(SparseNullVector)

        fully_fixed_groups = {t for t, inds in self.group_fixed_inds.items() if inds is None}

        energy: torch.Tensor | None = None
        frozen_objectives: list[FrozenTermObjective] = []
        for term, kernel in zip(self.terms, self.kernels):
            # Compute the newest term formulation
            term.update(self)
            active_group_names = term.group_names().difference(fully_fixed_groups)
            term_return = term.forward(
                variables,
                jacobian=bool(active_group_names),
                active_group_names=active_group_names,
            )
            # A stable block ordering keeps the sparse system identical across
            # Python hash seeds and makes repeated BA experiments reproducible.
            term_group_names = _ordered_group_names(active_group_names)

            if kernel is not None:
                term_return.apply_robust_kernel(kernel)

            if self.compute_energy or freeze_objective:
                cur_energy = term_return.residual().sum()
                energy = cur_energy if energy is None else energy + cur_energy
            if freeze_objective:
                frozen_objectives.append(term_return.freeze_objective())

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
        marginalized_group_names = _ordered_group_names(
            group_name
            for group_name, marginalized in self.group_marginalized.items()
            if marginalized and group_name in all_group_names
        )
        regular_group_names = _ordered_group_names(set(all_group_names).difference(marginalized_group_names))

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

        return x_dict, energy, frozen_objectives

    def _apply_step(self, variables: dict[str, Any], x_dict: dict[str, SparseBlockVector]) -> None:
        for group_name, update in x_dict.items():
            self.group_retractor[group_name].oplus(
                variables[group_name],
                update.inds,
                update.data,
            )

    @staticmethod
    def _snapshot_variables(variables: dict[str, Any]) -> dict[str, torch.Tensor]:
        snapshots: dict[str, torch.Tensor] = {}
        for group_name, variable in variables.items():
            data = variable if isinstance(variable, torch.Tensor) else variable.data
            if not isinstance(data, torch.Tensor):
                raise TypeError(f"Cannot snapshot variable group {group_name!r} of type {type(variable)!r}")
            snapshots[group_name] = data.detach().clone()
        return snapshots

    @staticmethod
    def _restore_variables(variables: dict[str, Any], snapshots: dict[str, torch.Tensor]) -> None:
        for group_name, snapshot in snapshots.items():
            variable = variables[group_name]
            data = variable if isinstance(variable, torch.Tensor) else variable.data
            data.copy_(snapshot)

    def _candidate_energy(
        self,
        variables: dict[str, Any],
        frozen_objectives: list[FrozenTermObjective],
    ) -> torch.Tensor:
        if len(frozen_objectives) != len(self.terms):
            raise RuntimeError(
                f"Expected one frozen objective per term, got {len(frozen_objectives)} for {len(self.terms)} terms"
            )
        energy: torch.Tensor | None = None
        for term, frozen_objective in zip(self.terms, frozen_objectives):
            candidate = term.forward(variables, jacobian=False, active_group_names=set())
            cur_energy = frozen_objective.evaluate(candidate)
            energy = cur_energy if energy is None else energy + cur_energy
        if energy is None:
            raise RuntimeError("Cannot evaluate an adaptive step without solver terms")
        return energy

    @staticmethod
    def _scaled_damping(
        damping: SparseBlockVector | RelativeBlockDamping | float,
        scale: float,
    ) -> SparseBlockVector | RelativeBlockDamping | float:
        def clamp(value: float) -> float:
            return min(value * scale, 1e12)

        if isinstance(damping, SparseBlockVector):
            return SparseBlockVector(
                inds=damping.inds,
                data=torch.clamp(damping.data * scale, max=1e12),
            )
        if isinstance(damping, RelativeBlockDamping):
            return RelativeBlockDamping(tuple(clamp(factor) for factor in damping.factors))
        return clamp(damping)

    @staticmethod
    def _validate_adaptive_damping(damping: SparseBlockVector | RelativeBlockDamping | float) -> None:
        if isinstance(damping, SparseBlockVector):
            if not bool(torch.all(torch.isfinite(damping.data) & (damping.data > 0.0))):
                raise ValueError("Adaptive damping must contain only finite, strictly positive factors")
            return
        factors = damping.factors if isinstance(damping, RelativeBlockDamping) else (damping,)
        if not all(torch.isfinite(torch.tensor(factor)) and factor > 0.0 for factor in factors):
            raise ValueError("Adaptive damping must contain only finite, strictly positive factors")

    @staticmethod
    def _updates_are_finite(x_dict: dict[str, SparseBlockVector]) -> bool:
        return all(bool(torch.isfinite(update.data).all()) for update in x_dict.values())

    def run_inplace(self, variables: dict[str, Any]) -> float:
        x_dict, energy, _ = self._linearize_and_solve(variables, freeze_objective=False)
        self._apply_step(variables, x_dict)
        return energy.item() if energy is not None else 0.0

    def run_adaptive_inplace(
        self,
        variables: dict[str, Any],
        *,
        damping_group: str | None = None,
        damping_groups: Iterable[str] | None = None,
        max_trials: int = 8,
        damping_up: float = 10.0,
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-9,
    ) -> AdaptiveStepResult:
        """Try a joint nonlinear step and atomically reject energy increases.

        Candidate residuals are evaluated with the valid/robust weights frozen
        at each trial's linearization point.  All variable groups are restored
        between trials because Schur-complement updates are coupled.  Rejected
        trials increase damping; accepted trials retain their trial damping.
        Use ``damping_group`` for the legacy single-group path or
        ``damping_groups`` to scale every listed active group together.
        """

        if max_trials < 1:
            raise ValueError("max_trials must be at least 1")
        if damping_up <= 1.0:
            raise ValueError("adaptive damping factor must be greater than 1")

        if (damping_group is None) == (damping_groups is None):
            raise ValueError("Specify exactly one of damping_group or damping_groups")
        if damping_groups is not None and isinstance(damping_groups, str):
            raise TypeError("damping_groups must be an iterable of group names, not a string")

        requested_groups = [damping_group] if damping_group is not None else list(damping_groups or ())
        fully_fixed_groups = {name for name, inds in self.group_fixed_inds.items() if inds is None}
        term_group_names = set().union(*(term.group_names() for term in self.terms)) if self.terms else set()
        requested_group_set = set(requested_groups)
        inactive_groups = requested_group_set.difference(term_group_names).union(
            requested_group_set.intersection(fully_fixed_groups)
        )
        if inactive_groups:
            requested = damping_group if damping_group is not None else requested_groups
            raise ValueError(
                f"Adaptive damping group(s) {_ordered_group_names(inactive_groups)!r} from {requested!r} "
                "are not active in the solved system"
            )
        active_damping_groups = _ordered_group_names(requested_group_set)
        if not active_damping_groups:
            raise ValueError("At least one adaptive damping group is required")

        damping_was_present = {name: name in self.group_damping for name in active_damping_groups}
        initial_dampings = {name: self.group_damping.get(name, 0.0) for name in active_damping_groups}
        for name, damping in initial_dampings.items():
            try:
                self._validate_adaptive_damping(damping)
            except ValueError as exc:
                raise ValueError(f"Invalid adaptive damping for group {name!r}: {exc}") from exc

        def restore_initial_dampings() -> None:
            for name, damping in initial_dampings.items():
                if damping_was_present[name]:
                    self.group_damping[name] = damping
                else:
                    self.group_damping.pop(name, None)

        snapshots = self._snapshot_variables(variables)
        last_pre_energy = float("nan")

        try:
            for attempt in range(1, max_trials + 1):
                self._restore_variables(variables, snapshots)
                trial_scale = damping_up ** (attempt - 1)
                trial_dampings = {
                    name: self._scaled_damping(damping, trial_scale) for name, damping in initial_dampings.items()
                }
                self.group_damping.update(trial_dampings)

                x_dict, pre_energy_tensor, frozen_objectives = self._linearize_and_solve(
                    variables,
                    freeze_objective=True,
                )
                missing_groups = set(active_damping_groups).difference(x_dict)
                if missing_groups:
                    raise ValueError(
                        f"Adaptive damping group(s) {_ordered_group_names(missing_groups)!r} "
                        "are not active in the solved system"
                    )
                if pre_energy_tensor is None:
                    raise RuntimeError("Adaptive solve did not produce a pre-step energy")
                last_pre_energy = pre_energy_tensor.item()

                if self._updates_are_finite(x_dict):
                    self._apply_step(variables, x_dict)
                    candidate_energy = self._candidate_energy(variables, frozen_objectives)
                    post_energy = candidate_energy.item()
                    tolerance = absolute_tolerance + relative_tolerance * abs(last_pre_energy)
                    accepted = bool(torch.isfinite(candidate_energy)) and post_energy <= last_pre_energy + tolerance
                else:
                    post_energy = float("nan")
                    accepted = False

                if accepted:
                    # Be conservative across BA iterations: acceptance never
                    # lowers damping, while every rejected trial ratchets it up.
                    self.group_damping.update(trial_dampings)
                    return AdaptiveStepResult(last_pre_energy, post_energy, True, attempt)

            self._restore_variables(variables, snapshots)
            rejected_dampings = {
                name: self._scaled_damping(damping, damping_up**max_trials)
                for name, damping in initial_dampings.items()
            }
            self.group_damping.update(rejected_dampings)
            return AdaptiveStepResult(last_pre_energy, last_pre_energy, False, max_trials)
        except Exception:
            # A failed candidate evaluation, retraction, or linear solve must
            # not leak a partially updated pose/depth/intrinsics state.
            self._restore_variables(variables, snapshots)
            restore_initial_dampings()
            raise
