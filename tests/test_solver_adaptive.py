# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vipe.slam.ba.solver import Solver
from vipe.slam.ba.terms import ConcreteTermEvalReturn, SolverTerm
from vipe.slam.maths.matrix import RelativeBlockDamping, SparseBlockMatrix, SparseDenseBlockMatrix
from vipe.slam.maths.vector import SparseBlockVector


class QuadraticResidualTerm(SolverTerm):
    """One-dimensional residual r(x) = x^2 - 1."""

    def __init__(self, *, dynamic_support: bool = False, raise_on_candidate: bool = False) -> None:
        self.dynamic_support = dynamic_support
        self.raise_on_candidate = raise_on_candidate

    def group_names(self) -> set[str]:
        return {"x"}

    def forward(
        self,
        variables,
        jacobian: bool = True,
        active_group_names: set[str] | None = None,
    ) -> ConcreteTermEvalReturn:
        if not jacobian and self.raise_on_candidate:
            raise RuntimeError("candidate evaluation failed")
        x = variables["x"][0, 0]
        residual = (x.square() - 1.0).reshape(1, 1)
        weight = torch.ones_like(residual)
        if self.dynamic_support:
            weight = weight * (x.abs() < 1.0)

        jacobians: dict[str, SparseBlockMatrix] = {}
        if jacobian and (active_group_names is None or "x" in active_group_names):
            jacobians["x"] = SparseDenseBlockMatrix(
                i_inds=torch.tensor([0], device=x.device),
                j_inds=torch.tensor([0], device=x.device),
                data=(2.0 * x).reshape(1, 1, 1),
            )
        return ConcreteTermEvalReturn(J=jacobians, w=weight, r=residual)


class CoupledQuadraticResidualTerm(SolverTerm):
    """Two nonlinear residuals with a small cross-group consistency term."""

    def __init__(self, *, raise_on_candidate: bool = False) -> None:
        self.raise_on_candidate = raise_on_candidate

    def group_names(self) -> set[str]:
        return {"x", "y"}

    def forward(
        self,
        variables,
        jacobian: bool = True,
        active_group_names: set[str] | None = None,
    ) -> ConcreteTermEvalReturn:
        if not jacobian and self.raise_on_candidate:
            raise RuntimeError("coupled candidate evaluation failed")

        x = variables["x"][0, 0]
        y = variables["y"][0, 0]
        coupling = 0.1
        residual = torch.stack([x.square() - 1.0, y.square() - 1.0, coupling * (x - y)]).reshape(1, 3)
        jacobians: dict[str, SparseBlockMatrix] = {}
        if jacobian:
            active_groups = self.group_names() if active_group_names is None else active_group_names
            block_ind = torch.zeros(1, dtype=torch.long, device=x.device)
            if "x" in active_groups:
                jacobians["x"] = SparseDenseBlockMatrix(
                    i_inds=block_ind,
                    j_inds=block_ind,
                    data=torch.stack([2.0 * x, x.new_zeros(()), x.new_tensor(coupling)]).reshape(1, 3, 1),
                )
            if "y" in active_groups:
                jacobians["y"] = SparseDenseBlockMatrix(
                    i_inds=block_ind,
                    j_inds=block_ind,
                    data=torch.stack([y.new_zeros(()), 2.0 * y, y.new_tensor(-coupling)]).reshape(1, 3, 1),
                )
        return ConcreteTermEvalReturn(J=jacobians, w=torch.ones_like(residual), r=residual)


def build_solver(*, dynamic_support: bool = False, raise_on_candidate: bool = False) -> Solver:
    solver = Solver()
    solver.add_term(
        QuadraticResidualTerm(
            dynamic_support=dynamic_support,
            raise_on_candidate=raise_on_candidate,
        )
    )
    solver.set_damping("x", damping=1e-6, ep=1e-9)
    return solver


def build_coupled_solver(*, raise_on_candidate: bool = False) -> tuple[Solver, SparseBlockVector]:
    solver = Solver()
    solver.add_term(CoupledQuadraticResidualTerm(raise_on_candidate=raise_on_candidate))
    solver.set_damping("x", damping=1e-6, ep=1e-9)
    # At the test's linearization point H_yy = (2*0.1)^2 + 0.1^2 = 0.05,
    # so this absolute damping matches x's initial relative damping.
    y_damping = SparseBlockVector(inds=torch.tensor([0]), data=torch.tensor([[5e-8]]))
    solver.set_damping("y", damping=y_damping, ep=1e-9)
    return solver, y_damping


def test_normal_solver_applies_step_once():
    solver = build_solver()
    variable = torch.tensor([[2.0]])

    solver.run_inplace({"x": variable})

    torch.testing.assert_close(variable, torch.tensor([[1.25]]), rtol=1e-5, atol=1e-5)


def test_adaptive_solver_retries_overshoot_until_energy_decreases():
    solver = build_solver()
    variable = torch.tensor([[0.1]])
    initial_energy = float((variable.square() - 1.0).square())
    initial_damping = solver.group_damping["x"]

    result = solver.run_adaptive_inplace({"x": variable}, damping_group="x", max_trials=8)

    assert result.accepted
    assert result.attempts > 1
    assert result.post_energy < initial_energy
    assert float((variable.square() - 1.0).square()) < initial_energy
    assert solver.group_damping["x"] == initial_damping * 10.0 ** (result.attempts - 1)


def test_adaptive_solver_keeps_damping_after_first_trial_acceptance():
    solver = build_solver()
    variable = torch.tensor([[2.0]])
    initial_damping = solver.group_damping["x"]

    result = solver.run_adaptive_inplace({"x": variable}, damping_group="x")

    assert result.accepted
    assert result.attempts == 1
    assert solver.group_damping["x"] == initial_damping


def test_adaptive_solver_restores_state_when_all_trials_fail():
    solver = build_solver()
    variable = torch.tensor([[0.1]])
    original = variable.clone()
    initial_damping = solver.group_damping["x"]

    result = solver.run_adaptive_inplace({"x": variable}, damping_group="x", max_trials=1)

    assert not result.accepted
    torch.testing.assert_close(variable, original, rtol=0, atol=0)
    assert solver.group_damping["x"] == initial_damping * 10.0


def test_adaptive_solver_uses_frozen_support_for_candidate_energy():
    solver = build_solver(dynamic_support=True)
    variable = torch.tensor([[0.1]])
    original = variable.clone()

    result = solver.run_adaptive_inplace({"x": variable}, damping_group="x", max_trials=1)

    # The overshooting candidate has |x| > 1 and would hide its error by
    # setting its recomputed support weight to zero.  Frozen weights reject it.
    assert not result.accepted
    torch.testing.assert_close(variable, original, rtol=0, atol=0)


def test_adaptive_solver_restores_state_when_candidate_evaluation_raises():
    solver = build_solver(raise_on_candidate=True)
    variable = torch.tensor([[0.1]])
    original = variable.clone()
    original_damping = solver.group_damping["x"]

    with pytest.raises(RuntimeError, match="candidate evaluation failed"):
        solver.run_adaptive_inplace({"x": variable}, damping_group="x", max_trials=1)

    torch.testing.assert_close(variable, original, rtol=0, atol=0)
    assert solver.group_damping["x"] == original_damping


def test_adaptive_solver_requires_positive_damping():
    solver = build_solver()
    solver.set_damping("x", damping=0.0, ep=1e-9)

    with pytest.raises(ValueError, match="strictly positive"):
        solver.run_adaptive_inplace({"x": torch.tensor([[0.1]])}, damping_group="x")


def test_adaptive_damping_scale_one_is_identity_for_all_supported_types():
    sparse = SparseBlockVector(inds=torch.tensor([2]), data=torch.tensor([[1.0e-15, 2.0e-15]]))
    relative = RelativeBlockDamping((1.0e-15, 2.0e-15))

    assert Solver._scaled_damping(1.0e-15, 1.0) == 1.0e-15
    assert Solver._scaled_damping(relative, 1.0) == relative
    scaled_sparse = Solver._scaled_damping(sparse, 1.0)
    assert isinstance(scaled_sparse, SparseBlockVector)
    assert scaled_sparse.inds is sparse.inds
    assert torch.equal(scaled_sparse.data, sparse.data)


def test_adaptive_solver_rejects_a_fixed_damping_group():
    solver = build_solver()
    solver.set_fixed("x")

    with pytest.raises(ValueError, match="not active"):
        solver.run_adaptive_inplace({"x": torch.tensor([[0.1]])}, damping_group="x")


def test_adaptive_solver_rejects_unknown_group_mixed_with_active_group():
    solver = build_solver()

    with pytest.raises(ValueError, match="typo.*not active"):
        solver.run_adaptive_inplace(
            {"x": torch.tensor([[0.1]])},
            damping_groups=("x", "typo"),
        )


def test_coupled_adaptive_solver_needs_all_groups_damped_and_ratchets_each_group():
    single_group_solver, initial_single_y_damping = build_coupled_solver()
    single_group_variables = {"x": torch.tensor([[0.1]]), "y": torch.tensor([[0.1]])}

    single_group_result = single_group_solver.run_adaptive_inplace(
        single_group_variables,
        damping_group="x",
        max_trials=8,
    )

    assert not single_group_result.accepted
    torch.testing.assert_close(single_group_variables["x"], torch.tensor([[0.1]]), rtol=0, atol=0)
    torch.testing.assert_close(single_group_variables["y"], torch.tensor([[0.1]]), rtol=0, atol=0)
    assert single_group_solver.group_damping["y"] is initial_single_y_damping

    all_group_solver, initial_y_damping = build_coupled_solver()
    all_group_variables = {"x": torch.tensor([[0.1]]), "y": torch.tensor([[0.1]])}
    initial_x_damping = all_group_solver.group_damping["x"]

    all_group_result = all_group_solver.run_adaptive_inplace(
        all_group_variables,
        damping_groups=("x", "y"),
        max_trials=8,
    )

    assert all_group_result.accepted
    assert all_group_result.attempts > 1
    trial_scale = 10.0 ** (all_group_result.attempts - 1)
    assert all_group_solver.group_damping["x"] == initial_x_damping * trial_scale
    ratcheted_y_damping = all_group_solver.group_damping["y"]
    assert isinstance(ratcheted_y_damping, SparseBlockVector)
    assert ratcheted_y_damping.inds is initial_y_damping.inds
    torch.testing.assert_close(ratcheted_y_damping.data, initial_y_damping.data * trial_scale)


def test_coupled_adaptive_solver_atomically_restores_all_dampings_on_exception():
    solver, initial_y_damping = build_coupled_solver(raise_on_candidate=True)
    variables = {"x": torch.tensor([[0.1]]), "y": torch.tensor([[0.1]])}
    originals = {name: value.clone() for name, value in variables.items()}
    initial_x_damping = solver.group_damping["x"]

    with pytest.raises(RuntimeError, match="coupled candidate evaluation failed"):
        solver.run_adaptive_inplace(variables, damping_groups=("x", "y"), max_trials=8)

    torch.testing.assert_close(variables["x"], originals["x"], rtol=0, atol=0)
    torch.testing.assert_close(variables["y"], originals["y"], rtol=0, atol=0)
    assert solver.group_damping["x"] == initial_x_damping
    assert solver.group_damping["y"] is initial_y_damping
