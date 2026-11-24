# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from vipe.slam.ba.solver import Solver
from vipe.slam.ba.terms import ConcreteTermEvalReturn, SolverTerm
from vipe.slam.maths.matrix import SparseBlockMatrix, SparseDenseBlockMatrix


class LinearResidualTerm(SolverTerm):
    def __init__(self, coefficients: dict[str, float], target: float) -> None:
        self.coefficients = coefficients
        self.target = target

    def group_names(self) -> set[str]:
        return set(self.coefficients)

    def forward(
        self,
        variables,
        jacobian: bool = True,
        active_group_names: set[str] | None = None,
    ) -> ConcreteTermEvalReturn:
        first = next(iter(variables.values()))
        prediction = sum(self.coefficients[name] * variables[name][0, 0] for name in self.coefficients)
        residual = (prediction - self.target).reshape(1, 1)
        jacobians: dict[str, SparseBlockMatrix] = {}
        if jacobian:
            active = self.group_names() if active_group_names is None else active_group_names
            for name, coefficient in self.coefficients.items():
                if name in active:
                    jacobians[name] = SparseDenseBlockMatrix(
                        i_inds=torch.zeros(1, dtype=torch.long),
                        j_inds=torch.zeros(1, dtype=torch.long),
                        data=first.new_tensor([[[coefficient]]]),
                    )
        return ConcreteTermEvalReturn(J=jacobians, w=torch.ones_like(residual), r=residual)


def solve_case() -> list[float]:
    solver = Solver()
    # These two terms deliberately expose the same pose/intrinsics pair through
    # differently sized group sets. A raw set iteration can reverse that pair
    # and split its Hessian contributions across (a,b) and (b,a).
    solver.add_term(LinearResidualTerm({"pose": 1.0, "intrinsics": 1.0}, 1.0))
    solver.add_term(LinearResidualTerm({"pose": 2.0, "intrinsics": 3.0, "dense_disp": 4.0}, 2.0))
    variables = {
        "pose": torch.zeros((1, 1), dtype=torch.float64),
        "intrinsics": torch.zeros((1, 1), dtype=torch.float64),
        "dense_disp": torch.zeros((1, 1), dtype=torch.float64),
    }
    for name in variables:
        solver.set_damping(name, damping=1.0e-3, ep=1.0e-6)
    solver.run_inplace(variables)
    return [variables[name].item() for name in ("pose", "intrinsics", "dense_disp")]


def dense_reference() -> torch.Tensor:
    jacobian = torch.tensor([[1.0, 1.0, 0.0], [2.0, 3.0, 4.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    lhs = jacobian.T @ jacobian
    lhs += torch.diag(torch.diagonal(lhs) * 1.0e-3 + 1.0e-6)
    return torch.linalg.solve(lhs, jacobian.T @ target)


def test_solver_group_order_aggregates_cross_terms_across_hash_seeds() -> None:
    expected = dense_reference()
    script = Path(__file__).resolve()
    observed = []
    for seed in (0, 39, 123):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        completed = subprocess.run(
            [sys.executable, str(script), "--solve-case"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        actual = torch.tensor(json.loads(completed.stdout), dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=1.0e-7, atol=1.0e-8)
        observed.append(actual)
    for actual in observed[1:]:
        torch.testing.assert_close(actual, observed[0], rtol=0.0, atol=0.0)


if __name__ == "__main__" and sys.argv[1:] == ["--solve-case"]:
    print(json.dumps(solve_case()))
