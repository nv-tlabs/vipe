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

import torch


class RobustKernel(ABC):
    """
    Per-residual M-estimator applied inside the BA solver.

    Given a residual tensor ``x`` (typically shape ``(n_terms, res_dim)``),
    ``apply(x)`` returns a multiplicative reweighting of the same shape.
    ``Solver.run_inplace`` folds that reweighting into the information
    weight via ``ConcreteTermEvalReturn.apply_robust_kernel``, producing
    one IRLS step per BA iteration.
    """

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return a per-residual weight of the same shape as ``x``."""


class HuberRobustKernel(RobustKernel):
    """
    Huber M-estimator weight: ``w(r) = 1`` if ``|r| <= k`` else ``k / |r|``.

    ``threshold`` (``k``) is the residual magnitude at which the loss
    transitions from quadratic (inliers) to linear (outliers).  For the
    dense-flow residuals used by ``DenseDepthFlowTerm``, which operate
    on the /8 feature grid, a value around ``3.0`` is a reasonable
    default (roughly 24 full-resolution pixels of flow mismatch).
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = float(threshold)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = x.abs()
        k = self.threshold
        return torch.where(abs_x <= k, torch.ones_like(x), k / abs_x.clamp_min(1e-8))


def build_robust_kernel(name: str | None, threshold: float) -> RobustKernel | None:
    """
    Instantiate a :class:`RobustKernel` by short name.

    Returns ``None`` to indicate vanilla L2 (no reweighting) so that the
    caller can use ``solver.add_term(term, kernel=build_robust_kernel(...))``
    uniformly and rely on ``Solver.run_inplace`` to skip the kernel when
    it is ``None``.

    Parameters
    ----------
    name : str | None
        Kernel short name.  ``None`` / ``"none"`` / ``"null"`` / ``"off"``
        / ``""`` all map to the L2 (no-kernel) case.  Currently supported:
        ``"huber"``.
    threshold : float
        Kernel threshold parameter (Huber ``k``).
    """
    if name is None:
        return None
    if isinstance(name, str) and name.lower() in ("none", "null", "off", ""):
        return None
    n = str(name).lower()
    if n == "huber":
        return HuberRobustKernel(threshold=threshold)
    raise ValueError(f"Unknown robust_kernel: {name!r}; expected 'huber' or None")
