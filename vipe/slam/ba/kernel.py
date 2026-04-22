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
    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return a per-residual weight of the same shape as ``x``."""

    def is_gnc(self) -> bool:
        return False

    def set_mu(self, mu: float) -> None:
        pass

    def update_mu(self, mu: float) -> float:
        return mu

    @property
    def mu_max(self) -> float:
        return float("inf")


class HuberRobustKernel(RobustKernel):
    """``w(r) = 1`` if ``|r| <= k`` else ``k / |r|``."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = float(threshold)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = x.abs()
        k = self.threshold
        return torch.where(abs_x <= k, torch.ones_like(x), k / abs_x.clamp_min(1e-8))


class TukeyRobustKernel(RobustKernel):
    """``w(r) = (1 - (r/c)^2)^2`` if ``|r| <= c`` else ``0``."""

    def __init__(self, c: float = 2.0) -> None:
        self.c = float(c)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c
        u = (x / c) ** 2
        return (1.0 - u).clamp(min=0.0) ** 2


class TLSGncRobustKernel(RobustKernel):
    """
    Graduated Non-Convexity with Truncated Least Squares.

    Yang, Antonante, Tzoumas, Carlone.  "Graduated Non-Convexity for Robust
    Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection."
    RA-L / ICRA, 2020.  Weight formulation and mu schedule follow GTSAM's
    ``GncOptimizer``:
    https://github.com/borglab/gtsam/blob/develop/gtsam/nonlinear/GncOptimizer.h

    At scalar ``mu`` with inlier threshold ``c_bar``:

        lo  = mu / (mu + 1)  * c_bar^2
        hi  = (mu + 1) / mu  * c_bar^2
        r^2 < lo   ->  w = 1
        r^2 > hi   ->  w = 0
        otherwise  ->  w = sqrt(c_bar^2 * mu * (mu + 1) / r^2) - mu

    Schedule: ``mu <- min(mu * mu_step, mu_max)``.
    ``mu -> inf`` recovers the hard-TLS indicator ``w = 1{r^2 < c_bar^2}``.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        mu_step: float = 3.0,
        mu_init: float = 1.0,
        mu_max: float = 1.0e6,
    ) -> None:
        self.c_bar = float(threshold)
        self.c_bar_sq = self.c_bar * self.c_bar
        self.mu_step = float(mu_step)
        self.mu_init = float(mu_init)
        self._mu_max = float(mu_max)
        self._mu: float | None = None

    def is_gnc(self) -> bool:
        return True

    @property
    def mu_max(self) -> float:
        return self._mu_max

    def set_mu(self, mu: float) -> None:
        self._mu = float(mu)

    def update_mu(self, mu: float) -> float:
        return min(mu * self.mu_step, self._mu_max)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        mu = self._mu
        c_sq = self.c_bar_sq
        r2 = x * x

        if mu is None or mu >= self._mu_max:
            return (r2 < c_sq).to(dtype=x.dtype)

        lo = (mu / (mu + 1.0)) * c_sq
        hi = ((mu + 1.0) / mu) * c_sq

        weights = torch.zeros_like(x)
        inlier_mask = r2 < lo
        weights = torch.where(inlier_mask, torch.ones_like(x), weights)

        transition_mask = (~inlier_mask) & (r2 <= hi)
        if transition_mask.any():
            r2_t = r2.clamp_min(1e-12)
            transition_w = torch.sqrt(c_sq * mu * (mu + 1.0) / r2_t) - mu
            weights = torch.where(transition_mask, transition_w.clamp_min(0.0), weights)

        return weights


def build_robust_kernel(
    name: str | None,
    threshold: float,
    *,
    gnc_mu_init: float = 1.0,
    gnc_mu_step: float = 3.0,
    gnc_mu_max: float = 1.0e6,
) -> RobustKernel | None:
    """Factory.  ``None`` / ``"none"`` / ``"null"`` / ``"off"`` / ``""`` -> L2."""
    if name is None:
        return None
    n = str(name).lower()
    if n in ("none", "null", "off", ""):
        return None
    if n == "huber":
        return HuberRobustKernel(threshold=threshold)
    if n == "tukey":
        return TukeyRobustKernel(c=threshold)
    if n in ("gnc_tls", "gnc-tls"):
        return TLSGncRobustKernel(
            threshold=threshold,
            mu_init=gnc_mu_init,
            mu_step=gnc_mu_step,
            mu_max=gnc_mu_max,
        )
    raise ValueError(
        f"Unknown robust_kernel: {name!r}; expected 'huber' | 'tukey' | 'gnc_tls' | None"
    )
