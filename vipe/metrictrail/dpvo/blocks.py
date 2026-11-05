# SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
# SPDX-License-Identifier: MIT
#
# Source: https://github.com/princeton-vl/DPVO
# See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

# Modified for MetricTrail integration.
# The following Apache-2.0 terms apply to NVIDIA modifications only.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from vipe.metrictrail import dpvo_scatter as torch_scatter  # [vipe] was: import torch_scatter

class LayerNorm1D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def forward(self, x):
        return self.norm(x.transpose(1,2)).transpose(1,2)

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        self.res = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)

class SoftAgg(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, self.dim)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]
            
        return self.h(y)

class SoftAggBasic(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAggBasic, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim,        1)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]
            
        return self.h(y)


### Gradient Clipping and Zeroing Operations ###

GRAD_CLIP = 0.1

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)

class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        grad_x = torch.where(torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x)
        return grad_x

class GradientZero(nn.Module):
    def __init__(self):
        super(GradientZero, self).__init__()

    def forward(self, x):
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        print(grad_x.abs().mean())
        return grad_x
