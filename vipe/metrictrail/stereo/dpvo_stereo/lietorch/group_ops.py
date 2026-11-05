# SPDX-FileCopyrightText: Copyright (c) 2021, princeton-vl
# SPDX-License-Identifier: BSD-3-Clause
#
# Source: princeton-vl/lietorch, as vendored by princeton-vl/DPVO.
# See the component LICENSE and THIRD_PARTY_LICENSES.md.

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

from vipe.metrictrail.dpvo_stereo_ext import lietorch_backends  # [vipe] was: import lietorch_backends
import torch
import torch.nn.functional as F



class GroupOp(torch.autograd.Function):
    """ group operation base class """

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        out = cls.forward_op(ctx.group_id, *inputs)
        return out

    @classmethod
    def backward(cls, ctx, grad):
        error_str = "Backward operation not implemented for {}".format(cls)
        assert cls.backward_op is not None, error_str

        inputs = ctx.saved_tensors
        grad = grad.contiguous()
        grad_inputs = cls.backward_op(ctx.group_id, grad, *inputs)
        return (None, ) + tuple(grad_inputs)
        

class Exp(GroupOp):
    """ exponential map """
    forward_op, backward_op = lietorch_backends.expm, lietorch_backends.expm_backward

class Log(GroupOp):
    """ logarithm map """
    forward_op, backward_op = lietorch_backends.logm, lietorch_backends.logm_backward

class Inv(GroupOp):
    """ group inverse """
    forward_op, backward_op = lietorch_backends.inv, lietorch_backends.inv_backward

class Mul(GroupOp):
    """ group multiplication """
    forward_op, backward_op = lietorch_backends.mul, lietorch_backends.mul_backward

class Adj(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = lietorch_backends.adj, lietorch_backends.adj_backward

class AdjT(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = lietorch_backends.adjT, lietorch_backends.adjT_backward

class Act3(GroupOp):
    """ action on point """
    forward_op, backward_op = lietorch_backends.act, lietorch_backends.act_backward

class Act4(GroupOp):
    """ action on point """
    forward_op, backward_op = lietorch_backends.act4, lietorch_backends.act4_backward

class Jinv(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = lietorch_backends.Jinv, None

class ToMatrix(GroupOp):
    """ convert to matrix representation """
    forward_op, backward_op = lietorch_backends.as_matrix, None




### conversion operations to/from Euclidean embeddings ###

class FromVec(torch.autograd.Function):
    """ convert vector into group object """

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        return inputs[0]

    @classmethod
    def backward(cls, ctx, grad):
        inputs = ctx.saved_tensors
        J = lietorch_backends.projector(ctx.group_id, *inputs)
        return None, torch.matmul(grad.unsqueeze(-2), torch.linalg.pinv(J)).squeeze(-2)

class ToVec(torch.autograd.Function):
    """ convert group object to vector """

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        return inputs[0]

    @classmethod
    def backward(cls, ctx, grad):
        inputs = ctx.saved_tensors
        J = lietorch_backends.projector(ctx.group_id, *inputs)
        return None, torch.matmul(grad.unsqueeze(-2), J).squeeze(-2)

