# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vipe.slam.maths.matrix import RelativeBlockDamping, SparseDenseBlockMatrix
from vipe.slam.maths.vector import SparseBlockVector


def test_relative_block_damping_scales_each_hessian_diagonal_independently():
    matrix = SparseDenseBlockMatrix(
        i_inds=torch.tensor([0]),
        j_inds=torch.tensor([0]),
        data=torch.tensor([[[4.0, 2.0], [2.0, 9.0]]]),
    )

    matrix.apply_damping_assume_coalesced(RelativeBlockDamping((0.1, 1.0)), ep=0.5)

    torch.testing.assert_close(matrix.data, torch.tensor([[[4.9, 2.0], [2.0, 18.5]]]))


def test_absolute_block_damping_adds_epsilon_only_to_diagonal():
    matrix = SparseDenseBlockMatrix(
        i_inds=torch.tensor([0]),
        j_inds=torch.tensor([0]),
        data=torch.tensor([[[4.0, 2.0], [2.0, 9.0]]]),
    )
    damping = SparseBlockVector(inds=torch.tensor([0]), data=torch.tensor([[1.0, 3.0]]))

    matrix.apply_damping_assume_coalesced(damping, ep=0.5)

    torch.testing.assert_close(matrix.data, torch.tensor([[[5.5, 2.0], [2.0, 12.5]]]))


@pytest.mark.parametrize("factors", [(), (float("nan"),), (-1.0,)])
def test_relative_block_damping_rejects_invalid_factors(factors: tuple[float, ...]):
    with pytest.raises(ValueError):
        RelativeBlockDamping(factors)
