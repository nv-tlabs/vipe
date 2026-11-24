# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import pytest
import torch

import vipe.slam.maths.matrix as matrix_module
import vipe.slam.maths.vector as vector_module
from vipe.slam.maths.matrix import SparseDenseBlockMatrix, SparseMDiagonalBlockMatrix
from vipe.slam.maths.vector import SparseBlockVector, SparseVectorSubview


@contextmanager
def _deterministic_algorithms(enabled: bool):
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)


def _reference_vector_coalesce(inds: torch.Tensor, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique_inds, inverse = torch.unique(inds, return_inverse=True)
    reduced = data.new_zeros((unique_inds.numel(), *data.shape[1:]))
    reduced.index_add_(0, inverse, data)
    return unique_inds, reduced


def _reference_matrix_coalesce(
    i_inds: torch.Tensor,
    j_inds: torch.Tensor,
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_pairs, inverse = torch.unique(torch.stack([i_inds, j_inds]), dim=1, return_inverse=True)
    reduced = data.new_zeros((unique_pairs.shape[1], *data.shape[1:]))
    reduced.index_add_(0, inverse, data)
    return unique_pairs[0], unique_pairs[1], reduced


def test_sparse_block_vector_coalesce_matches_previous_cpu_result_and_sorts_indices():
    inds = torch.tensor([5, 1, 5, 3, 1, 4, 5])
    data = torch.arange(7 * 9, dtype=torch.float64).reshape(7, 9) / 7.0
    expected_inds, expected_data = _reference_vector_coalesce(inds, data)

    actual = SparseBlockVector(inds=inds, data=data).coalesce()

    torch.testing.assert_close(actual.inds, expected_inds)
    torch.testing.assert_close(actual.data, expected_data)
    assert torch.all(actual.inds[1:] > actual.inds[:-1])


@pytest.mark.parametrize("matrix_cls", [SparseDenseBlockMatrix, SparseMDiagonalBlockMatrix])
def test_sparse_block_matrix_coalesce_matches_previous_cpu_result_and_sorts_pairs(matrix_cls):
    i_inds = torch.tensor([3, 1, 3, 0, 1, 3, 0, 3])
    j_inds = torch.tensor([2, 4, 2, 9, 3, 1, 9, 1])
    data = torch.arange(8 * 4 * 3, dtype=torch.float64).reshape(8, 4, 3) / 11.0
    expected_i, expected_j, expected_data = _reference_matrix_coalesce(i_inds, j_inds, data)

    actual = matrix_cls(i_inds=i_inds, j_inds=j_inds, data=data).coalesce()

    torch.testing.assert_close(actual.i_inds, expected_i)
    torch.testing.assert_close(actual.j_inds, expected_j)
    torch.testing.assert_close(actual.data, expected_data)
    assert list(zip(actual.i_inds.tolist(), actual.j_inds.tolist())) == sorted(
        zip(actual.i_inds.tolist(), actual.j_inds.tolist())
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_sparse_coalesce_supports_empty_inputs_and_preserves_block_shapes(deterministic: bool):
    with _deterministic_algorithms(deterministic):
        vector = SparseBlockVector(inds=torch.empty(0, dtype=torch.long), data=torch.empty((0, 7))).coalesce()
        dense = SparseDenseBlockMatrix(
            i_inds=torch.empty(0, dtype=torch.long),
            j_inds=torch.empty(0, dtype=torch.long),
            data=torch.empty((0, 2, 5)),
        ).coalesce()
        mdiagonal = SparseMDiagonalBlockMatrix(
            i_inds=torch.empty(0, dtype=torch.long),
            j_inds=torch.empty(0, dtype=torch.long),
            data=torch.empty((0, 6, 3)),
        ).coalesce()

    assert vector.inds.shape == (0,)
    assert vector.data.shape == (0, 7)
    assert dense.i_inds.shape == dense.j_inds.shape == (0,)
    assert dense.data.shape == (0, 2, 5)
    assert mdiagonal.i_inds.shape == mdiagonal.j_inds.shape == (0,)
    assert mdiagonal.data.shape == (0, 6, 3)


def test_sparse_vector_subview_ravel_matches_previous_cpu_result():
    vector = SparseBlockVector(
        inds=torch.tensor([4, 1, 4, 0, 1]),
        data=torch.arange(15, dtype=torch.float64).reshape(5, 3) / 5.0,
    )
    subview = SparseVectorSubview(vectors={"x": vector}, group_names=["x"])
    mapping = subview.get_ravel_mapping()
    _, expected_data = _reference_vector_coalesce(vector.inds, vector.data)

    actual = subview.ravel(mapping)

    torch.testing.assert_close(actual, expected_data.reshape(-1))


def test_default_coalesce_and_ravel_use_legacy_scatter_add(monkeypatch):
    vector_scatter_calls = 0
    matrix_scatter_calls = 0
    original_vector_scatter_add = vector_module.scatter_add
    original_matrix_scatter_add = matrix_module.scatter_add

    def count_vector_scatter(*args, **kwargs):
        nonlocal vector_scatter_calls
        vector_scatter_calls += 1
        return original_vector_scatter_add(*args, **kwargs)

    def count_matrix_scatter(*args, **kwargs):
        nonlocal matrix_scatter_calls
        matrix_scatter_calls += 1
        return original_matrix_scatter_add(*args, **kwargs)

    monkeypatch.setattr(vector_module, "scatter_add", count_vector_scatter)
    monkeypatch.setattr(matrix_module, "scatter_add", count_matrix_scatter)

    vector = SparseBlockVector(inds=torch.tensor([1, 0, 1]), data=torch.ones((3, 2)))
    dense = SparseDenseBlockMatrix(
        i_inds=torch.tensor([1, 0, 1]),
        j_inds=torch.tensor([2, 0, 2]),
        data=torch.ones((3, 2, 3)),
    )
    mdiagonal = SparseMDiagonalBlockMatrix(
        i_inds=torch.tensor([1, 0, 1]),
        j_inds=torch.tensor([2, 0, 2]),
        data=torch.ones((3, 4, 2)),
    )
    subview = SparseVectorSubview(vectors={"x": vector}, group_names=["x"])

    with _deterministic_algorithms(False):
        vector.coalesce()
        dense.coalesce()
        mdiagonal.coalesce()
        subview.ravel(subview.get_ravel_mapping())

    assert vector_scatter_calls == 2
    assert matrix_scatter_calls == 2


def test_deterministic_coalesce_and_ravel_bypass_scatter_add(monkeypatch):
    vector = SparseBlockVector(inds=torch.tensor([1, 0, 1]), data=torch.ones((3, 2)))
    dense = SparseDenseBlockMatrix(
        i_inds=torch.tensor([1, 0, 1]),
        j_inds=torch.tensor([2, 0, 2]),
        data=torch.ones((3, 2, 3)),
    )
    mdiagonal = SparseMDiagonalBlockMatrix(
        i_inds=torch.tensor([1, 0, 1]),
        j_inds=torch.tensor([2, 0, 2]),
        data=torch.ones((3, 4, 2)),
    )
    subview = SparseVectorSubview(vectors={"x": vector}, group_names=["x"])
    mapping = subview.get_ravel_mapping()

    with _deterministic_algorithms(False):
        expected_vector = vector.coalesce()
        expected_dense = dense.coalesce()
        expected_mdiagonal = mdiagonal.coalesce()
        expected_ravel = subview.ravel(mapping)

    def fail_scatter_add(*args, **kwargs):
        raise AssertionError("deterministic path must not call scatter_add")

    monkeypatch.setattr(vector_module, "scatter_add", fail_scatter_add)
    monkeypatch.setattr(matrix_module, "scatter_add", fail_scatter_add)

    with _deterministic_algorithms(True):
        actual_vector = vector.coalesce()
        actual_dense = dense.coalesce()
        actual_mdiagonal = mdiagonal.coalesce()
        actual_ravel = subview.ravel(mapping)

    torch.testing.assert_close(actual_vector.inds, expected_vector.inds)
    torch.testing.assert_close(actual_vector.data, expected_vector.data)
    torch.testing.assert_close(actual_dense.i_inds, expected_dense.i_inds)
    torch.testing.assert_close(actual_dense.j_inds, expected_dense.j_inds)
    torch.testing.assert_close(actual_dense.data, expected_dense.data)
    torch.testing.assert_close(actual_mdiagonal.i_inds, expected_mdiagonal.i_inds)
    torch.testing.assert_close(actual_mdiagonal.j_inds, expected_mdiagonal.j_inds)
    torch.testing.assert_close(actual_mdiagonal.data, expected_mdiagonal.data)
    torch.testing.assert_close(actual_ravel, expected_ravel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for deterministic coalesce coverage")
def test_cuda_coalesce_and_ravel_are_deterministic():
    generator = torch.Generator().manual_seed(7)
    vector = SparseBlockVector(
        inds=torch.randint(0, 19, (2048,), generator=generator).cuda(),
        data=torch.randn((2048, 7), generator=generator).cuda(),
    )
    i_inds = torch.randint(0, 13, (2048,), generator=generator).cuda()
    j_inds = torch.randint(0, 11, (2048,), generator=generator).cuda()
    dense = SparseDenseBlockMatrix(
        i_inds=i_inds,
        j_inds=j_inds,
        data=torch.randn((2048, 4, 6), generator=generator).cuda(),
    )
    mdiagonal = SparseMDiagonalBlockMatrix(
        i_inds=i_inds,
        j_inds=j_inds,
        data=torch.randn((2048, 5, 3), generator=generator).cuda(),
    )
    subview = SparseVectorSubview(vectors={"x": vector}, group_names=["x"])

    with _deterministic_algorithms(True):
        mapping = subview.get_ravel_mapping()
        baseline_vector = vector.coalesce()
        baseline_dense = dense.coalesce()
        baseline_mdiagonal = mdiagonal.coalesce()
        baseline_ravel = subview.ravel(mapping)

        for _ in range(4):
            actual_vector = vector.coalesce()
            actual_dense = dense.coalesce()
            actual_mdiagonal = mdiagonal.coalesce()
            actual_ravel = subview.ravel(mapping)

            assert torch.equal(actual_vector.inds, baseline_vector.inds)
            assert torch.equal(actual_vector.data, baseline_vector.data)
            assert torch.equal(actual_dense.i_inds, baseline_dense.i_inds)
            assert torch.equal(actual_dense.j_inds, baseline_dense.j_inds)
            assert torch.equal(actual_dense.data, baseline_dense.data)
            assert torch.equal(actual_mdiagonal.i_inds, baseline_mdiagonal.i_inds)
            assert torch.equal(actual_mdiagonal.j_inds, baseline_mdiagonal.j_inds)
            assert torch.equal(actual_mdiagonal.data, baseline_mdiagonal.data)
            assert torch.equal(actual_ravel, baseline_ravel)
