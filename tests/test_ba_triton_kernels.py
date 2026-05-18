import os
import unittest
from collections import defaultdict

try:
    import torch
except ImportError:  # pragma: no cover - lets discovery work without the runtime env
    torch = None


if torch is not None:
    from vipe.slam.maths import triton_kernels
    from vipe.slam.maths.matrix import (
        SparseDenseBlockMatrix,
        SparseMatrixSubview,
        SparseMDiagonalBlockMatrix,
        diagonal_schur_complement,
    )
    from vipe.slam.maths.vector import SparseBlockVector, SparseVectorSubview
else:
    triton_kernels = None
    SparseDenseBlockMatrix = None
    SparseMDiagonalBlockMatrix = None
    SparseMatrixSubview = None
    SparseBlockVector = None
    SparseVectorSubview = None
    diagonal_schur_complement = None


def _has_cuda_triton() -> bool:
    return (
        torch is not None
        and torch.cuda.is_available()
        and triton_kernels is not None
        and triton_kernels.triton_available()
    )


def _matching_indices(lhs_i, rhs_i):
    rhs_by_term = defaultdict(list)
    for idx, term in enumerate(rhs_i.cpu().tolist()):
        rhs_by_term[term].append(idx)

    lhs_matches, rhs_matches = [], []
    for idx, term in enumerate(lhs_i.cpu().tolist()):
        for rhs_idx in rhs_by_term.get(term, []):
            lhs_matches.append(idx)
            rhs_matches.append(rhs_idx)

    return (
        torch.tensor(lhs_matches, device=lhs_i.device, dtype=torch.long),
        torch.tensor(rhs_matches, device=rhs_i.device, dtype=torch.long),
    )


@unittest.skipUnless(_has_cuda_triton(), "CUDA Triton runtime is required")
class BATritonKernelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.device = torch.device("cuda")
        self._old_min_elements = os.environ.get("VIPE_BA_TRITON_MIN_MDIAG_DENSE_ELEMENTS")
        self._old_enable_weighted_outer = os.environ.get("VIPE_ENABLE_BA_WEIGHTED_OUTER_TRITON")
        self._old_enable_fused_schur = os.environ.get("VIPE_ENABLE_BA_FUSED_SCHUR")
        os.environ["VIPE_BA_TRITON_MIN_MDIAG_DENSE_ELEMENTS"] = "0"
        os.environ["VIPE_ENABLE_BA_WEIGHTED_OUTER_TRITON"] = "1"
        os.environ["VIPE_ENABLE_BA_FUSED_SCHUR"] = "1"

    def tearDown(self):
        if self._old_min_elements is None:
            os.environ.pop("VIPE_BA_TRITON_MIN_MDIAG_DENSE_ELEMENTS", None)
        else:
            os.environ["VIPE_BA_TRITON_MIN_MDIAG_DENSE_ELEMENTS"] = self._old_min_elements
        if self._old_enable_weighted_outer is None:
            os.environ.pop("VIPE_ENABLE_BA_WEIGHTED_OUTER_TRITON", None)
        else:
            os.environ["VIPE_ENABLE_BA_WEIGHTED_OUTER_TRITON"] = self._old_enable_weighted_outer
        if self._old_enable_fused_schur is None:
            os.environ.pop("VIPE_ENABLE_BA_FUSED_SCHUR", None)
        else:
            os.environ["VIPE_ENABLE_BA_FUSED_SCHUR"] = self._old_enable_fused_schur

    def assertClose(self, actual, expected, *, atol=3e-4, rtol=3e-4):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_dense_tmult_vec_kernel(self):
        data = torch.randn(11, 257, 6, device=self.device)
        vec = torch.randn(11, 257, device=self.device)

        actual = triton_kernels.dense_tmult_vec(data, vec)
        expected = torch.einsum("bij,bi->bj", data, vec)

        self.assertIsNotNone(actual)
        self.assertClose(actual, expected)

    def test_dense_dense_tmult_mat_kernel(self):
        lhs = torch.randn(9, 319, 6, device=self.device)
        rhs = torch.randn(9, 319, 6, device=self.device)

        actual = triton_kernels.dense_dense_tmult_mat(lhs, rhs)
        expected = torch.einsum("bij,bik->bjk", lhs, rhs)

        self.assertIsNotNone(actual)
        self.assertClose(actual, expected)

    def test_mdiag_dense_tmult_mat_kernel(self):
        diag = torch.randn(7, 131, 2, device=self.device)
        dense = torch.randn(7, 262, 6, device=self.device)

        actual = triton_kernels.mdiag_dense_tmult_mat(diag, dense)
        expected = (diag.unsqueeze(-1) * dense.reshape(7, 131, 2, 6)).sum(dim=-2)

        self.assertIsNotNone(actual)
        self.assertClose(actual, expected)

    def test_mdiag_tmult_vec_kernel(self):
        diag = torch.randn(7, 173, 2, device=self.device)
        vec = torch.randn(7, 346, device=self.device)

        actual = triton_kernels.mdiag_tmult_vec(diag, vec)
        expected = (diag.reshape(7, -1) * vec).reshape_as(diag).sum(dim=-1)

        self.assertIsNotNone(actual)
        self.assertClose(actual, expected)

    def test_weighted_dense_dense_tmult_mat_kernel(self):
        left = torch.randn(13, 6, 257, device=self.device)
        right = torch.randn(13, 4, 257, device=self.device)
        diag = torch.rand(13, 257, 1, device=self.device) + 0.01

        actual = triton_kernels.weighted_dense_dense_tmult_mat(left, right, diag)
        expected = torch.einsum("bik,bk,bjk->bij", left, diag.squeeze(-1), right)

        self.assertIsNotNone(actual)
        self.assertClose(actual, expected, atol=2e-3, rtol=2e-3)

    def test_weighted_dense_tmult_vec_kernel(self):
        left = torch.randn(13, 6, 257, device=self.device)
        diag = torch.rand(13, 257, 1, device=self.device) + 0.01
        vec = torch.randn(13, 257, device=self.device)

        actual = triton_kernels.weighted_dense_tmult_vec(left, diag, vec)
        expected = torch.einsum("bik,bk,bk->bi", left, diag.squeeze(-1), vec)

        self.assertIsNotNone(actual)
        self.assertClose(actual, expected, atol=2e-3, rtol=2e-3)

    def test_row_weighted_dense_dense_tmult_mat_kernel(self):
        left = torch.randn(13, 257, 6, device=self.device)
        right = torch.randn(13, 257, 4, device=self.device)
        weight = torch.rand(13, 257, device=self.device) + 0.01

        actual = triton_kernels.row_weighted_dense_dense_tmult_mat(left, right, weight)
        expected = torch.einsum("bik,bi,bin->bkn", left, weight, right)

        self.assertIsNotNone(actual)
        self.assertClose(actual, expected, atol=2e-3, rtol=2e-3)

    def test_sparse_dense_tmult_vec_matches_reference(self):
        i_inds = torch.tensor([2, 0, 2, 1, 4, 4], device=self.device)
        j_inds = torch.tensor([3, 1, 5, 2, 8, 9], device=self.device)
        data = torch.randn(i_inds.shape[0], 211, 6, device=self.device)
        matrix = SparseDenseBlockMatrix(i_inds=i_inds, j_inds=j_inds, data=data)
        vec = torch.randn(5, 211, device=self.device)

        actual = matrix.tmult_vec(vec)
        expected = torch.einsum("bij,bi->bj", data, vec[i_inds])

        torch.testing.assert_close(actual.inds, j_inds)
        self.assertClose(actual.data, expected)

    def test_sparse_dense_tmult_mat_matches_reference(self):
        lhs_i = torch.tensor([0, 1, 1, 3, 4, 4], device=self.device)
        lhs_j = torch.tensor([7, 8, 9, 10, 11, 12], device=self.device)
        rhs_i = torch.tensor([1, 0, 4, 4, 3], device=self.device)
        rhs_j = torch.tensor([13, 14, 15, 16, 17], device=self.device)
        lhs = SparseDenseBlockMatrix(
            i_inds=lhs_i,
            j_inds=lhs_j,
            data=torch.randn(lhs_i.shape[0], 193, 6, device=self.device),
        )
        rhs = SparseDenseBlockMatrix(
            i_inds=rhs_i,
            j_inds=rhs_j,
            data=torch.randn(rhs_i.shape[0], 193, 6, device=self.device),
        )
        lhs_matches, rhs_matches = _matching_indices(lhs_i, rhs_i)

        actual = lhs.tmult_mat(rhs)
        expected = torch.einsum("bij,bik->bjk", lhs.data[lhs_matches], rhs.data[rhs_matches])

        torch.testing.assert_close(actual.i_inds, lhs_j[lhs_matches])
        torch.testing.assert_close(actual.j_inds, rhs_j[rhs_matches])
        self.assertClose(actual.data, expected)

    def test_large_sparse_dense_tmult_mat_matches_reference(self):
        n_blocks = 6001
        lhs_i = torch.arange(n_blocks, device=self.device) % 997
        rhs_i = torch.arange(n_blocks - 1, -1, -1, device=self.device) % 997
        lhs_j = torch.arange(n_blocks, device=self.device) + 10_000
        rhs_j = torch.arange(n_blocks, device=self.device) + 20_000
        lhs = SparseDenseBlockMatrix(
            i_inds=lhs_i,
            j_inds=lhs_j,
            data=torch.randn(n_blocks, 5, 3, device=self.device),
        )
        rhs = SparseDenseBlockMatrix(
            i_inds=rhs_i,
            j_inds=rhs_j,
            data=torch.randn(n_blocks, 5, 4, device=self.device),
        )
        lhs_matches, rhs_matches = _matching_indices(lhs_i, rhs_i)

        actual = lhs.tmult_mat(rhs)
        expected = torch.einsum("bij,bik->bjk", lhs.data[lhs_matches], rhs.data[rhs_matches])

        torch.testing.assert_close(actual.i_inds, lhs_j[lhs_matches])
        torch.testing.assert_close(actual.j_inds, rhs_j[rhs_matches])
        self.assertClose(actual.data, expected)

    def test_sparse_mdiag_tmult_vec_matches_reference(self):
        i_inds = torch.tensor([2, 0, 2, 1, 4, 4], device=self.device)
        j_inds = torch.tensor([3, 1, 5, 2, 8, 9], device=self.device)
        data = torch.randn(i_inds.shape[0], 127, 2, device=self.device)
        matrix = SparseMDiagonalBlockMatrix(i_inds=i_inds, j_inds=j_inds, data=data)
        vec = torch.randn(5, 254, device=self.device)

        actual = matrix.tmult_vec(vec)
        expected = (data.reshape(data.shape[0], -1) * vec[i_inds]).reshape_as(data).sum(dim=-1)

        torch.testing.assert_close(actual.inds, j_inds)
        self.assertClose(actual.data, expected)

    def test_sparse_mdiag_dense_tmult_mat_uses_triton_correctly(self):
        diag_i = torch.tensor([0, 2, 2, 3, 5], device=self.device)
        diag_j = torch.tensor([4, 6, 7, 8, 9], device=self.device)
        dense_i = torch.tensor([2, 0, 5, 3], device=self.device)
        dense_j = torch.tensor([10, 11, 12, 13], device=self.device)
        diag = SparseMDiagonalBlockMatrix(
            i_inds=diag_i,
            j_inds=diag_j,
            data=torch.randn(diag_i.shape[0], 113, 2, device=self.device),
        )
        dense = SparseDenseBlockMatrix(
            i_inds=dense_i,
            j_inds=dense_j,
            data=torch.randn(dense_i.shape[0], 226, 6, device=self.device),
        )
        diag_matches, dense_matches = _matching_indices(diag_i, dense_i)

        actual = diag.tmult_mat(dense)
        expected = (diag.data[diag_matches].unsqueeze(-1) * dense.data[dense_matches].reshape(-1, 113, 2, 6)).sum(
            dim=-2
        )

        torch.testing.assert_close(actual.i_inds, diag_j[diag_matches])
        torch.testing.assert_close(actual.j_inds, dense_j[dense_matches])
        self.assertClose(actual.data, expected)

    def test_weighted_sparse_dense_dense_product_matches_scaled_reference(self):
        lhs_i = torch.tensor([0, 1, 1, 3, 4, 4], device=self.device)
        lhs_j = torch.tensor([7, 8, 9, 10, 11, 12], device=self.device)
        rhs_i = torch.tensor([1, 0, 4, 4, 3], device=self.device)
        rhs_j = torch.tensor([13, 14, 15, 16, 17], device=self.device)
        weight = torch.rand(5, 193, device=self.device)
        lhs = SparseDenseBlockMatrix(
            i_inds=lhs_i,
            j_inds=lhs_j,
            data=torch.randn(lhs_i.shape[0], 193, 6, device=self.device),
        )
        rhs = SparseDenseBlockMatrix(
            i_inds=rhs_i,
            j_inds=rhs_j,
            data=torch.randn(rhs_i.shape[0], 193, 4, device=self.device),
        )
        lhs_matches, rhs_matches = _matching_indices(lhs_i, rhs_i)

        actual = lhs.weighted_tmult_mat_by_rows(rhs, weight)
        expected = torch.einsum(
            "bij,bi,bik->bjk",
            lhs.data[lhs_matches],
            weight[lhs_i[lhs_matches]],
            rhs.data[rhs_matches],
        )

        torch.testing.assert_close(actual.i_inds, lhs_j[lhs_matches])
        torch.testing.assert_close(actual.j_inds, rhs_j[rhs_matches])
        self.assertClose(actual.data, expected)

    def test_weighted_sparse_dense_mdiag_product_matches_scaled_reference(self):
        dense_i = torch.tensor([0, 1, 1, 3, 4, 4], device=self.device)
        dense_j = torch.tensor([7, 8, 9, 10, 11, 12], device=self.device)
        diag_i = torch.tensor([1, 0, 4, 4, 3], device=self.device)
        diag_j = torch.tensor([13, 14, 15, 16, 17], device=self.device)
        weight = torch.rand(5, 194, device=self.device)
        dense = SparseDenseBlockMatrix(
            i_inds=dense_i,
            j_inds=dense_j,
            data=torch.randn(dense_i.shape[0], 194, 6, device=self.device),
        )
        diag = SparseMDiagonalBlockMatrix(
            i_inds=diag_i,
            j_inds=diag_j,
            data=torch.randn(diag_i.shape[0], 97, 2, device=self.device),
        )
        dense_matches, diag_matches = _matching_indices(dense_i, diag_i)
        dense_data = dense.data[dense_matches].reshape(-1, 97, 2, 6)
        weight_data = weight[dense_i[dense_matches]].reshape(-1, 97, 2)

        actual = dense.weighted_tmult_mat_by_rows(diag, weight)
        expected = (dense_data * weight_data.unsqueeze(-1) * diag.data[diag_matches].unsqueeze(-1)).sum(dim=2)

        torch.testing.assert_close(actual.i_inds, dense_j[dense_matches])
        torch.testing.assert_close(actual.j_inds, diag_j[diag_matches])
        self.assertClose(actual.data, expected.transpose(1, 2))

    def test_weighted_sparse_mdiag_mdiag_product_matches_scaled_reference(self):
        lhs_i = torch.tensor([0, 1, 1, 3, 4, 4], device=self.device)
        lhs_j = torch.tensor([7, 8, 9, 10, 11, 12], device=self.device)
        rhs_i = torch.tensor([1, 0, 4, 4, 3], device=self.device)
        rhs_j = torch.tensor([13, 14, 15, 16, 17], device=self.device)
        weight = torch.rand(5, 194, device=self.device)
        lhs = SparseMDiagonalBlockMatrix(
            i_inds=lhs_i,
            j_inds=lhs_j,
            data=torch.randn(lhs_i.shape[0], 97, 2, device=self.device),
        )
        rhs = SparseMDiagonalBlockMatrix(
            i_inds=rhs_i,
            j_inds=rhs_j,
            data=torch.randn(rhs_i.shape[0], 97, 2, device=self.device),
        )
        lhs_matches, rhs_matches = _matching_indices(lhs_i, rhs_i)

        actual = lhs.weighted_tmult_mat_by_rows(rhs, weight)
        expected = (lhs.data[lhs_matches] * weight[lhs_i[lhs_matches]].reshape(-1, 97, 2) * rhs.data[rhs_matches]).sum(
            dim=-1
        )

        torch.testing.assert_close(actual.i_inds, lhs_j[lhs_matches])
        torch.testing.assert_close(actual.j_inds, rhs_j[rhs_matches])
        self.assertClose(actual.data, expected.unsqueeze(-1))

    def test_public_sparse_vector_path_still_matches_reference(self):
        i_inds = torch.tensor([0, 1, 1, 3], device=self.device)
        j_inds = torch.tensor([4, 5, 6, 7], device=self.device)
        matrix = SparseDenseBlockMatrix(
            i_inds=i_inds,
            j_inds=j_inds,
            data=torch.randn(i_inds.shape[0], 97, 6, device=self.device),
        )
        vec = SparseBlockVector(
            inds=torch.tensor([1, 3], device=self.device),
            data=torch.randn(2, 97, device=self.device),
        )
        matrix_matches, vector_matches = _matching_indices(i_inds, vec.inds)

        actual = matrix.tmult_vec(vec)
        expected = torch.einsum("bij,bi->bj", matrix.data[matrix_matches], vec.data[vector_matches])

        torch.testing.assert_close(actual.inds, j_inds[matrix_matches])
        self.assertClose(actual.data, expected)

    def test_fused_diagonal_schur_matches_reference_path(self):
        pose_inds = torch.tensor([0, 1, 1, 2, 2], device=self.device)
        disp_inds = torch.tensor([4, 4, 5, 5, 6], device=self.device)
        e_data = torch.randn(pose_inds.shape[0], 6, 129, device=self.device)
        c_inv = torch.rand(3, 129, 1, device=self.device) + 0.1
        rhs_w_data = torch.randn(3, 129, device=self.device)

        lhs_h = SparseMatrixSubview(
            matrices={
                ("pose", "pose"): SparseDenseBlockMatrix(
                    i_inds=torch.tensor([0, 1, 2], device=self.device),
                    j_inds=torch.tensor([0, 1, 2], device=self.device),
                    data=torch.eye(6, device=self.device).repeat(3, 1, 1),
                )
            },
            row_group_names=["pose"],
            col_group_names=["pose"],
        )
        lhs_e = SparseMatrixSubview(
            matrices={
                ("pose", "dense_disp"): SparseDenseBlockMatrix(
                    i_inds=pose_inds,
                    j_inds=disp_inds,
                    data=e_data,
                )
            },
            row_group_names=["pose"],
            col_group_names=["dense_disp"],
        )
        lhs_c_inv = SparseMatrixSubview(
            matrices={
                ("dense_disp", "dense_disp"): SparseMDiagonalBlockMatrix(
                    i_inds=torch.tensor([4, 5, 6], device=self.device),
                    j_inds=torch.tensor([4, 5, 6], device=self.device),
                    data=c_inv,
                )
            },
            row_group_names=["dense_disp"],
            col_group_names=["dense_disp"],
        )
        rhs_v = SparseVectorSubview(
            vectors={
                "pose": SparseBlockVector(
                    inds=torch.tensor([0, 1, 2], device=self.device),
                    data=torch.randn(3, 6, device=self.device),
                )
            },
            group_names=["pose"],
        )
        rhs_w = SparseVectorSubview(
            vectors={
                "dense_disp": SparseBlockVector(
                    inds=torch.tensor([4, 5, 6], device=self.device),
                    data=rhs_w_data,
                )
            },
            group_names=["dense_disp"],
        )

        actual_lhs, actual_rhs = diagonal_schur_complement(lhs_h, lhs_e, lhs_c_inv, rhs_v, rhs_w)
        h_cinv = lhs_e @ lhs_c_inv
        expected_lhs = lhs_h - h_cinv @ lhs_e.transpose()
        expected_rhs = rhs_v - h_cinv * rhs_w

        actual_lhs_block = actual_lhs.get("pose", "pose").coalesce()
        expected_lhs_block = expected_lhs.get("pose", "pose").coalesce()
        actual_rhs_block = actual_rhs.vectors["pose"].coalesce()
        expected_rhs_block = expected_rhs.vectors["pose"].coalesce()

        torch.testing.assert_close(actual_lhs_block.i_inds, expected_lhs_block.i_inds)
        torch.testing.assert_close(actual_lhs_block.j_inds, expected_lhs_block.j_inds)
        self.assertClose(actual_lhs_block.data, expected_lhs_block.data, atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(actual_rhs_block.inds, expected_rhs_block.inds)
        self.assertClose(actual_rhs_block.data, expected_rhs_block.data, atol=3e-3, rtol=3e-3)


if __name__ == "__main__":
    unittest.main()
