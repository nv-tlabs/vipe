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

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-Triton installs
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and tl is not None and os.environ.get("VIPE_DISABLE_BA_TRITON", "0") != "1"


def _supported_cuda_float32(*tensors: torch.Tensor) -> bool:
    return triton_available() and all(t.is_cuda and t.dtype == torch.float32 for t in tensors)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _min_mdiag_dense_elements() -> int:
    return int(os.environ.get("VIPE_BA_TRITON_MIN_MDIAG_DENSE_ELEMENTS", "1000000"))


def _weighted_outer_triton_enabled() -> bool:
    return os.environ.get("VIPE_ENABLE_BA_WEIGHTED_OUTER_TRITON", "0") == "1"


if triton is not None and tl is not None:

    @triton.jit
    def _dense_tmult_vec_kernel(
        data,
        vec,
        out,
        K: tl.constexpr,
        C: tl.constexpr,
        DATA_SB: tl.constexpr,
        DATA_SK: tl.constexpr,
        DATA_SC: tl.constexpr,
        VEC_SB: tl.constexpr,
        VEC_SK: tl.constexpr,
        OUT_SB: tl.constexpr,
        OUT_SC: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.program_id(0)
        k_offsets = tl.arange(0, BLOCK_K)
        c_offsets = tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + k_offsets
            k_mask = k < K
            data_vals = tl.load(
                data + b * DATA_SB + k[:, None] * DATA_SK + c_offsets[None, :] * DATA_SC,
                mask=k_mask[:, None] & c_mask[None, :],
                other=0.0,
            )
            vec_vals = tl.load(
                vec + b * VEC_SB + k * VEC_SK,
                mask=k_mask,
                other=0.0,
            )
            acc += tl.sum(data_vals * vec_vals[:, None], axis=0)

        tl.store(out + b * OUT_SB + c_offsets * OUT_SC, acc, mask=c_mask)

    @triton.jit
    def _dense_dense_tmult_mat_kernel(
        a,
        b,
        out,
        K: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        A_SB: tl.constexpr,
        A_SK: tl.constexpr,
        A_SM: tl.constexpr,
        B_SB: tl.constexpr,
        B_SK: tl.constexpr,
        B_SN: tl.constexpr,
        OUT_SB: tl.constexpr,
        OUT_SM: tl.constexpr,
        OUT_SN: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        block = tl.program_id(0)
        m = tl.program_id(1)
        n = tl.program_id(2)
        k_offsets = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + k_offsets
            k_mask = k < K
            a_vals = tl.load(
                a + block * A_SB + k * A_SK + m * A_SM,
                mask=k_mask,
                other=0.0,
            )
            b_vals = tl.load(
                b + block * B_SB + k * B_SK + n * B_SN,
                mask=k_mask,
                other=0.0,
            )
            acc += a_vals * b_vals

        tl.store(out + block * OUT_SB + m * OUT_SM + n * OUT_SN, tl.sum(acc, axis=0))

    @triton.jit
    def _mdiag_dense_tmult_mat_kernel(
        diag,
        dense,
        out,
        R: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        DIAG_SB: tl.constexpr,
        DIAG_SR: tl.constexpr,
        DIAG_SD: tl.constexpr,
        DENSE_SB: tl.constexpr,
        DENSE_SK: tl.constexpr,
        DENSE_SC: tl.constexpr,
        OUT_SB: tl.constexpr,
        OUT_SR: tl.constexpr,
        OUT_SC: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        block = tl.program_id(0)
        r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        c = tl.arange(0, BLOCK_C)
        r_mask = r < R
        c_mask = c < C
        acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

        for d in range(0, D):
            diag_vals = tl.load(
                diag + block * DIAG_SB + r * DIAG_SR + d * DIAG_SD,
                mask=r_mask,
                other=0.0,
            )
            dense_vals = tl.load(
                dense + block * DENSE_SB + (r[:, None] * D + d) * DENSE_SK + c[None, :] * DENSE_SC,
                mask=r_mask[:, None] & c_mask[None, :],
                other=0.0,
            )
            acc += diag_vals[:, None] * dense_vals

        tl.store(
            out + block * OUT_SB + r[:, None] * OUT_SR + c[None, :] * OUT_SC,
            acc,
            mask=r_mask[:, None] & c_mask[None, :],
        )

    @triton.jit
    def _mdiag_tmult_vec_kernel(
        diag,
        vec,
        out,
        R: tl.constexpr,
        D: tl.constexpr,
        DIAG_SB: tl.constexpr,
        DIAG_SR: tl.constexpr,
        DIAG_SD: tl.constexpr,
        VEC_SB: tl.constexpr,
        VEC_SK: tl.constexpr,
        OUT_SB: tl.constexpr,
        OUT_SR: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        block = tl.program_id(0)
        r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r < R
        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

        for d in range(0, D):
            diag_vals = tl.load(
                diag + block * DIAG_SB + r * DIAG_SR + d * DIAG_SD,
                mask=r_mask,
                other=0.0,
            )
            vec_vals = tl.load(
                vec + block * VEC_SB + (r * D + d) * VEC_SK,
                mask=r_mask,
                other=0.0,
            )
            acc += diag_vals * vec_vals

        tl.store(out + block * OUT_SB + r * OUT_SR, acc, mask=r_mask)

    @triton.jit
    def _weighted_dense_dense_tmult_mat_kernel(
        left,
        right,
        diag,
        out,
        K: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        LEFT_SB: tl.constexpr,
        LEFT_SM: tl.constexpr,
        LEFT_SK: tl.constexpr,
        RIGHT_SB: tl.constexpr,
        RIGHT_SN: tl.constexpr,
        RIGHT_SK: tl.constexpr,
        DIAG_SB: tl.constexpr,
        DIAG_SK: tl.constexpr,
        OUT_SB: tl.constexpr,
        OUT_SM: tl.constexpr,
        OUT_SN: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        block = tl.program_id(0)
        m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m < M
        n_mask = n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            for kk in range(0, BLOCK_K):
                cur_k = k_start + kk
                k_mask = cur_k < K
                left_vals = tl.load(
                    left + block * LEFT_SB + m * LEFT_SM + cur_k * LEFT_SK,
                    mask=m_mask & k_mask,
                    other=0.0,
                )
                right_vals = tl.load(
                    right + block * RIGHT_SB + n * RIGHT_SN + cur_k * RIGHT_SK,
                    mask=n_mask & k_mask,
                    other=0.0,
                )
                diag_val = tl.load(
                    diag + block * DIAG_SB + cur_k * DIAG_SK,
                    mask=k_mask,
                    other=0.0,
                )
                acc += left_vals[:, None] * diag_val * right_vals[None, :]

        tl.store(
            out + block * OUT_SB + m[:, None] * OUT_SM + n[None, :] * OUT_SN,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _weighted_dense_tmult_vec_kernel(
        left,
        diag,
        vec,
        out,
        K: tl.constexpr,
        M: tl.constexpr,
        LEFT_SB: tl.constexpr,
        LEFT_SM: tl.constexpr,
        LEFT_SK: tl.constexpr,
        DIAG_SB: tl.constexpr,
        DIAG_SK: tl.constexpr,
        VEC_SB: tl.constexpr,
        VEC_SK: tl.constexpr,
        OUT_SB: tl.constexpr,
        OUT_SM: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        block = tl.program_id(0)
        m = tl.arange(0, BLOCK_M)
        k_offsets = tl.arange(0, BLOCK_K)
        m_mask = m < M
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + k_offsets
            k_mask = k < K
            left_vals = tl.load(
                left + block * LEFT_SB + m[:, None] * LEFT_SM + k[None, :] * LEFT_SK,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            diag_vals = tl.load(
                diag + block * DIAG_SB + k * DIAG_SK,
                mask=k_mask,
                other=0.0,
            )
            vec_vals = tl.load(
                vec + block * VEC_SB + k * VEC_SK,
                mask=k_mask,
                other=0.0,
            )
            acc += tl.sum(left_vals * diag_vals[None, :] * vec_vals[None, :], axis=1)

        tl.store(out + block * OUT_SB + m * OUT_SM, acc, mask=m_mask)

    @triton.jit
    def _row_weighted_dense_dense_tmult_mat_kernel(
        left,
        right,
        weight,
        out,
        K: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        LEFT_SB: tl.constexpr,
        LEFT_SK: tl.constexpr,
        LEFT_SM: tl.constexpr,
        RIGHT_SB: tl.constexpr,
        RIGHT_SK: tl.constexpr,
        RIGHT_SN: tl.constexpr,
        WEIGHT_SB: tl.constexpr,
        WEIGHT_SK: tl.constexpr,
        OUT_SB: tl.constexpr,
        OUT_SM: tl.constexpr,
        OUT_SN: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        block = tl.program_id(0)
        m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m < M
        n_mask = n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            for kk in range(0, BLOCK_K):
                cur_k = k_start + kk
                k_mask = cur_k < K
                left_vals = tl.load(
                    left + block * LEFT_SB + cur_k * LEFT_SK + m * LEFT_SM,
                    mask=m_mask & k_mask,
                    other=0.0,
                )
                right_vals = tl.load(
                    right + block * RIGHT_SB + cur_k * RIGHT_SK + n * RIGHT_SN,
                    mask=n_mask & k_mask,
                    other=0.0,
                )
                weight_val = tl.load(
                    weight + block * WEIGHT_SB + cur_k * WEIGHT_SK,
                    mask=k_mask,
                    other=0.0,
                )
                acc += left_vals[:, None] * weight_val * right_vals[None, :]

        tl.store(
            out + block * OUT_SB + m[:, None] * OUT_SM + n[None, :] * OUT_SN,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


def dense_tmult_vec(data: torch.Tensor, vec: torch.Tensor) -> torch.Tensor | None:
    if not _supported_cuda_float32(data, vec):
        return None
    if data.ndim != 3 or vec.ndim != 2 or data.shape[:2] != vec.shape:
        return None

    data = data.contiguous()
    vec = vec.contiguous()
    n_blocks, k, c = data.shape
    if c > 16:
        return None

    out = torch.empty((n_blocks, c), device=data.device, dtype=data.dtype)
    block_k = min(128, _next_power_of_2(k))
    block_c = _next_power_of_2(c)
    _dense_tmult_vec_kernel[(n_blocks,)](
        data,
        vec,
        out,
        K=k,
        C=c,
        DATA_SB=data.stride(0),
        DATA_SK=data.stride(1),
        DATA_SC=data.stride(2),
        VEC_SB=vec.stride(0),
        VEC_SK=vec.stride(1),
        OUT_SB=out.stride(0),
        OUT_SC=out.stride(1),
        BLOCK_K=block_k,
        BLOCK_C=block_c,
    )
    return out


def dense_dense_tmult_mat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | None:
    if not _supported_cuda_float32(a, b):
        return None
    if a.ndim != 3 or b.ndim != 3 or a.shape[:2] != b.shape[:2]:
        return None

    a = a.contiguous()
    b = b.contiguous()
    n_blocks, k, m = a.shape
    n = b.shape[2]
    if m > 16 or n > 16:
        return None

    out = torch.empty((n_blocks, m, n), device=a.device, dtype=a.dtype)
    block_k = max(16, min(64, _next_power_of_2(k)))
    _dense_dense_tmult_mat_kernel[(n_blocks, m, n)](
        a,
        b,
        out,
        K=k,
        M=m,
        N=n,
        A_SB=a.stride(0),
        A_SK=a.stride(1),
        A_SM=a.stride(2),
        B_SB=b.stride(0),
        B_SK=b.stride(1),
        B_SN=b.stride(2),
        OUT_SB=out.stride(0),
        OUT_SM=out.stride(1),
        OUT_SN=out.stride(2),
        BLOCK_K=block_k,
        BLOCK_M=1,
        BLOCK_N=1,
    )
    return out


def mdiag_dense_tmult_mat(diag: torch.Tensor, dense: torch.Tensor) -> torch.Tensor | None:
    if diag.ndim != 3 or dense.ndim != 3:
        return None

    n_blocks, r, d = diag.shape
    if dense.shape[0] != n_blocks or dense.shape[1] != r * d:
        return None

    c = dense.shape[2]
    if d > 4 or c > 16 or n_blocks * r < _min_mdiag_dense_elements():
        return None
    if not _supported_cuda_float32(diag, dense):
        return None

    diag = diag.contiguous()
    dense = dense.contiguous()

    out = torch.empty((n_blocks, r, c), device=diag.device, dtype=diag.dtype)
    block_r = 32
    _mdiag_dense_tmult_mat_kernel[(n_blocks, triton.cdiv(r, block_r))](
        diag,
        dense,
        out,
        R=r,
        D=d,
        C=c,
        DIAG_SB=diag.stride(0),
        DIAG_SR=diag.stride(1),
        DIAG_SD=diag.stride(2),
        DENSE_SB=dense.stride(0),
        DENSE_SK=dense.stride(1),
        DENSE_SC=dense.stride(2),
        OUT_SB=out.stride(0),
        OUT_SR=out.stride(1),
        OUT_SC=out.stride(2),
        BLOCK_R=block_r,
        BLOCK_C=_next_power_of_2(c),
    )
    return out


def mdiag_tmult_vec(diag: torch.Tensor, vec: torch.Tensor) -> torch.Tensor | None:
    if not _supported_cuda_float32(diag, vec):
        return None
    if diag.ndim != 3 or vec.ndim != 2:
        return None

    diag = diag.contiguous()
    vec = vec.contiguous()
    n_blocks, r, d = diag.shape
    if vec.shape != (n_blocks, r * d) or d > 4:
        return None

    out = torch.empty((n_blocks, r), device=diag.device, dtype=diag.dtype)
    block_r = 64
    _mdiag_tmult_vec_kernel[(n_blocks, triton.cdiv(r, block_r))](
        diag,
        vec,
        out,
        R=r,
        D=d,
        DIAG_SB=diag.stride(0),
        DIAG_SR=diag.stride(1),
        DIAG_SD=diag.stride(2),
        VEC_SB=vec.stride(0),
        VEC_SK=vec.stride(1),
        OUT_SB=out.stride(0),
        OUT_SR=out.stride(1),
        BLOCK_R=block_r,
    )
    return out


def weighted_dense_dense_tmult_mat(
    left: torch.Tensor,
    right: torch.Tensor,
    diag: torch.Tensor,
) -> torch.Tensor | None:
    if not _weighted_outer_triton_enabled():
        return None
    if left.ndim != 3 or right.ndim != 3 or diag.ndim != 3:
        return None

    n_blocks, m, k = left.shape
    if right.shape[0] != n_blocks or right.shape[2] != k:
        return None
    if diag.shape != (n_blocks, k, 1):
        return None

    n = right.shape[1]
    if m > 16 or n > 16 or k < 64:
        return None
    if not _supported_cuda_float32(left, right, diag):
        return None

    left = left.contiguous()
    right = right.contiguous()
    diag = diag.contiguous()

    out = torch.empty((n_blocks, m, n), device=left.device, dtype=left.dtype)
    block_m = min(8, _next_power_of_2(m))
    block_n = min(8, _next_power_of_2(n))
    block_k = 32
    _weighted_dense_dense_tmult_mat_kernel[(n_blocks, triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        left,
        right,
        diag,
        out,
        K=k,
        M=m,
        N=n,
        LEFT_SB=left.stride(0),
        LEFT_SM=left.stride(1),
        LEFT_SK=left.stride(2),
        RIGHT_SB=right.stride(0),
        RIGHT_SN=right.stride(1),
        RIGHT_SK=right.stride(2),
        DIAG_SB=diag.stride(0),
        DIAG_SK=diag.stride(1),
        OUT_SB=out.stride(0),
        OUT_SM=out.stride(1),
        OUT_SN=out.stride(2),
        BLOCK_K=block_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


def weighted_dense_tmult_vec(
    left: torch.Tensor,
    diag: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor | None:
    if left.ndim != 3 or diag.ndim != 3 or vec.ndim != 2:
        return None

    n_blocks, m, k = left.shape
    if diag.shape != (n_blocks, k, 1) or vec.shape != (n_blocks, k):
        return None
    if m > 16 or k < 64:
        return None
    if not _supported_cuda_float32(left, diag, vec):
        return None

    left = left.contiguous()
    diag = diag.contiguous()
    vec = vec.contiguous()

    out = torch.empty((n_blocks, m), device=left.device, dtype=left.dtype)
    block_k = 64
    _weighted_dense_tmult_vec_kernel[(n_blocks,)](
        left,
        diag,
        vec,
        out,
        K=k,
        M=m,
        LEFT_SB=left.stride(0),
        LEFT_SM=left.stride(1),
        LEFT_SK=left.stride(2),
        DIAG_SB=diag.stride(0),
        DIAG_SK=diag.stride(1),
        VEC_SB=vec.stride(0),
        VEC_SK=vec.stride(1),
        OUT_SB=out.stride(0),
        OUT_SM=out.stride(1),
        BLOCK_K=block_k,
        BLOCK_M=_next_power_of_2(m),
    )
    return out


def row_weighted_dense_dense_tmult_mat(
    left: torch.Tensor,
    right: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    if left.ndim != 3 or right.ndim != 3 or weight.ndim != 2:
        return None

    n_blocks, k, m = left.shape
    if right.shape[0] != n_blocks or right.shape[1] != k or weight.shape != (n_blocks, k):
        return None

    n = right.shape[2]
    if m > 16 or n > 16 or k < 64:
        return None
    if not _supported_cuda_float32(left, right, weight):
        return None

    left = left.contiguous()
    right = right.contiguous()
    weight = weight.contiguous()

    out = torch.empty((n_blocks, m, n), device=left.device, dtype=left.dtype)
    block_m = min(8, _next_power_of_2(m))
    block_n = min(8, _next_power_of_2(n))
    block_k = 32
    _row_weighted_dense_dense_tmult_mat_kernel[(n_blocks, triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        left,
        right,
        weight,
        out,
        K=k,
        M=m,
        N=n,
        LEFT_SB=left.stride(0),
        LEFT_SK=left.stride(1),
        LEFT_SM=left.stride(2),
        RIGHT_SB=right.stride(0),
        RIGHT_SK=right.stride(1),
        RIGHT_SN=right.stride(2),
        WEIGHT_SB=weight.stride(0),
        WEIGHT_SK=weight.stride(1),
        OUT_SB=out.stride(0),
        OUT_SM=out.stride(1),
        OUT_SN=out.stride(2),
        BLOCK_K=block_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out
