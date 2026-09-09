// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Device-resident assembly and solve of the Gauss-Newton normal equations
// used by the fused bundle adjustment (see geom_kernels.cu).
//
// The historical DROID-SLAM implementation copied every per-edge 6x6 block to
// the host, assembled an Eigen sparse matrix from triplets, and factorized it
// with SimplicialLLT on the CPU — forcing ~35 device synchronizations per BA
// call and leaving the GPU idle during the solve. This header replaces that
// path with:
//
//  - GpuBlockSystem: accumulates 6x6 pose blocks / 6-vectors (plus an optional
//    scalar intrinsics row) directly on the GPU via scatter-add into a dense
//    block grid, then damps and factorizes on the GPU. Verified bit-exact
//    against the Eigen path before that path was removed.
//  - Precomputed index patterns (AccumPattern / SchurPattern): the edge graph
//    is constant across the Gauss-Newton iterations of one BA call, so all
//    symbolic/index work is done once per call on the host (the cholespy
//    split: symbolic once, numeric many) and only numeric kernels run inside
//    the iteration loop — with zero host round-trips.
//
// Solver policy: the reduced camera system is small (6 x #keyframes plus at
// most one intrinsics DOF), so a dense factorization on the GPU beats a
// sparse one at these sizes. Systems up to VIPE_BA_GPU_FP64_MAX_DIM unknowns
// (default 2048) are factorized in float64, matching the Eigen path bit-for-
// bit up to summation order. Larger systems — large keyframe counts where
// consumer GPUs pay a 1:32 float64 penalty — are factorized in float32 and
// polished with float64 iterative refinement, which recovers ~float64 solve
// accuracy at float32 cost (the block data entering the solver is float32 to
// begin with).

#pragma once

#include <torch/extension.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace slam_ext {

// ------------------------------------------------------------------------------------
// Solver configuration (read once per process from the environment).
// ------------------------------------------------------------------------------------

// Largest system dimension factorized in float64; beyond it use float32 + refinement.
// Read per call (getenv is ~ns and BA calls are ~ms) so tests can override it
// via os.environ without process restarts.
inline int64_t gpu_solver_fp64_max_dim() {
    const char* value = std::getenv("VIPE_BA_GPU_FP64_MAX_DIM");
    return value != nullptr ? std::atoll(value) : 2048LL;
}

// ------------------------------------------------------------------------------------
// Symbolic patterns, built once per BA call from host copies of the edge lists.
// ------------------------------------------------------------------------------------

// Index pattern for accum_kernel: out[j] = sum over {i : ix[i] == jx[j]} of data[i].
// Equivalent to the index preprocessing the historical accum_cuda redid (on the CPU,
// with device transfers) on every invocation.
struct AccumPattern {
    torch::Tensor ptrs;  // (count + 1,) int64 CUDA — group offsets into `idxs`
    torch::Tensor idxs;  // (nnz,) int64 CUDA — source rows, grouped by output row
    int64_t count = 0;   // number of output rows
};

inline AccumPattern build_accum_pattern(torch::Tensor ix_cpu, torch::Tensor jx_cpu) {
    TORCH_CHECK(ix_cpu.device().is_cpu() && jx_cpu.device().is_cpu(), "pattern inputs must be on CPU");
    torch::Tensor order = torch::argsort(ix_cpu);

    const long* ix_data = ix_cpu.data_ptr<long>();
    const long* jx_data = jx_cpu.data_ptr<long>();
    const long* order_data = order.data_ptr<long>();

    const int64_t count = jx_cpu.size(0);
    std::vector<long> cols;
    torch::Tensor ptrs_cpu = torch::zeros({count + 1}, torch::TensorOptions().dtype(torch::kInt64));
    long* ptrs_data = ptrs_cpu.data_ptr<long>();

    int64_t i = 0;
    for (int64_t j = 0; j < count; j++) {
        while (i < ix_cpu.size(0) && ix_data[order_data[i]] <= jx_data[j]) {
            if (ix_data[order_data[i]] == jx_data[j]) cols.push_back(order_data[i]);
            i++;
        }
        ptrs_data[j + 1] = static_cast<long>(cols.size());
    }

    torch::Tensor idxs_cpu = torch::from_blob(cols.data(), {static_cast<long>(cols.size())},
                                              torch::TensorOptions().dtype(torch::kInt64))
                                 .clone();

    AccumPattern pattern;
    pattern.ptrs = ptrs_cpu.to(torch::kCUDA);
    pattern.idxs = idxs_cpu.to(torch::kCUDA);
    pattern.count = count;
    return pattern;
}

// Index pattern for the Schur complement kernels (EEt6x6 / Ev6x1) and for the
// optional pose-intrinsics coupling. For every pair of edges (n, m) whose targets
// fall inside the active window and that share the same depth block k, one 6x6
// contribution E_n Q_k E_m^T lands at pose block (jj[n]-t0, jj[m]-t0).
struct SchurPattern {
    torch::Tensor pair_idx;      // (npairs, 3) int64 CUDA — (edge n, edge m, depth k) per EEt block
    torch::Tensor pair_block_i;  // (npairs,) int64 CUDA — destination block row
    torch::Tensor pair_block_j;  // (npairs,) int64 CUDA — destination block col
    torch::Tensor rhs_depth;     // (n_edges, 1) int64 CUDA — depth column per edge for Ev6x1
    torch::Tensor rhs_block;     // (n_edges,) int64 CUDA — destination block row (jj - t0)
    // Pose-intrinsics coupling (rows of E / Q / Ef selected per in-window edge):
    torch::Tensor coupling_edge_rows;   // (nrows,) int64 CUDA
    torch::Tensor coupling_depth_rows;  // (nrows,) int64 CUDA
    torch::Tensor coupling_block;       // (nrows,) int64 CUDA — destination pose block row
    int64_t npairs = 0;
};

inline SchurPattern build_schur_pattern(torch::Tensor ii_cpu, torch::Tensor jj_cpu, torch::Tensor kk_cpu,
                                        const int t0, const int t1) {
    TORCH_CHECK(ii_cpu.device().is_cpu() && jj_cpu.device().is_cpu() && kk_cpu.device().is_cpu(),
                "pattern inputs must be on CPU");
    const int P = t1 - t0;
    const long* jj_data = jj_cpu.data_ptr<long>();
    const long* kk_data = kk_cpu.data_ptr<long>();

    // Edges grouped by their (in-window) target pose.
    std::vector<std::vector<long>> depth_of(P);
    std::vector<std::vector<long>> edge_of(P);
    std::vector<long> row_list, pose_list, depth_list;

    for (int64_t n = 0; n < jj_cpu.size(0); n++) {
        const long j = jj_data[n];
        if (j >= t0 && j < t1) {
            const long t = j - t0;
            depth_of[t].push_back(kk_data[n]);
            edge_of[t].push_back(n);
            row_list.push_back(n);
            pose_list.push_back(t);
            depth_list.push_back(kk_data[n]);
        }
    }

    std::vector<long> bi_list, bj_list, idx_list;
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            for (size_t k = 0; k < depth_of[i].size(); k++) {
                for (size_t l = 0; l < depth_of[j].size(); l++) {
                    if (depth_of[i][k] == depth_of[j][l]) {
                        bi_list.push_back(i);
                        bj_list.push_back(j);
                        idx_list.push_back(edge_of[i][k]);
                        idx_list.push_back(edge_of[j][l]);
                        idx_list.push_back(depth_of[i][k]);
                    }
                }
            }
        }
    }

    const auto long_opts = torch::TensorOptions().dtype(torch::kInt64);
    auto to_cuda = [&](std::vector<long>& v) {
        return torch::from_blob(v.data(), {static_cast<long>(v.size())}, long_opts).clone().to(torch::kCUDA);
    };

    SchurPattern pattern;
    pattern.npairs = static_cast<int64_t>(bi_list.size());
    pattern.pair_idx = to_cuda(idx_list).view({-1, 3});
    pattern.pair_block_i = to_cuda(bi_list);
    pattern.pair_block_j = to_cuda(bj_list);
    pattern.rhs_depth = kk_cpu.to(torch::kCUDA).view({-1, 1});
    pattern.rhs_block = (jj_cpu - t0).to(torch::kCUDA);
    pattern.coupling_edge_rows = to_cuda(row_list);
    pattern.coupling_depth_rows = to_cuda(depth_list);
    pattern.coupling_block = to_cuda(pose_list);
    return pattern;
}

// ------------------------------------------------------------------------------------
// GpuBlockSystem — the normal equations, assembled and solved on the device.
// ------------------------------------------------------------------------------------

class GpuBlockSystem {
   public:
    // pose_count P: number of free poses (6 DOF each); scalar_count: extra scalar
    // unknowns appended after the pose block (ViPE uses one, the shared focal).
    GpuBlockSystem(int pose_count, int scalar_count)
        : P_(pose_count), S_(scalar_count), dim_(6 * pose_count + scalar_count) {
        const auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
        // (P+1) x (P+1) grid of 6x6 blocks; row/col P is a spill slot that
        // absorbs contributions with out-of-range indices (fixed poses) and is
        // sliced away at solve time — no host-visible branching required.
        lhs_blocks_ = torch::zeros({P_ + 1, P_ + 1, 6, 6}, opts);
        rhs_blocks_ = torch::zeros({P_ + 1, 6}, opts);
        if (S_ > 0) {
            lhs_pose_scalar_ = torch::zeros({P_ + 1, 6, S_}, opts);
            lhs_scalar_ = torch::zeros({S_, S_}, opts);
            rhs_scalar_ = torch::zeros({S_}, opts);
        }
    }

    // Accumulate sign * blocks[n] at pose block (block_i[n], block_j[n]).
    // Blocks with either index outside [0, P) are dropped (spill slot).
    void add_pose_blocks(const torch::Tensor& blocks, const torch::Tensor& block_i, const torch::Tensor& block_j,
                         const double sign) {
        if (blocks.size(0) == 0) return;
        const auto valid = valid_mask(block_i) & valid_mask(block_j);
        lhs_blocks_.index_put_({spill(block_i, valid), spill(block_j, valid)},
                               blocks.to(torch::kFloat64) * sign, /*accumulate=*/true);
    }

    // Accumulate sign * vecs[n] (6-vectors) at pose block block_i[n].
    void add_rhs_blocks(const torch::Tensor& vecs, const torch::Tensor& block_i, const double sign) {
        if (vecs.size(0) == 0) return;
        const auto valid = valid_mask(block_i);
        rhs_blocks_.index_put_({spill(block_i, valid)}, vecs.to(torch::kFloat64) * sign, /*accumulate=*/true);
    }

    // Accumulate sign * vals[n] (6-vectors) into the pose-scalar coupling column.
    void add_pose_scalar(const torch::Tensor& vals, const torch::Tensor& block_i, const double sign) {
        if (S_ == 0 || vals.size(0) == 0) return;
        const auto valid = valid_mask(block_i);
        lhs_pose_scalar_.index_put_({spill(block_i, valid)},
                                    (vals.to(torch::kFloat64) * sign).unsqueeze(-1), /*accumulate=*/true);
    }

    // Accumulate a 0-dim device tensor into the scalar diagonal / rhs (no host sync).
    void add_scalar_diag(const torch::Tensor& value, const double sign) {
        if (S_ == 0) return;
        lhs_scalar_ += value.to(torch::kFloat64) * sign;
    }
    void add_scalar_rhs(const torch::Tensor& value, const double sign) {
        if (S_ == 0) return;
        rhs_scalar_ += value.to(torch::kFloat64) * sign;
    }

    // Damp (diag += ep + lm * diag, matching the Eigen path) and solve H x = b.
    // Returns (dx, df): pose update (P, 6) float32 and scalar update (S,) float32,
    // both on the device; a failed factorization yields zero updates, matching the
    // historical fallback, without a host round-trip.
    std::tuple<torch::Tensor, torch::Tensor> solve(const double pose_lm, const double pose_ep,
                                                   const double scalar_lm, const double scalar_ep) const {
        const auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);

        // Assemble the dense system from the block grid (spill slot sliced away).
        torch::Tensor H =
            lhs_blocks_.narrow(0, 0, P_).narrow(1, 0, P_).permute({0, 2, 1, 3}).reshape({6 * P_, 6 * P_});
        torch::Tensor b = rhs_blocks_.narrow(0, 0, P_).reshape({6 * P_});
        if (S_ > 0) {
            torch::Tensor coupling = lhs_pose_scalar_.narrow(0, 0, P_).reshape({6 * P_, S_});
            H = torch::cat({torch::cat({H, coupling}, 1), torch::cat({coupling.t(), lhs_scalar_}, 1)}, 0);
            b = torch::cat({b, rhs_scalar_});
        }
        H = H.contiguous();

        // Levenberg-Marquardt style damping with separate pose/scalar strengths.
        torch::Tensor lm = torch::full({dim_}, pose_lm, f64);
        torch::Tensor ep = torch::full({dim_}, pose_ep, f64);
        if (S_ > 0) {
            lm.narrow(0, 6 * P_, S_).fill_(scalar_lm);
            ep.narrow(0, 6 * P_, S_).fill_(scalar_ep);
        }
        torch::Tensor diag = H.diagonal();
        diag.add_(diag * lm + ep);

        torch::Tensor x = cholesky_solve_device(H, b.unsqueeze(1)).squeeze(1);

        torch::Tensor dx = x.narrow(0, 0, 6 * P_).view({P_, 6}).to(torch::kFloat32);
        torch::Tensor df = S_ > 0
                               ? x.narrow(0, 6 * P_, S_).to(torch::kFloat32)
                               : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        return {dx, df};
    }

   private:
    torch::Tensor valid_mask(const torch::Tensor& idx) const { return (idx >= 0) & (idx < P_); }

    torch::Tensor spill(const torch::Tensor& idx, const torch::Tensor& valid) const {
        return torch::where(valid, idx, torch::full_like(idx, P_));
    }

    // Dense Cholesky solve on the device. Small systems factor in float64 (parity
    // with the Eigen double path); large ones factor in float32 and are polished
    // with float64 iterative refinement (consumer GPUs run float64 at 1/32 rate).
    static torch::Tensor cholesky_solve_device(const torch::Tensor& H, const torch::Tensor& b) {
        const int64_t dim = H.size(0);
        torch::Tensor x;
        torch::Tensor info;

        if (dim <= gpu_solver_fp64_max_dim()) {
            auto [factor, factor_info] = at::linalg_cholesky_ex(H, /*upper=*/false, /*check_errors=*/false);
            info = factor_info;
            x = at::cholesky_solve(b, factor, /*upper=*/false);
        } else {
            torch::Tensor H32 = H.to(torch::kFloat32);
            auto [factor, factor_info] = at::linalg_cholesky_ex(H32, /*upper=*/false, /*check_errors=*/false);
            info = factor_info;
            x = at::cholesky_solve(b.to(torch::kFloat32), factor, /*upper=*/false).to(torch::kFloat64);
            for (int refinement = 0; refinement < 2; refinement++) {
                torch::Tensor residual = b - H.matmul(x);  // float64 residual
                x = x + at::cholesky_solve(residual.to(torch::kFloat32), factor, /*upper=*/false).to(torch::kFloat64);
            }
        }

        // On factorization failure return a zero update (historical Eigen fallback),
        // decided entirely on the device.
        return torch::where(info.eq(0), x, torch::zeros_like(x));
    }

    const int P_;
    const int S_;
    const int64_t dim_;
    torch::Tensor lhs_blocks_;
    torch::Tensor rhs_blocks_;
    torch::Tensor lhs_pose_scalar_;
    torch::Tensor lhs_scalar_;
    torch::Tensor rhs_scalar_;
};

}  // namespace slam_ext
