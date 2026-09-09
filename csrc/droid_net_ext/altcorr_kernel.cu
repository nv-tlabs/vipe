/**
 * This file includes code originally from the DROID-SLAM repository:
 * https://github.com/princeton-vl/DROID-SLAM
 * Licensed under the BSD-3 License. See THIRD_PARTY_LICENSES.md for details.
 */

#include <torch/extension.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>

// 8x8 tile = 64-thread blocks. The original DROID-SLAM 4x8 (32-thread, single
// warp) blocks cap GA10x occupancy at ~512/1536 threads per SM through the
// 16-resident-blocks limit; 64-thread blocks double that at unchanged shared
// memory per thread. All kernels below index tiles generically via
// BLOCK_HW / CHANNEL_STRIDE, so the tile size is a free parameter.
#define BLOCK_H 8
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H *BLOCK_W
#define CHANNEL_STRIDE 32

__forceinline__ __device__ bool within_bounds(int h, int w, int H, int W) { return h >= 0 && h < H && w >= 0 && w < W; }

template <typename scalar_t>
__global__ void altcorr_forward_kernel(const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fmap1,
                                       const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fmap2,
                                       const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> coords,
                                       torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> corr,
                                       int r) {
    const int b = blockIdx.x;
    const int h0 = blockIdx.y * blockDim.x;
    const int w0 = blockIdx.z * blockDim.y;
    const int tid = threadIdx.x * blockDim.y + threadIdx.y;

    const int H1 = fmap1.size(1);
    const int W1 = fmap1.size(2);
    const int H2 = fmap2.size(1);
    const int W2 = fmap2.size(2);
    const int N = coords.size(1);
    const int C = fmap1.size(3);

    __shared__ scalar_t f1[CHANNEL_STRIDE][BLOCK_HW];
    __shared__ scalar_t f2[CHANNEL_STRIDE][BLOCK_HW];

    __shared__ float x2s[BLOCK_HW];
    __shared__ float y2s[BLOCK_HW];

    for (int c = 0; c < C; c += CHANNEL_STRIDE) {
        for (int k = 0; k < BLOCK_HW; k += BLOCK_HW / CHANNEL_STRIDE) {
            int k1 = k + tid / CHANNEL_STRIDE;
            int h1 = h0 + k1 / BLOCK_W;
            int w1 = w0 + k1 % BLOCK_W;
            int c1 = tid % CHANNEL_STRIDE;

            if (within_bounds(h1, w1, H1, W1))
                f1[c1][k1] = fmap1[b][h1][w1][c + c1];

            else
                f1[c1][k1] = 0.0;
        }

        __syncthreads();

        for (int n = 0; n < N; n++) {
            int h1 = h0 + threadIdx.x;
            int w1 = w0 + threadIdx.y;
            if (within_bounds(h1, w1, H1, W1)) {
                x2s[tid] = coords[b][n][h1][w1][0];
                y2s[tid] = coords[b][n][h1][w1][1];
            }

            float dx = x2s[tid] - floor(x2s[tid]);
            float dy = y2s[tid] - floor(y2s[tid]);

            // [vipe] Barrier before the cross-thread reads of x2s/y2s below (h2/w2
            // for k1 != tid). With BLOCK_HW == 32 (one warp) this was accidentally
            // safe via SIMT lockstep; retiling to larger blocks (see BLOCK_H above)
            // spans multiple independently-scheduled warps, turning the missing
            // sync into a real, nondeterministic race between this write and that
            // read. Latent in the original DROID-SLAM kernel at any tile size >32.
            __syncthreads();

            int rd = 2 * r + 1;
            for (int iy = 0; iy < rd + 1; iy++) {
                for (int ix = 0; ix < rd + 1; ix++) {
                    for (int k = 0; k < BLOCK_HW; k += BLOCK_HW / CHANNEL_STRIDE) {
                        int k1 = k + tid / CHANNEL_STRIDE;
                        int h2 = static_cast<int>(floor(y2s[k1])) - r + iy;
                        int w2 = static_cast<int>(floor(x2s[k1])) - r + ix;
                        int c2 = tid % CHANNEL_STRIDE;

                        if (within_bounds(h2, w2, H2, W2))
                            f2[c2][k1] = fmap2[b][h2][w2][c + c2];

                        else
                            f2[c2][k1] = static_cast<scalar_t>(0.0);
                    }

                    __syncthreads();

                    scalar_t s = 0.0;
                    for (int k = 0; k < CHANNEL_STRIDE; k++) s += f1[k][tid] * f2[k][tid];

                    int ix_nw = H1 * W1 * ((iy - 1) + rd * (ix - 1));
                    int ix_ne = H1 * W1 * ((iy - 1) + rd * ix);
                    int ix_sw = H1 * W1 * (iy + rd * (ix - 1));
                    int ix_se = H1 * W1 * (iy + rd * ix);

                    // int ix_nw = ((iy-1) + rd*(ix-1));
                    // int ix_ne = ((iy-1) + rd*ix);
                    // int ix_sw = (iy + rd*(ix-1));
                    // int ix_se = (iy + rd*ix);

                    scalar_t nw = s * static_cast<scalar_t>((dy) * (dx));
                    scalar_t ne = s * static_cast<scalar_t>((dy) * (1 - dx));
                    scalar_t sw = s * static_cast<scalar_t>((1 - dy) * (dx));
                    scalar_t se = s * static_cast<scalar_t>((1 - dy) * (1 - dx));

                    // if (iy > 0 && ix > 0 && within_bounds(h1, w1, H1, W1))
                    //   corr[b][n][ix_nw][h1][w1] += nw;

                    // if (iy > 0 && ix < rd && within_bounds(h1, w1, H1, W1))
                    //   corr[b][n][ix_ne][h1][w1] += ne;

                    // if (iy < rd && ix > 0 && within_bounds(h1, w1, H1, W1))
                    //   corr[b][n][ix_sw][h1][w1] += sw;

                    // if (iy < rd && ix < rd && within_bounds(h1, w1, H1, W1))
                    //   corr[b][n][ix_se][h1][w1] += se;

                    scalar_t *corr_ptr = &corr[b][n][0][h1][w1];

                    if (iy > 0 && ix > 0 && within_bounds(h1, w1, H1, W1)) *(corr_ptr + ix_nw) += nw;

                    if (iy > 0 && ix < rd && within_bounds(h1, w1, H1, W1)) *(corr_ptr + ix_ne) += ne;

                    if (iy < rd && ix > 0 && within_bounds(h1, w1, H1, W1)) *(corr_ptr + ix_sw) += sw;

                    if (iy < rd && ix < rd && within_bounds(h1, w1, H1, W1)) *(corr_ptr + ix_se) += se;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void altcorr_index_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> corr, int r, int level_offset,
    float coord_scale) {
    const int bm = blockIdx.x;
    const int m = bm % coords.size(1);
    const int b = bm / coords.size(1);
    const int src = static_cast<int>(ii[m]);
    const int dst = static_cast<int>(jj[m]);
    const int h0 = blockIdx.y * blockDim.x;
    const int w0 = blockIdx.z * blockDim.y;
    const int tid = threadIdx.x * blockDim.y + threadIdx.y;

    const int H1 = coords.size(2);
    const int W1 = coords.size(3);
    const int H2 = fmap2.size(2);
    const int W2 = fmap2.size(3);
    const int C = fmap1.size(4);

    // The +1 is shared-memory padding, not correctness: only columns
    // 0..BLOCK_HW-1 are indexed. With BLOCK_HW = 32, same-column row
    // accesses such as f1[tid % CHANNEL_STRIDE][k1] map a warp to one
    // bank. A stride of 33 shifts each row by one bank, reducing conflicts
    // in the tile load/store pattern; dot-product reads f1[k][tid] are not
    // the main reason for the padding.
    __shared__ scalar_t f1[CHANNEL_STRIDE][BLOCK_HW + 1];
    __shared__ scalar_t f2[CHANNEL_STRIDE][BLOCK_HW + 1];

    __shared__ float x2s[BLOCK_HW];
    __shared__ float y2s[BLOCK_HW];

    const int h1 = h0 + threadIdx.x;
    const int w1 = w0 + threadIdx.y;
    const bool valid1 = within_bounds(h1, w1, H1, W1);

    if (valid1) {
        x2s[tid] = coords[b][m][h1][w1][0] / coord_scale;
        y2s[tid] = coords[b][m][h1][w1][1] / coord_scale;
    } else {
        x2s[tid] = 0.0f;
        y2s[tid] = 0.0f;
    }

    const float x2 = x2s[tid];
    const float y2 = y2s[tid];
    const float dx = x2 - floorf(x2);
    const float dy = y2 - floorf(y2);
    const int rd = 2 * r + 1;

    for (int c = 0; c < C; c += CHANNEL_STRIDE) {
        for (int k = 0; k < BLOCK_HW; k += BLOCK_HW / CHANNEL_STRIDE) {
            int k1 = k + tid / CHANNEL_STRIDE;
            int hk = h0 + k1 / BLOCK_W;
            int wk = w0 + k1 % BLOCK_W;
            int c1 = c + tid % CHANNEL_STRIDE;

            if (within_bounds(hk, wk, H1, W1) && c1 < C)
                f1[tid % CHANNEL_STRIDE][k1] = fmap1[b][src][hk][wk][c1];
            else
                f1[tid % CHANNEL_STRIDE][k1] = static_cast<scalar_t>(0.0);
        }

        __syncthreads();

        for (int iy = 0; iy < rd + 1; iy++) {
            for (int ix = 0; ix < rd + 1; ix++) {
                for (int k = 0; k < BLOCK_HW; k += BLOCK_HW / CHANNEL_STRIDE) {
                    int k1 = k + tid / CHANNEL_STRIDE;
                    int h2 = static_cast<int>(floorf(y2s[k1])) - r + iy;
                    int w2 = static_cast<int>(floorf(x2s[k1])) - r + ix;
                    int c2 = c + tid % CHANNEL_STRIDE;

                    if (within_bounds(h2, w2, H2, W2) && c2 < C)
                        f2[tid % CHANNEL_STRIDE][k1] = fmap2[b][dst][h2][w2][c2];
                    else
                        f2[tid % CHANNEL_STRIDE][k1] = static_cast<scalar_t>(0.0);
                }

                __syncthreads();

                if (valid1) {
                    float s = 0.0f;
                    for (int k = 0; k < CHANNEL_STRIDE; k++) {
                        s += static_cast<float>(f1[k][tid]) * static_cast<float>(f2[k][tid]);
                    }

                    float *corr_ptr = &corr[b][m][level_offset][h1][w1];

                    if (iy > 0 && ix > 0) *(corr_ptr + ((iy - 1) + rd * (ix - 1)) * H1 * W1) += s * dy * dx;

                    if (iy > 0 && ix < rd) *(corr_ptr + ((iy - 1) + rd * ix) * H1 * W1) += s * dy * (1.0f - dx);

                    if (iy < rd && ix > 0) *(corr_ptr + (iy + rd * (ix - 1)) * H1 * W1) += s * (1.0f - dy) * dx;

                    if (iy < rd && ix < rd) *(corr_ptr + (iy + rd * ix) * H1 * W1) += s * (1.0f - dy) * (1.0f - dx);
                }

                __syncthreads();
            }
        }
    }
}

// Thread-per-pixel variant of altcorr_index_forward_kernel for fp16 inputs
// with C <= 128. The staged/shared-memory version above serializes on
// 2 * (C/CHANNEL_STRIDE) * (rd+1)^2 = 512 block barriers per launch, while its
// shared tiles provide no cross-thread reuse (the layout is channels-last, so
// each thread's own gathers are already contiguous). Here each thread owns one
// output pixel: fmap1's channels sit in registers (vector-loaded, 16B chunks),
// fmap2 is gathered directly per offset and served by L1 — the (2r+2)^2
// windows of neighboring pixels/offsets overlap almost entirely, so the cache
// does what the shared tiles could not. No shared memory, no barriers.
__global__ void altcorr_index_forward_kernel_pix(
    const torch::PackedTensorAccessor32<at::Half, 5, torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<at::Half, 5, torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> corr, int r, int level_offset,
    float coord_scale) {
    const int M = coords.size(1);
    const int bm = blockIdx.x;
    const int m = bm % M;
    const int b = bm / M;
    const int src = static_cast<int>(ii[m]);
    const int dst = static_cast<int>(jj[m]);

    const int H1 = coords.size(2);
    const int W1 = coords.size(3);
    const int H2 = fmap2.size(2);
    const int W2 = fmap2.size(3);
    const int C = fmap1.size(4);

    const int idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (idx >= H1 * W1) return;
    const int h1 = idx / W1;
    const int w1 = idx % W1;

    const float x2 = coords[b][m][h1][w1][0] / coord_scale;
    const float y2 = coords[b][m][h1][w1][1] / coord_scale;
    const float dx = x2 - floorf(x2);
    const float dy = y2 - floorf(y2);
    const int x0 = static_cast<int>(floorf(x2));
    const int y0 = static_cast<int>(floorf(y2));
    const int rd = 2 * r + 1;

    // fmap1 channels of this pixel in registers (channels-last: contiguous).
    // Pixel base is C * sizeof(half) = 256B aligned for C = 128, so 16B
    // vector loads are safe.
    __half2 f1v[64];  // up to C = 128 halfs
    const __half2 *f1p = reinterpret_cast<const __half2 *>(&fmap1[b][src][h1][w1][0]);
#pragma unroll
    for (int c = 0; c < C / 2; c += 4) {
        const float4 v = *reinterpret_cast<const float4 *>(f1p + c);
        f1v[c] = reinterpret_cast<const __half2 *>(&v)[0];
        f1v[c + 1] = reinterpret_cast<const __half2 *>(&v)[1];
        f1v[c + 2] = reinterpret_cast<const __half2 *>(&v)[2];
        f1v[c + 3] = reinterpret_cast<const __half2 *>(&v)[3];
    }

    float *corr_ptr = &corr[b][m][level_offset][h1][w1];
    const int HW1 = H1 * W1;

    for (int iy = 0; iy < rd + 1; iy++) {
        for (int ix = 0; ix < rd + 1; ix++) {
            const int h2 = y0 - r + iy;
            const int w2 = x0 - r + ix;

            float s = 0.0f;
            if (within_bounds(h2, w2, H2, W2)) {
                const __half2 *f2p = reinterpret_cast<const __half2 *>(&fmap2[b][dst][h2][w2][0]);
                float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
                for (int c = 0; c < C / 2; c += 4) {
                    const float4 v = *reinterpret_cast<const float4 *>(f2p + c);
                    const __half2 *f2h = reinterpret_cast<const __half2 *>(&v);
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const float2 a = __half22float2(f1v[c + u]);
                        const float2 bb = __half22float2(f2h[u]);
                        acc.x += a.x * bb.x;
                        acc.y += a.y * bb.y;
                    }
                }
                s = acc.x + acc.y;
            }

            if (iy > 0 && ix > 0) *(corr_ptr + ((iy - 1) + rd * (ix - 1)) * HW1) += s * dy * dx;
            if (iy > 0 && ix < rd) *(corr_ptr + ((iy - 1) + rd * ix) * HW1) += s * dy * (1.0f - dx);
            if (iy < rd && ix > 0) *(corr_ptr + (iy + rd * (ix - 1)) * HW1) += s * (1.0f - dy) * dx;
            if (iy < rd && ix < rd) *(corr_ptr + (iy + rd * ix) * HW1) += s * (1.0f - dy) * (1.0f - dx);
        }
    }
}

template <typename scalar_t>
__global__ void altcorr_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fmap1_grad,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fmap2_grad,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> coords_grad, int r) {
    const int b = blockIdx.x;
    const int h0 = blockIdx.y * blockDim.x;
    const int w0 = blockIdx.z * blockDim.y;
    const int tid = threadIdx.x * blockDim.y + threadIdx.y;

    const int H1 = fmap1.size(1);
    const int W1 = fmap1.size(2);
    const int H2 = fmap2.size(1);
    const int W2 = fmap2.size(2);
    const int N = coords.size(1);
    const int C = fmap1.size(3);

    __shared__ scalar_t f1[CHANNEL_STRIDE][BLOCK_HW + 1];
    __shared__ scalar_t f2[CHANNEL_STRIDE][BLOCK_HW + 1];

    __shared__ scalar_t f1_grad[CHANNEL_STRIDE][BLOCK_HW + 1];
    __shared__ scalar_t f2_grad[CHANNEL_STRIDE][BLOCK_HW + 1];

    __shared__ scalar_t x2s[BLOCK_HW];
    __shared__ scalar_t y2s[BLOCK_HW];

    for (int c = 0; c < C; c += CHANNEL_STRIDE) {
        for (int k = 0; k < BLOCK_HW; k += BLOCK_HW / CHANNEL_STRIDE) {
            int k1 = k + tid / CHANNEL_STRIDE;
            int h1 = h0 + k1 / BLOCK_W;
            int w1 = w0 + k1 % BLOCK_W;
            int c1 = tid % CHANNEL_STRIDE;

            auto fptr = fmap1[b][h1][w1];
            if (within_bounds(h1, w1, H1, W1))
                f1[c1][k1] = fptr[c + c1];
            else
                f1[c1][k1] = 0.0;

            f1_grad[c1][k1] = 0.0;
        }

        __syncthreads();

        int h1 = h0 + threadIdx.x;
        int w1 = w0 + threadIdx.y;

        for (int n = 0; n < N; n++) {
            x2s[tid] = coords[b][n][h1][w1][0];
            y2s[tid] = coords[b][n][h1][w1][1];

            scalar_t dx = x2s[tid] - floor(x2s[tid]);
            scalar_t dy = y2s[tid] - floor(y2s[tid]);

            // [vipe] See the matching barrier in altcorr_forward_kernel above: without
            // it, the cross-thread reads of x2s/y2s a few lines down race the write
            // just above once BLOCK_HW spans more than one warp.
            __syncthreads();

            int rd = 2 * r + 1;
            for (int iy = 0; iy < rd + 1; iy++) {
                for (int ix = 0; ix < rd + 1; ix++) {
                    for (int k = 0; k < BLOCK_HW; k += BLOCK_HW / CHANNEL_STRIDE) {
                        int k1 = k + tid / CHANNEL_STRIDE;
                        int h2 = static_cast<int>(floor(y2s[k1])) - r + iy;
                        int w2 = static_cast<int>(floor(x2s[k1])) - r + ix;
                        int c2 = tid % CHANNEL_STRIDE;

                        auto fptr = fmap2[b][h2][w2];
                        if (within_bounds(h2, w2, H2, W2))
                            f2[c2][k1] = fptr[c + c2];
                        else
                            f2[c2][k1] = 0.0;

                        f2_grad[c2][k1] = 0.0;
                    }

                    __syncthreads();

                    const scalar_t *grad_ptr = &corr_grad[b][n][0][h1][w1];
                    scalar_t g = 0.0;

                    int ix_nw = H1 * W1 * ((iy - 1) + rd * (ix - 1));
                    int ix_ne = H1 * W1 * ((iy - 1) + rd * ix);
                    int ix_sw = H1 * W1 * (iy + rd * (ix - 1));
                    int ix_se = H1 * W1 * (iy + rd * ix);

                    if (iy > 0 && ix > 0 && within_bounds(h1, w1, H1, W1)) g += *(grad_ptr + ix_nw) * dy * dx;

                    if (iy > 0 && ix < rd && within_bounds(h1, w1, H1, W1)) g += *(grad_ptr + ix_ne) * dy * (1 - dx);

                    if (iy < rd && ix > 0 && within_bounds(h1, w1, H1, W1)) g += *(grad_ptr + ix_sw) * (1 - dy) * dx;

                    if (iy < rd && ix < rd && within_bounds(h1, w1, H1, W1))
                        g += *(grad_ptr + ix_se) * (1 - dy) * (1 - dx);

                    for (int k = 0; k < CHANNEL_STRIDE; k++) {
                        f1_grad[k][tid] += g * f2[k][tid];
                        f2_grad[k][tid] += g * f1[k][tid];
                    }

                    for (int k = 0; k < BLOCK_HW; k += BLOCK_HW / CHANNEL_STRIDE) {
                        int k1 = k + tid / CHANNEL_STRIDE;
                        int h2 = static_cast<int>(floor(y2s[k1])) - r + iy;
                        int w2 = static_cast<int>(floor(x2s[k1])) - r + ix;
                        int c2 = tid % CHANNEL_STRIDE;

                        scalar_t *fptr = &fmap2_grad[b][h2][w2][0];
                        if (within_bounds(h2, w2, H2, W2)) atomicAdd(fptr + c + c2, f2_grad[c2][k1]);
                    }
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_HW; k += BLOCK_HW / CHANNEL_STRIDE) {
            int k1 = k + tid / CHANNEL_STRIDE;
            int h1 = h0 + k1 / BLOCK_W;
            int w1 = w0 + k1 % BLOCK_W;
            int c1 = tid % CHANNEL_STRIDE;

            scalar_t *fptr = &fmap1_grad[b][h1][w1][0];
            if (within_bounds(h1, w1, H1, W1)) fptr[c + c1] += f1_grad[c1][k1];
        }
    }
}

std::vector<torch::Tensor> altcorr_cuda_forward(torch::Tensor fmap1, torch::Tensor fmap2, torch::Tensor coords,
                                                int radius) {
    const auto B = coords.size(0);
    const auto N = coords.size(1);
    const auto H = coords.size(2);
    const auto W = coords.size(3);

    const auto rd = 2 * radius + 1;
    auto opts = fmap1.options();
    auto corr = torch::zeros({B, N, rd * rd, H, W}, opts);

    const dim3 blocks(B, (H + BLOCK_H - 1) / BLOCK_H, (W + BLOCK_W - 1) / BLOCK_W);
    const dim3 threads(BLOCK_H, BLOCK_W);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "altcorr_forward_kernel", ([&] {
                                            altcorr_forward_kernel<scalar_t><<<blocks, threads>>>(
                                                fmap1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                                fmap2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                                coords.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                                                corr.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                                                radius);
                                        }));

    return {corr};
}

std::vector<torch::Tensor> altcorr_index_cuda_forward(torch::Tensor fmap1, std::vector<torch::Tensor> fmap2_pyramid,
                                                      torch::Tensor coords, torch::Tensor ii, torch::Tensor jj,
                                                      int radius) {
    const auto B = coords.size(0);
    const auto M = coords.size(1);
    const auto H = coords.size(2);
    const auto W = coords.size(3);
    const auto rd = 2 * radius + 1;
    const auto L = static_cast<int64_t>(fmap2_pyramid.size());

    auto corr = torch::zeros({B, M, L * rd * rd, H, W}, fmap1.options().dtype(torch::kFloat));

    const auto C = fmap1.size(4);
    // Debug escape hatch: VIPE_ALTCORR_PIX=0 forces the original staged kernel
    // (used for kernel A/B validation; the pix kernel is the default).
    const char *pix_env = std::getenv("VIPE_ALTCORR_PIX");
    const bool allow_pix = !(pix_env != nullptr && pix_env[0] == '0');
    if (allow_pix && fmap1.scalar_type() == torch::kHalf && C % 8 == 0 && C <= 128) {
        // Fast path: thread-per-pixel kernel (see altcorr_index_forward_kernel_pix).
        const int threads_pix = 128;
        const dim3 blocks_pix(B * M, (H * W + threads_pix - 1) / threads_pix);
        for (int64_t level = 0; level < L; ++level) {
            altcorr_index_forward_kernel_pix<<<blocks_pix, threads_pix>>>(
                fmap1.packed_accessor32<at::Half, 5, torch::RestrictPtrTraits>(),
                fmap2_pyramid[level].packed_accessor32<at::Half, 5, torch::RestrictPtrTraits>(),
                coords.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                ii.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                jj.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                corr.packed_accessor32<float, 5, torch::RestrictPtrTraits>(), radius,
                static_cast<int>(level * rd * rd), static_cast<float>(1 << level));
        }
        return {corr};
    }

    const dim3 blocks(B * M, (H + BLOCK_H - 1) / BLOCK_H, (W + BLOCK_W - 1) / BLOCK_W);
    const dim3 threads(BLOCK_H, BLOCK_W);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "altcorr_index_forward_kernel", ([&] {
                                            for (int64_t level = 0; level < L; ++level) {
                                                altcorr_index_forward_kernel<scalar_t><<<blocks, threads>>>(
                                                    fmap1.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                                                    fmap2_pyramid[level].packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                                                    coords.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                                                    ii.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                                    jj.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                                    corr.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                                                    radius, static_cast<int>(level * rd * rd),
                                                    static_cast<float>(1 << level));
                                            }
                                        }));

    return {corr};
}

std::vector<torch::Tensor> altcorr_cuda_backward(torch::Tensor fmap1, torch::Tensor fmap2, torch::Tensor coords,
                                                 torch::Tensor corr_grad, int radius) {
    const auto B = coords.size(0);
    const auto N = coords.size(1);

    const auto H1 = fmap1.size(1);
    const auto W1 = fmap1.size(2);
    const auto H2 = fmap2.size(1);
    const auto W2 = fmap2.size(2);
    const auto C = fmap1.size(3);

    auto opts = fmap1.options();
    auto fmap1_grad = torch::zeros({B, H1, W1, C}, opts);
    auto fmap2_grad = torch::zeros({B, H2, W2, C}, opts);
    auto coords_grad = torch::zeros({B, N, H1, W1, 2}, opts);

    const dim3 blocks(B, (H1 + BLOCK_H - 1) / BLOCK_H, (W1 + BLOCK_W - 1) / BLOCK_W);
    const dim3 threads(BLOCK_H, BLOCK_W);

    altcorr_backward_kernel<float>
        <<<blocks, threads>>>(fmap1.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                              fmap2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                              coords.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                              corr_grad.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                              fmap1_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                              fmap2_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                              coords_grad.packed_accessor32<float, 5, torch::RestrictPtrTraits>(), radius);

    return {fmap1_grad, fmap2_grad, coords_grad};
}
