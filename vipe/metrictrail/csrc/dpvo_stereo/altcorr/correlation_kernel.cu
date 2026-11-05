// SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
// SPDX-License-Identifier: MIT
//
// Source: https://github.com/princeton-vl/DPVO
// See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

// Modified for MetricTrail integration.
// The following Apache-2.0 terms apply to NVIDIA modifications only.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>
#include <THC/THCAtomics.cuh>
#include <vector>
#include <iostream>

using namespace torch::indexing;

#define THREADS 256
#define BLOCKS(n) (n + THREADS - 1) / THREADS

__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void patchify_forward_kernel(int R,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> net,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> patches)
{
  // diameter
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = net.size(1);
  const int H = net.size(2);
  const int W = net.size(3);

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < B * M * D * D) {
    const int ii = n % D; n /= D;
    const int jj = n % D; n /= D;
    const int  m = n % M; n /= M;

    const float x = coords[n][m][0];
    const float y = coords[n][m][1];
    const int i = static_cast<int>(floor(y)) + (ii - R);
    const int j = static_cast<int>(floor(x)) + (jj - R);

    if (within_bounds(i, j, H, W)) {
      for (int k=0; k<C; k++)
        patches[n][m][k][ii][jj] = net[n][k][i][j];
    }
  }
}

template <typename scalar_t>
__global__ void patchify_backward_kernel(int R,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> patch_gradient,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> gradient)
{
  // diameter
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = gradient.size(1);
  const int H = gradient.size(2);
  const int W = gradient.size(3);

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < B * M * D * D) {
    const int ii = n % D; n /= D;
    const int jj = n % D; n /= D;
    const int  m = n % M; n /= M;

    const float x = coords[n][m][0];
    const float y = coords[n][m][1];
    const int i = static_cast<int>(floor(y)) + (ii - R);
    const int j = static_cast<int>(floor(x)) + (jj - R);

    if (within_bounds(i, j, H, W)) {
      for (int k=0; k<C; k++)
        atomicAdd(&gradient[n][k][i][j], patch_gradient[n][m][k][ii][jj]);
    }
  }
}

template <typename scalar_t>
__global__ void corr_forward_kernel(int R,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> us,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> corr)
{
  // diameter
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int H = coords.size(3);
  const int W = coords.size(4);

  const int C = fmap1.size(2);
  const int H2 = fmap2.size(3);
  const int W2 = fmap2.size(4);

  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < B * M * H * W * D * D) {
    const int jj = n % D; n /= D;
    const int ii = n % D; n /= D;
    const int j0 = n % W; n /= W;
    const int i0 = n % H; n /= H;
    const int  m = n % M; n /= M;

    const int ix = us[m];
    const int jx = vs[m];

    const float x = coords[n][m][0][i0][j0];
    const float y = coords[n][m][1][i0][j0];

    const int i1 = static_cast<int>(floor(y)) + (ii - R);
    const int j1 = static_cast<int>(floor(x)) + (jj - R);

    scalar_t s = 0;
    if (within_bounds(i1, j1, H2, W2)) {

      #pragma unroll 8
      for (int i=0; i<C; i+=8) {
        scalar_t f1[8]; for (int j=0; j<8; j++) f1[j] = fmap1[n][ix][i+j][i0][j0];
        scalar_t f2[8]; for (int j=0; j<8; j++) f2[j] = fmap2[n][jx][i+j][i1][j1];

        #pragma unroll
        for (int j=0; j<8; j++) s += f1[j] * f2[j];
      }
    }

    corr[n][m][ii][jj][i0][j0] = s;
  }
}


template <typename scalar_t>
__global__ void corr_backward_kernel(int R,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> us,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> vs,
    const torch::PackedTensorAccessor32<float,6,torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap1_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap2_grad)
{
  // diameter
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int H = coords.size(3);
  const int W = coords.size(4);

  const int C = fmap1.size(2);
  const int H2 = fmap2.size(3);
  const int W2 = fmap2.size(4);

  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < B * M * H * W * D * D) {
    const int jj = n % D; n /= D;
    const int ii = n % D; n /= D;
    const int j0 = n % W; n /= W;
    const int i0 = n % H; n /= H;
    const int  m = n % M; n /= M;

    const int ix = us[m];
    const int jx = vs[m];

    const float x = coords[n][m][0][i0][j0];
    const float y = coords[n][m][1][i0][j0];

    const int i1 = static_cast<int>(floor(y)) + (ii - R);
    const int j1 = static_cast<int>(floor(x)) + (jj - R);

    const scalar_t g = (scalar_t) corr_grad[n][m][ii][jj][i0][j0];

    if (within_bounds(i1, j1, H2, W2)) {
      #pragma unroll 32
      for (int i=0; i<C; i++) {
        atomicAdd(&fmap1_grad[n][ix][i][i0][j0], g * fmap2[n][jx][i][i1][j1]);
        atomicAdd(&fmap2_grad[n][jx][i][i1][j1], g * fmap1[n][ix][i][i0][j0]);
      }
    }
  }
}


std::vector<torch::Tensor> corr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor ii,
  torch::Tensor jj,
  int radius)
{
  const int B = coords.size(0);
  const int M = coords.size(1);

  const int H = coords.size(3);
  const int W = coords.size(4);
  const int D = 2 * radius + 2;

  auto opts = fmap1.options();
  auto corr = torch::empty({B, M, D, D, H, W}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "corr_forward_kernel", ([&] {
      corr_forward_kernel<scalar_t><<<BLOCKS(B * M * H * W * D * D), THREADS>>>(radius,
        fmap1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        fmap2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        corr.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>());
  }));

  torch::Tensor x = coords.index({Slice(), Slice(), 0, None, None});
  torch::Tensor y = coords.index({Slice(), Slice(), 1, None, None});
  torch::Tensor dx = x - x.floor(); dx = dx.to(fmap1.dtype());
  torch::Tensor dy = y - y.floor(); dy = dy.to(fmap2.dtype());

  torch::Tensor out;
  out  = (1 - dx) * (1 - dy) * corr.index({Slice(), Slice(), Slice(0, D-1), Slice(0, D-1)});
  out +=     (dx) * (1 - dy) * corr.index({Slice(), Slice(), Slice(0, D-1), Slice(1, D-0)});
  out += (1 - dx) *     (dy) * corr.index({Slice(), Slice(), Slice(1, D-0), Slice(0, D-1)});
  out +=     (dx) *     (dy) * corr.index({Slice(), Slice(), Slice(1, D-0), Slice(1, D-0)});

  return { out.permute({0,1,3,2,4,5}) };
}


std::vector<torch::Tensor> corr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor ii,
  torch::Tensor jj,
  torch::Tensor grad,
  int radius)
{
  const int B = coords.size(0);
  const int M = coords.size(1);

  const int H = coords.size(3);
  const int W = coords.size(4);
  const int D = 2 * radius + 2;
   
  grad = grad.permute({0,1,3,2,4,5}).contiguous();
  torch::Tensor x = coords.index({Slice(), Slice(), 0, None, None});
  torch::Tensor y = coords.index({Slice(), Slice(), 1, None, None});
  torch::Tensor dx = x - x.floor();
  torch::Tensor dy = y - y.floor();

  auto opts = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
  torch::Tensor g1 = torch::zeros({B, M, D, D, H, W}, grad.options());
  torch::Tensor g2 = torch::zeros({B, M, D, D, H, W}, grad.options());
  torch::Tensor g3 = torch::zeros({B, M, D, D, H, W}, grad.options());
  torch::Tensor g4 = torch::zeros({B, M, D, D, H, W}, grad.options());
  
  g1.index_put_({Slice(), Slice(), Slice(0, D-1), Slice(0, D-1)}, (1 - dx) * (1 - dy) * grad);
  g2.index_put_({Slice(), Slice(), Slice(0, D-1), Slice(1, D-0)},     (dx) * (1 - dy) * grad); 
  g3.index_put_({Slice(), Slice(), Slice(1, D-0), Slice(0, D-1)}, (1 - dx) *     (dy) * grad);
  g4.index_put_({Slice(), Slice(), Slice(1, D-0), Slice(1, D-0)},     (dx) *     (dy) * grad);

  torch::Tensor corr_grad = g1 + g2 + g3 + g4;
  auto fmap1_grad = torch::zeros_like(fmap1);
  auto fmap2_grad = torch::zeros_like(fmap2);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "corr_backward_kernel", ([&] {
    corr_backward_kernel<scalar_t><<<BLOCKS(B * M * H * W * D * D), THREADS>>>(radius,
      fmap1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      fmap2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      corr_grad.packed_accessor32<float,6,torch::RestrictPtrTraits>(),
      fmap1_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      fmap2_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));

  return {fmap1_grad, fmap2_grad};
}

std::vector<torch::Tensor> patchify_cuda_forward(
  torch::Tensor net, torch::Tensor coords, int radius)
{
  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = net.size(1);
  const int D = 2 * radius + 2;

  auto opts = net.options();
  auto patches = torch::zeros({B, M, C, D, D}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(net.scalar_type(), "patchify_forward_kernel", ([&] {
      patchify_forward_kernel<scalar_t><<<BLOCKS(B * M * D * D), THREADS>>>(radius,
        net.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));

  return { patches };
}


std::vector<torch::Tensor> patchify_cuda_backward(
  torch::Tensor net,
  torch::Tensor coords,
  torch::Tensor gradient,
  int radius)
{
  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = net.size(1);
  const int H = net.size(2);
  const int W = net.size(3);
  const int D = 2 * radius + 2;
  
  torch::Tensor net_gradient = torch::zeros_like(net);
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(net.scalar_type(), "patchify_backward_kernel", ([&] {
    patchify_backward_kernel<scalar_t><<<BLOCKS(B * M * D * D), THREADS>>>(radius,
      gradient.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      net_gradient.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));

  return { net_gradient };
}