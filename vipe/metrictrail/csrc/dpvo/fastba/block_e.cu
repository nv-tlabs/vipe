// SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
// SPDX-License-Identifier: MIT
//
// Source: https://github.com/princeton-vl/DPVO
// See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

#include <torch/extension.h>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <Eigen/Core>
#include "block_e.cuh"

#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

#define NUM_THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + NUM_THREADS - 1) / NUM_THREADS)

inline void release_assert(const char *file, int line, bool condition, const std::string &msg)
{
  if (!condition)
  {
    std::cout << (std::string("Assertion failed: ") + file + " (" + std::to_string(line) + ")\n" + msg + "\n") << std::endl;
    exit(1);
  }
}

#define RASSERT(c) release_assert(__FILE__, __LINE__, c, "<no-message>")
#define MRASSERT(c, m) release_assert(__FILE__, __LINE__, c, m)

#define CREATE_IDX_ACC(t, d)              \
  const auto cpu_##t = t.to(torch::kCPU); \
  const auto acc_##t = cpu_##t.accessor<long, d>();

typedef Eigen::Array<long, -1, -1> IndexLookup;

EfficentE::EfficentE() : ppf(0), t0(0) {
    E_lookup = torch::empty({0, 0, 0}, mdtype);
    ij_xself = torch::empty({2, 0}, torch::dtype(torch::kInt64).device(torch::kCUDA));
}

EfficentE::EfficentE(const torch::Tensor &ii, const torch::Tensor &jj, const torch::Tensor &ku, const int patches_per_frame, const int t0) : ppf(patches_per_frame), t0(t0)
{
  const long n_frames = std::max(ii.max().item<long>(), jj.max().item<long>()) + 1;
  const auto ij_tuple = torch::_unique(torch::cat({ii * n_frames + jj, ii * n_frames + ii}), true, true);
  torch::Tensor ij_uniq = std::get<0>(ij_tuple);

  const long E = ii.size(0);
  ij_xself = std::get<1>(ij_tuple).view({2, E});
  E_lookup = torch::zeros({ij_uniq.size(0), ppf, 6}, mdtype);

  { // Create mapping from (frame, patch) -> index in vec
    patch_to_ku = torch::full({n_frames, ppf}, -1, torch::kInt64);
    auto patch_to_ku_acc = patch_to_ku.accessor<long, 2>();
    CREATE_IDX_ACC(ku, 1)
    for (int idx = 0; idx < cpu_ku.size(0); idx++)
    {
      const long k = acc_ku[idx]; // the actual uniq value. idx is the row in Q where it was found
      // RASSERT((patch_to_ku_acc[k / ppf][k % ppf] == idx) || (patch_to_ku_acc[k / ppf][k % ppf] == -1));
      patch_to_ku_acc[k / ppf][k % ppf] = idx;
    }
  }
  patch_to_ku = patch_to_ku.to(torch::kCUDA);

  { // Create mapping from (i,j) -> E_lookup
    IndexLookup frame_to_idx = IndexLookup::Constant(n_frames, n_frames, -1);
    CREATE_IDX_ACC(ii, 1)
    CREATE_IDX_ACC(jj, 1)
    CREATE_IDX_ACC(ij_xself, 2)

    for (int idx = 0; idx < E; idx++)
    {
      const long i = acc_ii[idx];
      const long j = acc_jj[idx];
      const long ijx = acc_ij_xself[0][idx];
      const long ijs = acc_ij_xself[1][idx];
      // RASSERT((frame_to_idx(i, j) == ijx) || (frame_to_idx(i, j) == -1));
      // RASSERT((frame_to_idx(i, i) == ijs) || (frame_to_idx(i, i) == -1));
      frame_to_idx(i, j) = ijx;
      frame_to_idx(i, i) = ijs;
    }

    // lookup table for edges
    const long E = cpu_ii.size(0);
    std::vector<std::unordered_set<long>> edge_lookup(n_frames);
    for (int x = 0; x < E; x++)
    {
      const long i = acc_ii[x];
      const long j = acc_jj[x];
      edge_lookup[i].insert(j);
      edge_lookup[i].insert(i);
      // RASSERT(j < n_frames);
      // RASSERT(i < n_frames);
      // MRASSERT(edge_lookup[i].size() < 30, "More edges than expected");
    }
    // std::cout << "#U" << std::endl;

    int count = 0;
    for (const auto &connected_frames : edge_lookup)
      count += (connected_frames.size() * connected_frames.size());

    // std::cout << "#V" << std::endl;
    index_tensor = torch::empty({count, 5}, torch::kInt64);
    auto index_tensor_acc = index_tensor.accessor<long, 2>();
    // std::cout << "#W" << std::endl;

    int cx = 0;
    for (int i = 0; i < n_frames; i++)
    {
      const auto &connected_frames = edge_lookup[i];
      for (const long &j1 : connected_frames)
      {
        for (const long &j2 : connected_frames)
        {
          index_tensor_acc[cx][0] = i;
          index_tensor_acc[cx][1] = j1;
          index_tensor_acc[cx][2] = j2;
          index_tensor_acc[cx][3] = frame_to_idx(i, j1);
          index_tensor_acc[cx][4] = frame_to_idx(i, j2);
          cx += 1;
        }
      }
    }
    index_tensor = index_tensor.to(torch::kCUDA);
    // RASSERT(cx == count);
  }

  {
    CREATE_IDX_ACC(ij_uniq, 1)
    const long count = ij_uniq.size(0);
    block_index_tensor = torch::empty({count, 2}, torch::kInt64);
    auto index_tensor_acc = block_index_tensor.accessor<long, 2>();
    for (int idx = 0; idx < count; idx++)
    {
      const long ij = acc_ij_uniq[idx];
      const long i = ij / n_frames;
      const long j = ij % n_frames;

      index_tensor_acc[idx][0] = i;
      index_tensor_acc[idx][1] = j;
    }
    block_index_tensor = block_index_tensor.to(torch::kCUDA);
  }
}

__global__ void EEt_kernel(
    torch::PackedTensorAccessor32<mtype, 2, torch::RestrictPtrTraits> EEt,
    const torch::PackedTensorAccessor32<mtype, 3, torch::RestrictPtrTraits> E_lookup,
    const torch::PackedTensorAccessor32<mtype, 1, torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> index_tensor,
    const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> patch_to_ku, const int t0, const int ppf)
{

  GPU_1D_KERNEL_LOOP(n, index_tensor.size(0) * ppf)
  {

    int k = n % ppf; // src patch
    int idx = n / ppf;
    int i = index_tensor[idx][0];  // src frame
    int j1 = index_tensor[idx][1]; // dest j1
    int j2 = index_tensor[idx][2]; // dest j2

    int j1_idx = index_tensor[idx][3]; // index for first slice
    int j2_idx = index_tensor[idx][4]; // index for second slice

    const auto j1_slice = E_lookup[j1_idx][k]; // 6
    const auto j2_slice = E_lookup[j2_idx][k]; // 6

    j1 = j1 - t0;
    j2 = j2 - t0;

    for (int xi = 0; xi < 6; xi++)
    {
      for (int xj = 0; xj < 6; xj++)
      {
        if ((j1 >= 0) && (j2 >= 0))
        {
          long q_idx = patch_to_ku[i][k];
          float q = Q[q_idx];
          atomicAdd(&EEt[6 * j1 + xi][6 * j2 + xj], j1_slice[xi] * j2_slice[xj] * q);
        }
      }
    }
  }
}

torch::Tensor EfficentE::computeEQEt(const int N, const torch::Tensor &Q) const
{
  torch::Tensor EEt = torch::zeros({6 * N, 6 * N}, mdtype);
  const auto tmp_Q = Q.view({-1});

  EEt_kernel<<<NUM_BLOCKS(index_tensor.size(0) * ppf), NUM_THREADS>>>(
      EEt.packed_accessor32<mtype, 2, torch::RestrictPtrTraits>(),
      E_lookup.packed_accessor32<mtype, 3, torch::RestrictPtrTraits>(),
      tmp_Q.packed_accessor32<mtype, 1, torch::RestrictPtrTraits>(),
      index_tensor.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
      patch_to_ku.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
      t0, ppf);

  return EEt;
}

__global__ void Ev_kernel(
    torch::PackedTensorAccessor32<mtype, 1, torch::RestrictPtrTraits> Ev,
    const torch::PackedTensorAccessor32<mtype, 3, torch::RestrictPtrTraits> E_lookup,
    const torch::PackedTensorAccessor32<mtype, 1, torch::RestrictPtrTraits> vec,
    const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> index_tensor,
    const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> patch_to_ku, const int t0, const int ppf)
{

  GPU_1D_KERNEL_LOOP(n, index_tensor.size(0) * ppf)
  {

    int k = n % ppf; // src patch
    int idx = n / ppf;
    int i = index_tensor[idx][0];
    int j = index_tensor[idx][1];

    auto slice = E_lookup[idx][k]; // 6
    long q_idx = patch_to_ku[i][k];
    float v = vec[q_idx];

    j = j - t0; // i not used anymore

    for (int r = 0; r < 6; r++)
    {
      if (j >= 0)
      {
        atomicAdd(&Ev[j * 6 + r], slice[r] * v);
      }
    }
  }
}

torch::Tensor EfficentE::computeEv(const int N, const torch::Tensor &vec) const
{
  torch::Tensor Ev = torch::zeros({6 * N}, mdtype);
  const auto tmp_vec = vec.view({-1});

  Ev_kernel<<<NUM_BLOCKS(E_lookup.size(0) * ppf), NUM_THREADS>>>(
      Ev.packed_accessor32<mtype, 1, torch::RestrictPtrTraits>(),
      E_lookup.packed_accessor32<mtype, 3, torch::RestrictPtrTraits>(),
      tmp_vec.packed_accessor32<mtype, 1, torch::RestrictPtrTraits>(),
      block_index_tensor.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
      patch_to_ku.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
      t0, ppf);

  Ev = Ev.view({-1, 1});
  return Ev;
}

__global__ void Etv_kernel(
    torch::PackedTensorAccessor32<mtype, 1, torch::RestrictPtrTraits> Etv,
    const torch::PackedTensorAccessor32<mtype, 3, torch::RestrictPtrTraits> E_lookup,
    const torch::PackedTensorAccessor32<mtype, 1, torch::RestrictPtrTraits> vec,
    const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> index_tensor,
    const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> patch_to_ku, const int t0, const int ppf)
{

  GPU_1D_KERNEL_LOOP(n, index_tensor.size(0) * ppf)
  {

    int k = n % ppf; // src patch
    int idx = n / ppf;
    int i = index_tensor[idx][0];
    int j = index_tensor[idx][1];

    auto slice = E_lookup[idx][k]; // 6
    long q_idx = patch_to_ku[i][k];

    j = j - t0; // i not used anymore

    for (int r = 0; r < 6; r++)
    {
      if (j >= 0)
      {
        float dp = slice[r] * vec[j * 6 + r];
        atomicAdd(&Etv[q_idx], dp);
      }
    }
  }
}

torch::Tensor EfficentE::computeEtv(const int M, const torch::Tensor &vec) const
{
  torch::Tensor Etv = torch::zeros({M}, mdtype);
  const auto tmp_vec = vec.view({-1});

  Etv_kernel<<<NUM_BLOCKS(E_lookup.size(0) * ppf), NUM_THREADS>>>(
      Etv.packed_accessor32<mtype, 1, torch::RestrictPtrTraits>(),
      E_lookup.packed_accessor32<mtype, 3, torch::RestrictPtrTraits>(),
      tmp_vec.packed_accessor32<mtype, 1, torch::RestrictPtrTraits>(),
      block_index_tensor.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
      patch_to_ku.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
      t0, ppf);

  Etv = Etv.view({-1, 1});
  return Etv;
}