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
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include "block_e.cuh"


#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x)


#define NUM_THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + NUM_THREADS - 1) / NUM_THREADS)

inline void release_assert(const char *file, int line, bool condition, const std::string &msg){
    if (!condition)
        throw std::runtime_error(std::string("Assertion failed: ") + file + " (" + std::to_string(line) + ")\n" + msg + "\n");
}

#define RASSERT(c) release_assert(__FILE__, __LINE__, c, "")
#define MRASSERT(c, m) release_assert(__FILE__, __LINE__, c, m)

void save(const char *filename, const torch::Tensor &data){
  const auto pickled = torch::pickle_save(data);
  std::ofstream fout(filename, std::ios::out | std::ios::binary);
  fout.write(pickled.data(), pickled.size());
  fout.close();
}

__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

__device__ void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

  
__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}


__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}



__global__ void pose_retr_kernel(const int t0, const int t1,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    torch::PackedTensorAccessor32<mtype,2,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(i, t1 - t0) {
    const float t = t0 + i;
    float t1[3], t0[3] = { poses[t][0], poses[t][1], poses[t][2] };
    float q1[4], q0[4] = { poses[t][3], poses[t][4], poses[t][5], poses[t][6] };

    float xi[6] = {
      update[i][0],
      update[i][1],
      update[i][2],
      update[i][3],
      update[i][4],
      update[i][5],
    };

    retrSE3(xi, t0, q0, t1, q1);

    poses[t][0] = t1[0];
    poses[t][1] = t1[1];
    poses[t][2] = t1[2];
    poses[t][3] = q1[0];
    poses[t][4] = q1[1];
    poses[t][5] = q1[2];
    poses[t][6] = q1[3];
  }
}


__global__ void patch_retr_kernel(
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> index,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(n, index.size(0)) {
    const int p = patches.size(2);
    const int ix = index[n];
  
    float d = patches[ix][2][0][0];
    d = d + update[n];
    d = (d > 20) ? 1.0 : d;
    d = max(d, 1e-4);

    for (int i=0; i<p; i++) {
      for (int j=0; j<p; j++) {
        patches[ix][2][i][j] = d;
      }
    }
  }
}


template <bool HAS_STEREO_EDGES>
__global__ void reprojection_residuals_and_hessian(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> lmbda,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> ij_xself,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ku,
    const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> edge_is_stereo,
    torch::PackedTensorAccessor32<double,1,torch::RestrictPtrTraits> r_total,
    torch::PackedTensorAccessor32<mtype,3,torch::RestrictPtrTraits> E_lookup,
    torch::PackedTensorAccessor32<mtype,2,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<mtype,2,torch::RestrictPtrTraits> E,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> u,
    const int t0, const int ppf, const float stereo_baseline_m)
{

  __shared__ float fx, fy, cx, cy;
  if (threadIdx.x == 0) {
    fx = intrinsics[0][0];
    fy = intrinsics[0][1];
    cx = intrinsics[0][2];
    cy = intrinsics[0][3];
  }

  bool eff_impl = (ppf > 0);

  __syncthreads();

  GPU_1D_KERNEL_LOOP(n, ii.size(0)) {
    int k = ku[n]; // inverse indices
    int ix = ii[n];
    int jx = jj[n];
    int kx = kk[n]; // actual
    const bool stereo_edge = HAS_STEREO_EDGES && edge_is_stereo[n];
    int ijx, ijs;
    if (eff_impl){
      ijx = ij_xself[0][n];
      ijs = ij_xself[1][n];
    }

    float ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    float tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    float qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    float qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    float Xi[4], Xj[4];
    Xi[0] = (patches[kx][0][1][1] - cx) / fx;
    Xi[1] = (patches[kx][1][1][1] - cy) / fy;
    Xi[2] = 1.0;
    Xi[3] = patches[kx][2][1][1];
    
    float tij[3], qij[4];
    if (stereo_edge) {
      // Fixed left-camera -> right-camera rig transform. The baseline is in
      // metric units and patches store metric inverse depth.
      tij[0] = -stereo_baseline_m;
      tij[1] = 0.0f;
      tij[2] = 0.0f;
      qij[0] = 0.0f;
      qij[1] = 0.0f;
      qij[2] = 0.0f;
      qij[3] = 1.0f;
    } else {
      relSE3(ti, qi, tj, qj, tij, qij);
    }
    actSE3(tij, qij, Xi, Xj);

    const float X = Xj[0];
    const float Y = Xj[1];
    const float Z = Xj[2];
    const float W = Xj[3];

    const float d = (Z >= 0.2) ? 1.0 / Z : 0.0; 
    const float d2 = d * d;

    const float x1 = fx * (X / Z) + cx;
    const float y1 = fy * (Y / Z) + cy;

    const float rx = target[n][0] - x1;
    const float ry = target[n][1] - y1;

    const bool in_bounds = (sqrt(rx*rx + ry*ry) < 128) && (Z > 0.2) &&
      (x1 > -64) && (y1 > -64) && (x1 < 2*cx + 64) && (y1 < 2*cy + 64);

    const float mask = in_bounds ? 1.0 : 0.0;

    ix = ix - t0;
    jx = jx - t0;

    for (int row=0; row<2; row++) {

      float *Jj, Ji[6], Jz, r, w;

      if (row == 0){

        r = target[n][0] - x1;
        w = mask * weight[n][0];

        Jz = fx * (tij[0] * d - tij[2] * (X * d2));
        Jj = (float[6]){fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};

      } else {

        r = target[n][1] - y1;
        w = mask * weight[n][1];

        Jz = fy * (tij[1] * d - tij[2] * (Y * d2));
        Jj = (float[6]){0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};

      }

      atomicAdd(&r_total[0], static_cast<double>(w * r * r));

      if (!stereo_edge) {
        adjSE3(tij, qij, Jj, Ji);

        for (int i=0; i<6; i++) {
          for (int j=0; j<6; j++) {
            if (ix >= 0)
              atomicAdd(&B[6*ix+i][6*ix+j],  w * Ji[i] * Ji[j]);
            if (jx >= 0)
              atomicAdd(&B[6*jx+i][6*jx+j],  w * Jj[i] * Jj[j]);
            if (ix >= 0 && jx >= 0) {
              atomicAdd(&B[6*ix+i][6*jx+j], -w * Ji[i] * Jj[j]);
              atomicAdd(&B[6*jx+i][6*ix+j], -w * Jj[i] * Ji[j]);
            }
          }
        }

        for (int i=0; i<6; i++) {
          if (eff_impl){
            atomicAdd(&E_lookup[ijs][kx % ppf][i],  -w * Jz * Ji[i]);
            atomicAdd(&E_lookup[ijx][kx % ppf][i],  w * Jz * Jj[i]);
          } else {
            if (ix >= 0)
              atomicAdd(&E[6*ix+i][k], -w * Jz * Ji[i]);
            if (jx >= 0)
              atomicAdd(&E[6*jx+i][k],  w * Jz * Jj[i]);
          }
        }

        for (int i=0; i<6; i++) {
          if (ix >= 0)
            atomicAdd(&v[6*ix+i], -w * r * Ji[i]);
          if (jx >= 0)
            atomicAdd(&v[6*jx+i],  w * r * Jj[i]);
        }
      }

      atomicAdd(&C[k], w * Jz * Jz);
      atomicAdd(&u[k], w *  r * Jz);
    }
  }
}


__global__ void metric_depth_residuals_and_hessian(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> depth_prior,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> depth_prior_weight,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kx,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> u,
    const float prior_lambda,
    const float huber_delta)
{
  GPU_1D_KERNEL_LOOP(n, kx.size(0)) {
    const int patch_index = kx[n];
    if (patch_index >= depth_prior.size(0))
      continue;

    const float base_weight = depth_prior_weight[patch_index];
    const float prior = depth_prior[patch_index];
    const float current = patches[patch_index][2][1][1];
    if (!(base_weight > 0.0f) || !(prior > 0.0f) || !(current > 0.0f) ||
        !isfinite(prior) || !isfinite(current) || !(prior_lambda > 0.0f))
      continue;

    // For r = log(d_prior) - log(d), dr/dd = -1/d. The normal-equation
    // right-hand side is -J^T r = r/d, so the positive 1/d factor below is
    // the RHS sensitivity (while J^T J is unchanged by the sign).
    const float residual = logf(prior) - logf(current);
    const float abs_residual = fabsf(residual);
    const float robust = (abs_residual <= huber_delta)
        ? 1.0f
        : huber_delta / max(abs_residual, 1e-6f);
    const float w = prior_lambda * base_weight * robust;
    const float J = 1.0f / max(current, 1e-4f);
    atomicAdd(&C[n], w * J * J);
    atomicAdd(&u[n], w * residual * J);
  }
}


template <bool HAS_STEREO_EDGES>
__global__ void reproject(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> edge_is_stereo,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const float stereo_baseline_m) {

  __shared__ float fx, fy, cx, cy;
  if (threadIdx.x == 0) {
    fx = intrinsics[0][0];
    fy = intrinsics[0][1];
    cx = intrinsics[0][2];
    cy = intrinsics[0][3];
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(n, ii.size(0)) {
    int ix = ii[n];
    int jx = jj[n];
    int kx = kk[n];
    const bool stereo_edge = HAS_STEREO_EDGES && edge_is_stereo[n];

    float ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    float tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    float qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    float qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    float tij[3], qij[4];
    if (stereo_edge) {
      tij[0] = -stereo_baseline_m;
      tij[1] = 0.0f;
      tij[2] = 0.0f;
      qij[0] = 0.0f;
      qij[1] = 0.0f;
      qij[2] = 0.0f;
      qij[3] = 1.0f;
    } else {
      relSE3(ti, qi, tj, qj, tij, qij);
    }

    float Xi[4], Xj[4];
    for (int i=0; i<patches.size(2); i++) {
      for (int j=0; j<patches.size(3); j++) {
        
        Xi[0] = (patches[kx][0][i][j] - cx) / fx;
        Xi[1] = (patches[kx][1][i][j] - cy) / fy;
        Xi[2] = 1.0;
        Xi[3] = patches[kx][2][i][j];

        actSE3(tij, qij, Xi, Xj);

        coords[n][0][i][j] = fx * (Xj[0] / Xj[2]) + cx;
        coords[n][1][i][j] = fy * (Xj[1] / Xj[2]) + cy;
        // coords[n][2][i][j] = 1.0 / Xj[2];

      }
    }
  }
}


bool prepare_stereo_edges(
    torch::Tensor &edge_is_stereo,
    const torch::Tensor &reference,
    const int64_t edge_count,
    const double stereo_baseline_m)
{
  RASSERT(std::isfinite(stereo_baseline_m));
  RASSERT(stereo_baseline_m >= 0.0);

  if (!edge_is_stereo.defined() || edge_is_stereo.numel() == 0) {
    edge_is_stereo = torch::empty(
        {0}, torch::dtype(torch::kBool).device(reference.device()));
    return false;
  }

  RASSERT(edge_is_stereo.scalar_type() == torch::kBool);
  RASSERT(edge_is_stereo.device() == reference.device());
  RASSERT(edge_is_stereo.dim() == 1);
  edge_is_stereo = edge_is_stereo.contiguous();
  RASSERT(edge_is_stereo.size(0) == edge_count);

  // A positive baseline is the host-side signal that the supplied mask may
  // contain stereo edges. This avoids synchronizing the CUDA stream just to
  // reduce the mask on every stereo BA iteration. An all-false mask remains
  // mathematically monocular even when it uses the runtime-mask kernel.
  if (stereo_baseline_m > 0.0)
    return true;

  // For baseline==0, validate that callers did not silently request an
  // invalid zero-baseline stereo factor, then collapse to the exact mono
  // template path.
  RASSERT(!edge_is_stereo.any().item<bool>());
  edge_is_stereo = torch::empty(
      {0}, torch::dtype(torch::kBool).device(reference.device()));
  return false;
}



std::vector<torch::Tensor> cuda_ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor depth_prior,
    torch::Tensor depth_prior_weight,
    double depth_prior_lambda,
    double depth_prior_huber,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk,
    const int PPF,
    const int t0, const int t1, const int iterations, bool eff_impl,
    torch::Tensor edge_is_stereo,
    const double stereo_baseline_m)
{

  const int num = ii.size(0);
  const bool has_stereo_edges = prepare_stereo_edges(
      edge_is_stereo, ii, num, stereo_baseline_m);

  auto ktuple = torch::_unique(kk, true, true);
  torch::Tensor kx = std::get<0>(ktuple);
  torch::Tensor ku = std::get<1>(ktuple);

  const int N = t1 - t0;    // number of poses
  const int M = kx.size(0); // number of patches
  const int P = patches.size(3); // patch size

  // auto opts = torch::TensorOptions()
  //   .dtype(torch::kFloat32).device(torch::kCUDA);

  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  target = target.view({-1, 2});
  weight = weight.view({-1, 2});
  depth_prior = depth_prior.view({-1});
  depth_prior_weight = depth_prior_weight.view({-1});

  RASSERT(depth_prior.size(0) == depth_prior_weight.size(0));
  RASSERT(depth_prior.size(0) == 0 || depth_prior.size(0) == patches.size(0));
  RASSERT(depth_prior_lambda >= 0.0);
  RASSERT(depth_prior_huber > 0.0);

  torch::Tensor B = torch::empty({6*N, 6*N}, mdtype);
  torch::Tensor E = torch::empty({0, 0}, mdtype);
  torch::Tensor C = torch::empty({M}, mdtype);

  torch::Tensor v = torch::empty({6*N}, mdtype);
  torch::Tensor u = torch::empty({1*M}, mdtype);

  torch::Tensor r_total = torch::empty({1}, torch::dtype(torch::kFloat64).device(torch::kCUDA));

  auto blockE = std::make_unique<EfficentE>();

  if (eff_impl)
    blockE = std::make_unique<EfficentE>(ii, jj, kx, PPF, t0);
  else
    E = torch::empty({6*N, 1*M}, mdtype);

  for (int itr=0; itr < iterations; itr++) {

    B.zero_();
    E.zero_();
    C.zero_();
    v.zero_();
    u.zero_();
    r_total.zero_();
    blockE->E_lookup.zero_();

    v = v.view({6*N});
    u = u.view({1*M});

#define LAUNCH_REPROJECTION_HESSIAN(HAS_STEREO) \
    reprojection_residuals_and_hessian<HAS_STEREO><<<NUM_BLOCKS(ii.size(0)), NUM_THREADS>>>( \
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(), \
        patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(), \
        intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(), \
        target.packed_accessor32<float,2,torch::RestrictPtrTraits>(), \
        weight.packed_accessor32<float,2,torch::RestrictPtrTraits>(), \
        lmbda.packed_accessor32<float,1,torch::RestrictPtrTraits>(), \
        blockE->ij_xself.packed_accessor32<long,2,torch::RestrictPtrTraits>(), \
        ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(), \
        jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(), \
        kk.packed_accessor32<long,1,torch::RestrictPtrTraits>(), \
        ku.packed_accessor32<long,1,torch::RestrictPtrTraits>(), \
        edge_is_stereo.packed_accessor32<bool,1,torch::RestrictPtrTraits>(), \
        r_total.packed_accessor32<double,1,torch::RestrictPtrTraits>(), \
        blockE->E_lookup.packed_accessor32<mtype,3,torch::RestrictPtrTraits>(), \
        B.packed_accessor32<mtype,2,torch::RestrictPtrTraits>(), \
        E.packed_accessor32<mtype,2,torch::RestrictPtrTraits>(), \
        C.packed_accessor32<mtype,1,torch::RestrictPtrTraits>(), \
        v.packed_accessor32<mtype,1,torch::RestrictPtrTraits>(), \
        u.packed_accessor32<mtype,1,torch::RestrictPtrTraits>(), t0, blockE->ppf, \
        static_cast<float>(stereo_baseline_m))

    if (has_stereo_edges)
      LAUNCH_REPROJECTION_HESSIAN(true);
    else
      LAUNCH_REPROJECTION_HESSIAN(false);

#undef LAUNCH_REPROJECTION_HESSIAN

    if (depth_prior_lambda > 0.0 && depth_prior.size(0) > 0 && M > 0) {
      metric_depth_residuals_and_hessian<<<NUM_BLOCKS(M), NUM_THREADS>>>(
        patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        depth_prior.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        depth_prior_weight.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        C.packed_accessor32<mtype,1,torch::RestrictPtrTraits>(),
        u.packed_accessor32<mtype,1,torch::RestrictPtrTraits>(),
        static_cast<float>(depth_prior_lambda),
        static_cast<float>(depth_prior_huber));
    }

    // std::cout << "Total residuals: " << r_total.item<double>() << std::endl;
    v = v.view({6*N, 1});
    u = u.view({1*M, 1});

    torch::Tensor Q = 1.0 / (C + lmbda).view({1, M});

    if (t1 - t0 == 0) {

      torch::Tensor Qt = torch::transpose(Q, 0, 1);
      torch::Tensor dZ = Qt * u;

      dZ = dZ.view({M});

      patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS>>>(
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        dZ.packed_accessor32<mtype,1,torch::RestrictPtrTraits>());

    }  else {

      torch::Tensor dX, dZ, Qt = torch::transpose(Q, 0, 1);
      torch::Tensor I = torch::eye(6*N, mdtype);

      if (eff_impl) {

        torch::Tensor EQEt = blockE->computeEQEt(N, Q);
        torch::Tensor EQu = blockE->computeEv(N, Qt * u);

        torch::Tensor S = B - EQEt;
        torch::Tensor y = v - EQu;

        S += I * (1e-4 * S + 1.0);
        torch::Tensor U = std::get<0>(at::linalg_cholesky_ex(S));
        dX = torch::cholesky_solve(y, U);
        torch::Tensor EtdX = blockE->computeEtv(M, dX);
        dZ = Qt * (u - EtdX);

      } else {

        torch::Tensor EQ = E * Q;
        torch::Tensor Et = torch::transpose(E, 0, 1);

        torch::Tensor S = B - torch::matmul(EQ, Et);
        torch::Tensor y = v - torch::matmul(EQ,  u);

        S += I * (1e-4 * S + 1.0);
        torch::Tensor U = std::get<0>(at::linalg_cholesky_ex(S));
        dX = torch::cholesky_solve(y, U);
        dZ = Qt * (u - torch::matmul(Et, dX));

      }

      dX = dX.view({N, 6});
      dZ = dZ.view({M});

      pose_retr_kernel<<<NUM_BLOCKS(N), NUM_THREADS>>>(t0, t1,
          poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          dX.packed_accessor32<mtype,2,torch::RestrictPtrTraits>());

      patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS>>>(
          kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
          patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
          dZ.packed_accessor32<mtype,1,torch::RestrictPtrTraits>());
    }
  }
  
  return {};
}


torch::Tensor cuda_reproject(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj, 
    torch::Tensor kk,
    torch::Tensor edge_is_stereo,
    const double stereo_baseline_m)
{

  const int N = ii.size(0);
  const bool has_stereo_edges = prepare_stereo_edges(
      edge_is_stereo, ii, N, stereo_baseline_m);
  const int P = patches.size(3); // patch size

  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  auto opts = torch::TensorOptions()
    .dtype(torch::kFloat32).device(torch::kCUDA);

  torch::Tensor coords = torch::empty({N, 2, P, P}, opts);

#define LAUNCH_REPROJECT(HAS_STEREO) \
  reproject<HAS_STEREO><<<NUM_BLOCKS(N), NUM_THREADS>>>( \
      poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(), \
      patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(), \
      intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(), \
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(), \
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(), \
      kk.packed_accessor32<long,1,torch::RestrictPtrTraits>(), \
      edge_is_stereo.packed_accessor32<bool,1,torch::RestrictPtrTraits>(), \
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(), \
      static_cast<float>(stereo_baseline_m))

  if (has_stereo_edges)
    LAUNCH_REPROJECT(true);
  else
    LAUNCH_REPROJECT(false);

#undef LAUNCH_REPROJECT

  return coords.view({1, N, 2, P, P});

}
