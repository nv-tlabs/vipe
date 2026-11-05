// SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
// SPDX-License-Identifier: MIT
//
// Source: https://github.com/princeton-vl/DPVO
// See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> corr_cuda_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor ii,
    torch::Tensor jj,
    int radius);

std::vector<torch::Tensor> corr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor ii,
  torch::Tensor jj,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> patchify_cuda_forward(
    torch::Tensor net, torch::Tensor coords, int radius);

std::vector<torch::Tensor> patchify_cuda_backward(
    torch::Tensor net, torch::Tensor coords, torch::Tensor gradient, int radius);

std::vector<torch::Tensor> corr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor ii,
    torch::Tensor jj, int radius) {
  return corr_cuda_forward(fmap1, fmap2, coords, ii, jj, radius);
}

std::vector<torch::Tensor> corr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor corr_grad, int radius) {
  return corr_cuda_backward(fmap1, fmap2, coords, ii, jj, corr_grad, radius);
}

std::vector<torch::Tensor> patchify_forward(
    torch::Tensor net, torch::Tensor coords, int radius) {
  return patchify_cuda_forward(net, coords, radius);
}

std::vector<torch::Tensor> patchify_backward(
    torch::Tensor net, torch::Tensor coords, torch::Tensor gradient, int radius) {
  return patchify_cuda_backward(net, coords, gradient, radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &corr_forward, "CORR forward");
  m.def("backward", &corr_backward, "CORR backward");

  m.def("patchify_forward", &patchify_forward, "PATCHIFY forward");
  m.def("patchify_backward", &patchify_backward, "PATCHIFY backward");
}