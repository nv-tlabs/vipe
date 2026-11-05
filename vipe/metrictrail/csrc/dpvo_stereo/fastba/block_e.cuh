// SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
// SPDX-License-Identifier: MIT
//
// Source: https://github.com/princeton-vl/DPVO
// See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

const auto mdtype = torch::dtype(torch::kFloat32).device(torch::kCUDA);

typedef float mtype;

class EfficentE
{
private:
    torch::Tensor block_index_tensor, index_tensor, patch_to_ku;
    const int t0;

public:
    const int ppf;
    torch::Tensor E_lookup, ij_xself;

    EfficentE(const torch::Tensor &ii, const torch::Tensor &jj, const torch::Tensor &ku, const int patches_per_frame, const int t0);

    EfficentE();

    torch::Tensor computeEQEt(const int N, const torch::Tensor &Q) const;
    torch::Tensor computeEv(const int N, const torch::Tensor &vec) const;
    torch::Tensor computeEtv(const int M, const torch::Tensor &vec) const;
};