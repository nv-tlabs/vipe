/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>

#include "cuda_kdtree.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")

std::vector<torch::Tensor> nearestNeighbours(torch::Tensor query, torch::Tensor tree, int knn) {
    CHECK_CUDA(query);
    CHECK_IS_FLOAT(query);
    CHECK_CUDA(tree);
    CHECK_IS_FLOAT(tree);
    TORCH_CHECK(tree.size(0) >= knn, "knn is too small compared to the size of point cloud!");

    torch::Tensor strided_tree = tree;
    torch::Tensor strided_query = query;
    torch::Device device = tree.device();
    // Use int64_t (not `long`) — on Windows MSVC `long` is 32-bit which
    // would silently truncate point-cloud sizes ≥ 2^31.  GCC/Linux `long`
    // is 64-bit so this is a no-op there.
    int64_t n_queries = query.size(0);
    int64_t n_tree = tree.size(0);

    if (strided_tree.stride(0) != 4) {
        strided_tree = torch::zeros({n_tree, 4}, torch::dtype(torch::kFloat32).device(device));
        strided_tree.index_put_(
            {torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, tree.size(1))}, tree);
    }

    if (strided_query.stride(0) != 3) {
        strided_query = torch::zeros({n_queries, 3}, torch::dtype(torch::kFloat32).device(device));
        strided_query.index_put_(
            {torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, query.size(1))}, query);
    }

    auto *knn_index = new tinyflann::KDTreeCuda3dIndex<tinyflann::CudaL2>(strided_tree.data_ptr<float>(), n_tree);
    knn_index->buildIndex();
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    torch::Tensor dist = torch::empty(n_queries * knn, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor indices = torch::empty(n_queries * knn, torch::dtype(torch::kInt32).device(device));

    knn_index->knnSearch(strided_query.data_ptr<float>(), n_queries, 3, indices.data_ptr<int>(), dist.data_ptr<float>(),
                         knn);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    delete knn_index;

    return {dist.reshape({n_queries, knn}), indices.reshape({n_queries, knn})};
}
