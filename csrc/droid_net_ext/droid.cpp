/**
 * This file includes code originally from the DROID-SLAM repository:
 * https://github.com/princeton-vl/DROID-SLAM
 * Licensed under the BSD-3 License. See THIRD_PARTY_LICENSES.md for details.
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> corr_index_cuda_forward(torch::Tensor volume, torch::Tensor coords, int radius);
std::vector<torch::Tensor> corr_index_cuda_backward(torch::Tensor volume, torch::Tensor coords, torch::Tensor corr_grad,
                                                    int radius);
std::vector<torch::Tensor> altcorr_cuda_forward(torch::Tensor fmap1, torch::Tensor fmap2, torch::Tensor coords,
                                                int radius);
std::vector<torch::Tensor> altcorr_cuda_backward(torch::Tensor fmap1, torch::Tensor fmap2, torch::Tensor coords,
                                                 torch::Tensor corr_grad, int radius);
std::vector<torch::Tensor> altcorr_index_cuda_forward(torch::Tensor fmap1, std::vector<torch::Tensor> fmap2_pyramid,
                                                      torch::Tensor coords, torch::Tensor ii, torch::Tensor jj,
                                                      int radius);

// c++ python binding
std::vector<torch::Tensor> corr_index_forward(torch::Tensor volume, torch::Tensor coords, int radius) {
    CHECK_INPUT(volume);
    CHECK_INPUT(coords);

    return corr_index_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> corr_index_backward(torch::Tensor volume, torch::Tensor coords, torch::Tensor corr_grad,
                                               int radius) {
    CHECK_INPUT(volume);
    CHECK_INPUT(coords);
    CHECK_INPUT(corr_grad);

    auto volume_grad = corr_index_cuda_backward(volume, coords, corr_grad, radius);
    return {volume_grad};
}

std::vector<torch::Tensor> altcorr_forward(torch::Tensor fmap1, torch::Tensor fmap2, torch::Tensor coords, int radius) {
    CHECK_INPUT(fmap1);
    CHECK_INPUT(fmap2);
    CHECK_INPUT(coords);

    return altcorr_cuda_forward(fmap1, fmap2, coords, radius);
}

std::vector<torch::Tensor> altcorr_index_forward(torch::Tensor fmap1, std::vector<torch::Tensor> fmap2_pyramid,
                                                 torch::Tensor coords, torch::Tensor ii, torch::Tensor jj,
                                                 int radius) {
    CHECK_INPUT(fmap1);
    CHECK_INPUT(coords);
    CHECK_INPUT(ii);
    CHECK_INPUT(jj);
    for (const auto& fmap2 : fmap2_pyramid) {
        CHECK_INPUT(fmap2);
    }

    return altcorr_index_cuda_forward(fmap1, fmap2_pyramid, coords, ii, jj, radius);
}

std::vector<torch::Tensor> altcorr_backward(torch::Tensor fmap1, torch::Tensor fmap2, torch::Tensor coords,
                                            torch::Tensor corr_grad, int radius) {
    CHECK_INPUT(fmap1);
    CHECK_INPUT(fmap2);
    CHECK_INPUT(coords);
    CHECK_INPUT(corr_grad);

    return altcorr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
}

void pybind_droid_net_ext(py::module& m) {
    // correlation volume kernels
    m.def("altcorr_forward", &altcorr_forward, "ALTCORR forward");
    m.def("altcorr_index_forward", &altcorr_index_forward, "ALTCORR indexed forward");
    m.def("altcorr_backward", &altcorr_backward, "ALTCORR backward");
    m.def("corr_index_forward", &corr_index_forward, "INDEX forward");
    m.def("corr_index_backward", &corr_index_backward, "INDEX backward");
}
