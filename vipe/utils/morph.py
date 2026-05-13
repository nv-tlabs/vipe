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


import torch


class MorphOp:
    """Morphological dilation / erosion operator for 4D input tensors of shape (B,C,H,W) with variable output channel count (single morphological operation and step and square kernels only).

    Makes use of unfolded tensors to replace kernel convolution with matrix multiplication

    Ref: [1] Tensorflow reference implementation: "Morphological Networks for Image De-raining (Ranjan Mondal et al. (2019))" https://github.com/ranjanZ/2D-Morphological-Network
         [2] Vector-based pytorch implementation: "Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019))" https://github.com/jlebensold/iclr_2019_buffalo-3
         [3] Torch-model-based implementation: pytorch morphological dilation2d and erosion2d https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d
    """

    def __init__(
        self,
        c_out: int,
        type_str: str,
        device: torch.device,
        kernel_size: int = 5,
        use_soft_max: bool = False,
        soft_max_beta=20,
    ):
        """
        c_out: the number of target output channels after applying the operation [int]
        type: either "dilation2d" or "erosion2d" [str]
        kernel_size: the spatial size of the morphological operation [int]
        use_soft_max: using the soft max rather the torch.max(), ref: [2] [bool]
        soft_max_beta: used by soft_max [float]
        """
        self.c_out = c_out
        self.type_str = type_str

        self.kernel_size = kernel_size
        self.use_soft_max = use_soft_max
        self.soft_max_beta = soft_max_beta

        assert self.type_str in [
            "dilation2d",
            "erosion2d",
        ], f"MorphOp: invalid type {self.type_str}"

        # Unfold operator to replace convolution with matrix multiplication
        self.unfold = torch.nn.Unfold(kernel_size).to(device)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply morphological operation on input tensor of shape (B,C,H,W)
        """

        # add padding to inputs depending on kernel sizes (pad last two H/W dimensions of input)
        h, w = input.shape[-2:]

        pad_N = self.kernel_size - 1
        pad_start = pad_N // 2
        pad_end = pad_N - pad_start
        x = torch.nn.functional.pad(input, (pad_start, pad_end, pad_start, pad_end), mode="replicate")

        # perform unfold to apply morphological operation via
        # simple patch-based operation instead of convolution
        x = self.unfold(x).unsqueeze(1)  # (B, 1, Cin*k, L), with number of patches L

        # apply actual morphological operation
        if self.type_str == "erosion2d":
            x = -x  # (B, Cout, Cin*k, L)

        # combine internal dimensions to output channel number
        if self.use_soft_max:
            x = torch.logsumexp(x * self.soft_max_beta, dim=2, keepdim=False) / self.soft_max_beta  # (B, Cout, L)
        else:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)

        if self.type_str == "erosion2d":
            x = -1 * x  # (B, Cout, L)

        # use view instead of fold to avoid creating a copy
        return x.view(-1, self.c_out, h, w)  # (B, Cout, sqrt(L), sqrt(L))


def dilate(input: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Dilate the input tensor using a square kernel of size kernel_size
    input and output are [(B), H, W], either float 0-1 or bool
    """
    batch_dim = input.dim() == 3
    if not batch_dim:
        input = input.unsqueeze(0)

    is_bool = input.dtype == torch.bool
    if is_bool:
        input = input.float()

    op = MorphOp(1, "dilation2d", input.device, kernel_size)
    res = op(input.unsqueeze(1)).squeeze(1)

    res = (res > 0.5) if is_bool else res

    return res if batch_dim else res.squeeze(0)


def erode(input: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Erode the input tensor using a square kernel of size kernel_size
    input and output are [(B), H, W], either float 0-1 or bool
    """
    batch_dim = input.dim() == 3
    if not batch_dim:
        input = input.unsqueeze(0)

    is_bool = input.dtype == torch.bool
    if is_bool:
        input = input.float()

    op = MorphOp(1, "erosion2d", input.device, kernel_size)
    res = op(input.unsqueeze(1)).squeeze(1)

    res = (res > 0.5) if is_bool else res

    return res if batch_dim else res.squeeze(0)
