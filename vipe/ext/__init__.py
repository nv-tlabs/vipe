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

import os

from torch.utils.cpp_extension import load

from vipe.ext.specs import get_cpp_flags, get_cuda_flags, get_sources

try:
    import vipe_ext as _C

    vipe_ext_not_found = False
except ImportError:
    vipe_ext_not_found = True

if vipe_ext_not_found or os.environ.get("VIPE_EXT_JIT", "0") == "1":
    _C = load(
        name="vipe_ext_jit",
        sources=get_sources(),
        extra_cflags=get_cpp_flags(),
        extra_cuda_cflags=get_cuda_flags(),
        verbose=True,
    )

# Reference to submodules
droid_net_ext = _C.droid_net_ext
grounding_dino_ext = _C.grounding_dino_ext
utils_ext = _C.utils_ext
slam_ext = _C.slam_ext
scatter_ext = _C.scatter_ext
lietorch_ext = _C.lietorch_ext
corr_ext = _C.corr_ext
