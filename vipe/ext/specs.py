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
from pathlib import Path


def get_sources() -> list[str]:
    csrc_path = Path(__file__).parent.parent.parent / "csrc"
    return [str(p) for p in csrc_path.glob("**/*") if p.suffix in [".cpp", ".cu"]]


def _additional_include_flags() -> list[str]:
    if "CONDA_PREFIX" in os.environ:
        conda_prefix = Path(os.environ["CONDA_PREFIX"])
        include_paths = [
            conda_prefix / "include",
            conda_prefix / "nvvm" / "include",
            *sorted((conda_prefix / "targets").glob("*/include")),
        ]
        flags: list[str] = []
        for include_path in include_paths:
            if include_path.exists():
                flags += ["-isystem", str(include_path)]
        return flags
    return []


def get_cpp_flags() -> list[str]:
    return ["-O3", "-DWITH_CUDA"] + _additional_include_flags()


def get_cuda_flags() -> list[str]:
    return ["-O3", "-DWITH_CUDA", "--use_fast_math"] + _additional_include_flags()
