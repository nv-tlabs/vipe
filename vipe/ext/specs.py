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
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"


def _csrc_path() -> Path:
    return Path(__file__).parent.parent.parent / "csrc"


def get_sources() -> list[str]:
    csrc_path = _csrc_path()
    return [str(p) for p in csrc_path.glob("**/*") if p.suffix in [".cpp", ".cu"]]


def _eigen_include_flags() -> list[str]:
    if os.environ.get("USE_SYSTEM_EIGEN", "0") == "1":
        return []

    include_path = _csrc_path() / "include"
    eigen_path = include_path / "eigen3" / "Eigen"
    if not eigen_path.exists():
        eigen_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_tar_path = Path(temp_dir) / "eigen.tar.gz"
            extracted_dir = Path(temp_dir) / "eigen-extracted"
            urlretrieve(EIGEN_URL, tmp_tar_path)
            with tarfile.open(tmp_tar_path, "r:gz") as tar:
                tar.extractall(path=extracted_dir)
            shutil.move(str(extracted_dir / "eigen-3.4.0" / "Eigen"), eigen_path)

    return ["-isystem", str(include_path.resolve())]


def _additional_include_flags() -> list[str]:
    flags = _eigen_include_flags()
    if "CONDA_PREFIX" in os.environ:
        conda_prefix = Path(os.environ["CONDA_PREFIX"])
        include_paths = [
            conda_prefix / "include",
            conda_prefix / "nvvm" / "include",
            *sorted((conda_prefix / "targets").glob("*/include")),
        ]
        for include_path in include_paths:
            if include_path.exists():
                flags += ["-isystem", str(include_path)]
    return flags


def get_cpp_flags() -> list[str]:
    return ["-O3", "-DWITH_CUDA"] + _additional_include_flags()


def get_cuda_flags() -> list[str]:
    return ["-O3", "-DWITH_CUDA", "--use_fast_math"] + _additional_include_flags()
