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
import platform
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"

# Windows MSVC (cl.exe) and the host compiler used for nvcc-on-Windows do not
# understand GCC-style flags like ``-O3`` or ``-isystem``.  Detect once at
# import time so the get_*_flags helpers can return the correct dialect.
_IS_WINDOWS = platform.system() == "Windows"


def _csrc_path() -> Path:
    return Path(__file__).parent.parent.parent / "csrc"


def get_sources() -> list[str]:
    csrc_path = _csrc_path()
    return [str(p) for p in csrc_path.glob("**/*") if p.suffix in [".cpp", ".cu"]]


def _include_flag_for_host(path: str) -> list[str]:
    """Emit an include-path flag accepted by both the host compiler and nvcc.

    Why a single helper for both: ``_additional_include_flags()`` is consumed
    by ``get_cpp_flags()`` (cl.exe / g++ direct invocation) **and** by
    ``get_cuda_flags()`` (nvcc), so the same flag string is passed to both.
    nvcc does not understand MSVC's slash-prefixed ``/I`` form — it raises
    ``nvcc fatal: A single input file is required ...`` because it sees the
    ``/I<path>`` as an extra positional argument.

    Both MSVC ``cl.exe`` (modern versions, ≥ VS2017) and nvcc accept the
    GCC-style ``-I<path>`` form (no space).  So we emit that everywhere
    instead of branching on OS.  We keep the helper for symmetry with the
    earlier ``-isystem`` form (which we drop on Windows because cl.exe
    treats it as an unknown option and silently demotes the next arg to
    "unrecognised source file").
    """
    if _IS_WINDOWS:
        # `-I<path>` with no space — accepted by both cl.exe and nvcc on Win.
        return [f"-I{path}"]
    # Linux / macOS: keep `-isystem` so Eigen template noise doesn't pollute
    # the build log.  Both GCC and Clang understand this on POSIX hosts; nvcc
    # forwards it to the host compiler unchanged.
    return ["-isystem", path]


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

    return _include_flag_for_host(str(include_path.resolve()))


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
                flags += _include_flag_for_host(str(include_path))
    return flags


def get_cpp_flags() -> list[str]:
    # MSVC's cl.exe doesn't recognise ``-O3`` (issues D9002 warning then
    # ignores it).  Use ``/O2`` (max speed) instead on Windows; GCC/Clang
    # keep ``-O3`` on POSIX.
    opt = "/O2" if _IS_WINDOWS else "-O3"
    flags = [opt, "-DWITH_CUDA"]
    # Windows: optionally emit PDB symbols so post-mortem crash analyzers
    # (cdb / WinDbg) can resolve module offsets to source file + line.
    # ``/Zi`` writes full debug info to a separate .pdb file (paired with
    # ``/DEBUG`` in setup.py's extra_link_args for the final consolidated
    # PDB next to vipe_ext.pyd).
    # ``/FS`` allows multiple cl.exe processes to write to the same PDB
    # serially (needed since ninja parallelises the build).
    # Opt-in via ``VIPE_DEBUG_SYMBOLS=1`` in environment (off by default).
    if _IS_WINDOWS and os.environ.get("VIPE_DEBUG_SYMBOLS", "0") == "1":
        flags.extend(["/Zi", "/FS"])
    return flags + _additional_include_flags()


def get_cuda_flags() -> list[str]:
    # nvcc accepts ``-O3`` everywhere — it forwards host-specific flags via
    # ``-Xcompiler`` automatically.  We leave the nvcc opt level alone.
    flags = ["-O3", "-DWITH_CUDA", "--use_fast_math"]
    # Windows: mirror the host-side ``/Zi`` so .cu host stubs also produce
    # debug info.  Use ``-Xcompiler`` so nvcc passes the flag through to
    # MSVC unchanged.  We deliberately do NOT add ``-G`` (CUDA device-side
    # debug info) because it inflates the build by ~5x and the typical
    # crash investigations are in CPU code, not in CUDA kernels.
    if _IS_WINDOWS and os.environ.get("VIPE_DEBUG_SYMBOLS", "0") == "1":
        flags.extend(["-Xcompiler", "/Zi", "-Xcompiler", "/FS"])
    return flags + _additional_include_flags()
