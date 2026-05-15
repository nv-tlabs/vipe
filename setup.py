import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

try:
    import torch
    import torch.version
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cuda_version = torch.version.cuda

    assert cuda_version is not None, "Pytorch CUDA is required for this installation."

except ImportError:
    raise ValueError("Pytorch not found, please install it first.")

PACKAGE_NAME = "vipe"
SOURCE_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

coder_finder_path = f"{PACKAGE_NAME}/ext/specs.py"
code_finder_namespace = {"__file__": coder_finder_path}
with open(coder_finder_path, "r") as fh:
    exec(fh.read(), code_finder_namespace)
get_sources = code_finder_namespace["get_sources"]
get_cpp_flags = code_finder_namespace["get_cpp_flags"]
get_cuda_flags = code_finder_namespace["get_cuda_flags"]


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        self._copy_configs()

    def _copy_configs(self) -> None:
        if not SOURCE_CONFIG_DIR.is_dir():
            raise RuntimeError(f"Missing config source directory: {SOURCE_CONFIG_DIR}")

        target = Path(self.build_lib) / PACKAGE_NAME / "_configs"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            SOURCE_CONFIG_DIR,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (target / "__init__.py").write_text(
            "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n\n"
            '"""Package data target for build-time generated ViPE configs."""\n',
            encoding="utf-8",
        )


# Setup CUDA_HOME for conda environment for consistency
if "CONDA_PREFIX" in os.environ:
    conda_nvcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc")
    if os.path.exists(conda_nvcc_path):
        os.environ["PYTORCH_NVCC"] = conda_nvcc_path

cpp_flags = get_cpp_flags()
cuda_flags = get_cuda_flags()

packages = find_packages()
setup(
    packages=packages,
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            f"{PACKAGE_NAME}_ext",
            sources=get_sources(),  # type: ignore
            extra_compile_args={"cxx": cpp_flags, "nvcc": cuda_flags},  # type: ignore
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True), "build_py": build_py},
)
