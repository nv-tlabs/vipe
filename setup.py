import os
import shutil
import tarfile
import tempfile
from urllib.request import urlretrieve

from setuptools import find_packages, setup

try:
    import torch
    import torch.version
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cuda_version = torch.version.cuda

    assert cuda_version is not None, "Pytorch CUDA is required for this installation."

except ImportError:
    raise ValueError("Pytorch not found, please install it first.")

PACKAGE_NAME = "vipe"

coder_finder_path = f"{PACKAGE_NAME}/ext/specs.py"
code_finder_namespace = {"__file__": coder_finder_path}
with open(coder_finder_path, "r") as fh:
    exec(fh.read(), code_finder_namespace)
get_sources = code_finder_namespace["get_sources"]
get_cpp_flags = code_finder_namespace["get_cpp_flags"]
get_cuda_flags = code_finder_namespace["get_cuda_flags"]

# Setup CUDA_HOME for conda environment for consistency
if "CONDA_PREFIX" in os.environ:
    conda_nvcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc")
    if os.path.exists(conda_nvcc_path):
        os.environ["PYTORCH_NVCC"] = conda_nvcc_path

# Download the put Eigen 3.4 in a correct place
cpp_flags = get_cpp_flags()
cuda_flags = get_cuda_flags()
if os.environ.get("USE_SYSTEM_EIGEN", "0") == "0":
    eigen_include_dir = "csrc/include/eigen3"
    eigen_url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"

    if not os.path.exists(eigen_include_dir):
        os.makedirs(eigen_include_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_tar_path = os.path.join(temp_dir, "eigen.gz")
            extracted_dir = os.path.join(temp_dir, "eigen-extracted")
            urlretrieve(eigen_url, tmp_tar_path)
            with tarfile.open(tmp_tar_path, "r:gz") as tar:
                tar.extractall(path=extracted_dir)

            shutil.move(os.path.join(extracted_dir, "eigen-3.4.0", "Eigen"), eigen_include_dir)

    # Use full path
    additional_include_path = os.path.join(os.path.dirname(__file__), "csrc/include")
    cpp_flags += ["-isystem", additional_include_path]
    cuda_flags += ["-isystem", additional_include_path]

packages = find_packages()
setup(
    packages=packages,
    ext_modules=[
        CUDAExtension(
            f"{PACKAGE_NAME}_ext",
            sources=get_sources(),  # type: ignore
            extra_compile_args={"cxx": cpp_flags, "nvcc": cuda_flags},  # type: ignore
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
