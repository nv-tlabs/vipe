import os
import re

from setuptools import find_packages, setup


try:
    import torch
    import torch.version

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    torch_version = torch.version.__version__.split(".")[:2]
    cuda_version = torch.version.cuda

    # This will be e.g. "+pt23cu121"
    assert cuda_version is not None, "Pytorch CUDA is required for this installation."
    version_suffix = f"+pt{torch_version[0]}{torch_version[1]}cu{cuda_version.replace('.', '')}"

except ImportError:
    raise ValueError("Pytorch not found, please install it first.")

PACKAGE_NAME = "vipe"

# Avoid directly importing the package
with open(f"{PACKAGE_NAME}/__init__.py", "r") as fh:
    __version__ = re.findall(r"__version__ = \"(.*?)\"", fh.read())[0]
__version__ += version_suffix

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

packages = find_packages()
setup(
    packages=packages,
    version=__version__,
    ext_modules=[
        CUDAExtension(
            f"{PACKAGE_NAME}_ext",
            include_dirs=[os.path.join(os.path.dirname(__file__), 'thirdparty/eigen')],
            sources=get_sources(),  # type: ignore
            extra_compile_args={"cxx": get_cpp_flags(), "nvcc": get_cuda_flags()},  # type: ignore
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
