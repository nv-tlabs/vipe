# Installation

## From PyPI

Published ViPE releases are available directly from PyPI:

```bash
pip install nvidia-vipe
```

This installs the `vipe` Python package and the `vipe` command-line tool. ViPE releases are published as source distributions, so pip builds the native CUDA extensions during installation. The environment needs a CUDA-enabled PyTorch build and a CUDA toolkit with `nvcc`.

## From Source

ViPE uses [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) for CUDA/native tooling and [uv](https://docs.astral.sh/uv/) for the local Python environment in `.venv`.

```bash
# Create a conda environment for uv, CUDA, and native build dependencies.
conda env create -f envs/cu128.yml
conda activate cu128

# Create .venv, install Python runtime dependencies, and build the package.
uv sync
```

For development, include the `dev` dependency group:

```bash
conda activate cu128
uv sync --dev

uv run --dev pre-commit install
uv run --dev ruff format .
uv run --dev ruff check .
uv run --dev mypy
```

To work on the documentation locally, install the docs dependency group:

```bash
uv sync --dev --group docs
uv run --group docs mkdocs serve
```
