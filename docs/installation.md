# Installation

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
