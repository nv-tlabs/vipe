"""Repository paths, hashing and small I/O helpers shared by all evaluators."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DATASETS_CONFIG = PACKAGE_DIR / "datasets.json"


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Absolute paths are kept; relative paths are resolved against the repo root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: object) -> str | None:
    """SHA-256 of a string identifier (``None`` for non-strings).

    Preparation profile ids stored in the prepared GT (manifests, NPZ labels)
    are compared by digest, so the evaluator does not spell them out.
    """

    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def dataset_defaults() -> dict[str, Any]:
    """Default GT roots relative to the repository root (package data ``datasets.json``)."""

    return read_json(DATASETS_CONFIG)
