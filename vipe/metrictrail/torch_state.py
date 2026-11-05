# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep MetricTrail's process-wide torch settings inside one pipeline run.

Inside a run MetricTrail sets its precision and seed policy process-wide: the
SharedK calibration turns TF32 off for cuBLAS and cuDNN, asks for ``highest``
float32 matmul precision and seeds Python, numpy and torch; the tracker reseeds
per sequence; the vendored DPVO calls ``torch.set_num_threads(2)``. Those settings
stay in force for the whole run (calibration, tracking, a re-track).
:func:`preserve_torch_state` puts the caller's values back afterwards, so a ViPE
pipeline run later in the same process (Python API use) sees exactly the state it
would have seen without MetricTrail.
"""

from __future__ import annotations

import functools
import logging
import random
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import numpy as np
import torch

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

#: Plain ``torch.backends`` flags saved and restored (``module path``, ``attribute``).
BACKEND_FLAGS: tuple[tuple[str, str], ...] = (
    ("cuda.matmul", "allow_fp16_reduced_precision_reduction"),
    ("cuda.matmul", "allow_bf16_reduced_precision_reduction"),
    ("cudnn", "benchmark"),
    ("cudnn", "deterministic"),
)

#: Owners of PyTorch's ``fp32_precision`` settings (torch >= 2.9), parents before children.
FP32_PRECISION_OWNERS = (
    "",
    "cuda.matmul",
    "cudnn",
    "cudnn.conv",
    "cudnn.rnn",
    "mkldnn",
    "mkldnn.matmul",
    "mkldnn.conv",
    "mkldnn.rnn",
)


def _backend(path: str) -> Any:
    target: Any = torch.backends
    for part in filter(None, path.split(".")):
        target = getattr(target, part)
    return target


def _tf32_settings() -> dict[str, Any] | None:
    """The TF32 switches MetricTrail itself uses (PyTorch's legacy ``allow_tf32`` API).

    None when PyTorch refuses to report them, which torch >= 2.9 does after a caller
    has set cuDNN conv and RNN to different values through its ``fp32_precision`` API.
    """

    try:
        return {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }
    except RuntimeError:
        return None


def _fp32_precision_settings() -> dict[str, str] | None:
    """PyTorch's per-backend ``fp32_precision`` values, or None before torch 2.9."""

    values: dict[str, str] = {}
    try:
        for path in FP32_PRECISION_OWNERS:
            values[path] = str(getattr(_backend(path), "fp32_precision"))
    except AttributeError:
        return None
    return values


def capture_torch_state() -> dict[str, Any]:
    """The precision flags, intra-op thread count and RNG states MetricTrail may change."""

    flags = {}
    for path, name in BACKEND_FLAGS:
        try:
            flags[(path, name)] = getattr(_backend(path), name)
        except AttributeError:
            continue
    state: dict[str, Any] = {
        "tf32": _tf32_settings(),
        "fp32_precision": _fp32_precision_settings(),
        "backend_flags": flags,
        "num_threads": torch.get_num_threads(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def restore_torch_state(state: dict[str, Any]) -> None:
    """Put back a :func:`capture_torch_state` snapshot."""

    tf32 = state["tf32"]
    if tf32 is not None:
        # Through the API MetricTrail used to change them, so that every getter of that
        # API reports the caller's values again. The precision setter also rewrites
        # cuBLAS' TF32 switch, so it goes first; a switch that already has the caller's
        # value is not written again (torch >= 2.9 then refuses to report the precision
        # 'medium' through this API).
        torch.set_float32_matmul_precision(tf32["float32_matmul_precision"])
        if bool(torch.backends.cuda.matmul.allow_tf32) != tf32["cuda_matmul_allow_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = tf32["cuda_matmul_allow_tf32"]
        if bool(torch.backends.cudnn.allow_tf32) != tf32["cudnn_allow_tf32"]:
            torch.backends.cudnn.allow_tf32 = tf32["cudnn_allow_tf32"]
    if state["fp32_precision"] is not None:
        # torch >= 2.9: the legacy setters above write explicit values into the
        # fp32_precision settings (an inherited 'none' becomes 'ieee'), so those are
        # written back afterwards, parents before children.
        values = state["fp32_precision"]
        for path in FP32_PRECISION_OWNERS:
            if path in values:
                setattr(_backend(path), "fp32_precision", values[path])
    for (path, name), value in state["backend_flags"].items():
        setattr(_backend(path), name, value)
    if torch.get_num_threads() != state["num_threads"]:
        torch.set_num_threads(state["num_threads"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


@contextmanager
def preserved_torch_state() -> Iterator[None]:
    """Restore the caller's torch state when the block exits (normally or by an exception)."""

    state = capture_torch_state()
    try:
        yield
    except BaseException:
        # Never let a failed restore (e.g. after a CUDA error) mask the run's own error.
        try:
            restore_torch_state(state)
        except Exception as error:  # pragma: no cover - only after a broken CUDA context
            logger.warning("could not restore the torch state after a failed MetricTrail run: %s", error)
        raise
    restore_torch_state(state)


def preserve_torch_state(function: F) -> F:
    """Decorator form of :func:`preserved_torch_state`."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with preserved_torch_state():
            return function(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
