# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial shared K from the depth provider's own focal (no FoV given).

The provider predicts a pinhole K on each of the first ``num_frames`` RGB frames
and the component-wise median is the shared initial K. It is also the K kept when the calibration falls back
(too few keyframes, or a terminal endpoint that fails the convergence gate).

The principal point is the image centre. MoGe-2 always predicts exactly the
centre (its normalised centre is the constant 0.5), so for the default provider
the explicit centring below is the identity; for any other provider it pins the
centre the way the bundle adjustment keeps it (only the focal is optimised).
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
import torch

from .protocol import validate_k4


class IndexedImages(Protocol):
    """Random access to (1,3,H,W) float32 images; a list works, so does a lazy converter."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> torch.Tensor: ...


def centred_component_median(rows: np.ndarray, image_hw: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Component-wise median of per-frame K4 rows with the principal point at the image centre."""

    predictions = np.asarray(rows, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != 4 or len(predictions) < 1:
        raise ValueError(f"expected (N,4) per-frame K rows, got {predictions.shape}")
    height, width = image_hw
    shared = validate_k4(np.median(predictions, axis=0), image_hw)
    centre = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    provider_offset = np.abs(predictions[:, 2:] - centre[None])
    shared = shared.copy()
    shared[2:] = centre
    return validate_k4(shared, image_hw), {
        "principal_point_policy": "image_centre",
        "provider_principal_point_max_offset_px": float(provider_offset.max()),
    }


def estimate_initial_k(
    depth_model: Any,
    frames_bchw: IndexedImages,
    image_hw: tuple[int, int],
    *,
    num_frames: int,
    source: str,
    model_load_seconds: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the free-FoV initializer on the first ``num_frames`` frames.

    ``frames_bchw`` is an indexable sequence of (1,3,H,W) float32 images in
    [0, 1] on the provider's device; it is read one image at a time, so it may
    convert lazily. ``depth_model`` must implement ``estimate_free_fov_k4``.
    """

    if not hasattr(depth_model, "estimate_free_fov_k4"):
        raise TypeError(
            f"{type(depth_model).__name__} cannot predict its own focal; SharedK calibration needs a depth "
            "provider implementing estimate_free_fov_k4 (for example moge2-l)"
        )
    count = min(int(num_frames), len(frames_bchw))
    if count < 1:
        raise ValueError("initializer frame count must be positive")
    started = time.perf_counter()
    forward_seconds = 0.0
    rows: list[np.ndarray] = []
    for index in range(count):
        image = frames_bchw[index]
        frame_started = time.perf_counter()
        rows.append(np.asarray(depth_model.estimate_free_fov_k4(image), dtype=np.float64))
        del image
        forward_seconds += time.perf_counter() - frame_started
    predictions = np.stack([validate_k4(row, image_hw) for row in rows])
    shared, centring = centred_component_median(predictions, image_hw)
    precision = getattr(depth_model, "precision", None)
    return shared, {
        "source": source,
        "num_frames": count,
        "moge_fp32": precision == "FP32",
        "precision": precision,
        "model_load_seconds": float(model_load_seconds),
        "forward_seconds": float(forward_seconds),
        "wall_seconds": time.perf_counter() - started + float(model_load_seconds),
        "K4_per_frame_px": predictions.tolist(),
        **centring,
    }
