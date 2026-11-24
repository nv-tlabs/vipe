# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ViPE adapter for the unmodified Apple Depth Pro 2 (https://github.com/apple-aiml-research/ml-depth-pro-2) package.

"""
``depthpro2-{l,h}`` is stateless FP32 for calibration and DROID keyframe replay.
The ``-stream`` suffix uses a sequence-local causal 16-frame cache. Only weights
are shared; call ``reset_stream`` before another sequence or tracking replay.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

import numpy as np
import torch

from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from vipe.utils.cameras import CameraType


def checkpoint_directory(value: str | Path | None = None) -> Path:
    """Resolve local checkpoints without changing Apple's global presets."""
    default = Path(__file__).resolve().parents[4] / "ckpt" / "depth-pro-2"
    return Path(value or os.environ.get("DEPTHPRO2_CHECKPOINT_DIR", str(default))).expanduser().resolve()


@dataclass
class DepthPro2Weights:
    """Cached weights and a precision guard, never per-video key/value state."""

    model: torch.nn.Module
    checkpoint_dir: Path
    lock: Any = field(default_factory=RLock, repr=False)


def load_depth_pro2_weights(name: str = "depthpro2-h", *, checkpoint_dir: str | Path | None = None) -> DepthPro2Weights:
    from depth_pro_2 import create_model_and_transforms
    from depth_pro_2.model import PRESETS

    if name not in PRESETS:
        raise ValueError(f"unknown Depth Pro 2 preset: {name}")
    directory = checkpoint_directory(checkpoint_dir)
    preset = PRESETS[name]
    depth_path = directory / str(preset.checkpoint_uri).rsplit("/", 1)[-1]
    fov_path = directory / str(preset.fov_checkpoint_uri).rsplit("/", 1)[-1]
    for path in (depth_path, fov_path):
        if not path.is_file():
            raise FileNotFoundError(f"Depth Pro 2 checkpoint missing: {path}; set DEPTHPRO2_CHECKPOINT_DIR")
    config = replace(preset, checkpoint_uri=str(depth_path), fov_checkpoint_uri=str(fov_path))
    model, _ = create_model_and_transforms(config, precision=torch.float32)
    return DepthPro2Weights(model=model, checkpoint_dir=directory)


class DepthPro2Model(DepthEstimationModel):
    """Focal-conditioned metric z depth, using ImageNet-normalized FP32 RGB.

    No resizing, uint8 round-trip, masks or sensor intrinsics are introduced;
    dynamic/world masks remain the existing pipeline's job.
    """

    def __init__(
        self,
        name: str = "depthpro2-h",
        *,
        streaming: bool = False,
        use_fp16: bool = True,
        window: int = 16,
        checkpoint_dir: str | Path | None = None,
        weights: DepthPro2Weights | None = None,
    ) -> None:
        if name not in {"depthpro2-s", "depthpro2-b", "depthpro2-l", "depthpro2-h"}:
            raise ValueError(f"unknown Depth Pro 2 preset: {name}")
        if window < 1:
            raise ValueError("Depth Pro 2 window must be positive")
        self.name, self.streaming, self.window = name, bool(streaming), int(window)
        # Calibration is always FP32, regardless of the generic depth_fp16 flag.
        self.inference_dtype = torch.float16 if streaming and use_fp16 else torch.float32
        self.weights = weights if weights is not None else load_depth_pro2_weights(name, checkpoint_dir=checkpoint_dir)
        self.model = self.weights.model
        wrong_dtype = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if wrong_dtype:
            raise TypeError(f"Depth Pro 2 must retain FP32 weights: {wrong_dtype[:5]}")
        self.reset_stream()

    @property
    def network(self) -> torch.nn.Module:
        return self.model

    @property
    def depth_type(self) -> DepthType:
        # Shared-K re-infers every retained frame after updating focal length.
        return DepthType.MODEL_METRIC_DEPTH

    def reset_stream(self) -> None:
        """Discard causal history before another sequence, retry, or replay."""
        self._stream = None
        self._stream_focal: float | None = None
        self._stream_hw: tuple[int, int] | None = None
        self.frames_seen = 0

    @contextmanager
    def _precision(self, dtype: torch.dtype) -> Iterator[None]:
        # Apple's precision flags live on modules, not streams. Guard the flags
        # so FP32 calibration and FP16 tracking safely reuse one FP32 network.
        with self.weights.lock, torch.inference_mode():
            previous, fov = self.model.precision, self.model.fov
            previous_fov = None if fov is None else fov.precision
            self.model.precision = dtype
            if fov is not None:
                fov.precision = dtype
            try:
                device = next(self.model.parameters()).device
                with torch.autocast(device_type=device.type, enabled=False):
                    yield
            finally:
                self.model.precision = previous
                if fov is not None:
                    fov.precision = previous_fov

    def _normalize(self, rgb_chw: torch.Tensor) -> torch.Tensor:
        if rgb_chw.dtype != torch.float32 or rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3:
            raise TypeError(f"Depth Pro 2 requires FP32 RGB (3,H,W), got {rgb_chw.dtype} {tuple(rgb_chw.shape)}")
        if min(rgb_chw.shape[-2:]) < 1:
            raise ValueError("Depth Pro 2 requires non-empty frames")
        if not bool((torch.isfinite(rgb_chw) & (rgb_chw >= 0) & (rgb_chw <= 1)).all().item()):
            raise ValueError("Depth Pro 2 RGB must be finite and within [0,1]")
        device = next(self.model.parameters()).device
        rgb = rgb_chw.to(device=device, dtype=torch.float32).contiguous()
        mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = rgb.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        return (rgb - mean) / std

    def _infer_frame(self, rgb_chw: torch.Tensor, focal: float) -> torch.Tensor:
        normalized = self._normalize(rgb_chw)
        image_hw = (int(rgb_chw.shape[-2]), int(rgb_chw.shape[-1]))
        with self._precision(self.inference_dtype):
            if not self.streaming:
                prediction = self.model.infer(normalized, f_px=focal)
            else:
                if self._stream is None:
                    self._stream = self.model.stream(mode="unbuffered", window=self.window, f_px=focal)
                    self._stream_focal, self._stream_hw = focal, image_hw
                if self._stream_focal != focal or self._stream_hw != image_hw:
                    raise ValueError(
                        "Depth Pro 2 stream focal/size changed; reset_stream() before replay or a new video"
                    )
                try:
                    predictions = self._stream.step(normalized)
                    if len(predictions) != 1:
                        raise RuntimeError("unbuffered Depth Pro 2 must return exactly one prediction per frame")
                    prediction = predictions[0]
                except Exception:
                    self.reset_stream()
                    raise
        if prediction.get("depth") is None:
            raise RuntimeError("Depth Pro 2 returned no metric depth despite its bound focal")
        depth = prediction["depth"].detach().to(device=rgb_chw.device, dtype=torch.float32)
        if tuple(depth.shape) != image_hw:
            raise ValueError(f"Depth Pro 2 depth shape {tuple(depth.shape)} does not match {image_hw}")
        valid = torch.isfinite(depth) & (depth >= 0.2) & (depth <= 500.0)
        if not bool(valid.any().item()):
            raise RuntimeError("Depth Pro 2 produced no valid metric depth")
        self.frames_seen += 1
        return torch.where(valid, depth, torch.zeros_like(depth))

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb, intrinsics = src.rgb, src.intrinsics
        if src.camera_type != CameraType.PINHOLE or rgb is None or rgb.dtype != torch.float32:
            raise TypeError("Depth Pro 2 requires FP32 pinhole RGB")
        if rgb.ndim not in (3, 4) or rgb.shape[-1] != 3:
            raise ValueError(f"expected RGB (H,W,3) or (B,H,W,3), got {tuple(rgb.shape)}")
        if intrinsics is None or tuple(intrinsics.shape) != (4,):
            raise ValueError("Depth Pro 2 requires one explicit shared [fx,fy,cx,cy]")
        focal = float(intrinsics[0].item())
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError(f"Depth Pro 2 requires finite positive fx, got {focal}")
        is_image = rgb.ndim == 3
        batch = rgb[None] if is_image else rgb
        if batch.shape[0] < 1:
            raise ValueError("Depth Pro 2 requires at least one image")
        # infer(T,C,H,W) would cross-attend. Calibration batches must instead
        # be independent single-frame calls. Streaming batches are chronological.
        depth = torch.stack([self._infer_frame(frame.permute(2, 0, 1), focal) for frame in batch])
        return DepthEstimationResult(metric_depth=depth[0] if is_image else depth)

    def estimate_free_fov_k4(self, rgb_bchw: torch.Tensor) -> np.ndarray:
        """Centered fx=fy from the FP32 FOV head only, without depth/history."""
        if rgb_bchw.ndim != 4 or rgb_bchw.shape[0] != 1:
            raise ValueError(f"expected a (1,3,H,W) image, got {tuple(rgb_bchw.shape)}")
        normalized = self._normalize(rgb_bchw[0])
        if self.model.fov is None:
            raise RuntimeError("Depth Pro 2 model has no FOV estimator")
        with self._precision(torch.float32):
            focal = float(self.model.fov.focal_length_px(normalized[None]).item())
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError(f"Depth Pro 2 predicted an invalid focal length: {focal}")
        height, width = normalized.shape[-2:]
        return np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {
            "model": "Depth Pro 2",
            "registry_name": self.name + ("-stream" if self.streaming else ""),
            "precision": "FP16" if self.inference_dtype == torch.float16 else "FP32",
            "weights_precision": "FP32",
            "depth_type": self.depth_type.name,
            "depth_convention": "metric optical-axis z in meters",
            "mode": "unbuffered" if self.streaming else "stateless_single_frame",
            "window": self.window if self.streaming else 1,
            "focal": "explicit camera fx, fixed for each stream",
            "free_fov": "FP32 FOV head only; centered fx=fy; no depth inference",
            "valid_rule": "finite & 0.2 <= depth <= 500.0 m; invalid pixels are 0",
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "checkpoint_dir": str(self.weights.checkpoint_dir),
            "GT_used": False,
        }
