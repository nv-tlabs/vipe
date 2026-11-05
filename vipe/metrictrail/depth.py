# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MetricTrail's metric depth provider on top of ViPE's depth registry.

MetricTrail uses ViPE's ``vipe.priors.depth`` registry as its depth provider.
The default, ``moge2-l``, goes through :class:`MetricTrailMoGe2`, a subclass of
ViPE's ``MogeModel`` (version 2, the vendored MoGe-2 code and checkpoint
resolution are reused unchanged) that makes MetricTrail's exact MoGe-2 call:

- ``resolution_level=9`` with ``num_tokens=None``;
- ``use_fp16`` as configured (autocast only, weights stay FP32);
- ``fov_x`` computed from the bound focal in float64 Python arithmetic
  (``math.atan``), not in float32 torch arithmetic;
- ``force_projection=True`` and ``apply_mask=False``;
- the metric validity rule ``MoGe mask & finite & 0.2 m <= depth <= 500 m``;
  invalid pixels become 0 ("no prior") in :meth:`estimate`, and a frame without
  a single valid pixel is an error.

Any other name is handed to ViPE's ``make_depth_model`` unchanged, so other
registry backends can be selected by configuration alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import torch

from vipe.priors.depth import make_depth_model
from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult
from vipe.priors.depth.moge import MogeModel
from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional
from vipe.utils.model_cache import ModelCache

from .calibration.protocol import validate_k4

#: MoGe-2 inference arguments shared by every MetricTrail call (``fov_x`` and
#: ``use_fp16`` are supplied per call).
MOGE2_INFER_KWARGS: dict[str, Any] = {
    "num_tokens": None,
    "resolution_level": 9,
    "force_projection": True,
    "apply_mask": False,
}

METRIC_DEPTH_MIN_M = 0.2
METRIC_DEPTH_MAX_M = 500.0


def fov_x_degrees(focal_px: float, width: int) -> float:
    """FoV along the image width (x), in degrees, from a focal in pixels; float64 (MetricTrail's rule)."""

    return math.degrees(2.0 * math.atan(float(width) / (2.0 * float(focal_px))))


def normalized_to_pixel_k4(value: Any, image_hw: tuple[int, int]) -> np.ndarray:
    """MoGe's normalised 3x3 K (optionally batched by one) to pixel ``[fx, fy, cx, cy]``."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (1, 3, 3):
        matrix = matrix[0]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid normalized MoGe K: {matrix.shape}")
    height, width = image_hw
    return validate_k4(
        [
            matrix[0, 0] * width,
            matrix[1, 1] * height,
            matrix[0, 2] * width,
            matrix[1, 2] * height,
        ],
        image_hw,
    )


@runtime_checkable
class FreeFovIntrinsicsEstimator(Protocol):
    """A depth provider that can predict its own pinhole K from one RGB image.

    SharedK calibration uses it for the initial focal and for the fallback when
    the bundle-adjustment focal cannot be trusted.
    """

    def estimate_free_fov_k4(self, rgb_bchw: torch.Tensor) -> np.ndarray: ...


@dataclass(frozen=True)
class MetricDepthPrediction:
    """Raw MoGe-2 depth plus the MetricTrail validity mask, both shaped like the input batch."""

    depth: torch.Tensor
    valid: torch.Tensor
    fov_x_degrees: float | None


def _squeeze_channel(tensor: torch.Tensor) -> torch.Tensor:
    while tensor.ndim > 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim == 2:
        tensor = tensor[None]
    return tensor


class MetricTrailMoGe2(MogeModel):
    """ViPE's MoGe-2 prior with MetricTrail's exact inference call and validity rule.

    ``network`` lets two wrappers share one set of weights (for example an FP32
    calibration wrapper and an FP16 tracking wrapper); when omitted the network
    is built by ``MogeModel`` itself (``VIPE_MOGE2_CHECKPOINT``, the torch hub
    cache, or HuggingFace, in that order).
    """

    def __init__(
        self,
        variant: str = "l",
        *,
        use_fp16: bool = True,
        min_depth_m: float = METRIC_DEPTH_MIN_M,
        max_depth_m: float = METRIC_DEPTH_MAX_M,
        network: torch.nn.Module | None = None,
    ) -> None:
        if network is None:
            super().__init__(version=2, variant=variant, num_tokens=None, use_fp16=use_fp16, fp16_weights=False)
        else:
            DepthEstimationModel.__init__(self)
            self.version = 2
            self.num_tokens = None
            self.use_fp16 = use_fp16
            self.model = network
        if not 0.0 < float(min_depth_m) < float(max_depth_m):
            raise ValueError(f"invalid metric depth range: [{min_depth_m}, {max_depth_m}]")
        self.variant = variant
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        wrong_dtype = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if wrong_dtype:
            raise TypeError(f"MoGe-2 must keep FP32 weights (use_fp16 is autocast only): {wrong_dtype[:5]}")

    @property
    def network(self) -> torch.nn.Module:
        return self.model

    @property
    def precision(self) -> str:
        return "FP16" if self.use_fp16 else "FP32"

    def infer_raw(self, rgb: torch.Tensor, *, fov_x: float | None) -> Mapping[str, torch.Tensor]:
        """One MoGe-2 forward with MetricTrail's arguments. ``rgb`` is (3,H,W) or (B,3,H,W) in [0, 1]."""

        with torch.inference_mode():
            output = self.model.infer(rgb, fov_x=fov_x, use_fp16=self.use_fp16, **MOGE2_INFER_KWARGS)
        if not isinstance(output, Mapping) or "depth" not in output:
            raise KeyError("MoGe-2 output lacks depth")
        return output

    def predict_metric(self, rgb_bchw: torch.Tensor, *, focal_px: float) -> MetricDepthPrediction:
        """K-conditioned metric depth and validity for a (B,3,H,W) batch."""

        if rgb_bchw.ndim != 4:
            raise ValueError(f"expected a (B,3,H,W) batch, got {tuple(rgb_bchw.shape)}")
        fov = fov_x_degrees(focal_px, int(rgb_bchw.shape[-1]))
        output = self.infer_raw(rgb_bchw, fov_x=fov)
        depth = _squeeze_channel(output["depth"].detach().float())
        expected = (int(rgb_bchw.shape[0]), int(rgb_bchw.shape[-2]), int(rgb_bchw.shape[-1]))
        if tuple(depth.shape) != expected:
            raise ValueError(f"MoGe-2 depth shape drifted: {tuple(depth.shape)} versus {expected}")
        valid = torch.isfinite(depth)
        mask = output.get("mask")
        if mask is not None:
            mask = _squeeze_channel(mask.detach().bool())
            if mask.shape != depth.shape:
                raise ValueError(f"MoGe-2 mask/depth shape mismatch: {tuple(mask.shape)} versus {tuple(depth.shape)}")
            valid &= mask
        valid &= depth >= self.min_depth_m
        valid &= depth <= self.max_depth_m
        return MetricDepthPrediction(depth=depth, valid=valid, fov_x_degrees=fov)

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb = unpack_optional(src.rgb)
        intrinsics = unpack_optional(src.intrinsics)
        if src.camera_type != CameraType.PINHOLE or rgb.dtype != torch.float32:
            raise TypeError("MetricTrail MoGe-2 requires FP32 pinhole RGB")
        if rgb.ndim == 3:
            rgb, remove_batch = rgb[None], True
        elif rgb.ndim == 4:
            remove_batch = False
        else:
            raise ValueError(f"unexpected ViPE RGB shape {tuple(rgb.shape)}")
        rgb_bchw = rgb.moveaxis(-1, 1).contiguous()
        prediction = self.predict_metric(rgb_bchw, focal_px=float(intrinsics[0].item()))
        if int(prediction.valid.sum().item()) == 0:
            raise RuntimeError("MoGe-2 produced no valid metric depth")
        metric = torch.where(prediction.valid, prediction.depth, torch.zeros_like(prediction.depth))
        if remove_batch:
            metric = metric[0]
        return DepthEstimationResult(metric_depth=metric)

    def estimate_free_fov_k4(self, rgb_bchw: torch.Tensor) -> np.ndarray:
        """MoGe-2's own pinhole K (no FoV given) for a (1,3,H,W) image, in pixels (float64)."""

        if rgb_bchw.ndim != 4 or rgb_bchw.shape[0] != 1:
            raise ValueError(f"expected a (1,3,H,W) image, got {tuple(rgb_bchw.shape)}")
        output = self.infer_raw(rgb_bchw, fov_x=None)
        if "intrinsics" not in output:
            raise KeyError("free-FoV MoGe-2 output lacks intrinsics")
        image_hw = (int(rgb_bchw.shape[-2]), int(rgb_bchw.shape[-1]))
        return normalized_to_pixel_k4(output["intrinsics"].detach().float().cpu().numpy(), image_hw)

    def describe(self) -> dict[str, Any]:
        return {
            "model": "MoGe-2",
            "registry_name": f"moge2-{self.variant}",
            "precision": self.precision,
            "depth_type": self.depth_type.name,
            "infer_kwargs": {**MOGE2_INFER_KWARGS, "use_fp16": self.use_fp16},
            "fov_x": "float64 math.atan of the bound focal",
            "valid_rule": f"mask & finite & {self.min_depth_m} <= depth <= {self.max_depth_m} m",
        }


def is_moge2_name(name: str) -> bool:
    return name == "moge2" or name.startswith("moge2-")


def _moge2_variant(name: str) -> str:
    return name.split("-", 1)[1] if "-" in name else "l"


def make_metrictrail_depth_model(
    name: str = "moge2-l",
    *,
    use_fp16: bool = True,
    model_cache: ModelCache | None = None,
) -> DepthEstimationModel:
    """Build MetricTrail's depth provider.

    ``moge2[-variant]`` returns a :class:`MetricTrailMoGe2` (weights shared
    through ``model_cache`` across precisions); every other name is ViPE's
    ``make_depth_model(name)``, cached under ViPE's own ``depth/<name>`` key.
    """

    if is_moge2_name(name):
        variant = _moge2_variant(name)
        if model_cache is None:
            return MetricTrailMoGe2(variant, use_fp16=use_fp16)
        network = model_cache.get(
            f"metrictrail/depth/moge2-{variant}/network",
            lambda: MetricTrailMoGe2(variant, use_fp16=use_fp16).network,
        )
        return MetricTrailMoGe2(variant, use_fp16=use_fp16, network=network)
    if model_cache is None:
        return make_depth_model(name)
    return model_cache.get(f"depth/{name}", lambda: make_depth_model(name))
