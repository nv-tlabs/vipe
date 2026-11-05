# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The calibration engine: ViPE's own keyframe SLAM driven frame by frame.

:class:`ViPESharedFocalEngine` builds an unmodified ``vipe.slam.system.SLAMSystem``
(its ``GraphBuffer``, motion filter, frontend, backend and fused bundle
adjustment) and steps it the way ``SLAMSystem.run`` pass 1 does, but stops at the
calibration endpoints and exposes the shared focal in between. This needs a few
private ``SLAMSystem`` methods.

The ViPE runtime (CUDA extension, network weights) is imported lazily so that
the module itself can be imported on a CPU-only machine.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

from vipe._paths import get_config_path
from vipe.config.base_schema import BaseConfigSchema
from vipe.config.slam import SLAMConfig
from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from vipe.utils.cameras import CameraType
from vipe.utils.model_cache import ModelCache

from .protocol import FINAL_STEPS, OFFICIAL_MILESTONES

#: SLAM settings forced for calibration on top of whatever ``slam`` config is given.
SHARED_FOCAL_SLAM_OVERRIDES: dict[str, Any] = {
    # ViPE's own keyframe-depth prior is not built; the calibration attaches its
    # metric depth provider explicitly (MODEL_METRIC_DEPTH, relatched per focal).
    "keyframe_depth": None,
    "optimize_intrinsics": True,
    "visualize": False,
    "frontend_backend_iters": list(OFFICIAL_MILESTONES),
    "backend_iters": FINAL_STEPS[1],
}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def calibration_slam_config(slam: BaseConfigSchema | Mapping[str, Any] | None = None) -> Any:
    """The SLAM config used for calibration, as the hydra config ``SLAMSystem`` takes.

    ``slam=None`` reads ViPE's ``configs/slam/default.yaml`` (the SLAM group of
    ViPE's default pipeline). ``slam=None``, a ``SLAMConfig`` or a plain dict is
    validated as ``SLAMConfig``; a hydra config (e.g. a composed ``slam`` group)
    is copied.
    The SharedK overrides (no ViPE keyframe-depth prior, focal optimisation on,
    fused BA forced, milestones 16/64/256, 24 final backend iterations) are
    applied on top in every case.
    """

    config: Any
    if slam is None:
        # Read the group file directly instead of composing it with hydra: this also
        # works inside a running hydra application (e.g. ``run.py``), where a second
        # hydra initialisation is refused. The file has no defaults list or
        # interpolation; only its mandatory ``optimize_intrinsics`` is filled in.
        # Loaded the way ViPE's scripts/generate_config_docs.py loads preset files.
        raw = OmegaConf.to_container(OmegaConf.load(get_config_path() / "slam" / "default.yaml"), resolve=False)
        assert isinstance(raw, dict)
        raw.update(optimize_intrinsics=True, keyframe_depth=None)
        config = SLAMConfig.model_validate(raw).to_dictconfig()
    elif isinstance(slam, BaseConfigSchema):
        config = slam.to_dictconfig()
    elif isinstance(slam, dict):
        config = SLAMConfig.model_validate(slam).to_dictconfig()
    else:
        config = copy.deepcopy(slam)
    # Every overridden key exists in a SLAM config, so struct mode is fine.
    for key, value in SHARED_FOCAL_SLAM_OVERRIDES.items():
        config[key] = value
    config.ba.fused = True
    return config


class AuditedDepthModel(DepthEstimationModel):
    """Counts and times the K-conditioned depth inferences ViPE's graph buffer makes.

    The relatch audit needs the number of metric-depth calls around every backend
    run; the record also keeps every conditioning focal. Inference itself is
    delegated unchanged.
    """

    METRIC_PURPOSE = "vipe_k_conditioned_metric_depth"
    INITIALIZER_PURPOSE = "moge2_free_fov_initializer"

    def __init__(self, inner: DepthEstimationModel, device: torch.device, *, model_load_seconds: float = 0.0) -> None:
        super().__init__()
        self.inner = inner
        self.device = device
        self.model_load_seconds = float(model_load_seconds)
        self.calls: list[dict[str, Any]] = []

    @property
    def depth_type(self) -> DepthType:
        return self.inner.depth_type

    @property
    def supported_camera_types(self) -> list[CameraType]:
        return self.inner.supported_camera_types

    @property
    def precision(self) -> str | None:
        return getattr(self.inner, "precision", None)

    @property
    def metric_depth_calls(self) -> int:
        return sum(1 for call in self.calls if call["purpose"] == self.METRIC_PURPOSE)

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb = src.rgb
        focal = None if src.intrinsics is None else float(src.intrinsics[0].item())
        _synchronize(self.device)
        started = time.perf_counter()
        result = self.inner.estimate(src)
        _synchronize(self.device)
        elapsed = time.perf_counter() - started
        call: dict[str, Any] = {
            "purpose": self.METRIC_PURPOSE,
            "conditioning_focal_px": focal,
            "forward_seconds": elapsed,
            "batch_size": int(rgb.shape[0]) if rgb is not None and rgb.ndim == 4 else 1,
        }
        if result.metric_depth is not None:
            depth = result.metric_depth
            call["output_hw"] = [int(depth.shape[-2]), int(depth.shape[-1])]
            call["valid_depth_fraction"] = float((depth > 0).sum().item()) / float(depth.numel())
        self.calls.append(call)
        return result

    def estimate_free_fov_k4(self, rgb_bchw: torch.Tensor) -> np.ndarray:
        estimator = getattr(self.inner, "estimate_free_fov_k4", None)
        if estimator is None:
            raise TypeError(
                f"{type(self.inner).__name__} cannot predict its own focal; SharedK calibration needs a depth "
                "provider implementing estimate_free_fov_k4 (for example moge2-l)"
            )
        _synchronize(self.device)
        started = time.perf_counter()
        k4 = estimator(rgb_bchw)
        _synchronize(self.device)
        self.calls.append(
            {
                "purpose": self.INITIALIZER_PURPOSE,
                "conditioning_focal_px": None,
                "forward_seconds": time.perf_counter() - started,
                "batch_size": int(rgb_bchw.shape[0]),
            }
        )
        return k4

    def diagnostics(self, *, include_calls: bool = False) -> dict[str, Any]:
        metric_calls = [call for call in self.calls if call["purpose"] == self.METRIC_PURPOSE]
        focal_values = [
            float(call["conditioning_focal_px"]) for call in metric_calls if call["conditioning_focal_px"] is not None
        ]
        describe = getattr(self.inner, "describe", None)
        result: dict[str, Any] = {
            "model": describe()["model"] if callable(describe) else type(self.inner).__name__,
            "precision": self.precision,
            "depth_type": self.depth_type.name,
            "model_load_seconds": self.model_load_seconds,
            "total_calls": len(self.calls),
            "initializer_calls": len(self.calls) - len(metric_calls),
            "metric_depth_calls": len(metric_calls),
            "metric_depth_forward_seconds": float(sum(call["forward_seconds"] for call in metric_calls)),
            "conditioning_focal_units": "vipe_internal_resized_pixels",
            "distinct_conditioning_focals_internal_px": sorted(set(focal_values)),
            "K_change_requires_full_reinfer": self.depth_type == DepthType.MODEL_METRIC_DEPTH,
        }
        if callable(describe):
            result["provider"] = describe()
        if include_calls:
            result["calls"] = list(self.calls)
        return result


class ViPESharedFocalEngine:
    """:class:`~vipe.metrictrail.calibration.driver.SharedFocalEngine` on ViPE's ``SLAMSystem``."""

    def __init__(
        self,
        *,
        slam_config: Any,
        depth_model: AuditedDepthModel,
        image_hw: tuple[int, int],
        device: torch.device,
        model_cache: ModelCache | None = None,
    ) -> None:
        from vipe.ext.lietorch import SE3
        from vipe.slam.system import SLAMSystem, StandardResizeStreamProcessor

        if depth_model.depth_type != DepthType.MODEL_METRIC_DEPTH:
            raise TypeError("shared-K calibration requires a MODEL_METRIC_DEPTH provider (re-inferred per focal)")
        self.device = device
        self.depth_model = depth_model
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.resizer = StandardResizeStreamProcessor()
        height, width = self.resizer.update_frame_size(self.image_hw)
        self.internal_hw = (int(height), int(width))

        system = SLAMSystem(device=device, config=slam_config, model_cache=model_cache)
        system.rig = SE3.Identity(1)
        # What SLAMSystem.run sets before building its components.
        system.config.update(
            {
                "height": self.internal_hw[0],
                "width": self.internal_hw[1],
                "n_views": 1,
                "has_init_pose": False,
                "camera_type": CameraType.PINHOLE,
            }
        )
        system._build_components()
        # Keyframe depth is attached here (slam.keyframe_depth is null), exactly
        # where SLAMSystem._build_components would have put its own prior.
        system.metric_depth = depth_model
        system.backend.depth_model = depth_model
        self.system = system
        self._pending: tuple[int, torch.Tensor, torch.Tensor | None, list[Any]] | None = None
        self._gpu_start: torch.cuda.Event | None = None
        self._gpu_end: torch.cuda.Event | None = None

    def candidate(self, frame_index: int, frame: Any) -> bool:
        frame = self.resizer(frame_index, frame)
        frame_list = [frame]
        inputs, buffer_masks = self.system._precompute_features(frame_list)
        self.system.sparse_tracks.track_image(frame_list)
        verdict = bool(self.system.motion_filter.check(inputs, buffer_masks))
        self._pending = (frame_index, inputs, buffer_masks, frame_list)
        return verdict

    def add_keyframe(self, frame_index: int) -> None:
        if self._pending is None or self._pending[0] != frame_index:
            raise RuntimeError("add_keyframe must follow candidate() for the same frame")
        _, inputs, buffer_masks, frame_list = self._pending
        self.system._add_keyframe(frame_index, inputs, buffer_masks, frame_list, phase=1)

    def run_frontend(self) -> None:
        self.system.frontend.run()

    def run_backend(self, steps: int, *, update_depth: bool) -> None:
        self.system.backend.run(steps, update_depth=update_depth)

    def accepted_keyframes(self) -> int:
        return int(self.system.buffer.n_frames)

    def recovered_k(self) -> np.ndarray:
        intrinsics = self.system.buffer.intrinsics[0].detach().float()
        return self.resizer.recover_intrinsics(intrinsics).cpu().numpy()

    def depth_calls(self) -> int:
        return self.depth_model.metric_depth_calls

    def begin_benchmark(self) -> None:
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._gpu_start = torch.cuda.Event(enable_timing=True)
        self._gpu_end = torch.cuda.Event(enable_timing=True)
        self._gpu_start.record()

    def end_benchmark(self) -> dict[str, float]:
        if self.device.type != "cuda" or self._gpu_start is None or self._gpu_end is None:
            return {"gpu_event_seconds": 0.0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        self._gpu_end.record()
        torch.cuda.synchronize(self.device)
        return {
            "gpu_event_seconds": self._gpu_start.elapsed_time(self._gpu_end) / 1000.0,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }

    def depth_diagnostics(self) -> dict[str, Any]:
        return self.depth_model.diagnostics()

    def backend_contract(self) -> dict[str, Any]:
        precision = (self.depth_model.precision or "unknown").lower()
        provider = self.depth_model.diagnostics()["model"].lower().replace("-", "")
        return {
            "network": "released ViPE network",
            "keyframe_depth": f"{provider}_{precision}_k_conditioned",
            "fused_bundle_adjustment": bool(self.system.config.ba.fused),
            "fused_solver": str(self.system.config.ba.get("solver", "legacy")),
        }
