# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Depth Anything 3 API module.

This module provides the main API for Depth Anything 3, including model loading,
inference, and export capabilities. It supports both single and nested model architectures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from vipe.priors.depth.dav3.cfg import AttrDict, create_object, load_config
from vipe.priors.depth.dav3.registry import MODEL_REGISTRY
from vipe.priors.depth.dav3.utils import InputProcessor, affine_inverse, align_poses_umeyama, logger

torch.backends.cudnn.benchmark = False
# logger.info("CUDNN Benchmark Disabled")

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"

_REPO_TO_MODEL_NAME = {
    "depth-anything/DA3METRIC-LARGE": "da3metric-large",
    "depth-anything/DA3-GIANT": "da3-giant",
}


@dataclass
class Prediction:
    depth: np.ndarray
    is_metric: int
    sky: np.ndarray | None = None
    conf: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    processed_images: np.ndarray | None = None
    aux: dict[str, Any] | None = None
    scale_factor: float | None = None


class OutputProcessor:
    """Convert DAv3 tensor outputs to the small Prediction object ViPE uses."""

    def __call__(self, model_output: dict[str, torch.Tensor]) -> Prediction:
        return Prediction(
            depth=model_output["depth"].squeeze(0).squeeze(-1).cpu().numpy(),
            sky=self._extract_sky(model_output),
            conf=self._extract_conf(model_output),
            extrinsics=self._extract_extrinsics(model_output),
            intrinsics=self._extract_intrinsics(model_output),
            is_metric=getattr(model_output, "is_metric", 0),
            aux=self._extract_aux(model_output),
            scale_factor=model_output.get("scale_factor", None),
        )

    def _extract_conf(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        conf = model_output.get("depth_conf", None)
        return conf.squeeze(0).cpu().numpy() if conf is not None else None

    def _extract_extrinsics(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        extrinsics = model_output.get("extrinsics", None)
        return extrinsics.squeeze(0).cpu().numpy() if extrinsics is not None else None

    def _extract_intrinsics(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        intrinsics = model_output.get("intrinsics", None)
        return intrinsics.squeeze(0).cpu().numpy() if intrinsics is not None else None

    def _extract_sky(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        sky = model_output.get("sky", None)
        return sky.squeeze(0).cpu().numpy() >= 0.5 if sky is not None else None

    def _extract_aux(self, model_output: dict[str, torch.Tensor]) -> AttrDict:
        ret = AttrDict()
        aux = model_output.get("aux", None)
        if aux is not None:
            for key, value in aux.items():
                ret[key] = value.squeeze(0).cpu().numpy() if isinstance(value, torch.Tensor) else value
        return ret


def _model_name_from_repo_id(repo_id: str) -> str:
    if repo_id in _REPO_TO_MODEL_NAME:
        return _REPO_TO_MODEL_NAME[repo_id]
    return repo_id.rsplit("/", 1)[-1].lower()


def _load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "safetensors is required to load DAv3 checkpoints. "
                "Run `uv sync` to install the default ViPE dependencies."
            ) from exc
        return load_file(str(path), device="cpu")

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported DAv3 checkpoint format at {path}")
    return state


class DepthAnything3(nn.Module):
    """
    Depth Anything 3 main API class.

    This class provides a high-level interface for depth estimation using Depth Anything 3.
    It supports both single and nested model architectures with metric scaling capabilities.

    Features:
    - Hugging Face Hub integration via PyTorchModelHubMixin
    - Support for multiple model presets (vitb, vitg, nested variants)
    - Automatic mixed precision inference
    - Export capabilities for various formats (GLB, PLY, NPZ, etc.)
    - Camera pose estimation and metric depth scaling

    Usage:
        # Load from Hugging Face Hub
        model = DepthAnything3.from_pretrained("huggingface/model-name")

        # Or create with specific preset
        model = DepthAnything3(preset="vitg")

        # Run inference
        prediction = model.inference(images, export_dir="output", export_format="glb")
    """

    def __init__(self, model_name: str = "da3-large", **kwargs):
        """
        Initialize DepthAnything3 with specified preset.

        Args:
        model_name: The name of the model preset to use.
                    Examples: 'da3-giant', 'da3-large', 'da3metric-large', 'da3nested-giant-large'.
        **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.model_name = model_name

        # Build the underlying network
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()

        # Initialize processors
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()

        # Device management (set by user)
        self.device = None

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        model_name: str | None = None,
        weights_path: str | None = None,
        **kwargs,
    ) -> "DepthAnything3":
        """Load a vendored DAv3 inference model from a Hugging Face checkpoint."""
        model = cls(model_name=model_name or _model_name_from_repo_id(repo_id), **kwargs)
        if weights_path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "huggingface_hub is required to fetch DAv3 weights. Run `uv sync` or pass `weights_path=...`."
                ) from exc
            weights_path = hf_hub_download(repo_id=repo_id, filename=SAFETENSORS_NAME)

        state = _load_state_dict_file(weights_path)
        model._load_flexible_state_dict(state)
        return model

    def _load_flexible_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        def strip_prefix(prefix: str) -> dict[str, torch.Tensor]:
            return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}

        candidates = [
            (self, state),
            (self.model, state),
            (self.model, strip_prefix("model.")),
        ]
        target, target_state = max(
            candidates,
            key=lambda candidate: len(set(candidate[0].state_dict()).intersection(candidate[1])),
        )
        missing, unexpected = target.load_state_dict(target_state, strict=False)
        if missing:
            logger.warn(f"DAv3 checkpoint did not populate {len(missing)} parameters/buffers.")
        if unexpected:
            logger.warn(f"DAv3 checkpoint had {len(unexpected)} unexpected parameters/buffers.")

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.
            extrinsics: Optional camera extrinsics with shape ``(B, N, 4, 4)``.
            intrinsics: Optional camera intrinsics with shape ``(B, N, 3, 3)``.
            export_feat_layers: Layer indices to return intermediate features for.

        Returns:
            Dictionary containing model predictions
        """
        # Determine optimal autocast dtype
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=image.device.type, dtype=autocast_dtype):
                return self.model(image, extrinsics, intrinsics, export_feat_layers, infer_gs)

    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        render_exts: np.ndarray | None = None,
        render_ixts: np.ndarray | None = None,
        render_hw: tuple[int, int] | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] | None = None,
        # GLB export parameters
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Feat_vis export parameters
        feat_vis_fps: int = 15,
        # Other export parameters, e.g., gs_ply, gs_video
        export_kwargs: dict | None = None,
    ) -> Prediction:
        """
        Run inference on input images.

        Args:
            image: List of input images (numpy arrays, PIL Images, or file paths)
            extrinsics: Camera extrinsics (N, 4, 4)
            intrinsics: Camera intrinsics (N, 3, 3)
            align_to_input_ext_scale: whether to align the input pose scale to the prediction
            infer_gs: Enable the 3D Gaussian branch (needed for `gs_ply`/`gs_video` exports)
            render_exts: Optional render extrinsics for Gaussian video export
            render_ixts: Optional render intrinsics for Gaussian video export
            render_hw: Optional render resolution for Gaussian video export
            process_res: Processing resolution
            process_res_method: Resize method for processing
            export_dir: Directory to export results
            export_format: Export format (mini_npz, npz, glb, ply, gs, gs_video)
            export_feat_layers: Layer indices to export intermediate features from
            conf_thresh_percentile: [GLB] Lower percentile for adaptive confidence threshold (default: 40.0) # noqa: E501
            num_max_points: [GLB] Maximum number of points in the point cloud (default: 1,000,000)
            show_cameras: [GLB] Show camera wireframes in the exported scene (default: True)
            feat_vis_fps: [FEAT_VIS] Frame rate for output video (default: 15)
            export_kwargs: additional arguments to export functions.

        Returns:
            Prediction object containing depth maps and camera parameters
        """
        if infer_gs:
            raise NotImplementedError("The vendored DAv3 path only includes depth inference, not 3DGS export.")

        # Preprocess images
        imgs_cpu, extrinsics, intrinsics = self._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )

        # Prepare tensors for model
        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)

        # Normalize extrinsics
        ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

        # Run model forward pass
        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        raw_output = self._run_model_forward(imgs, ex_t_norm, in_t, export_feat_layers, infer_gs)

        # Convert raw output to prediction
        prediction = self._convert_to_prediction(raw_output)

        # Align prediction to extrinsincs
        prediction = self._align_to_input_extrinsics_intrinsics(
            extrinsics, intrinsics, prediction, align_to_input_ext_scale
        )

        # Add processed images for visualization
        prediction = self._add_processed_images(prediction, imgs_cpu)

        # Export if requested
        if export_dir is not None:
            raise NotImplementedError("The vendored DAv3 path only includes inference, not result export.")

        return prediction

    def _preprocess_inputs(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Preprocess input images using input processor."""
        start_time = time.time()
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
        )
        end_time = time.time()
        logger.info(
            "Processed Images Done taking",
            end_time - start_time,
            "seconds. Shape: ",
            imgs_cpu.shape,
        )
        return imgs_cpu, extrinsics, intrinsics

    def _prepare_model_inputs(
        self,
        imgs_cpu: torch.Tensor,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepare tensors for model input."""
        device = self._get_model_device()

        # Move images to model device
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

        # Convert camera parameters to tensors
        ex_t = extrinsics.to(device, non_blocking=True)[None].float() if extrinsics is not None else None
        in_t = intrinsics.to(device, non_blocking=True)[None].float() if intrinsics is not None else None

        return imgs, ex_t, in_t

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> Prediction:
        """Align depth map to input extrinsics"""
        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction

    def _run_model_forward(
        self,
        imgs: torch.Tensor,
        ex_t: torch.Tensor | None,
        in_t: torch.Tensor | None,
        export_feat_layers: Sequence[int] | None = None,
        infer_gs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run model forward pass."""
        device = imgs.device
        need_sync = device.type == "cuda"
        if need_sync:
            torch.cuda.synchronize(device)
        start_time = time.time()
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else None
        output = self.forward(imgs, ex_t, in_t, feat_layers, infer_gs)
        if need_sync:
            torch.cuda.synchronize(device)
        end_time = time.time()
        logger.info(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")
        return output

    def _convert_to_prediction(self, raw_output: dict[str, torch.Tensor]) -> Prediction:
        """Convert raw model output to Prediction object."""
        start_time = time.time()
        output = self.output_processor(raw_output)
        end_time = time.time()
        logger.info(f"Conversion to Prediction Done. Time: {end_time - start_time} seconds")
        return output

    def _add_processed_images(self, prediction: Prediction, imgs_cpu: torch.Tensor) -> Prediction:
        """Add processed images to prediction for visualization."""
        # Convert from (N, 3, H, W) to (N, H, W, 3) and denormalize
        processed_imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)

        # Denormalize from ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_imgs = processed_imgs * std + mean
        processed_imgs = np.clip(processed_imgs, 0, 1)
        processed_imgs = (processed_imgs * 255).astype(np.uint8)

        prediction.processed_images = processed_imgs
        return prediction

    def _export_results(self, prediction: Prediction, export_format: str, export_dir: str, **kwargs) -> None:
        """Export results to specified format and directory."""
        raise NotImplementedError("The vendored DAv3 path only includes inference, not result export.")

    def _get_model_device(self) -> torch.device:
        """
        Get the device where the model is located.

        Returns:
            Device where the model parameters are located

        Raises:
            ValueError: If no tensors are found in the model
        """
        if self.device is not None:
            return self.device

        # Find device from parameters
        for param in self.parameters():
            self.device = param.device
            return param.device

        # Find device from buffers
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device

        raise ValueError("No tensor found in model")
