# This file includes code originally from the Geocalib repository:
# https://github.com/cvg/GeoCalib
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

"""Simple interface for GeoCalib model."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .camera import BaseCamera
from .geocalib import GeoCalib as Model
from .utils import ImagePreprocessor, load_image


class GeoCalib(nn.Module):
    """Simple interface for GeoCalib model."""

    def __init__(self, weights: str = "pinhole"):
        """Initialize the model with optional config overrides.

        Args:
            weights (str): Weights to load. Can be "pinhole", "distorted" or path to a checkpoint.
            Note that in case of custom weights, the architecture must match the original model.
            If this is not the case, use the extractor from the 'siclib' package
            (from siclib.models.extractor import GeoCalib).
        """
        super().__init__()
        if weights in {"pinhole", "distorted"}:
            url = f"https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-{weights}.tar"

            # load checkpoint
            model_dir = f"{torch.hub.get_dir()}/geocalib"
            state_dict = torch.hub.load_state_dict_from_url(
                url, model_dir, map_location="cpu", file_name=f"{weights}.tar"
            )
        elif Path(weights).exists():
            state_dict = torch.load(weights, map_location="cpu")
        else:
            raise ValueError(f"Invalid weights: {weights}")

        self.model = Model()
        self.model.flexible_load(state_dict["model"])
        self.model.eval()

        self.image_processor = ImagePreprocessor({"resize": 320, "edge_divisible_by": 32})

    def load_image(self, path: Path) -> torch.Tensor:
        """Load image from path."""
        return load_image(path)

    def _post_process(
        self,
        camera: BaseCamera,
        img_data: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
    ) -> tuple[BaseCamera, dict[str, torch.Tensor]]:
        """Post-process model output by undoing scaling and cropping."""
        camera = camera.undo_scale_crop(img_data)

        w, h = camera.size.unbind(-1)
        h = h[0].round().int().item()
        w = w[0].round().int().item()

        for k in ["latitude_field", "up_field"]:
            out[k] = interpolate(out[k], size=(h, w), mode="bilinear")
        for k in ["up_confidence", "latitude_confidence"]:
            out[k] = interpolate(out[k][:, None], size=(h, w), mode="bilinear")[:, 0]

        inverse_scales = 1.0 / img_data["scales"]
        zero = camera.new_zeros(camera.f.shape[0])
        out["focal_uncertainty"] = out.get("focal_uncertainty", zero) * inverse_scales[1]
        return camera, out

    @torch.no_grad()
    def calibrate(
        self,
        img: torch.Tensor,
        camera_model: str = "pinhole",
        priors: Optional[Dict[str, torch.Tensor]] = None,
        shared_intrinsics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Perform calibration with online resizing.

        Assumes input image is in range [0, 1] and in RGB format.

        Args:
            img (torch.Tensor): Input image, shape (C, H, W) or (1, C, H, W)
            camera_model (str, optional): Camera model. Defaults to "pinhole".
            priors (Dict[str, torch.Tensor], optional): Prior parameters. Defaults to {}.
            shared_intrinsics (bool, optional): Whether to share intrinsics. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: camera and gravity vectors and uncertainties.
        """
        if len(img.shape) == 3:
            img = img[None]  # add batch dim
        if not shared_intrinsics:
            assert len(img.shape) == 4 and img.shape[0] == 1

        img_data = self.image_processor(img)

        if priors is None:
            priors = {}

        prior_values = {}
        if prior_focal := priors.get("focal"):
            prior_focal = prior_focal[None] if len(prior_focal.shape) == 0 else prior_focal
            prior_values["prior_focal"] = prior_focal * img_data["scales"][1]

        if "gravity" in priors:
            prior_gravity = priors["gravity"]
            prior_gravity = prior_gravity[None] if len(prior_gravity.shape) == 0 else prior_gravity
            prior_values["prior_gravity"] = prior_gravity

        self.model.optimizer.set_camera_model(camera_model)
        self.model.optimizer.shared_intrinsics = shared_intrinsics

        out = self.model(img_data | prior_values)

        camera, gravity = out["camera"], out["gravity"]
        camera, out = self._post_process(camera, img_data, out)

        return {
            "camera": camera,
            "gravity": gravity,
            "covariance": out["covariance"],
            **{k: out[k] for k in out.keys() if "field" in k},
            **{k: out[k] for k in out.keys() if "confidence" in k},
            **{k: out[k] for k in out.keys() if "uncertainty" in k},
        }
