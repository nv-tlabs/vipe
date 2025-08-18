from typing import Literal

import numpy as np
import torch

from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .unik3d import UniK3D
from .utils.camera import Spherical


class Unik3DModel(DepthEstimationModel):
    def __init__(self, type: Literal["s", "b", "l"] = "l") -> None:
        super().__init__()
        self.model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-vit{type}")
        self.model.resolution_level = 9
        self.model.interpolation_mode = "bilinear"
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DISTANCE

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.focal_length is None, "This is only intended for 360 panoramas"

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb = rgb.moveaxis(-1, 1) * 255.0
        H, W = rgb.shape[-2:]
        hfov2 = np.pi
        camera = Spherical(params=torch.tensor([0, 0, 0, 0, W, H, hfov2, H / W * hfov2]).float().cuda())
        outputs = self.model.infer(rgb, camera=camera, normalize=True)

        pred_distance = outputs["distance"][0]
        confidence = outputs["confidence"][0]

        if not batch_dim:
            pred_distance, confidence = pred_distance[0], confidence[0]

        return DepthEstimationResult(
            metric_depth=pred_distance,
            confidence=confidence,
        )
