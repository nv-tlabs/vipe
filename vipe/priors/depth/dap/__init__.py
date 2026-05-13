# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapter for "Depth Any Panoramas" (DAP), Lin et al., CVPR 2026.

https://github.com/Insta360-Research-Team/DAP

Pretrained weights are pulled from https://huggingface.co/Insta360-Research/DAP-weights
on first use (cached by `huggingface_hub`).
"""

import logging
from typing import Literal

import torch
import torch.nn.functional as F

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .model import make_dap_model

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
logger = logging.getLogger(__name__)


class DAPModel(DepthEstimationModel):
    """Depth Any Panoramas — equirectangular metric depth from a single panorama.

    Expects float32 RGB in [0, 1], shape (H, W, 3) or (B, H, W, 3). Returns
    metric distance (radial, not depth-from-image-plane) populated into the
    ``metric_depth`` field of :class:`DepthEstimationResult`, plus the inverse
    of DAP's occlusion/invalid mask as ``confidence``.

    The raw model output is not strictly metric (it is depth / 100 in the
    pretrained checkpoint), so we leave the SLAM-driven alignment in
    :class:`MergedPanoramaVideoStream` to recover absolute scale.
    """

    HF_REPO = "Insta360-Research/DAP-weights"
    HF_FILENAME = "model.pth"

    def __init__(
        self,
        midas_model_type: Literal["vits", "vitb", "vitl"] = "vitl",
        input_size: int = 518,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()
        if weights_path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "huggingface_hub is required to fetch DAP weights. "
                    "Install it with `pip install huggingface_hub` or pass "
                    "`weights_path=...` explicitly."
                ) from exc
            weights_path = hf_hub_download(repo_id=self.HF_REPO, filename=self.HF_FILENAME)

        model = make_dap_model(
            midas_model_type=midas_model_type,
            fine_tune_type="hypersim",
            min_depth=0.01,
            max_depth=1.0,
            train_decoder=True,
        )

        state = torch.load(weights_path, map_location="cpu")
        # Strip DataParallel "module." prefix if present.
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module.") :]: v for k, v in state.items()}
        m_state = model.state_dict()
        missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if k in m_state}, strict=False)
        if missing:
            logger.warning("DAP checkpoint did not populate %d model parameters/buffers.", len(missing))
        if unexpected:
            logger.warning("DAP checkpoint had %d unexpected parameters/buffers.", len(unexpected))

        self.model = model.cuda().eval()
        self.input_size = input_size
        self._patch_size = int(getattr(self.model.core, "patch_size", 14))

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DISTANCE

    @property
    def supported_camera_types(self) -> list[CameraType]:
        return [CameraType.PANORAMA]

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.intrinsics is None, "This is only intended for 360 panoramas"
        assert src.camera_type == CameraType.PANORAMA, "DAP only supports 360 panoramas"

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        _, H, W, _ = rgb.shape

        # MiDaS-style "lower_bound" resize: fit the larger side to (2*input_size, input_size),
        # keep aspect ratio, round to a multiple of patch_size.
        target_h, target_w = self.input_size, self.input_size * 2
        scale = max(target_h / H, target_w / W)
        new_h = max(self._patch_size, int(round(H * scale / self._patch_size)) * self._patch_size)
        new_w = max(self._patch_size, int(round(W * scale / self._patch_size)) * self._patch_size)

        x = rgb.moveaxis(-1, 1)  # (B, 3, H, W)
        x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)
        x = (x - _IMAGENET_MEAN.to(x)) / _IMAGENET_STD.to(x)

        with torch.inference_mode():
            outputs = self.model(x)

        pred = outputs["pred_depth"]  # (B, 1, h, w), >= 0
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)
        pred_distance = pred[:, 0]

        confidence: torch.Tensor | None = None
        if "pred_mask" in outputs:
            # DAP's pred_mask is the *invalid* probability; flip to confidence.
            mask = F.interpolate(outputs["pred_mask"].float(), size=(H, W), mode="bilinear", align_corners=True)
            confidence = (1.0 - mask)[:, 0].clamp(0.0, 1.0)

        if not batch_dim:
            pred_distance = pred_distance[0]
            if confidence is not None:
                confidence = confidence[0]

        return DepthEstimationResult(
            metric_depth=pred_distance,
            confidence=confidence,
        )
