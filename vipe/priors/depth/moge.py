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

import os
from pathlib import Path

import torch

try:
    from moge.model.v1 import MoGeModel
except ModuleNotFoundError:
    MoGeModel = None

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType

# MoGe-2 checkpoints, resolved from HuggingFace like every other prior in this
# package (Depth Anything 3, MoGe-1). `moge-2-vitl` is the metric,
# no-normals ViT-L.
MOGE2_HF_REPOS = {"l": "Ruicheng/moge-2-vitl"}
MOGE2_HF_FILENAME = "model.pt"

#: Optional offline fast path: a checkpoint already sitting in the torch hub cache
#: under this name is used without contacting HuggingFace.
MOGE2_CHECKPOINT_NAMES = {"l": "moge-2-vitl.pt"}


def focal_length_to_fov_degrees(focal_length: float, image_width: float) -> float:
    """Compute horizontal field of view from focal length."""
    fov_rad = 2 * torch.atan(torch.tensor(image_width / (2 * focal_length)))
    fov_deg = torch.rad2deg(fov_rad)
    return fov_deg.item()


class MogeModel(DepthEstimationModel):
    """https://github.com/microsoft/MoGe

    ``version=1`` runs MoGe-1 through the optional pip package. ``version=2`` runs
    MoGe-2, whose inference code is vendored under ``moge2/`` (no pip dependency),
    with the checkpoint resolved locally or from HuggingFace on first use.

    The two differ in more than weights: MoGe-2 has a dedicated ``metric_scale``
    head, so its depth is genuinely metric, whereas MoGe-1's is only metric once
    something downstream solves the scale.
    """

    def __init__(
        self,
        version: int = 1,
        variant: str = "l",
        num_tokens: int | None = None,
        use_fp16: bool = True,
        fp16_weights: bool = False,
    ) -> None:
        super().__init__()
        self.version = version
        # Inference knobs, surfaced here because they are the dominant
        # accuracy/throughput lever: for moge-2-vitl the token range is
        # [1200, 3600] and num_tokens=None means resolution_level=9, i.e. 3600.
        # `use_fp16` is upstream's autocast around the ViT forward; `fp16_weights`
        # additionally casts the weights, which makes infer() skip autocast and
        # run natively in half precision (~12% faster, no measured accuracy cost).
        self.num_tokens = num_tokens
        self.use_fp16 = use_fp16

        if version == 1:
            if MoGeModel is None:
                raise RuntimeError(
                    "moge is not found in the environment. You can install it via pip install `git+https://github.com/microsoft/MoGe.git`"
                )
            self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
        elif version == 2:
            from .moge2 import MoGeModel as MoGe2Model

            self.model = MoGe2Model.from_pretrained(self._checkpoint_path(variant))
        else:
            raise ValueError(f"Unknown MoGe version: {version}")

        self.model = self.model.cuda().eval()
        if fp16_weights:
            if version != 2:
                raise ValueError("fp16_weights is only supported for MoGe-2")
            self.model = self.model.half()

    @staticmethod
    def _checkpoint_path(variant: str) -> Path:
        """Local path of the MoGe-2 checkpoint, downloading it on first use.

        Resolution order, cheapest first:

        1. ``VIPE_MOGE2_CHECKPOINT`` -- an explicit file, for air-gapped runs or
           for pinning an exact checkpoint in a benchmark.
        2. ``$TORCH_HOME/hub/moge2/<name>.pt`` -- an offline copy placed by hand.
        3. HuggingFace, cached by ``huggingface_hub`` -- the automatic path, and
           what every other prior in this package does (DA3, MoGe-1), so
           a fresh install needs no manual download step.
        """
        # Truthiness, not `is not None`: an exported-but-empty VIPE_MOGE2_CHECKPOINT
        # would otherwise resolve to Path("") == ".", which exists as a directory and
        # silently becomes the "checkpoint".
        if override := os.environ.get("VIPE_MOGE2_CHECKPOINT", "").strip():
            path = Path(override).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"VIPE_MOGE2_CHECKPOINT does not point at a file: {path}")
            return path

        if variant not in MOGE2_HF_REPOS:
            raise ValueError(f"Unknown MoGe-2 variant '{variant}', expected one of {sorted(MOGE2_HF_REPOS)}")

        cached = Path(torch.hub.get_dir()) / "moge2" / MOGE2_CHECKPOINT_NAMES[variant]
        if cached.exists():
            return cached

        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "huggingface_hub is required to fetch MoGe-2 weights. Install it, or set "
                "VIPE_MOGE2_CHECKPOINT to a local checkpoint."
            ) from exc

        return Path(hf_hub_download(repo_id=MOGE2_HF_REPOS[variant], filename=MOGE2_HF_FILENAME))

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.camera_type == CameraType.PINHOLE, "MoGe only supports pinhole cameras"

        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        w = rgb.shape[2]
        input_image_for_depth = rgb.moveaxis(-1, 1)
        fov_x = focal_length_to_fov_degrees(focal_length, w)

        with torch.no_grad():
            if self.version == 1:
                moge_output_full = self.model.infer(input_image_for_depth, fov_x=fov_x)
            else:
                moge_output_full = self.model.infer(
                    input_image_for_depth,
                    fov_x=fov_x,
                    num_tokens=self.num_tokens,
                    use_fp16=self.use_fp16,
                    # Masking is applied below instead: MoGe-2's own apply_mask
                    # writes +inf into invalid pixels, and the SLAM buffer turns
                    # a positive depth into a real disparity prior -- an inf
                    # would silently become a "point at infinity" constraint
                    # rather than "no prior". force_projection only recomputes
                    # `points`, which we discard, and provably leaves `depth`
                    # untouched.
                    apply_mask=False,
                    force_projection=False,
                )

        moge_depth_hw_full = moge_output_full["depth"]
        moge_mask_hw_full = moge_output_full["mask"]

        if self.version == 1:
            # Process depth
            moge_depth_tensor = torch.nan_to_num(moge_depth_hw_full, nan=1e4)
            moge_depth_tensor = torch.clamp(moge_depth_tensor, min=0, max=1e4)

            moge_depth_tensor = moge_depth_tensor * moge_mask_hw_full.float()
        else:
            # Zero means "no prior" to the SLAM buffer, so every non-finite or
            # non-positive prediction must land on exactly 0.
            moge_depth_tensor = torch.nan_to_num(moge_depth_hw_full, nan=0.0, posinf=0.0, neginf=0.0)
            moge_depth_tensor = torch.where(
                moge_mask_hw_full & (moge_depth_tensor > 0),
                moge_depth_tensor,
                torch.zeros_like(moge_depth_tensor),
            )

        if not batch_dim:
            moge_depth_tensor = moge_depth_tensor.squeeze(0)
            moge_mask_hw_full = moge_mask_hw_full.squeeze(0)

        return DepthEstimationResult(metric_depth=moge_depth_tensor)
