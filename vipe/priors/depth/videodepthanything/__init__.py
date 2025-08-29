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

import numpy as np
import torch

from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .video_depth import VideoDepthAnything


class VideoDepthAnythingDepthModel(DepthEstimationModel):
    """
    https://github.com/DepthAnything/Video-Depth-Anything
    """

    def __init__(self, model: str = "vitl", input_size: int = 518) -> None:
        super().__init__()

        self.model_config = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }[model]

        self.is_metric = False
        if model == "vits":
            self.ckpt_url = "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
            self.use_fp32 = True
        elif model == "vitl":
            self.ckpt_url = "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth"
            self.use_fp32 = False
        else:
            raise ValueError(f"Model {model} not supported")

        self.input_size = input_size

        self.model = VideoDepthAnything(**self.model_config)
        self.model.load_state_dict(
            torch.hub.load_state_dict_from_url(self.ckpt_url, map_location="cpu"),
            strict=True,
        )
        self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.AFFINE_DISP

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        frame_list: list[np.ndarray] = unpack_optional(src.video_frame_list)
        depths = self.model.infer_video_depth(frame_list, input_size=self.input_size, fp32=self.use_fp32)  # [T, H, W]
        depths = torch.from_numpy(depths).float().cuda()
        return DepthEstimationResult(relative_inv_depth=depths)
