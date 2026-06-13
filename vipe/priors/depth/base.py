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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from vipe.utils.cameras import CameraType


class DepthType(Enum):
    """
    Type of depth estimated.
    """

    # Inverse metric depth (DepthPro, Metric3D), scale is determined by focal.
    METRIC_DEPTH = "metric_depth"

    # Metric depth (UniDepth), scale is determined by focal but require re-running the model.
    MODEL_METRIC_DEPTH = "model_metric_depth"

    # Metric distance, scale is determined by focal but require re-running the model.
    MODEL_METRIC_DISTANCE = "model_metric_distance"

    # Affine-invariant inverse depth (DepthAnything, MoGE), affine needs to be solved per-estimation.
    AFFINE_DISP = "affine_disp"

    # Scale-invariant depth (DUSt3R), scale needs to be solved per-estimation.
    SCALE_DISP = "scale_disp"


@dataclass(slots=True, kw_only=True)
class DepthEstimationResult:
    """
    Dataclass for depth estimation results.

    - relative_inv_depth: The estimated inverse depth map ([B,], H, W) in relative scale.
        One has to estimate scale and offset of such estimation
    - metric_depth: The estimated depth map ([B,], H, W) in metric scale.
    - confidence: The confidence map ([B,], H, W).
    """

    relative_inv_depth: torch.Tensor | None = None
    metric_depth: torch.Tensor | None = None
    confidence: torch.Tensor | None = None


@dataclass(slots=True, kw_only=True)
class DepthEstimationInput:
    """
    Dataclass for depth estimation inputs.

    - rgb: The source image ([B,], H, W, 3), should be within 0-1 float.
    - video_frame_list: The list of video frames (H, W, 3), should be within 0-1 float, length is T.
        we use numpy here mainly to enforce CPU tensor.
    - intrinsics: The intrinsics of the camera.
    - camera_type: The type of camera.
    """

    rgb: torch.Tensor | None = None
    video_frame_list: list[np.ndarray] | None = None
    prompt_metric_depth: torch.Tensor | None = None
    intrinsics: torch.Tensor | None = None
    camera_type: CameraType = CameraType.PINHOLE
    frame_idx: int | None = None


class DepthEstimationModel(ABC):
    """
    Unified interface for depth prediction models.
    """

    @property
    def depth_type(self) -> DepthType:
        """
        Type of depth estimated.
        """
        raise NotImplementedError

    @property
    def supported_camera_types(self) -> list[CameraType]:
        """
        Supported camera types.
        """
        return [CameraType.PINHOLE]

    @abstractmethod
    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """
        Estimate a single optical flow result from two images.
        """


class DummyDepthModel(DepthEstimationModel):
    """
    Dummy model that would not predict anything.
    """

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        return DepthEstimationResult()
