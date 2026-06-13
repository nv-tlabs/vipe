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

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


def make_depth_model(model: str):
    if "-" not in model:
        model_name, model_sub = model, ""
    else:
        model_name, model_sub = model.split("-")

    if model_name == "metric3d":
        from .metric3d import Metric3DDepthModel

        return Metric3DDepthModel(version=2, model=model_sub)

    elif model_name == "unidepth":
        from .unidepth import UniDepth2Model

        return UniDepth2Model(type=model_sub)

    elif model_name == "moge":
        from .moge import MogeModel

        return MogeModel()

    elif model_name == "dav3":
        from .dav3 import DepthAnything3Model

        return DepthAnything3Model()


    elif model_name == "pi3x_moge2" or (model_name == "pi3x" and model_sub == "moge2"):
        from .pi3x_moge2 import Pi3XMoGe2DepthModel
        return Pi3XMoGe2DepthModel()

    elif model_name == "cached":
        import os
        from .cached import CachedDepthModel
        cache_path = os.environ.get("SANA_WM_CACHED_DEPTH_PATH", "")
        if not cache_path:
            raise ValueError("cached depth model requires SANA_WM_CACHED_DEPTH_PATH env var")
        return CachedDepthModel(cache_path=cache_path)

    else:
        raise ValueError(f"Unknown depth model: {model}")
