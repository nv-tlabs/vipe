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

from pathlib import Path

from omegaconf import OmegaConf

from vipe.pipeline import make_pipeline


__version__ = "0.1.1"
__version_info__ = (0, 1, 1)

if not OmegaConf.has_resolver("eq"):
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
if not OmegaConf.has_resolver("neq"):
    OmegaConf.register_new_resolver("neq", lambda a, b: a != b)


def get_config_path() -> Path:
    return Path(__file__).parent / "configs"
