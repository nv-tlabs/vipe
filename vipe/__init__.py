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

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from omegaconf import OmegaConf

from vipe.pipeline import make_pipeline


def _version_info(version_string: str) -> tuple[int, ...]:
    release = version_string.split("+", 1)[0].split("-", 1)[0]
    return tuple(int(part) for part in release.split(".") if part.isdigit())


try:
    __version__ = version("vipe")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
__version_info__ = _version_info(__version__)

if not OmegaConf.has_resolver("eq"):
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
if not OmegaConf.has_resolver("neq"):
    OmegaConf.register_new_resolver("neq", lambda a, b: a != b)


def get_config_path() -> Path:
    return Path(__file__).parent.parent / "configs"
