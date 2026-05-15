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
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from vipe._paths import get_config_path

if TYPE_CHECKING:
    from vipe.pipeline import make_pipeline as make_pipeline

__all__ = ["__version__", "__version_info__", "get_config_path", "make_pipeline"]


def _version_info(version_string: str) -> tuple[int, ...]:
    release = version_string.split("+", 1)[0].split("-", 1)[0]
    return tuple(int(part) for part in release.split(".") if part.isdigit())


try:
    __version__ = version("nvidia-vipe")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
__version_info__ = _version_info(__version__)

if not OmegaConf.has_resolver("eq"):
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
if not OmegaConf.has_resolver("neq"):
    OmegaConf.register_new_resolver("neq", lambda a, b: a != b)


def __getattr__(name: str) -> Any:
    if name == "make_pipeline":
        from vipe.pipeline import make_pipeline

        return make_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
