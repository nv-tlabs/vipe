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

"""Cache for heavy, stateless models, owned by the pipeline.

The annotation pipeline is run once per video stream, and several large
networks (the metric depth model, GeoCalib, and the SAM / GroundingDINO / AOT
networks behind TrackAnything) were previously rebuilt from scratch for every
stream. Loading those weights from disk and transferring them to the GPU is the
dominant cost of the gap between "Processing <name>" and the first progress
bar tick.

These networks are used strictly for inference (``eval()`` + ``no_grad``), so a
single instance can be shared across all streams. A ``Pipeline`` holds one
``ModelCache`` and threads it into the processors and the SLAM system it builds;
the cache builds each model once, on first request, and hands back the same
instance afterwards. Only the *weight-holding* networks are cached here;
per-video state (SAM image embeddings, the AOT memory bank, tracking
bookkeeping) is rebuilt per stream by the callers so nothing leaks between
videos.
"""

import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelCache:
    """Caches fully-constructed models keyed by an arbitrary string.

    The cache stores whatever the builder returns by reference; it does not copy
    weights, so a cached model shares its GPU memory across every caller that
    requests the same key.
    """

    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def get(self, key: str, builder: Callable[[], T]) -> T:
        """Return the cached model for ``key``, building it with ``builder`` on a miss."""
        if key not in self._models:
            logger.info(f"Building and caching model '{key}'")
            self._models[key] = builder()
        return self._models[key]  # type: ignore[return-value]

    def clear(self) -> None:
        """Drop all cached models (e.g. to release GPU memory)."""
        self._models.clear()
