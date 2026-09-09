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

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CachedFrame:
    """CPU copy of the per-frame tensors the infill needs (SLAM resolution)."""

    # (n_views, 3, H, W) fp16 image, as produced by SLAMSystem._precompute_features.
    images: torch.Tensor
    # (n_views, H // 8, W // 8) bool invalidity mask, or None if the stream has no masks.
    buffer_masks: torch.Tensor | None


class RollingFrameCache:
    """Bounded CPU cache of raw frames between two retirements.

    The long-sequence recipe consumes the video in a single pass, but the
    infill for a retiring keyframe chunk needs the raw frames of that chunk's
    time span again (their DROID features are re-encoded into spare buffer
    slots). This cache holds exactly the frames whose poses have not been
    filled yet; every retirement pops its span, so the size stays at
    O(retire_chunk / keyframe_rate) frames.
    """

    def __init__(self) -> None:
        self._frames: dict[int, CachedFrame] = {}

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, frame_idx: int, images: torch.Tensor, buffer_masks: torch.Tensor | None) -> None:
        """Store a frame's tensors on CPU (they arrive on the SLAM device)."""
        self._frames[frame_idx] = CachedFrame(
            images=images.cpu(),
            buffer_masks=buffer_masks.cpu() if buffer_masks is not None else None,
        )

    def pop(self, frame_idx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Remove and return a frame's tensors, moved back to the SLAM device."""
        cached = self._frames.pop(frame_idx)
        images = cached.images.to(device)
        masks = cached.buffer_masks.to(device) if cached.buffer_masks is not None else None
        return images, masks
