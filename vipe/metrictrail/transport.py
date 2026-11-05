# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DPVO raster transport: reflect-pad DPVO's input to a multiple of 4.

The depth model and SharedK see the input raster unchanged; only DPVO's
transport is padded: reflected image border, replicated depth, no metric unary
on the border (mask False), principal point shifted by the left/top padding.
Rasters that are already divisible by 4 take the unpadded path. Examples:
280x504 and 288x512 -> no padding; 388x518 -> (1, 1, 0, 0); 434x574 and
378x518 -> (1, 1, 1, 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as torch_f


def dpvo_transport_padding(input_hw: Sequence[int]) -> tuple[int, int, int, int]:
    """Reflect padding (left, right, top, bottom) that makes DPVO's raster divisible by 4."""

    height, width = (int(input_hw[0]), int(input_hw[1]))
    pad_w = (-width) % 4
    pad_h = (-height) % 4
    return (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)


def transport_hw(input_hw: Sequence[int], padding_ltrb: Sequence[int]) -> tuple[int, int]:
    left, right, top, bottom = (int(value) for value in padding_ltrb)
    return (int(input_hw[0]) + top + bottom, int(input_hw[1]) + left + right)


class ReflectPaddedDPVOTransport:
    """Pad only DPVO's transport while the depth model and SharedK keep the raster."""

    def __init__(self, slam: Any, source_hw: Sequence[int], padding_ltrb: Sequence[int]) -> None:
        self.slam = slam
        self.source_hw = (int(source_hw[0]), int(source_hw[1]))
        self.padding_ltrb = tuple(int(item) for item in padding_ltrb)
        self.transport_hw = transport_hw(self.source_hw, self.padding_ltrb)

    def __call__(self, timestamp: int, image_bgr: torch.Tensor, intrinsics: torch.Tensor, **kwargs: Any) -> Any:
        metric_depth = kwargs.pop("metric_depth")
        metric_mask = kwargs.pop("metric_mask")
        if kwargs:
            raise TypeError(f"unexpected padded DPVO transport inputs: {sorted(kwargs)}")
        if tuple(image_bgr.shape) != (3, *self.source_hw):
            raise ValueError("padded DPVO transport received a noncanonical image raster")
        if tuple(metric_depth.shape) != self.source_hw or tuple(metric_mask.shape) != self.source_hw:
            raise ValueError("padded DPVO transport received noncanonical metric depth")
        if tuple(intrinsics.shape) != (4,):
            raise ValueError("padded DPVO transport intrinsics must be [fx,fy,cx,cy]")
        image = torch_f.pad(image_bgr, self.padding_ltrb, mode="reflect").contiguous()
        depth = torch_f.pad(metric_depth[None, None], self.padding_ltrb, mode="replicate")[0, 0].contiguous()
        valid = torch_f.pad(metric_mask[None, None], self.padding_ltrb, mode="constant", value=False)[0, 0].contiguous()
        shifted = intrinsics.clone()
        shifted[2] += float(self.padding_ltrb[0])
        shifted[3] += float(self.padding_ltrb[2])
        left, right, top, bottom = self.padding_ltrb
        if (
            tuple(image.shape) != (3, *self.transport_hw)
            or tuple(depth.shape) != self.transport_hw
            or tuple(valid.shape) != self.transport_hw
            or (top > 0 and bool(valid[:top].any()))
            or (bottom > 0 and bool(valid[-bottom:].any()))
            or (left > 0 and bool(valid[:, :left].any()))
            or (right > 0 and bool(valid[:, -right:].any()))
        ):
            raise AssertionError("padded DPVO transport contract drifted")
        return self.slam(int(timestamp), image, shifted, metric_depth=depth, metric_mask=valid)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.slam, name)
