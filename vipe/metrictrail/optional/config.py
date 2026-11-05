# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed optional intrinsics and file-based loop-closure settings."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from vipe.config.base_schema import BaseConfigSchema, Field

#: How the source image was turned into the input raster (see ``intrinsics.map_intrinsics``).
#: ``cover_crop``: one scale ``max(W/w, H/h)``, size rounded, integer centre crop (KITTI 280x504).
#: ``cover_affine``: the same scale with a sub-pixel centre crop (LaMARia).
#: ``stretch``: independent x/y scales, no crop (TUM 388x518; ViPE ``VideoFrame.resize``).
RasterMap = Literal["cover_crop", "cover_affine", "stretch"]


class KnownIntrinsicsConfig(BaseConfigSchema):
    """A known (GT/sensor) pinhole K that replaces the RGB-only SharedK calibration (one source, source pixels)."""

    # Give exactly one source: ``fx fy cx cy``, ``path``, or ``from_stream``. The K
    # is in source-image pixels. ``source_hw`` names that raster (for a file, the
    # file's own raster member is used when present); when it differs from the
    # input raster, ``raster_map`` must say how the input was made from the source.

    fx: float | None = Field(default=None, gt=0.0, description="Focal length x (source pixels).")
    fy: float | None = Field(default=None, gt=0.0, description="Focal length y (source pixels).")
    cx: float | None = Field(default=None, description="Principal point x (source pixels).")
    cy: float | None = Field(default=None, description="Principal point y (source pixels).")
    path: str | None = Field(
        default=None,
        description="K file: .txt (4 numbers fx fy cx cy, or a 3x3 matrix), .json or .npz "
        "(see docs/metrictrail_optional.md for the accepted keys).",
    )
    from_stream: bool = Field(
        default=False,
        description="Use the intrinsics carried by the input stream's frames (already at the input raster).",
    )
    source_hw: tuple[int, int] | None = Field(
        default=None,
        description="(height, width) of the raster the K refers to. Null: the K file's raster member, else the "
        "input raster (no mapping).",
    )
    raster_map: RasterMap | None = Field(
        default=None,
        description="How the input raster was made from the source image; required when the two differ.",
    )
    stream_tolerance_px: float = Field(
        default=1.0e-3,
        ge=0.0,
        description="from_stream only: largest per-frame deviation (pixels) from the first frame's K. MetricTrail "
        "tracks one shared K.",
    )

    @model_validator(mode="after")
    def _one_source(self) -> "KnownIntrinsicsConfig":
        explicit = [self.fx, self.fy, self.cx, self.cy]
        given = sum(value is not None for value in explicit)
        if given not in (0, 4):
            raise ValueError("known intrinsics: give all of fx, fy, cx, cy or none of them")
        sources = int(given == 4) + int(self.path is not None) + int(self.from_stream)
        if sources != 1:
            raise ValueError("known intrinsics: give exactly one of (fx, fy, cx, cy), path, from_stream")
        if self.from_stream and (self.source_hw is not None or self.raster_map is not None):
            raise ValueError("known intrinsics: stream intrinsics are already at the input raster")
        if self.source_hw is not None and min(self.source_hw) <= 0:
            raise ValueError(f"known intrinsics: invalid source_hw {self.source_hw}")
        return self


class LoopClosureOptions(BaseConfigSchema):
    """CPU loop closure from externally supplied frame pairs."""

    enabled: bool = Field(default=False, description="Apply file-based loop closure after saving the trajectory.")
    candidates: str | None = Field(
        default=None, description="External candidate JSON with frame pairs for the saved trajectory."
    )
    preset: Literal["kitti", "vbr"] = Field(default="kitti", description="CPU pose-graph weighting preset.")
    overwrite: bool = Field(default=False, description="Replace existing pose_lc artifacts.")

    @model_validator(mode="after")
    def _candidate_inputs(self) -> "LoopClosureOptions":
        if self.enabled and self.candidates is None:
            raise ValueError("enabled loop closure requires candidates")
        return self


class MetricTrailOptionalConfig(BaseConfigSchema):
    """MetricTrail's optional inputs/outputs; all off by default (the RGB-only MetricTrail pipeline)."""

    intrinsics: KnownIntrinsicsConfig | None = Field(
        default=None,
        description="Known (GT/sensor) intrinsics replacing SharedK. Null: RGB-only SharedK calibration.",
    )
    loop_closure: LoopClosureOptions = Field(
        default={},
        validate_default=True,
        description="CPU loop closure from external candidate files; writes pose_lc/<name>.{npz,json}. "
        "Disabled by default.",
    )
