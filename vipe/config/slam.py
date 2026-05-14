# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from vipe.config.base_schema import BaseConfigSchema, Field


class BAConfig(BaseConfigSchema):
    dense_disp_alpha: float = Field(ge=0.0)
    intrinsics_damping_scale: float = Field(gt=0.0)
    robust_kernel: Literal["huber", "tukey", "gnc_tls"] | None
    robust_kernel_threshold: float = Field(gt=0.0)
    gnc_mu_init: float = Field(gt=0.0)
    gnc_mu_step: float = Field(gt=0.0)
    gnc_mu_max: float = Field(gt=0.0)
    gnc_n_mu_steps: int = Field(ge=1)
    gnc_gn_iters_per_mu: int = Field(ge=1)


class SparseTracksConfig(BaseConfigSchema):
    name: Literal["dummy", "cuvslam"]


class SLAMConfig(BaseConfigSchema):
    buffer: int = Field(ge=1)
    beta: float = Field(ge=0.0)
    filter_thresh: float = Field(ge=0.0)
    warmup: int = Field(ge=0)
    keyframe_thresh: float = Field(ge=0.0)
    frontend_thresh: float = Field(ge=0.0)
    frontend_window: int = Field(ge=1)
    frontend_radius: int = Field(ge=1)
    frontend_nms: int = Field(ge=1)
    seq_init: bool
    frontend_backend_iters: list[int]
    backend_thresh: float = Field(ge=0.0)
    backend_radius: int = Field(ge=1)
    backend_nms: int = Field(ge=1)
    backend_iters: int = Field(ge=1)
    init_disp: float = Field(gt=0.0)
    optimize_intrinsics: bool
    optimize_rig_rotation: bool
    cross_view: bool
    cross_view_idx: list[int] | None
    adaptive_cross_view: bool
    infill_chunk_size: int = Field(ge=1)
    infill_dense_disp: bool
    map_filter_thresh: float = Field(ge=0.0)
    visualize: bool
    keyframe_depth: str | None
    ba: BAConfig
    sparse_tracks: SparseTracksConfig

    @field_validator("frontend_backend_iters")
    @classmethod
    def validate_frontend_backend_iters(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("frontend_backend_iters must not be empty")
        if any(item < 1 for item in value):
            raise ValueError("frontend_backend_iters must contain positive frame counts")
        return value

    @field_validator("cross_view_idx")
    @classmethod
    def validate_cross_view_idx(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(item < 0 for item in value):
            raise ValueError("cross_view_idx must contain non-negative view indices")
        return value
