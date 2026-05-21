# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from vipe.config.base_schema import BaseConfigSchema, Field


class BAConfig(BaseConfigSchema):
    """Bundle-adjustment solver and robust loss options."""

    dense_disp_alpha: float = Field(
        ge=0.0,
        description="Weight for dense-disparity regularization during bundle adjustment.",
    )
    fused: bool = Field(description="Use the fused CUDA bundle-adjustment path; unsupported layouts raise an error.")
    intrinsics_damping_scale: float = Field(
        gt=0.0,
        description="Multiplier for damping applied to optimized camera intrinsics.",
    )
    robust_kernel: Literal["huber", "tukey", "gnc_tls"] | None = Field(
        description="Robust loss for dense-flow residuals. Set to null for L2 residuals."
    )
    robust_kernel_threshold: float = Field(
        gt=0.0,
        description="Robust-kernel threshold in 1/8-resolution feature-map pixels.",
    )
    gnc_mu_init: float = Field(gt=0.0, description="Initial mu value for the GNC-TLS robust-kernel schedule.")
    gnc_mu_step: float = Field(gt=0.0, description="Multiplicative mu update step for GNC-TLS.")
    gnc_mu_max: float = Field(gt=0.0, description="Maximum mu value for GNC-TLS.")
    gnc_n_mu_steps: int = Field(ge=1, description="Number of GNC-TLS continuation steps.")
    gnc_gn_iters_per_mu: int = Field(
        ge=1,
        description="Number of Gauss-Newton iterations to run for each GNC-TLS mu value.",
    )


class SparseTracksConfig(BaseConfigSchema):
    """Sparse feature-track backend used to seed or support SLAM."""

    name: Literal["dummy", "cuvslam"] = Field(
        description="Sparse-track provider. dummy disables external sparse tracks; cuvslam uses NVIDIA cuVSLAM."
    )


class SLAMConfig(BaseConfigSchema):
    """DROID-style SLAM frontend, backend, map extraction, and metric-depth options."""

    buffer: int = Field(ge=1, description="Maximum number of keyframes stored in the SLAM graph buffer.")
    beta: float = Field(
        ge=0.0,
        description="Relative weighting of translation and rotation when measuring frame motion from optical flow.",
    )
    filter_thresh: float = Field(
        ge=0.0,
        description="Motion-filter threshold for accepting an incoming frame as a candidate keyframe.",
    )
    warmup: int = Field(ge=0, description="Number of accepted keyframes used before regular frontend updates begin.")
    keyframe_thresh: float = Field(
        ge=0.0,
        description="Frontend motion threshold below which the second-newest keyframe is removed.",
    )
    frontend_thresh: float = Field(ge=0.0, description="Distance threshold for adding frontend proximity edges.")
    frontend_window: int = Field(ge=1, description="Number of recent keyframes kept active in the frontend window.")
    frontend_radius: int = Field(ge=1, description="Frame-neighborhood radius forced into the frontend graph.")
    frontend_nms: int = Field(ge=1, description="Non-max suppression radius for frontend proximity edges.")
    seq_init: bool = Field(description="Initialize poses sequentially before regular graph optimization.")
    frontend_backend_iters: list[int] = Field(
        description="Accepted-keyframe counts at which the backend runs during frontend initialization."
    )
    backend_thresh: float = Field(ge=0.0, description="Distance threshold for adding backend proximity edges.")
    backend_radius: int = Field(ge=1, description="Frame-neighborhood radius forced into the backend graph.")
    backend_nms: int = Field(ge=1, description="Non-max suppression radius for backend proximity edges.")
    backend_iters: int = Field(ge=1, description="Number of backend optimization iterations.")
    init_disp: float = Field(gt=0.0, description="Initial inverse-depth value assigned to new keyframes.")
    optimize_intrinsics: bool = Field(description="Optimize camera intrinsics during bundle adjustment.")
    optimize_rig_rotation: bool = Field(description="Optimize rig rotations for multi-view inputs.")
    cross_view: bool = Field(
        description="Add cross-view reprojection factors for multi-view or panorama-derived inputs."
    )
    cross_view_idx: list[int] | None = Field(
        description="Optional cross-view index selection. Set to null to use the default neighboring-view selection."
    )
    adaptive_cross_view: bool = Field(description="Recompute cross-view pairs in the backend using current geometry.")
    infill_chunk_size: int = Field(ge=1, description="Chunk size for dense trajectory and disparity infill.")
    infill_dense_disp: bool = Field(description="Also optimize dense disparity while filling non-keyframe outputs.")
    map_filter_thresh: float = Field(
        ge=0.0,
        description="Depth-consistency threshold used when filtering SLAM-map points and extracting dense disparity.",
    )
    visualize: bool = Field(description="Stream SLAM internals to rerun for debugging.")
    keyframe_depth: str | None = Field(
        description="Metric depth model used on keyframes to recover scale. Examples include metric3d-small, "
        "unidepth-l, moge, and dav3. Set to null to skip keyframe metric-depth recovery."
    )
    ba: BAConfig = Field(description="Bundle-adjustment solver options.")
    sparse_tracks: SparseTracksConfig = Field(description="Sparse-track backend options.")

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
