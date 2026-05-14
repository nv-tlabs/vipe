# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

from vipe.config.base_schema import BaseConfigSchema, Field
from vipe.config.slam import SLAMConfig

FrameAttributeName = Literal["rgb", "instance", "depth", "pcd", "rectified"]


class InstanceInitConfig(BaseConfigSchema):
    kf_gap_sec: float = Field(gt=0.0)
    phrases: list[str] = Field(min_length=1)
    add_sky: bool


class DefaultInitConfig(BaseConfigSchema):
    camera_type: Literal["pinhole", "panorama", "simple_divisional", "mei"]
    intrinsics: Literal["geocalib", "gt"]
    instance: InstanceInitConfig | None


class PanoramaInitConfig(BaseConfigSchema):
    instance: InstanceInitConfig | None


class VirtualCameraConfig(BaseConfigSchema):
    height: int = Field(ge=1)
    fovx: float = Field(gt=0.0, lt=180.0)
    fovy: float = Field(gt=0.0, lt=180.0)
    num_views: int = Field(ge=1)
    top: bool
    bottom: bool


class PostConfig(BaseConfigSchema):
    depth_align_model: str | None


class OutputConfig(BaseConfigSchema):
    path: str
    skip_exists: bool
    save_artifacts: bool
    save_slam_map: bool = False
    save_viz: bool
    viz_downsample: int = Field(ge=1)
    viz_attributes: list[list[FrameAttributeName]] = Field(min_length=1)


class DefaultPipelineConfig(BaseConfigSchema):
    instance: Literal["vipe.pipeline.default.DefaultAnnotationPipeline"]
    init: DefaultInitConfig
    slam: SLAMConfig
    post: PostConfig
    output: OutputConfig


class PanoramaPipelineConfig(BaseConfigSchema):
    instance: Literal["vipe.pipeline.panorama.PanoramaAnnotationPipeline"]
    init: PanoramaInitConfig
    virtual: VirtualCameraConfig
    slam: SLAMConfig
    output: OutputConfig
    post: PostConfig


PipelineConfig = Annotated[DefaultPipelineConfig | PanoramaPipelineConfig, Field(discriminator="instance")]
