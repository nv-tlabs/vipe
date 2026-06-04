# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import model_validator

from vipe.config.base_schema import BaseConfigSchema, Field
from vipe.config.slam import SLAMConfig

FrameAttributeName = Literal["rgb", "instance", "depth", "pcd", "rectified"]


class InstanceInitConfig(BaseConfigSchema):
    """Object and sky-mask initialization used by the segmentation stage."""

    kf_gap_sec: float = Field(
        gt=0.0,
        description="Minimum time gap, in seconds, between keyframes used to initialize instance segmentation.",
    )
    phrases: list[str] = Field(
        min_length=1,
        description="Text prompts passed to the open-vocabulary detector for objects that should be segmented.",
    )
    add_sky: bool = Field(
        description="Add a sky mask to the instance segmentation output when the detector supports it."
    )


class DefaultInitConfig(BaseConfigSchema):
    """Initialization options for the default pinhole and wide-angle pipelines."""

    camera_type: Literal["pinhole", "panorama", "simple_divisional", "mei"] = Field(
        description="Camera model used by SLAM and projection code. Use mei for wide-angle/fisheye input."
    )
    intrinsics: Literal["geocalib", "gt"] = Field(
        description="Source of camera intrinsics. geocalib estimates intrinsics; gt expects each frame to provide them."
    )
    instance: InstanceInitConfig | None = Field(
        description="Instance-segmentation initialization. Set to null to skip instance masks."
    )
    async_prefetch: bool = Field(
        default=True,
        description="Prefetch initialized frames asynchronously before SLAM. Set false to use serialized caching.",
    )
    prefetch_queue_size: int = Field(
        default=16,
        ge=1,
        description="Maximum number of initialized frames the async producer may keep ready ahead of SLAM.",
    )


class PanoramaInitConfig(BaseConfigSchema):
    """Initialization options for the panorama pipeline."""

    instance: InstanceInitConfig | None = Field(
        description="Instance-segmentation initialization. Set to null to skip instance masks."
    )


class VirtualCameraConfig(BaseConfigSchema):
    """Perspective cameras sampled from a 360-degree panorama for SLAM."""

    height: int = Field(ge=1, description="Height, in pixels, of each virtual perspective view.")
    fovx: float = Field(gt=0.0, lt=180.0, description="Horizontal field of view of each virtual view, in degrees.")
    fovy: float = Field(gt=0.0, lt=180.0, description="Vertical field of view of each virtual view, in degrees.")
    num_views: int = Field(ge=1, description="Number of evenly spaced horizontal virtual views.")
    top: bool = Field(description="Add an upward-looking virtual view.")
    bottom: bool = Field(description="Add a downward-looking virtual view.")


class PostConfig(BaseConfigSchema):
    """Depth post-processing options."""

    depth_align_model: str | None = Field(
        description="Depth model or alignment recipe used after SLAM. Examples include adaptive_unidepth-l, "
        "adaptive_unidepth-l_svda, adaptive_moge_vda, mvd_dav3, dap, and unik3d. Set to null for pose-only output."
    )


class OutputConfig(BaseConfigSchema):
    """Output paths and artifact/visualization controls."""

    path: str = Field(
        description="Directory where ViPE writes artifacts, visualization videos, and optional SLAM maps."
    )
    skip_exists: bool = Field(description="Skip a sequence when the expected output already exists.")
    save_artifacts: bool = Field(
        description="Save reusable RGB, pose, intrinsics, depth, and mask artifacts for visualization or downstream use."
    )
    save_slam_map: bool = Field(
        default=False,
        description="Save the sparse SLAM reconstruction map for lightweight COLMAP conversion.",
    )
    save_viz: bool = Field(description="Render MP4 visualization videos for the configured visualization attributes.")
    viz_downsample: int = Field(ge=1, description="Downsample factor applied when rendering visualization videos.")
    viz_attributes: list[list[FrameAttributeName]] = Field(
        min_length=1,
        description="Groups of frame attributes to render into visualization videos. Each inner list becomes one panel.",
    )


class DefaultPipelineConfig(BaseConfigSchema):
    """Default annotation pipeline for pinhole and wide-angle videos."""

    instance: Literal["vipe.pipeline.default.DefaultAnnotationPipeline"] = Field(
        description="Implementation class for the default annotation pipeline."
    )
    init: DefaultInitConfig = Field(description="Initial camera and instance-mask setup.")
    slam: SLAMConfig = Field(description="SLAM and bundle-adjustment configuration.")
    post: PostConfig = Field(description="Depth alignment and post-processing configuration.")
    output: OutputConfig = Field(description="Output artifact and visualization configuration.")

    @model_validator(mode="after")
    def normalize_fused_ba(self) -> DefaultPipelineConfig:
        # Monocular pipeline: always single-view; the fused kernel additionally needs a
        # pinhole camera model.
        self.slam.ba.fused = self.slam.resolve_fused(
            single_view=True,
            pinhole=self.init.camera_type == "pinhole",
        )
        return self


class PanoramaPipelineConfig(BaseConfigSchema):
    """Annotation pipeline for 360-degree panorama videos."""

    instance: Literal["vipe.pipeline.panorama.PanoramaAnnotationPipeline"] = Field(
        description="Implementation class for the panorama annotation pipeline."
    )
    init: PanoramaInitConfig = Field(description="Initial instance-mask setup for panorama input.")
    virtual: VirtualCameraConfig = Field(description="Virtual perspective views projected from each panorama frame.")
    slam: SLAMConfig = Field(description="SLAM and bundle-adjustment configuration for virtual views.")
    output: OutputConfig = Field(description="Output artifact and visualization configuration.")
    post: PostConfig = Field(description="Panorama depth estimation and post-processing configuration.")

    @model_validator(mode="after")
    def normalize_fused_ba(self) -> PanoramaPipelineConfig:
        # Virtual views are pinhole, but multiple oriented views form a non-identity rig
        # that the fused kernel cannot handle; only the degenerate single-view panorama
        # (one centered, identity-rig view) is fused-eligible.
        n_views = self.virtual.num_views + int(self.virtual.top) + int(self.virtual.bottom)
        self.slam.ba.fused = self.slam.resolve_fused(
            single_view=n_views == 1,
            pinhole=True,
        )
        return self


PipelineConfig = Annotated[DefaultPipelineConfig | PanoramaPipelineConfig, Field(discriminator="instance")]
