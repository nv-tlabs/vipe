# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import field_validator

from vipe.config.base_schema import BaseConfigSchema, Field


class BaseStreamListConfig(BaseConfigSchema):
    base_path: str
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=-1)
    frame_skip: int = Field(ge=1)
    cached: bool

    @field_validator("frame_end")
    @classmethod
    def validate_frame_end(cls, value: int) -> int:
        if value == -1 or value >= 0:
            return value
        raise ValueError("frame_end must be -1 or a non-negative frame index")


class RawMP4StreamListConfig(BaseStreamListConfig):
    instance: Literal["vipe.streams.raw_mp4_stream.RawMP4StreamList"]


class FrameDirStreamListConfig(BaseStreamListConfig):
    instance: Literal["vipe.streams.frame_dir_stream.FrameDirStreamList"]


StreamsConfig = Annotated[RawMP4StreamListConfig | FrameDirStreamListConfig, Field(discriminator="instance")]
