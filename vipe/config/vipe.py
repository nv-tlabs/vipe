# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from vipe.config.base_schema import BaseConfigSchema
from vipe.config.pipeline import PipelineConfig
from vipe.config.streams import StreamsConfig


class ViPEConfig(BaseConfigSchema):
    ray: bool
    prefilter: bool
    streams: StreamsConfig
    pipeline: PipelineConfig
