# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pydantic import (
    ConfigDict,
    Field,  # re-export
)

__all__ = ["BaseConfigSchema", "Field", "config_to_primitive"]


def config_to_primitive(config: Any, resolve: bool = True) -> Any:
    if config is None:
        return None
    if isinstance(config, BaseConfigSchema):
        return config.model_dump()
    if isinstance(config, PydanticBaseModel):
        return config.model_dump()
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=resolve)
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, list):
        return list(config)
    return config


class BaseConfigSchema(PydanticBaseModel):
    """Base class for typed ViPE config structs."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_dictconfig(self) -> DictConfig:
        return OmegaConf.create(self.model_dump())

    def __hash__(self) -> int:
        return hash((type(self), self.__repr__()))
