# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from vipe._paths import get_config_path
from vipe.config.vipe import ViPEConfig


def _default_config_dir() -> Path:
    return get_config_path()


def register_config_resolvers() -> None:
    if not OmegaConf.has_resolver("eq"):
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    if not OmegaConf.has_resolver("neq"):
        OmegaConf.register_new_resolver("neq", lambda a, b: a != b)


def parse_untyped_config(
    config_name: str = "default",
    hydra_args: Sequence[str] = (),
    config_dir: str | Path | None = None,
) -> DictConfig:
    register_config_resolvers()

    config_dir_path = Path(config_dir) if config_dir is not None else Path("configs")
    config_registry_dir = (Path(config_dir).resolve() if config_dir is not None else _default_config_dir()).resolve()
    config_name_path = Path(str(config_name))
    hydra_args_list = list(hydra_args)

    if config_name_path.is_absolute():
        compose_config_dir = config_name_path.parent
        compose_config_name = config_name_path.name
        hydra_args_list.append(f"hydra.searchpath=[{config_registry_dir}]")
    else:
        if config_name_path.is_relative_to(config_dir_path):
            config_name_path = config_name_path.relative_to(config_dir_path)
        compose_config_dir = config_registry_dir
        compose_config_name = str(config_name_path)

    with hydra.initialize_config_dir(config_dir=str(compose_config_dir), version_base=None):
        config = hydra.compose(config_name=compose_config_name, overrides=hydra_args_list)

    OmegaConf.resolve(config)
    return config


def validate_typed_config(config: DictConfig, config_name: str = "default") -> ViPEConfig:
    register_config_resolvers()
    OmegaConf.resolve(config)
    return ViPEConfig.model_validate(config, context={"config_name": config_name})


def parse_typed_config(
    config_name: str = "default",
    hydra_args: Sequence[str] = (),
    config_dir: str | Path | None = None,
) -> ViPEConfig:
    untyped_config = parse_untyped_config(config_name=config_name, hydra_args=hydra_args, config_dir=config_dir)
    return ViPEConfig.model_validate(untyped_config, context={"config_name": config_name})
