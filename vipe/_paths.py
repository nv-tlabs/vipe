# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def get_config_path() -> Path:
    repo_configs = Path(__file__).resolve().parents[1] / "configs"
    if repo_configs.is_dir():
        return repo_configs
    return Path(__file__).resolve().parent / "_configs"
