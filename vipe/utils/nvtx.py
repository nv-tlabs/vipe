# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager
from typing import Iterator

import torch


def stage_nvtx_enabled() -> bool:
    return os.environ.get("VIPE_STAGE_NVTX") == "1" and torch.cuda.is_available()


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    pushed = False
    if stage_nvtx_enabled():
        torch.cuda.nvtx.range_push(name)
        pushed = True
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()
