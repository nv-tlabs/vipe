# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Vendored inference subset of MoGe-2 (https://github.com/microsoft/MoGe), MIT License.
# See VENDORED.md for the pinned upstream commit and the exact local modifications.

from .v2 import MoGeModel

__all__ = ["MoGeModel"]
