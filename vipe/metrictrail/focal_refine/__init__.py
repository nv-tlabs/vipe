# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focal refinement (stage B) for MetricTrail: refine the shared focal from revisit pairs.

- :mod:`.routing`: GT-free gates on stage A's output deciding whether to enter.
- :mod:`.stage_b`: the RoMa revisit-pair fit and its accept gate (no ViPE imports).
- :mod:`.stage_b_cli`: the stage-B process entry point (run in its own interpreter).
- :mod:`.refine`: the flow used by the pipeline (route -> one attempt -> accept gate).
"""
