# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional MetricTrail inputs and outputs, kept outside the pipeline definition.

Every option here is off by default. With all of them off, ``pipeline=metrictrail``
runs plain MetricTrail (RGB-only SharedK, MoGe-2 metric depth, DINO-fused DPVO).
See ``docs/metrictrail_optional.md``.

- Known intrinsics (input): :mod:`vipe.metrictrail.optional.intrinsics`. A
  GT/sensor K in source-image pixels, from the config (``fx fy cx cy``), a file
  (txt, json, npz) or the input stream's own frame intrinsics. When given, it
  replaces the RGB-only SharedK calibration. The K is mapped onto the input
  raster with the same resize and crop as the frames, and the run record states
  ``intrinsics_source: known``.
- Loop closure (output): :mod:`vipe.metrictrail.optional.loop_closure` applies
  CPU pose-graph correction from external frame-pair files. It writes
  ``pose_lc/<name>.npz`` alongside the original saved trajectory. The same
  operation is available through its CLI or ``pipeline.optional.loop_closure``.
- Stereo (input): the separate runner ``python -m vipe.metrictrail.stereo`` in
  :mod:`vipe.metrictrail.stereo`. It is not a pipeline option.

The typed switches live in :mod:`vipe.metrictrail.optional.config`
(:class:`MetricTrailOptionalConfig`, ``pipeline.optional``). ViPE's dynamic
masking (``pipeline.init.instance``) and depth-alignment fusion
(``pipeline.post.depth_align_model``) are pipeline fields with ViPE's names, not
part of this package (``docs/metrictrail_dynamic_and_fusion.md``).

Submodules are imported on demand; this package imports nothing eagerly.
"""
