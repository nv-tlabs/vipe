# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MetricTrail on ViPE.

MetricTrail = MoGe-2 metric depth + DINO-fused DPVO + RGB-only shared-K
calibration (+ optional focal refinement). It runs as the ViPE pipeline
``pipeline=metrictrail`` (``vipe infer -p metrictrail``); see
``docs/metrictrail.md``. Submodules are imported on demand, so
``import vipe.metrictrail`` does not load ViPE's or DPVO's CUDA extensions.

- :mod:`vipe.metrictrail.pipeline`: :class:`MetricTrailPipeline`, the one pipeline.
- :mod:`vipe.metrictrail.config`: its typed config (registered in ViPE's
  ``PipelineConfig`` union).
- :mod:`vipe.metrictrail.slam`: :class:`DPVOSLAMSystem`, DINO-fused DPVO with the
  overlapped metric-depth producer, returning ViPE's ``SLAMOutput``.
- :mod:`vipe.metrictrail.depth`: the metric depth provider (ViPE depth registry;
  default ``moge2-l`` with MetricTrail's exact MoGe-2 call).
- :mod:`vipe.metrictrail.calibration`: RGB-only shared-K calibration on ViPE's
  keyframe SLAM, as an init ``StreamProcessor``.
- :mod:`vipe.metrictrail.focal_refine`: optional focal refinement (stage B, RoMa).
- :mod:`vipe.metrictrail.dpvo`: vendored DPVO (``VENDORED.md``); its CUDA
  extensions are built by ``python -m vipe.metrictrail.build``.
- :mod:`vipe.metrictrail.streams`: the full-sequence guarantee (detects inputs
  whose tail was cut, e.g. by ViPE's stock ``frame_end: 1000``).
- :mod:`vipe.metrictrail.raster`: the input raster (``pipeline.init.input_raster``;
  by default a cover-fit to 280x504 with an integer centre crop, ViPE's bilinear resize).
- :mod:`vipe.metrictrail.depth_writer`: the asynchronous metric-depth writer
  (``pipeline.output.save_metric_depth``).
- :mod:`vipe.metrictrail.runtime_record`: the ``runtime`` block of the records (tracking
  wall clock and frames/s, CUDA peak memory after model load, attention backend).
- :mod:`vipe.metrictrail.dynamic`: ViPE's dynamic-object masks on DPVO
  (``pipeline.init.instance``, off by default).
- :mod:`vipe.metrictrail.optional`: optional inputs/outputs in one config block,
  ``pipeline.optional`` (known K; CPU loop closure from external candidate files),
  all off by default (``docs/metrictrail_optional.md``).
- :mod:`vipe.metrictrail.stereo`: optional stereo runner outside the pipeline registry
  (``python -m vipe.metrictrail.stereo``; known K + baseline, rectified left/right
  input, the vendored stereo DPVO tree ``stereo/dpvo_stereo``).
"""

# ViPE's config package imports this package's config module to register the
# MetricTrail pipeline schema. Importing it here first means the config package
# is always fully initialised before ``vipe.metrictrail.config`` starts, whichever
# of the two is imported first (no partially initialised module in the cycle).
import vipe.config  # noqa: F401  (pydantic/hydra only; no CUDA extension)
