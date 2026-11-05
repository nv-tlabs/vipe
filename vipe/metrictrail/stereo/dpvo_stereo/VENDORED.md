# Vendored DPVO (rectified-stereo extension)

Based on [DPVO](https://github.com/princeton-vl/DPVO) (MIT, `LICENSE` in this directory). The
`lietorch/` subtree is [lietorch](https://github.com/princeton-vl/lietorch) (BSD-3-Clause). This is
a separate copy from `vipe/metrictrail/dpvo/`, used only by the stereo runner
(`python -m vipe.metrictrail.stereo`).

NVIDIA modifications are identified by Apache-2.0 source headers; upstream source retains
its original terms. See [the third-party notices](../../../../THIRD_PARTY_LICENSES.md).

## Differences from DPVO

- **Added:** metric-depth bundle adjustment (as in the mono copy) and rectified-stereo factors (the
  `STEREO` / `STEREO_WEIGHT` options, the `right_image` / `stereo_baseline_m` arguments of
  `DPVO.__call__`, same-timestamp left-to-right edges). Both are off by default, which keeps
  DPVO's original behaviour.
- **Imports:** the same replacements as the mono copy (`vipe/metrictrail/dpvo/VENDORED.md`), with
  the CUDA extensions loaded from `vipe.metrictrail.dpvo_stereo_ext`.
- **CUDA/C++** (`vipe/metrictrail/csrc/dpvo_stereo/`): the same `x.scalar_type()` change;
  `fastba/` carries the stereo factors. Built by `python -m vipe.metrictrail.build --stereo`.
- **Not included** (unused): the same files as in the mono copy.
