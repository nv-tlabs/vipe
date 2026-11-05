# Vendored DPVO (metric-depth extension)

Based on [DPVO](https://github.com/princeton-vl/DPVO) (MIT, `LICENSE` in this directory). The
`lietorch/` subtree is [lietorch](https://github.com/princeton-vl/lietorch) (BSD-3-Clause).

NVIDIA modifications are identified by Apache-2.0 source headers; upstream source retains
its original terms. See [the third-party notices](../../../THIRD_PARTY_LICENSES.md).

## Differences from DPVO

- **Added:** metric-depth bundle adjustment (`metric_depth.py`; the `metric_depth` / `metric_mask`
  arguments of `DPVO.__call__`) and a metric global-BA backend (`config.py`, `burst_fixes.py`,
  `graph_guards.py`, `remeasure_burnin.py`, `remeasure_gates.py`, `scale_latent.py`). Every new
  option defaults to DPVO's original behaviour.
- **Imports:** the CUDA extensions are loaded from `vipe.metrictrail.dpvo_ext`, `torch_scatter` is
  replaced by the plain-PyTorch `vipe.metrictrail.dpvo_scatter`, and `yacs` and `numba`
  are imported on first use through `vipe.metrictrail.dpvo_deps`. Each rewritten import
  keeps the original import in a comment on the same line.
- **CUDA/C++** (`vipe/metrictrail/csrc/dpvo/`): `x.type()` -> `x.scalar_type()` in the dispatch
  macros, needed for torch >= 2.9. Built by `python -m vipe.metrictrail.build` with DPVO's
  compile flags.
- **Not included** (unused): `data_readers/`, `logger.py`, `plot_utils.py`, `stream.py`, the
  lietorch tests and `loop_closure/`. MIT edge-selection helpers are in `edge_utils.py`.

`tests/metrictrail/test_vendoring.py` checks the import and dispatch changes.
