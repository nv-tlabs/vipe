# Vendored: MoGe-2 inference subset

Upstream: <https://github.com/microsoft/MoGe>
Pinned commit: `925b8ed835a7a9cdb7578ba15c658a0afc969030` (2026-07-21)
License: MIT, except `dinov2/` which is Apache-2.0 (Meta Platforms, Inc.)

Only the inference path is vendored. MoGe is a git-only package, so a pip
dependency would break every normal ViPE install the moment `moge2-l` became a
default; carrying the code follows what this repo already does for `dav2`,
`dav3` and `videodepthanything`.

## Layout

Upstream's `moge/model/*.py` and `moge/utils/geometry_*.py` are flattened into
this directory; `moge/model/dinov2/` keeps its relative path.

| here | upstream |
|---|---|
| `v2.py` | `moge/model/v2.py` |
| `modules.py` | `moge/model/modules.py` |
| `utils.py` | `moge/model/utils.py` |
| `geometry_torch.py` | `moge/utils/geometry_torch.py` |
| `geometry_numpy.py` | `moge/utils/geometry_numpy.py` |
| `dinov2/**` | `moge/model/dinov2/**` (minus `utils/`) |
| `_utils3d.py` | **new** -- see below |
| `__init__.py` | **new** -- re-exports `MoGeModel` |

## Local modifications

Two kinds, both mechanical: import rewrites, and deletion of code unreachable on
the inference path. **No functional line was changed** -- every retained line is
upstream's.

### Import rewrites (marked `# [vipe]` in-file)

1. `v2.py` -- `import utils3d` → `from . import _utils3d as utils3d`. (The geometry
   modules needed this too before pruning; their last `utils3d` call site is now gone.)
2. `v2.py`, `modules.py` --
   `from ..utils.geometry_torch import ...` → `from .geometry_torch import ...`
   (consequence of the flattening).

Plus three import *deletions*, which carry no marker because nothing is left to mark:
`from .tools import timeit` in both geometry modules (imported but never used;
`moge/utils/tools.py` is not vendored), `from .dino_head import DINOHead` in
`dinov2/layers/__init__.py`, and the now-unused imports the deletions below orphaned.

### Deletions

Whole top-level definitions, removed because nothing on the inference path
reaches them. Verified with `coverage` over a real `estimate()` call (both
batched and single-frame, and with `apply_mask`/`force_projection` both ways),
then re-verified by re-running `scripts/moge2_sanity.py` and confirming the
outputs are unchanged to the last digit. 2604 → 1984 lines.

| file | removed |
|---|---|
| `geometry_numpy.py` | everything except `solve_optimal_focal_shift`, `solve_optimal_shift` (the two the torch `recover_focal_shift` calls) -- 260 → 45 lines |
| `geometry_torch.py` | everything except `normalized_view_plane_uv`, `angle_diff_vec3`, `recover_focal_shift` -- 233 → 86 lines |
| `dinov2/models/vision_transformer.py` | `vit_small`, `vit_base`, `vit_giant2` |
| `dinov2/hub/backbones.py` | every factory except `dinov2_vitl14` |
| `dinov2/models/__init__.py` | `build_model`, `build_model_from_cfg` (training-only) |
| `dinov2/layers/dino_head.py` | whole file -- imported by `layers/__init__.py`, never instantiated |

Deleting the geometry leaves also removed the last call sites of five of the
seven `utils3d` symbols, so `_utils3d.py` no longer carries stubs for them.

**Deliberately kept:** `dinov2/layers/block.py` still holds the stochastic-depth
and nested-tensor helpers. They are dead at inference (`self.training` guards),
but their call sites sit inside `Block.forward`; removing the functions alone
would leave a `NameError` behind a live branch, and removing the branches means
editing the model's core forward path. ~80 lines is not worth that.

## Re-syncing to a newer MoGe

The deletions above are the cost of the smaller tree: a version bump is now
re-copy, re-apply the four import rewrites, then re-apply the deletion table.
`tests/test_moge2_vendoring.py` checks the import rewrites and that the tree is
self-contained; `scripts/moge2_sanity.py` is the numerical gate.

## Not vendored

- `moge/{scripts,train,test}/` -- training, CLI and demo code.
- `moge/utils/{tools,io,vis,panorama,alignment,data_augmentation,webfile,webzipfile}.py`.
- `moge/model/v1.py` -- MoGe-1 still comes from the optional pip package, for the
  `lyra` preset only (see `../moge.py`).
- `moge/model/dinov2/utils/` -- training helpers, not imported by the inference path.
- `utils3d` -- see below.

## The `utils3d` shim

Upstream pins `utils3d` to a *git commit*
(`EasternJournalist/utils3d@3fab839f0be9931dac7c8488eb0e1600c236e183`), not the
PyPI release, and the PyPI package has a different namespace layout -- so
depending on PyPI `utils3d` would be a silent-breakage risk rather than a
convenience.

Before pruning, seven `utils3d` symbols were referenced. Removing the
non-inference code took five of those call sites with it, so `_utils3d.py` now
vendors exactly the four functions still used -- `intrinsics_from_focal_center`,
`uv_map`, `unproject_cv`, `depth_map_to_point_map` -- copied verbatim from that
commit, with no stubs.

`tests/test_moge2_vendoring.py` pins this: it asserts the module imports with
neither `utils3d` nor pip `moge` on `sys.path`, and that every `utils3d.pt.*`
reference left in the tree is covered by the shim -- so a MoGe bump that
reintroduces one fails a test instead of raising at runtime.

Note that `recover_focal_shift` (the torch one) *is* called even when `fov_x` is
given -- it still solves for the shift -- but it uses `scipy`, already a ViPE
dependency, which is why `geometry_numpy.py` is vendored at all.

## Checkpoint

`Ruicheng/moge-2-vitl` → `model.pt`, 326M params, ~1.2 GB.

```
sha256  3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5
```

Resolved by `MogeModel._checkpoint_path` in `../moge.py`, cheapest first:
`VIPE_MOGE2_CHECKPOINT` if set, else an offline copy at
`$TORCH_HOME/hub/moge2/moge-2-vitl.pt`, else HuggingFace via `hf_hub_download`
(the automatic path, matching every other prior in this package). The checkpoint
is self-describing (`{'model_config', 'model'}`), so no architecture constants
are duplicated on our side.
