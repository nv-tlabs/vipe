# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guards on the vendored MoGe-2 subset in ``vipe/priors/depth/moge2``.

The vendoring rests on two claims that are cheap to state and expensive to have
silently broken by a future upstream bump:

1. The package is self-contained -- neither the pip ``moge`` package nor
   ``utils3d`` is needed at import or inference time.
2. Only four ``utils3d`` symbols are reachable on ViPE's inference path, which is
   what lets ``_utils3d.py`` be forty lines instead of a dependency.

See ``vipe/priors/depth/moge2/VENDORED.md``.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import unittest
from pathlib import Path


class Moge2VendoringTest(unittest.TestCase):
    def test_imports_without_pip_moge_or_utils3d(self) -> None:
        """The vendored tree must not fall back to the optional pip packages."""
        real_import = builtins.__import__
        blocked = ("utils3d", "moge")

        def guarded(name, *args, **kwargs):
            root = name.split(".")[0]
            if root in blocked:
                raise AssertionError(f"vendored moge2 must not import {name!r}")
            return real_import(name, *args, **kwargs)

        for module in [m for m in sys.modules if m.startswith("vipe.priors.depth.moge2")]:
            del sys.modules[module]

        builtins.__import__ = guarded
        try:
            module = importlib.import_module("vipe.priors.depth.moge2")
            self.assertTrue(hasattr(module, "MoGeModel"))
        finally:
            builtins.__import__ = real_import

    def test_local_modifications_are_marked_and_minimal(self) -> None:
        """Every edit to upstream code carries a `# [vipe]` marker (VENDORED.md lists them)."""
        root = Path(__file__).resolve().parents[1] / "vipe" / "priors" / "depth" / "moge2"
        self.assertTrue((root / "VENDORED.md").is_file(), "vendored code must document its provenance")

        marked = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*.py")
            if "# [vipe]" in path.read_text(encoding="utf-8")
        }
        self.assertEqual(
            marked,
            {"v2.py", "modules.py"},
            "the set of import-rewritten upstream files changed -- update VENDORED.md to match",
        )

        # Nothing may reach for the real utils3d or the flattened-away moge.utils.
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("\nimport utils3d", source, f"{path.name} imports the real utils3d")
            self.assertNotIn("from ..utils.", source, f"{path.name} still uses the upstream package layout")

        # Pruned: dino_head.py was deleted, so layers/__init__ must not import it.
        self.assertFalse((root / "dinov2" / "layers" / "dino_head.py").exists())
        self.assertNotIn("dino_head", (root / "dinov2" / "layers" / "__init__.py").read_text())

    def test_shim_exposes_exactly_the_symbols_still_referenced(self) -> None:
        """The shim must cover every `utils3d.` reference left in the tree, and no more.

        Pruning the non-inference code removed the call sites of five of upstream's
        seven utils3d symbols, so the shim carries four functions and no stubs. If a
        MoGe bump reintroduces a reference, this fails instead of raising at runtime.
        """
        import re

        from vipe.priors.depth.moge2 import _utils3d

        root = Path(__file__).resolve().parents[1] / "vipe" / "priors" / "depth" / "moge2"
        referenced = set()
        for path in root.rglob("*.py"):
            if path.name == "_utils3d.py":
                continue
            referenced |= set(re.findall(r"utils3d\.pt\.(\w+)", path.read_text(encoding="utf-8")))

        provided = {n for n in vars(_utils3d.pt) if not n.startswith("_")}
        self.assertTrue(
            referenced <= provided,
            f"utils3d symbols referenced but not vendored: {sorted(referenced - provided)}",
        )
        self.assertFalse(hasattr(_utils3d, "np"), "the numpy namespace should be gone after pruning")

    def test_vendored_intrinsics_matches_reference(self) -> None:
        """`intrinsics_from_focal_center` must reproduce upstream's batched behaviour.

        Upstream gets broadcasting from decorators that were not vendored, so this
        pins the `[B]` focal + 0-dim principal-point call MoGe-2 actually makes.
        """
        import torch

        from vipe.priors.depth.moge2._utils3d import intrinsics_from_focal_center

        fx = torch.tensor([2.0, 4.0])
        fy = torch.tensor([3.0, 5.0])
        c = torch.tensor(0.5)
        K = intrinsics_from_focal_center(fx, fy, c, c)

        self.assertEqual(tuple(K.shape), (2, 3, 3))
        expected = torch.tensor([[2.0, 0.0, 0.5], [0.0, 3.0, 0.5], [0.0, 0.0, 1.0]])
        torch.testing.assert_close(K[0], expected)


class Moge2ModelTest(unittest.TestCase):
    """End-to-end checks. Skipped unless the (1.2 GB) checkpoint is already local."""

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from vipe.priors.depth.moge import MogeModel

        try:
            MogeModel._checkpoint_path("l")
        except FileNotFoundError as exc:
            raise unittest.SkipTest(f"MoGe-2 checkpoint not available locally: {exc}")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

        # Shared GPU: another job holding the card is not a defect in this code, so
        # check up front rather than letting an OOM deep inside a forward pass read
        # as a regression. ~3 GB covers the 1.3 GB of weights plus activations.
        free_gb = torch.cuda.mem_get_info()[0] / 2**30
        if free_gb < 3.0:
            raise unittest.SkipTest(f"GPU busy: only {free_gb:.1f} GiB free, need ~3 GiB")

        cls.model = MogeModel(version=2, variant="l")

    def test_depth_type_and_registry_name(self) -> None:
        from vipe.priors.depth import DepthType, make_depth_model

        model = make_depth_model("moge2-l")
        self.assertEqual(model.depth_type, DepthType.MODEL_METRIC_DEPTH)
        self.assertEqual(model.version, 2)

    def test_estimate_returns_zero_not_inf_for_invalid_pixels(self) -> None:
        """Zero means 'no prior' to the SLAM buffer; +inf would become a real constraint."""
        import torch

        from vipe.priors.depth import DepthEstimationInput
        from vipe.utils.cameras import CameraType

        torch.manual_seed(0)
        rgb = torch.rand(2, 240, 320, 3, device="cuda")
        intrinsics = torch.tensor([250.0, 250.0, 160.0, 120.0], device="cuda")

        depth = self.model.estimate(
            DepthEstimationInput(rgb=rgb, intrinsics=intrinsics, camera_type=CameraType.PINHOLE)
        ).metric_depth

        self.assertIsNotNone(depth)
        self.assertEqual(tuple(depth.shape), (2, 240, 320))
        self.assertTrue(bool(torch.isfinite(depth).all()), "estimate() leaked inf/nan into the depth prior")
        self.assertTrue(bool((depth >= 0).all()))

    def test_batched_matches_per_frame(self) -> None:
        """The backend calls estimate() with keyframe_depth_batch_size frames at once."""
        import torch

        from vipe.priors.depth import DepthEstimationInput
        from vipe.utils.cameras import CameraType

        torch.manual_seed(0)
        rgb = torch.rand(3, 240, 320, 3, device="cuda")
        intrinsics = torch.tensor([250.0, 250.0, 160.0, 120.0], device="cuda")
        src = lambda x: DepthEstimationInput(rgb=x, intrinsics=intrinsics, camera_type=CameraType.PINHOLE)  # noqa: E731

        batched = self.model.estimate(src(rgb)).metric_depth
        single = torch.stack([self.model.estimate(src(rgb[i])).metric_depth for i in range(3)])

        torch.testing.assert_close(batched, single, rtol=2e-2, atol=1e-2)


class Moge2CheckpointResolutionTest(unittest.TestCase):
    """Weight resolution must behave like the other priors: automatic, overridable."""

    def _path(self, **env):
        import os
        from unittest import mock

        from vipe.priors.depth.moge import MogeModel

        with mock.patch.dict(os.environ, env, clear=False):
            return MogeModel._checkpoint_path("l")

    def test_explicit_override_wins(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            self.assertEqual(str(self._path(VIPE_MOGE2_CHECKPOINT=f.name)), f.name)

    def test_empty_override_is_treated_as_unset(self) -> None:
        """An exported-but-empty var must not resolve to Path("") == "." (a directory)."""
        resolved = self._path(VIPE_MOGE2_CHECKPOINT="")
        self.assertNotEqual(str(resolved), ".")
        self.assertTrue(str(resolved).endswith(".pt"), resolved)

    def test_override_pointing_at_a_directory_is_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                self._path(VIPE_MOGE2_CHECKPOINT=tmp_dir)

    def test_override_pointing_at_a_missing_file_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self._path(VIPE_MOGE2_CHECKPOINT="/nonexistent/moge2.pt")

    def test_unknown_variant_is_rejected(self) -> None:
        import os
        from unittest import mock

        from vipe.priors.depth.moge import MogeModel

        with mock.patch.dict(os.environ, {"VIPE_MOGE2_CHECKPOINT": ""}, clear=False):
            with self.assertRaises(ValueError):
                MogeModel._checkpoint_path("xl")


if __name__ == "__main__":
    unittest.main()
