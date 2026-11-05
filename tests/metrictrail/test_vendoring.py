# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored DPVO: import rewrites, C++/CUDA (one allowed rewrite), extension build plan and loader (CPU)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import torch

from vipe.metrictrail import build
from vipe.metrictrail.dpvo_scatter import scatter_softmax, scatter_sum

PACKAGE = Path(__file__).resolve().parents[2] / "vipe" / "metrictrail"
VENDORED = PACKAGE / "dpvo"
FOREIGN_MODULES = {"cuda_corr", "cuda_ba", "lietorch_backends", "torch_scatter", "yacs", "numba", "pypose"}
MARKER = re.compile(r"  # \[vipe\] was: (.+)$")
#: The only change to the C++/CUDA sources: torch >= 2.9 dropped the ``DeprecatedTypeProperties``
#: overload behind the dispatch macros, so their dtype argument ``x.type()`` became ``x.scalar_type()``
#: (same kernel instantiation, no value change). Sites per file, identical in both trees; every other
#: C++/CUDA file is vendored byte for byte.
SCALAR_TYPE_REWRITE = re.compile(r"\b(a|X|fmap1|net)\.scalar_type\(\)")
SCALAR_TYPE_SITES = {
    "altcorr/correlation_kernel.cu": 4,
    "lietorch/src/lietorch_cpu.cpp": 19,
    "lietorch/src/lietorch_gpu.cu": 19,
}
#: A dispatch macro fed a ``.type()`` (what torch 2.9 no longer compiles).
DEPRECATED_DISPATCH = re.compile(r"DISPATCH_[A-Z_]+\((?:[^;]*?,\s*)?\w+\.type\(\)")


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_foreign_top_level_imports_remain() -> None:
    files = sorted(VENDORED.rglob("*.py"))
    assert len(files) == 25
    for path in files:
        assert not (_imports(path) & FOREIGN_MODULES), path


def test_every_rewrite_is_marked() -> None:
    marked = 0
    for path in VENDORED.rglob("*.py"):
        for line in path.read_text().splitlines():
            if "vipe.metrictrail" in line:
                assert MARKER.search(line), f"{path}: unmarked rewrite {line!r}"
                marked += 1
    assert marked == 9


def test_csrc_trees_are_complete() -> None:
    for tree in build.TREES:
        assert build.missing_sources(tree) == []
    plan = build.plan(["dpvo", "dpvo_stereo"], None)
    assert plan["compile_args"] == {"cxx": ["-O3"], "nvcc": ["-O3"]}
    assert set(plan["trees"]["dpvo"]["extensions"]) == {"cuda_corr", "cuda_ba", "lietorch_backends"}


def test_csrc_dispatch_uses_scalar_type_for_torch_2_9() -> None:
    """The one allowed C++/CUDA rewrite is in place in both trees, at exactly its sites."""

    for tree in build.TREES:
        root = PACKAGE / "csrc" / tree
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text()
            relative = path.relative_to(root).as_posix()
            assert not DEPRECATED_DISPATCH.search(text), (tree, relative)
            assert len(SCALAR_TYPE_REWRITE.findall(text)) == SCALAR_TYPE_SITES.get(relative, 0), (tree, relative)


def test_extension_loader_reports_missing_build(monkeypatch: pytest.MonkeyPatch) -> None:
    from vipe.metrictrail import dpvo_ext
    from vipe.metrictrail.dpvo_ext_loader import DPVOExtensionNotBuilt

    if list((PACKAGE / "dpvo_ext").glob("cuda_corr*.so")):
        pytest.skip("the DPVO extensions are built here")
    monkeypatch.delenv("VIPE_METRICTRAIL_DPVO_JIT", raising=False)
    with pytest.raises(DPVOExtensionNotBuilt, match="python -m vipe.metrictrail.build"):
        dpvo_ext.cuda_corr  # noqa: B018
    with pytest.raises(AttributeError):
        dpvo_ext.not_an_extension  # noqa: B018


def test_scatter_port_matches_a_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    src = torch.randn(1, 50, 8, generator=generator)
    index = torch.randint(0, 7, (50,), generator=generator)
    softmax = scatter_softmax(src, index, dim=1)
    for segment in range(7):
        rows = index == segment
        if rows.any():
            torch.testing.assert_close(softmax[0, rows], torch.softmax(src[0, rows], dim=0), rtol=0, atol=1e-6)
    summed = scatter_sum(src, index, dim=1, dim_size=9)
    assert summed.shape == (1, 9, 8)
    torch.testing.assert_close(summed[0, 3], src[0, index == 3].sum(dim=0))
