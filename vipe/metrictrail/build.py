# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build MetricTrail's DPVO CUDA extensions: ``python -m vipe.metrictrail.build``.

The DPVO kernels are compiled on their own, next to (not inside) ViPE's
``vipe_ext``: ViPE's ``setup.py`` is not involved, and ViPE's build flags
(``-DWITH_CUDA``, ``--use_fast_math``) are not used. Each tree is compiled with
DPVO's own ``setup.py`` recipe -- three extension modules, ``-O3`` for both
compilers, Eigen 3.4.0 headers -- from the vendored sources under ``vipe/metrictrail/csrc/<tree>/``,
changed only by ``x.type()`` -> ``x.scalar_type()`` in the dispatch macros (torch >= 2.9
removed the overload ``.type()`` needed; no value changes; ``vipe/metrictrail/dpvo/VENDORED.md``):

=====================  ==============================================  ======================
tree                   package of the compiled modules                 built
=====================  ==============================================  ======================
``dpvo``               ``vipe.metrictrail.dpvo_ext``                    always
``dpvo_stereo``        ``vipe.metrictrail.dpvo_stereo_ext``             only with ``--stereo``
=====================  ==============================================  ======================

Each package holds ``cuda_corr``, ``cuda_ba`` and ``lietorch_backends`` (DPVO's
module names; the sources keep ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)``, so
they must stay three modules). Fully qualified, they cannot clash with a
separately installed DPVO or with each other.

Eigen: ViPE's own build helper (``vipe/ext/specs.py``, ``_eigen_include_flags``)
is called, unchanged, so the headers ViPE downloads for ``vipe_ext``
(``csrc/include/eigen3``) are reused. ``--eigen-include DIR`` (or
``EIGEN_INCLUDE_DIR``) points at any other Eigen 3.4.0 root instead;
``USE_SYSTEM_EIGEN=1`` (ViPE's switch) uses ``$CONDA_PREFIX/include/eigen3`` (else pass ``--eigen-include``).
Whichever root is used, its version is read from its headers and anything other
than 3.4.0 (DPVO's recipe; conda-forge's unpinned ``eigen`` is 5.x) is refused
unless ``--allow-any-eigen`` (or ``VIPE_METRICTRAIL_ALLOW_ANY_EIGEN=1``) is given.
The version is printed with the build.
``TORCH_CUDA_ARCH_LIST``, ``MAX_JOBS`` and ``CUDA_HOME`` are honoured by PyTorch.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
CSRC_ROOT = PACKAGE_ROOT / "csrc"

#: DPVO's compiler flags (``DPVO/setup.py``): no ``--use_fast_math``, no ``-DWITH_CUDA``.
COMPILE_ARGS: dict[str, list[str]] = {"cxx": ["-O3"], "nvcc": ["-O3"]}

#: The Eigen release of DPVO's build recipe (and of ViPE's own download).
EIGEN_VERSION = "3.4.0"
ALLOW_ANY_EIGEN_ENV = "VIPE_METRICTRAIL_ALLOW_ANY_EIGEN"

#: tree -> package that receives the compiled modules.
TREES: dict[str, str] = {
    "dpvo": "vipe.metrictrail.dpvo_ext",
    "dpvo_stereo": "vipe.metrictrail.dpvo_stereo_ext",
}


@dataclass(frozen=True)
class ExtensionSpec:
    """One DPVO extension module, as declared in ``DPVO/setup.py``."""

    name: str
    sources: tuple[str, ...]
    include_dirs: tuple[str, ...] = ()
    needs_eigen: bool = False


EXTENSIONS: tuple[ExtensionSpec, ...] = (
    ExtensionSpec("cuda_corr", ("altcorr/correlation.cpp", "altcorr/correlation_kernel.cu")),
    ExtensionSpec(
        "cuda_ba",
        ("fastba/ba.cpp", "fastba/ba_cuda.cu", "fastba/block_e.cu"),
        needs_eigen=True,
    ),
    ExtensionSpec(
        "lietorch_backends",
        ("lietorch/src/lietorch.cpp", "lietorch/src/lietorch_gpu.cu", "lietorch/src/lietorch_cpu.cpp"),
        include_dirs=("lietorch/include",),
        needs_eigen=True,
    ),
)
EXTENSION_NAMES = tuple(spec.name for spec in EXTENSIONS)


def tree_root(tree: str) -> Path:
    if tree not in TREES:
        raise ValueError(f"unknown DPVO tree {tree!r}; expected one of {sorted(TREES)}")
    return CSRC_ROOT / tree


def missing_sources(tree: str) -> list[str]:
    root = tree_root(tree)
    paths = [source for spec in EXTENSIONS for source in spec.sources]
    paths += [include for spec in EXTENSIONS for include in spec.include_dirs]
    return [path for path in paths if not (root / path).exists()]


def _vipe_specs_module() -> Any:
    """ViPE's ``vipe/ext/specs.py``, loaded by path.

    Importing ``vipe.ext.specs`` would run ``vipe/ext/__init__.py``, which loads
    (or JIT-compiles) ``vipe_ext``; ViPE's ``setup.py`` loads the file the same
    way for the same reason.
    """

    path = PACKAGE_ROOT.parent / "ext" / "specs.py"
    spec = importlib.util.spec_from_file_location("_vipe_ext_specs_for_metrictrail", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load ViPE's build helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eigen_version(include_dir: str | Path) -> str | None:
    """``major.minor.patch`` of the Eigen headers under ``include_dir``, or None if unreadable.

    Eigen <= 3.4 defines ``EIGEN_WORLD/MAJOR/MINOR_VERSION`` in
    ``Eigen/src/Core/util/Macros.h`` (3.4.0 = 3, 4, 0); Eigen >= 5 moved them to
    ``Eigen/Version`` with semantic ``EIGEN_MAJOR/MINOR/PATCH_VERSION`` (5.0.1 = 5, 0, 1;
    its WORLD version stays 3).
    """

    root = Path(include_dir)
    for header in (root / "Eigen" / "Version", root / "Eigen" / "src" / "Core" / "util" / "Macros.h"):
        try:
            text = header.read_text(errors="replace")
        except OSError:
            continue
        found = {
            name: int(value)
            for name, value in re.findall(
                r"^\s*#\s*define\s+EIGEN_(WORLD|MAJOR|MINOR|PATCH)_VERSION\s+(\d+)", text, re.M
            )
        }
        if "PATCH" in found and "MAJOR" in found and "MINOR" in found:
            return f"{found['MAJOR']}.{found['MINOR']}.{found['PATCH']}"
        if {"WORLD", "MAJOR", "MINOR"} <= found.keys():
            return f"{found['WORLD']}.{found['MAJOR']}.{found['MINOR']}"
    return None


def check_eigen_version(include_dir: str | Path, *, allow_any: bool | None = None) -> str | None:
    """Refuse Eigen headers other than :data:`EIGEN_VERSION`; returns the version found.

    ``allow_any`` (default: ``$VIPE_METRICTRAIL_ALLOW_ANY_EIGEN == "1"``) turns the
    refusal into a warning on stderr.
    """

    version = eigen_version(include_dir)
    if version == EIGEN_VERSION:
        return version
    if allow_any is None:
        allow_any = os.environ.get(ALLOW_ANY_EIGEN_ENV, "0") == "1"
    message = (
        f"Eigen headers in {include_dir} are version {version or 'unknown'}, but DPVO's kernels are built against "
        f"Eigen {EIGEN_VERSION}"
    )
    if not allow_any:
        raise RuntimeError(
            f"{message}. Use ViPE's own copy (unset USE_SYSTEM_EIGEN and EIGEN_INCLUDE_DIR), pass "
            f"--eigen-include with an Eigen {EIGEN_VERSION} root, or accept this version with --allow-any-eigen "
            f"(or {ALLOW_ANY_EIGEN_ENV}=1)."
        )
    print(f"WARNING: {message}; building anyway ({ALLOW_ANY_EIGEN_ENV} / --allow-any-eigen).", file=sys.stderr)
    return version


def eigen_include_dir(override: str | Path | None = None, *, allow_any_version: bool | None = None) -> Path:
    """Directory that contains ``Eigen/Core`` (DPVO includes ``<Eigen/...>``); its version is checked."""

    candidates: list[Path] = []
    explicit = override or os.environ.get("EIGEN_INCLUDE_DIR")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    elif os.environ.get("USE_SYSTEM_EIGEN", "0") == "1":
        if "CONDA_PREFIX" in os.environ:
            candidates.append(Path(os.environ["CONDA_PREFIX"]) / "include" / "eigen3")
    else:
        flags = _vipe_specs_module()._eigen_include_flags()  # downloads Eigen 3.4.0 once, as for vipe_ext
        include_roots = [Path(flags[index + 1]) for index, flag in enumerate(flags[:-1]) if flag == "-isystem"]
        candidates.extend(root / "eigen3" for root in include_roots)
    for candidate in candidates:
        if (candidate / "Eigen" / "Core").is_file():
            resolved = candidate.resolve()
            check_eigen_version(resolved, allow_any=allow_any_version)
            return resolved
    raise FileNotFoundError(
        f"Eigen headers not found (looked in {[str(path) for path in candidates]}); pass --eigen-include DIR"
    )


def extension_kwargs(tree: str, spec: ExtensionSpec, eigen_dir: Path | None) -> dict[str, Any]:
    """Absolute sources / include dirs and DPVO's flags for one extension (JIT loader and build)."""

    root = tree_root(tree)
    include_dirs = [str(root / include) for include in spec.include_dirs]
    if spec.needs_eigen:
        if eigen_dir is None:
            raise ValueError(f"{spec.name} needs the Eigen headers")
        include_dirs.append(str(eigen_dir))
    return {
        "sources": [str(root / source) for source in spec.sources],
        "include_dirs": include_dirs,
        "extra_compile_args": {key: list(value) for key, value in COMPILE_ARGS.items()},
    }


def output_root() -> Path:
    """Directory that contains the ``vipe`` package (the compiled modules land inside it)."""

    return PACKAGE_ROOT.parents[1]


def build_tree(tree: str, *, eigen_dir: Path, build_lib: Path, verbose: bool = False) -> list[Path]:
    """Compile the three modules of ``tree`` into ``build_lib/<package path>/``."""

    from setuptools import setup  # type: ignore[import-untyped]
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    package = TREES[tree]
    modules = []
    for spec in EXTENSIONS:
        kwargs = extension_kwargs(tree, spec, eigen_dir)
        modules.append(
            CUDAExtension(
                f"{package}.{spec.name}",
                # Relative to the tree, as in DPVO's setup.py (run from the tree below).
                sources=list(spec.sources),
                include_dirs=kwargs["include_dirs"],
                extra_compile_args=kwargs["extra_compile_args"],
            )
        )
    with tempfile.TemporaryDirectory(prefix="vipe-metrictrail-build-") as build_temp:
        previous = Path.cwd()
        os.chdir(tree_root(tree))
        try:
            setup(
                name=f"vipe-metrictrail-{tree.replace('_', '-')}-ext",
                ext_modules=modules,
                cmdclass={"build_ext": BuildExtension},
                script_args=[
                    *([] if verbose else ["--quiet"]),
                    "build_ext",
                    "--build-lib",
                    str(build_lib),
                    "--build-temp",
                    build_temp,
                ],
            )
        finally:
            os.chdir(previous)
    package_dir = build_lib.joinpath(*package.split("."))
    built = sorted(path for path in package_dir.glob("*.so") if path.name.split(".", 1)[0] in EXTENSION_NAMES)
    if len(built) != len(EXTENSIONS):
        raise RuntimeError(f"expected {len(EXTENSIONS)} modules in {package_dir}, found {[p.name for p in built]}")
    return built


def verify_tree(tree: str) -> dict[str, str]:
    """Import the compiled modules of ``tree`` from inside this package; returns their files."""

    import torch  # noqa: F401  (the modules link against libtorch)

    package = TREES[tree]
    found: dict[str, str] = {}
    for name in EXTENSION_NAMES:
        sys.modules.pop(f"{package}.{name}", None)
        module = importlib.import_module(f"{package}.{name}")
        origin = Path(getattr(module, "__file__", "") or "").resolve()
        if PACKAGE_ROOT not in origin.parents:
            raise RuntimeError(f"{package}.{name} resolved to {origin}, outside {PACKAGE_ROOT}")
        found[name] = str(origin)
    return found


def plan(trees: Sequence[str], eigen_dir: Path | None) -> dict[str, Any]:
    return {
        "output_root": str(output_root()),
        "eigen_include_dir": None if eigen_dir is None else str(eigen_dir),
        "compile_args": COMPILE_ARGS,
        "trees": {
            tree: {
                "package": TREES[tree],
                "sources": str(tree_root(tree)),
                "missing_sources": missing_sources(tree),
                "extensions": {
                    spec.name: {
                        "sources": list(spec.sources),
                        "include_dirs": list(spec.include_dirs) + (["<eigen>"] if spec.needs_eigen else []),
                    }
                    for spec in EXTENSIONS
                },
            }
            for tree in trees
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vipe.metrictrail.build", description=__doc__.splitlines()[0])
    parser.add_argument("--stereo", action="store_true", help="also build the optional stereo DPVO tree")
    parser.add_argument("--stereo-only", action="store_true", help="build only the stereo DPVO tree")
    parser.add_argument("--eigen-include", type=Path, default=None, help="directory containing Eigen/Core")
    parser.add_argument(
        "--allow-any-eigen",
        action="store_true",
        help=f"build against Eigen headers other than {EIGEN_VERSION} (warns instead of refusing)",
    )
    parser.add_argument("--build-lib", type=Path, default=None, help="install root (default: next to vipe/)")
    parser.add_argument("--dry-run", action="store_true", help="print the build plan and exit (no compiler)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    trees = ["dpvo_stereo"] if args.stereo_only else ["dpvo", "dpvo_stereo"] if args.stereo else ["dpvo"]
    for tree in trees:
        missing = missing_sources(tree)
        if missing:
            raise SystemExit(
                f"{tree}: missing sources {missing} under {tree_root(tree)}; this ViPE install does not carry "
                "vipe/metrictrail/csrc (reinstall from a source checkout or from an sdist built with its MANIFEST.in)"
            )
    if args.dry_run:
        print(json.dumps(plan(trees, None), indent=2))
        return 0

    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if torch.version.cuda is None:
        raise SystemExit("PyTorch was built without CUDA; the DPVO extensions need a CUDA build")
    if CUDA_HOME is None or shutil.which("nvcc", path=str(Path(CUDA_HOME) / "bin")) is None:
        raise SystemExit("nvcc not found; set CUDA_HOME to the CUDA toolkit matching PyTorch")
    try:
        eigen_dir = eigen_include_dir(args.eigen_include, allow_any_version=args.allow_any_eigen or None)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    build_lib = (args.build_lib or output_root()).resolve()
    print(
        f"torch {torch.__version__}, CUDA {torch.version.cuda}, CUDA_HOME {CUDA_HOME}, "
        f"Eigen {eigen_version(eigen_dir) or 'unknown'} ({eigen_dir})"
    )
    for tree in trees:
        print(f"=== building {tree} -> {TREES[tree]}")
        for path in build_tree(tree, eigen_dir=eigen_dir, build_lib=build_lib, verbose=args.verbose):
            print(f"  {path}")
        if build_lib == output_root():
            for name, origin in verify_tree(tree).items():
                print(f"  import ok: {TREES[tree]}.{name} ({Path(origin).name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
