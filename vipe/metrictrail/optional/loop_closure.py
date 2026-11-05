# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU PyPose loop closure from saved ViPE poses and external candidate files."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import LoopClosureOptions

logger = logging.getLogger(__name__)
LC_IMPLEMENTATION = "external_candidates_pypose_identity_se3_pgo_v1"
OUTPUT_DIR = "pose_lc"
LC_OUTPUT_SUFFIXES = (".npz", ".json", ".candidates.json", ".identity_edges.json")


def run_loop_closure(
    output_dir: str | Path,
    name: str | Sequence[str] | None = None,
    options: LoopClosureOptions | None = None,
) -> list[dict[str, Any]]:
    """Correct one saved trajectory using external candidate frame pairs."""

    options = LoopClosureOptions() if options is None else options
    if options.candidates is None:
        raise ValueError("candidates are required")
    root = Path(output_dir).expanduser().resolve()
    names = [name] if isinstance(name, str) else list(name or [])
    if not names:
        names = sorted(path.stem for path in (root / "pose").glob("*.npz"))
    if len(names) != 1:
        raise ValueError("a candidate file applies to one trajectory; select exactly one name")
    from .loop_trajectory import run_on_vipe_output

    return [
        run_on_vipe_output(
            root,
            names[0],
            candidates=Path(options.candidates),
            preset=options.preset,
            overwrite=options.overwrite,
        )
    ]


def maybe_run_loop_closure(
    options: LoopClosureOptions | None, output_dir: str | Path, name: str
) -> dict[str, Any] | None:
    if options is None or not options.enabled:
        return None
    return run_loop_closure(output_dir, name, options)[0]


class LoopClosureHook:
    """Apply explicit file-based loop closure after the pipeline saves its poses."""

    after_save = True

    def __init__(self, options: LoopClosureOptions) -> None:
        if options.candidates is None:
            raise ValueError("loop closure requires candidates")
        self.options = options

    def __call__(self, pipeline: Any, video_stream: Any, final: Any, record: dict[str, Any]) -> None:
        report = maybe_run_loop_closure(self.options, pipeline.out_path, video_stream.name())
        if report is not None:
            logger.info("%s: loop closure %s", video_stream.name(), report["status"])


def lc_pose_path(output_dir: str | Path, name: str) -> Path:
    return lc_output_paths(output_dir, name)[0]


def lc_output_paths(output_dir: str | Path, name: str) -> list[Path]:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("name must be a single filename component")
    return [Path(output_dir) / OUTPUT_DIR / f"{name}{suffix}" for suffix in LC_OUTPUT_SUFFIXES]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="CPU loop closure from external frame pairs; writes pose_lc/.")
    value.add_argument("output_dir", type=Path)
    value.add_argument("--name", help="trajectory name (optional when pose/ has exactly one NPZ)")
    value.add_argument(
        "--candidates",
        required=True,
        help="external candidate JSON with frame pairs for the input trajectory",
    )
    value.add_argument("--preset", choices=("kitti", "vbr"), default="kitti")
    value.add_argument("--overwrite", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    options = LoopClosureOptions(
        enabled=True,
        candidates=args.candidates,
        preset=args.preset,
        overwrite=args.overwrite,
    )
    records = run_loop_closure(args.output_dir, args.name, options)
    print(json.dumps(records, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
