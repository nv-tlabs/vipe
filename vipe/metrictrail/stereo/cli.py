# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line of the stereo runner.

    python -m vipe.metrictrail.stereo --left LEFT --right RIGHT \\
        --intrinsics FX FY CX CY --baseline B --output DIR [--name N] [--config YAML] [--device cuda]

``LEFT``/``RIGHT`` are a video file or a directory of frames (ViPE's
``RawMp4Stream`` / ``FrameDirStream``), rectified, of the same size and length;
two frame directories must also pair by frame number. Nothing is resized: the
right frames must already be on the left raster (raw KITTI ``image_3`` PNGs, for
example, are cover-fit to it beforehand, ``INTER_AREA`` + centre crop).
``--intrinsics`` is the known pinhole K of the left camera in pixels of those
frames, ``--baseline`` the stereo baseline in metres. The whole sequence is
tracked (no frame cap); a stream that stops before its source ends is refused
(``on_truncated_input``). The artifacts go to ``DIR`` in ViPE's layout
(:mod:`vipe.metrictrail.stereo.artifacts`); ``--name`` sets the artifact name and
the per-sequence seed (default: the left stream's name).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..torch_state import preserve_torch_state
from .config import StereoRunnerConfig, load_stereo_config

if TYPE_CHECKING:
    from vipe.streams.base import VideoStream

logger = logging.getLogger(__name__)

RUN_SCHEMA = "metrictrail_vipe_stereo_run_v1"


@dataclass(frozen=True)
class StereoRequest:
    """One stereo run, as given on the command line."""

    left: Path
    right: Path
    intrinsics: np.ndarray
    baseline_m: float
    output: Path
    name: str | None
    config: StereoRunnerConfig
    config_source: str
    device: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vipe.metrictrail.stereo",
        description="MetricTrail stereo tracking (DINO-fused stereo DPVO + MoGe-2 metric depth, known K).",
    )
    parser.add_argument("--left", required=True, help="Left video file or frame directory.")
    parser.add_argument("--right", required=True, help="Right video file or frame directory (rectified).")
    parser.add_argument(
        "--intrinsics",
        required=True,
        nargs=4,
        type=float,
        metavar=("FX", "FY", "CX", "CY"),
        help="Known pinhole K of the left camera, in pixels of the input frames.",
    )
    parser.add_argument("--baseline", required=True, type=float, help="Stereo baseline in metres.")
    parser.add_argument("--output", required=True, help="Output directory (ViPE artifact layout).")
    parser.add_argument("--name", default=None, help="Artifact name and seed key (default: the left stream's name).")
    parser.add_argument(
        "--config", default=None, help="YAML with StereoRunnerConfig keys (default: the packaged stereo.yaml)."
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    return parser


def parse_request(argv: Sequence[str] | None = None) -> StereoRequest:
    parser = build_parser()
    args = parser.parse_args(argv)
    intrinsics = np.asarray(args.intrinsics, dtype=np.float32)
    if not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0):
        parser.error(f"--intrinsics must be finite with a positive focal length, got {args.intrinsics}")
    if not math.isfinite(args.baseline) or args.baseline <= 0.0:
        parser.error(f"--baseline must be a positive number of metres, got {args.baseline}")
    left, right = Path(args.left).expanduser(), Path(args.right).expanduser()
    for flag, path in (("--left", left), ("--right", right)):
        if not path.exists():
            parser.error(f"{flag} {path} does not exist")
    if args.name is not None and not args.name.strip():
        parser.error("--name must not be empty")
    return StereoRequest(
        left=left,
        right=right,
        intrinsics=intrinsics,
        baseline_m=float(args.baseline),
        output=Path(args.output).expanduser(),
        name=args.name,
        config=load_stereo_config(args.config),
        config_source=str(args.config) if args.config else "vipe/metrictrail/stereo/stereo.yaml",
        device=str(args.device),
    )


def open_stream(path: Path, name: str | None = None) -> VideoStream:
    """ViPE's full-sequence stream of a frame directory or a video file."""

    if path.is_dir():
        from vipe.streams.frame_dir_stream import FrameDirStream

        return FrameDirStream(path, name=name)
    if path.is_file():
        from vipe.streams.raw_mp4_stream import RawMp4Stream

        return RawMp4Stream(path, name=name)
    raise FileNotFoundError(f"no video file or frame directory at {path}")


def frame_keys(stream: VideoStream) -> list[str] | None:
    """Per-frame pairing keys of a frame-directory stream (the digits of each file stem), else ``None``."""

    files = getattr(stream, "frame_files", None)
    if files is None:
        return None
    keys = []
    for file in files:
        stem = Path(file).stem
        digits = re.findall(r"\d+", stem)
        keys.append("/".join(str(int(group)) for group in digits) if digits else stem)
    return keys


def check_stereo_pairing(left: VideoStream, right: VideoStream) -> str:
    """Refuse frame directories whose frames do not pair by name.

    Two frame directories must list the same frame numbers (the digits of each file
    stem, so ``left_0001.png`` pairs with ``right_0001.png``); a video has no names,
    and only its length is checked (by the runner).
    """

    left_keys, right_keys = frame_keys(left), frame_keys(right)
    if left_keys is None or right_keys is None:
        return "unchecked_not_frame_directories"
    if left_keys != right_keys:
        mismatch = next(
            (index for index, pair in enumerate(zip(left_keys, right_keys)) if pair[0] != pair[1]),
            min(len(left_keys), len(right_keys)),
        )
        raise ValueError(
            f"left and right frames do not pair by name at position {mismatch} "
            f"(left={len(left_keys)} frames, right={len(right_keys)} frames)"
        )
    return "frame_numbers_match"


@preserve_torch_state  # TF32-off / seeds / DPVO thread count do not outlive the run
def run_stereo(
    request: StereoRequest,
    *,
    stream_factory: Callable[[Path, str | None], VideoStream] = open_stream,
    system_factory: Callable[[torch.device, StereoRunnerConfig], Any] | None = None,
) -> dict[str, Any]:
    """Track one stereo sequence and write its artifacts; returns the run record."""

    from vipe.metrictrail.depth_producer import TRACKING_OK
    from vipe.metrictrail.streams import require_full_sequence

    from .artifacts import save_stereo_artifacts

    if system_factory is None:
        from .runner import StereoDPVOSystem

        system_factory = StereoDPVOSystem
    started = time.perf_counter()
    config = request.config
    left = stream_factory(request.left, request.name)
    right = stream_factory(request.right, None)
    name = request.name or left.name()
    coverage = {
        "left": require_full_sequence(left, on_truncation=config.on_truncated_input),
        "right": require_full_sequence(right, on_truncation=config.on_truncated_input),
    }
    pairing = check_stereo_pairing(left, right)
    system = system_factory(torch.device(request.device), config)
    final = system.run(left, right, intrinsics=request.intrinsics, baseline_m=request.baseline_m, name=name)
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "success",
        "name": name,
        "runner": "vipe.metrictrail.stereo",
        "config": config.model_dump(),
        "config_source": request.config_source,
        "inputs": {"left": str(request.left), "right": str(request.right)},
        "input_coverage": coverage,
        "stereo_pairing": pairing,
        "intrinsics_source": "known",
        "sensor_K_or_GT_read": True,
        "bound_K_fx_fy_cx_cy": np.asarray(request.intrinsics, dtype=np.float32).tolist(),
        "stereo_baseline_m": float(request.baseline_m),
        "loop_closure": False,
        "tracking": final.record,
        "tracking_status": final.tracking_status,
        # nonfinite_recovery: whether the FP32 recovery ran, its attempts and the kept seed.
        "nonfinite_recovery": final.record.get("nonfinite_recovery", {"enabled": False}),
        # The tracking stage's wall clock, frames/s and CUDA peak memory (vipe.metrictrail.runtime_record).
        "runtime": final.record.get("runtime"),
    }
    if final.tracking_status != TRACKING_OK:
        logger.warning(f"{name}: {final.tracking_status} (DPVO never initialized; near-static trajectory)")
    record["runtime_seconds"] = time.perf_counter() - started
    artifact_path = save_stereo_artifacts(
        request.output, name, final, request.intrinsics, record, save_slam_map=config.save_slam_map
    )
    logger.info(f"Saved MetricTrail stereo artifacts to {artifact_path.base_path} ({name})")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    request = parse_request(argv)
    record = run_stereo(request)
    print(
        json.dumps(
            {
                "name": record["name"],
                "output": str(request.output),
                "tracking_status": record["tracking_status"],
                "num_frames": record["tracking"].get("num_frames"),
            }
        )
    )
    return 0
