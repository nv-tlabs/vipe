"""The shared, hash-locked Sim(3) trajectory evaluator.

``vendored/trajectory_eval.py`` is a third-party trajectory evaluator (Apache-2.0,
``licenses/LICENSE.trajectory_eval``) whose executable code is unchanged from
upstream.  It is loaded from its file
only after its SHA-256 matches ``TRAJECTORY_SHA256``; an edited copy is refused
instead of silently changing the metric.

``TrajectoryEvaluator(align=True, correct_scale=True).evaluate`` aligns the
prediction to GT with evo's Umeyama Sim(3) on the matched frames (GT rows with
NaN/Inf are dropped first), and reports ``ate`` = APE translation RMSE plus
delta-1 RPE (``rpe_trans``, ``rpe_rot`` in degrees).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np

from .common import PACKAGE_DIR, file_identity, sha256_file

TRAJECTORY_RELATIVE = Path("vendored/trajectory_eval.py")
TRAJECTORY_PATH = PACKAGE_DIR / TRAJECTORY_RELATIVE
TRAJECTORY_SHA256 = "8385eee2b8b2c2e6578b0c031df05b948454f12c0646ec3cf04a8e55c7e0bcf2"

_MODULE: Any = None


def load_trajectory_module() -> Any:
    """Import the vendored trajectory evaluator once, after checking its hash."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    actual = sha256_file(TRAJECTORY_PATH)
    if actual != TRAJECTORY_SHA256:
        raise RuntimeError(
            "vendored trajectory evaluator changed; refusing an unreviewed "
            f"metric backend: {actual} != {TRAJECTORY_SHA256} ({TRAJECTORY_PATH})"
        )
    # The evaluator imports matplotlib.pyplot at module level; never ask for a display.
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location(
        "metrictrail_eval_vendored_trajectory_eval", TRAJECTORY_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {TRAJECTORY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def trajectory_evaluator_class() -> Any:
    return load_trajectory_module().TrajectoryEvaluator


class ArrayTrajectoryLoader:
    """In-memory loader with the two calls ``TrajectoryEvaluator.evaluate`` makes.

    ``frame_ids`` defaults to ``np.arange(N, dtype=np.int32)`` (the loaders of
    the Oxford-long, 7-Scenes and Wild-SLAM evaluators); VBR and TUM pass the
    prediction's ordinal frame ids so GT rows are selected by id.
    """

    def __init__(self, poses: np.ndarray, frame_ids: np.ndarray | None = None) -> None:
        self.poses = poses
        self.ids = (
            np.arange(len(poses), dtype=np.int32) if frame_ids is None else frame_ids
        )
        self.artifact = type("Artifact", (), {"traj_file": Path("in-memory")})()

    def load_trajectory(self) -> np.ndarray:
        return self.poses

    def get_frame_indices(self) -> np.ndarray:
        return self.ids


def evo_version() -> str:
    try:
        return importlib.metadata.version("evo")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def backend_identity() -> dict[str, Any]:
    return {
        "trajectory_eval_py": {
            **file_identity(TRAJECTORY_PATH),
            "path": str(TRAJECTORY_RELATIVE),
            "expected_sha256": TRAJECTORY_SHA256,
        },
        "evo_version": evo_version(),
        "numpy_version": np.__version__,
    }
