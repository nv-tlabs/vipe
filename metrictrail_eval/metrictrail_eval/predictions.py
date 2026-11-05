"""Locate one case's prediction NPZ under a run directory or a prediction directory.

A MetricTrail run directory (``outputs/<config>``) holds
``<stage>/inference/<case>.npz`` (KITTI local, every LC stage) or
``<stage>/inference/predictions/<case>.npz`` (the other local stages, the
historical LaMaria LC).  Anything else is treated as a plain prediction
directory holding ``<case>.npz`` or ``predictions/<case>.npz``; ``--stage``
then only labels the result.
"""

from __future__ import annotations

from pathlib import Path

STAGES = ("local", "LC")


def is_run_dir(pred: Path, stage: str) -> bool:
    return (Path(pred) / stage / "inference").is_dir()


def find_prediction(pred: Path, stage: str, case: str) -> Path | None:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    pred = Path(pred)
    if is_run_dir(pred, stage):
        base = pred / stage / "inference"
    else:
        base = pred
    candidates = [base / f"{case}.npz", base / "predictions" / f"{case}.npz"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(
            f"ambiguous prediction for case {case}: {[str(p) for p in existing]}"
        )
    return existing[0] if existing else None
