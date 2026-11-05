"""Registry of the dataset modules (one module per dataset in this package).

Every dataset module defines ``NAME``, ``CASES`` (canonical order), ``UNITS``,
``PROTOCOLS`` / ``DEFAULT_PROTOCOL`` and the functions
``find_prediction(pred, stage, case, protocol)``,
``evaluate_case(case, prediction, gt_root, protocol=..., stage=...)`` (returns a
dict whose ``ate`` is the headline value in ``UNITS``),
``evaluate_case_aligned(...)`` (the same computation, returning the same dict
plus the ``aligned.AlignedTrajectory`` its ATE was computed on; used by
``plot``) and ``macro(values)``.
"""

from __future__ import annotations

from types import ModuleType

from . import kitti, lamaria, oxford, sevenscenes, tum, vbr, wild_slam

REGISTRY: dict[str, ModuleType] = {
    module.NAME: module
    for module in (kitti, vbr, oxford, lamaria, tum, sevenscenes, wild_slam)
}


def get(name: str) -> ModuleType:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}") from None
