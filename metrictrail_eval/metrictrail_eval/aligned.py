"""The aligned trajectory behind one sequence's ATE (for plotting, ``plot.py``).

Every dataset module's ``evaluate_case_aligned`` returns, next to the very
result dict that ``evaluate_case`` returns (both run the same code), an
:class:`AlignedTrajectory`: the frame pairs the ATE was computed on, the GT
camera centres of those pairs, the prediction's camera centres after the
scorer's own Sim(3), and that Sim(3).  Nothing here is used to compute a
metric; :meth:`AlignedTrajectory.rmse` only restates the ATE from the drawn
points so that a plot can check it shows what was scored.

How the aligned centres are obtained depends on the backend that scored the
sequence:

* evo paths (KITTI ``evo``, Wild-SLAM): the positions of the evo trajectories
  after ``align(correct_scale=True)``, i.e. the arrays the APE was computed on;
* the shared ``TrajectoryEvaluator`` (VBR, Oxford-Spires, TUM, 7-Scenes): its
  returned ``traj_transform`` ``[sR | t]`` applied to the prediction's centres,
  on the pairs it evaluated (GT rows selected by the prediction's frame ids,
  pairs with a non-finite GT or prediction pose dropped, as inside the
  evaluator);
* closed-form Umeyama (KITTI ``umeyama``, LaMaria): the aligned array of the
  scorer itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AlignedTrajectory:
    """Camera centres in metres; ``sim3`` maps prediction coordinates to GT coordinates."""

    frame_ids: np.ndarray  # [M] int64: frame ids of the evaluated pairs, in evaluation order
    gt_xyz: np.ndarray  # [M, 3] GT camera centres of the evaluated pairs
    pred_xyz: np.ndarray  # [M, 3] prediction camera centres after the scorer's Sim(3)
    gt_ids: np.ndarray  # [K] int64: every GT frame id of the sequence (context for drawing)
    gt_all_xyz: np.ndarray  # [K, 3] every GT camera centre, NaN where the GT has no pose
    sim3: np.ndarray  # [4, 4] [sR | t] of the scorer's alignment
    scale: float
    backend: str

    def rmse(self) -> float:
        """RMSE of the drawn centre errors (m); equals the scorer's ATE up to rounding."""

        errors = np.linalg.norm(self.pred_xyz - self.gt_xyz, axis=1)
        return float(np.sqrt(np.mean(errors**2)))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "frame_ids": self.frame_ids,
            "gt_xyz": self.gt_xyz,
            "pred_xyz": self.pred_xyz,
            "gt_ids": self.gt_ids,
            "gt_all_xyz": self.gt_all_xyz,
            "sim3": self.sim3,
        }


def _sim3_matrix(rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = scale * np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def _context(gt_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_all = np.asarray(gt_all, dtype=np.float64)
    xyz = gt_all[:, :3, 3].copy()
    xyz[~np.isfinite(gt_all.reshape(len(gt_all), -1)).all(axis=1)] = np.nan
    return np.arange(len(gt_all), dtype=np.int64), xyz


def from_evo(
    ref: Any, est: Any, alignment: tuple[np.ndarray, np.ndarray, float], gt_all: np.ndarray, backend: str
) -> AlignedTrajectory:
    """From the evo trajectories an APE was computed on (``est`` already aligned to ``ref``)."""

    rotation, translation, scale = alignment
    gt_ids, gt_all_xyz = _context(gt_all)
    return AlignedTrajectory(
        frame_ids=np.rint(np.asarray(est.timestamps, dtype=np.float64)).astype(np.int64),
        gt_xyz=np.array(ref.positions_xyz, dtype=np.float64),
        pred_xyz=np.array(est.positions_xyz, dtype=np.float64),
        gt_ids=gt_ids,
        gt_all_xyz=gt_all_xyz,
        sim3=_sim3_matrix(rotation, translation, float(scale)),
        scale=float(scale),
        backend=backend,
    )


def from_shared_evaluator(
    gt_all: np.ndarray, pred_c2w: np.ndarray, pred_ids: np.ndarray | None, transform: np.ndarray
) -> AlignedTrajectory:
    """From the shared evaluator's inputs and its returned ``traj_transform``.

    Pair selection as in ``TrajectoryEvaluator.evaluate``: ``gt_all[pred_ids]``
    (``pred_ids`` None = ``arange(N)``), then pairs whose GT or prediction pose has
    a NaN/Inf entry are dropped.
    """

    gt_all = np.asarray(gt_all)
    pred = np.asarray(pred_c2w, dtype=np.float64)
    ids = np.arange(len(pred), dtype=np.int64) if pred_ids is None else np.asarray(pred_ids, dtype=np.int64)
    gt = np.asarray(gt_all[ids], dtype=np.float64)
    keep = np.isfinite(gt.reshape(len(gt), -1)).all(axis=1) & np.isfinite(pred.reshape(len(pred), -1)).all(axis=1)
    transform = np.asarray(transform, dtype=np.float64)
    pred_xyz = pred[keep, :3, 3] @ transform[:3, :3].T + transform[:3, 3]
    gt_ids, gt_all_xyz = _context(gt_all)
    return AlignedTrajectory(
        frame_ids=ids[keep],
        gt_xyz=gt[keep, :3, 3],
        pred_xyz=pred_xyz,
        gt_ids=gt_ids,
        gt_all_xyz=gt_all_xyz,
        sim3=transform,
        scale=float(np.cbrt(np.linalg.det(transform[:3, :3]))),
        backend="trajectory_eval",
    )


def from_umeyama(
    aligned_c2w: np.ndarray,
    target_c2w: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
    gt_all: np.ndarray,
    backend: str,
) -> AlignedTrajectory:
    """From a closed-form Umeyama scorer's aligned prediction and (sanitised) target."""

    gt_ids, gt_all_xyz = _context(gt_all)
    return AlignedTrajectory(
        frame_ids=np.arange(len(aligned_c2w), dtype=np.int64),
        gt_xyz=np.array(np.asarray(target_c2w)[:, :3, 3], dtype=np.float64),
        pred_xyz=np.array(np.asarray(aligned_c2w)[:, :3, 3], dtype=np.float64),
        gt_ids=gt_ids,
        gt_all_xyz=gt_all_xyz,
        sim3=_sim3_matrix(rotation, translation, float(scale)),
        scale=float(scale),
        backend=backend,
    )
