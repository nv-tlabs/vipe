"""Data-free self-test of the metric backends.

    python -m metrictrail_eval.selftest

Checks the vendored trajectory evaluator's hash, then on a synthetic
trajectory that (1) GT vs GT gives ATE 0, (2) a random Sim(3) copy of GT gives
ATE ~0, (3) 1 cm isotropic noise gives ATE ~1 cm, and (4) the evo paths
(KITTI ``evo``, Wild-SLAM), the shared TrajectoryEvaluator (VBR / Oxford / TUM /
7-Scenes) and the two closed-form Umeyama ports (KITTI ``umeyama``, LaMaria)
agree to 1e-9 m.  If open3d is installed, the point-cloud core is checked too:
a Sim(3) copy of a synthetic cloud is registered back (Umeyama + voxel + ICP)
to Chamfer ~0 and F1 100 %, and 1 cm noise gives a Chamfer near the expected
nearest-neighbour distance.
"""

from __future__ import annotations

import logging
import math
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from . import kitti, lamaria, points, sim3_ate, sim3_umeyama, wild_slam


def _trajectory(rng: np.random.Generator, n: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 4.0 * math.pi, n)
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, :3, 3] = np.stack([10 * np.cos(t), 6 * np.sin(t), 0.5 * t], axis=1)
    poses[:, :3, :3] = Rotation.from_euler("zyx", np.stack([t, 0.1 * np.sin(t), 0.05 * t], 1)).as_matrix()
    poses[:, :3, 3] += rng.normal(scale=0.01, size=(n, 3))
    return poses


def _sim3(rng: np.random.Generator, poses: np.ndarray, scale: float) -> np.ndarray:
    rotation = Rotation.random(random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()
    translation = rng.normal(size=3) * 5.0
    out = poses.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses[:, :3, :3])
    out[:, :3, 3] = scale * np.einsum("ij,nj->ni", rotation, poses[:, :3, 3]) + translation
    return out


def _all_backends(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    n = len(gt)
    logger = logging.getLogger("metrictrail_eval.selftest")
    shared = sim3_ate.trajectory_evaluator_class()(align=True, correct_scale=True).evaluate(
        sim3_ate.ArrayTrajectoryLoader(gt), sim3_ate.ArrayTrajectoryLoader(pred), logger
    )
    return {
        "evo (kitti)": kitti.ate_evo(pred, gt, np.arange(n, dtype=np.float64)),
        "evo (wild_slam)": wild_slam.full_traj_ate(pred, gt)["ate_rmse_m"],
        "trajectory_eval": float(shared["ate"]),
        "umeyama (kitti)": float(sim3_umeyama.compute_pose_metrics(pred, gt)["ate_rmse"]),
        "umeyama (lamaria)": lamaria.compute_pose_metrics(pred, gt)["ate_rmse_m"],
    }


def _points_check(rng: np.random.Generator) -> list[str]:
    try:
        import open3d  # noqa: F401
    except ImportError:
        print("SKIP points: open3d not installed")
        return []
    # A box-shaped room surface, 2 cm sampling.
    grid = np.stack(np.meshgrid(np.arange(0, 4, 0.02), np.arange(0, 3, 0.02)), -1).reshape(-1, 2)
    walls = [np.c_[grid, np.zeros(len(grid))], np.c_[grid[:, :1], np.zeros(len(grid)), grid[:, 1:]],
             np.c_[np.zeros(len(grid)), grid]]
    gt = np.concatenate(walls).astype(np.float32)
    rotation = Rotation.random(random_state=7).as_matrix()
    sim3 = np.eye(4)
    sim3[:3, :3] = 0.6 * rotation
    sim3[:3, 3] = [1.0, -2.0, 0.5]
    failures = []
    for name, noise, expected, tolerance in (("points_sim3", 0.0, 0.0, 1e-5), ("points_noise_1cm", 0.01, 0.0075, 0.003)):
        target = gt + rng.normal(scale=noise / math.sqrt(3.0), size=gt.shape).astype(np.float32) if noise else gt
        pred = points.apply_transform(gt, sim3).astype(np.float32)
        aligned = points.apply_transform(pred, points.umeyama_registration(pred, target))
        pred_voxel = points.voxel_downsample(aligned, 0.01)
        gt_voxel = points.voxel_downsample(target, 0.01)
        icp = points.icp_registration(pred_voxel, gt_voxel, 0.05)
        metrics = points.evaluate_pointcloud(points.apply_transform(pred_voxel, icp), gt_voxel, (0.05,))
        ok = abs(metrics["chamfer"] - expected) <= tolerance and metrics["f1_0.05"] > 99.0
        failures += [] if ok else [name]
        print(f"{'PASS' if ok else 'FAIL'} {name}: chamfer {metrics['chamfer']:.2e} m, F1@5cm {metrics['f1_0.05']:.2f} %")
    return failures


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    sim3_ate.load_trajectory_module()
    print(f"trajectory_eval.py sha256 OK ({sim3_ate.TRAJECTORY_SHA256[:12]}...)")
    rng = np.random.default_rng(20261008)
    gt = _trajectory(rng)
    failures = []
    checks = {
        "gt_vs_gt": (gt, 0.0, 1e-9),
        "random_sim3": (_sim3(rng, gt, 2.3), 0.0, 1e-8),
    }
    noisy = gt.copy()
    noisy[:, :3, 3] += rng.normal(scale=0.01 / math.sqrt(3.0), size=(len(gt), 3))
    checks["noise_1cm"] = (_sim3(rng, noisy, 0.7), 0.01, 0.002)
    for name, (pred, expected, tolerance) in checks.items():
        values = _all_backends(pred, gt)
        spread = max(values.values()) - min(values.values())
        ok = all(abs(v - expected) <= tolerance for v in values.values()) and spread <= 1e-9
        failures += [] if ok else [name]
        detail = ", ".join(f"{k} {v:.3e}" for k, v in values.items())
        print(f"{'PASS' if ok else 'FAIL'} {name}: spread {spread:.1e} m; {detail}")
    failures += _points_check(rng)
    print("selftest:", "OK" if not failures else f"FAILED {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
