# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two pose artifact npz files (as written by ViPE pipelines).

Reports ATE (after Sim3 alignment) and relative rotation error so that a
faster pipeline can be validated against the default pipeline's trajectory.

Usage: python scripts/compare_pose_artifacts.py ref_pose.npz test_pose.npz
"""

import argparse
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vipe.utils.io import read_pose_artifacts


def umeyama_align(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (s, R, t) minimizing ||target - (s * R @ source + t)||^2."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    cov = xt.T @ xs / len(source)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (xs**2).sum(1).mean()
    t = mu_t - s * R @ mu_s
    return float(s), R, t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", type=Path, help="Reference pose npz (e.g. default pipeline output)")
    parser.add_argument("test", type=Path, help="Test pose npz (e.g. pose-only pipeline output)")
    args = parser.parse_args()

    ref_inds, ref_traj = read_pose_artifacts(args.ref)
    test_inds, test_traj = read_pose_artifacts(args.test)

    common, ref_pos, test_pos = np.intersect1d(ref_inds, test_inds, return_indices=True)
    assert len(common) > 0, "No common frame indices between the two trajectories"
    ref_mat = ref_traj.matrix().cpu().numpy()[ref_pos]
    test_mat = test_traj.matrix().cpu().numpy()[test_pos]
    print(f"Comparing {len(common)} common frames")

    # Sim3-align test onto ref (pose accuracy is defined up to a global similarity).
    s, R, t = umeyama_align(test_mat[:, :3, 3], ref_mat[:, :3, 3])
    ref_t = ref_mat[:, :3, 3]
    test_t = (s * (R @ test_mat[:, :3, 3].T)).T + t
    ate = np.linalg.norm(ref_t - test_t, axis=-1)

    # Rotation error via frame-to-frame relative rotations, which are invariant
    # to the global alignment.
    ref_rot = ref_mat[:, :3, :3]
    test_rot = test_mat[:, :3, :3]
    rel_ref = np.einsum("nij,nik->njk", ref_rot[:-1], ref_rot[1:])
    rel_test = np.einsum("nij,nik->njk", test_rot[:-1], test_rot[1:])
    rel_err = np.einsum("nij,nik->njk", rel_ref, rel_test)
    cos = np.clip((np.trace(rel_err, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(cos))

    traj_extent = float(np.linalg.norm(ref_t.max(0) - ref_t.min(0)))
    print(f"Trajectory extent (ref): {traj_extent:.4f} m")
    print(f"ATE  RMSE: {np.sqrt((ate**2).mean()):.6f} m | mean: {ate.mean():.6f} | max: {ate.max():.6f}")
    print(f"Rot  mean: {rot_err_deg.mean():.6f} deg | max: {rot_err_deg.max():.6f} deg")
    if traj_extent > 0:
        print(f"ATE RMSE / extent: {100.0 * np.sqrt((ate**2).mean()) / traj_extent:.4f} %")


if __name__ == "__main__":
    main()
