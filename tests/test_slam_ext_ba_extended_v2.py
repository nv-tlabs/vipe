import os
import unittest

import torch

from vipe.ext import slam_ext


def _require_cuda_slam_ext() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for slam_ext BA tests")
    if not hasattr(slam_ext, "ba_extended_v2"):
        raise RuntimeError("slam_ext.ba_extended_v2 is required for slam_ext BA v2 tests")


def _make_ba_inputs(device: str = "cuda"):
    torch.manual_seed(2026)
    n_frames, height, width, n_edges = 4, 4, 5, 5

    poses = torch.zeros(n_frames, 7, device=device)
    poses[:, 6] = 1.0
    poses[:, 0] = torch.linspace(0.0, 0.03, n_frames, device=device)

    disps = torch.ones(n_frames, height, width, device=device) * 0.8
    disps_sens = torch.zeros_like(disps)
    intrinsics = torch.tensor([40.0, 40.0, 2.0, 2.0], device=device)
    targets = torch.rand(n_edges, 2, height, width, device=device) * 3.0
    weights = torch.rand(n_edges, 2, height, width, device=device)
    ii = torch.tensor([1, 2, 2, 3, 3], device=device, dtype=torch.long)
    jj = torch.tensor([0, 0, 1, 1, 2], device=device, dtype=torch.long)
    kx = torch.unique(torch.cat([torch.arange(1, 4, device=device), ii]))
    eta = torch.ones(n_frames, height, width, device=device)[kx].contiguous() * 0.1

    return poses, disps, intrinsics, disps_sens, targets, weights, eta, ii, jj


class SlamExtBAExtendedV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda_slam_ext()

    def test_v2_matches_legacy_when_optional_features_are_disabled(self):
        """The fp64 dense-solve path (small system, below VIPE_BA_GPU_FP64_MAX_DIM)
        should reproduce the legacy Eigen/SimplicialLLT solve to tight tolerance for
        the plain motion+depth update case (no intrinsics optimization, where the two
        paths use an identical linear retraction and only differ in solve backend)."""
        inputs = _make_ba_inputs()
        poses0, disps0, intrinsics0, disps_sens, targets, weights, eta, ii, jj = inputs
        poses1 = poses0.clone()
        disps1 = disps0.clone()
        depth_active = torch.ones(poses0.shape[0], device=poses0.device)

        slam_ext.ba_extended(
            poses0,
            disps0,
            intrinsics0.clone(),
            disps_sens,
            targets,
            weights,
            eta,
            ii,
            jj,
            depth_active,
            1,
            4,
            2,
            1e-3,
            0.1,
            False,
            0.001,
            False,
            1e-6,
            1e-6,
            0.125,
        )
        slam_ext.ba_extended_v2(
            poses1,
            disps1,
            intrinsics0.clone(),
            disps_sens,
            targets,
            weights,
            eta,
            ii,
            jj,
            depth_active,
            1,
            4,
            2,
            1e-3,
            0.1,
            False,
            0.001,
            False,
            1e-6,
            1e-6,
            0.125,
        )

        torch.testing.assert_close(poses1, poses0, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(disps1, disps0, atol=1e-4, rtol=1e-4)

    def test_v2_matches_legacy_repeatedly(self):
        # Determinism guard, in the same spirit as the altcorr retile fix: a single
        # run of the parity test above cannot distinguish "flaky" from
        # "deterministically correct" for a device-resident solver whose scatter-add
        # assembly (GpuBlockSystem.add_pose_blocks / add_rhs_blocks) could in
        # principle race across edges landing in the same block. It doesn't (ATen's
        # index_put_ with accumulate=True is a well-defined scatter-add), but a
        # fixed-seed problem run many times is a cheap, direct way to keep that true.
        for _ in range(10):
            self.test_v2_matches_legacy_when_optional_features_are_disabled()

    def test_v2_motion_only_matches_legacy(self):
        inputs = _make_ba_inputs()
        poses0, disps0, intrinsics0, disps_sens, targets, weights, eta, ii, jj = inputs
        poses1 = poses0.clone()
        depth_active = torch.ones(poses0.shape[0], device=poses0.device)

        slam_ext.ba_extended(
            poses0,
            disps0.clone(),
            intrinsics0.clone(),
            disps_sens,
            targets,
            weights,
            eta,
            ii,
            jj,
            depth_active,
            1,
            4,
            1,
            1e-3,
            0.1,
            True,
            0.001,
            False,
            1e-6,
            1e-6,
            0.125,
        )
        slam_ext.ba_extended_v2(
            poses1,
            disps0.clone(),
            intrinsics0.clone(),
            disps_sens,
            targets,
            weights,
            eta,
            ii,
            jj,
            depth_active,
            1,
            4,
            1,
            1e-3,
            0.1,
            True,
            0.001,
            False,
            1e-6,
            1e-6,
            0.125,
        )

        torch.testing.assert_close(poses1, poses0, atol=1e-4, rtol=1e-4)

    def test_v2_optimizes_intrinsics_and_masks_limited_disparities(self):
        """optimize_intrinsics uses a different (log-space, positive-reals) focal
        retraction in v2 than the legacy linear retraction, so poses/disps only need
        to be finite and sane here, not numerically identical to legacy."""
        inputs = _make_ba_inputs()
        poses, disps, intrinsics, disps_sens, targets, weights, eta, ii, jj = inputs
        original_disps = disps.clone()
        original_intrinsics = intrinsics.clone()
        depth_active = torch.ones(poses.shape[0], device=poses.device)
        depth_active[1] = 0.0

        slam_ext.ba_extended_v2(
            poses,
            disps,
            intrinsics,
            disps_sens,
            targets,
            weights,
            eta,
            ii,
            jj,
            depth_active,
            1,
            4,
            1,
            1e-3,
            0.1,
            False,
            0.001,
            True,
            1e-6,
            1e-6,
            0.125,
        )

        self.assertTrue(torch.isfinite(intrinsics).all().item())
        self.assertTrue((intrinsics[:2] > 0).all().item())
        self.assertGreater((intrinsics[:2] - original_intrinsics[:2]).abs().max().item(), 0.0)
        torch.testing.assert_close(disps[1], original_disps[1], atol=0.0, rtol=0.0)

    def test_v2_returns_kernel_energy_when_requested(self):
        inputs = _make_ba_inputs()
        poses, disps, intrinsics, disps_sens, targets, weights, eta, ii, jj = inputs
        depth_active = torch.ones(poses.shape[0], device=poses.device)

        _, _, energy = slam_ext.ba_extended_v2(
            poses,
            disps,
            intrinsics,
            disps_sens,
            targets,
            weights,
            eta,
            ii,
            jj,
            depth_active,
            1,
            4,
            2,
            1e-3,
            0.1,
            False,
            0.001,
            False,
            1e-6,
            1e-6,
            0.125,
            True,
        )

        self.assertEqual(tuple(energy.shape), (2,))
        self.assertTrue(torch.isfinite(energy).all().item())
        self.assertGreater(energy[0].item(), 0.0)

    def test_v2_matches_legacy_above_fp64_threshold(self):
        """Force the float32 + iterative-refinement solve path (system dim above
        VIPE_BA_GPU_FP64_MAX_DIM) via the env var override, and confirm it still
        agrees with the legacy solve to a looser (float32-appropriate) tolerance."""
        old_value = os.environ.get("VIPE_BA_GPU_FP64_MAX_DIM")
        os.environ["VIPE_BA_GPU_FP64_MAX_DIM"] = "1"
        try:
            inputs = _make_ba_inputs()
            poses0, disps0, intrinsics0, disps_sens, targets, weights, eta, ii, jj = inputs
            poses1 = poses0.clone()
            disps1 = disps0.clone()
            depth_active = torch.ones(poses0.shape[0], device=poses0.device)

            slam_ext.ba_extended(
                poses0,
                disps0,
                intrinsics0.clone(),
                disps_sens,
                targets,
                weights,
                eta,
                ii,
                jj,
                depth_active,
                1,
                4,
                2,
                1e-3,
                0.1,
                False,
                0.001,
                False,
                1e-6,
                1e-6,
                0.125,
            )
            slam_ext.ba_extended_v2(
                poses1,
                disps1,
                intrinsics0.clone(),
                disps_sens,
                targets,
                weights,
                eta,
                ii,
                jj,
                depth_active,
                1,
                4,
                2,
                1e-3,
                0.1,
                False,
                0.001,
                False,
                1e-6,
                1e-6,
                0.125,
            )

            torch.testing.assert_close(poses1, poses0, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(disps1, disps0, atol=1e-3, rtol=1e-3)
        finally:
            if old_value is None:
                os.environ.pop("VIPE_BA_GPU_FP64_MAX_DIM", None)
            else:
                os.environ["VIPE_BA_GPU_FP64_MAX_DIM"] = old_value


if __name__ == "__main__":
    unittest.main()
