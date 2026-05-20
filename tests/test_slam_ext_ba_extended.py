import unittest

import torch

from vipe.ext import slam_ext


def _require_cuda_slam_ext() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for slam_ext BA tests")
    if not hasattr(slam_ext, "ba_extended"):
        raise RuntimeError("slam_ext.ba_extended is required for slam_ext BA tests")


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


class SlamExtBAExtendedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda_slam_ext()

    def test_extended_matches_legacy_when_optional_features_are_disabled(self):
        inputs = _make_ba_inputs()
        poses0, disps0, intrinsics0, disps_sens, targets, weights, eta, ii, jj = inputs
        poses1 = poses0.clone()
        disps1 = disps0.clone()
        depth_active = torch.ones(poses0.shape[0], device=poses0.device)

        slam_ext.ba(
            poses0,
            disps0,
            intrinsics0.clone(),
            disps_sens,
            targets,
            weights,
            eta,
            ii,
            jj,
            1,
            4,
            1,
            1e-3,
            0.1,
            False,
            0.001,
        )
        slam_ext.ba_extended(
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
            1,
            1e-3,
            0.1,
            False,
            0.001,
            False,
            1e-6,
            1e-6,
            0.125,
        )

        torch.testing.assert_close(poses1, poses0, atol=0.0, rtol=0.0)
        torch.testing.assert_close(disps1, disps0, atol=0.0, rtol=0.0)

    def test_extended_optimizes_intrinsics_and_masks_limited_disparities(self):
        inputs = _make_ba_inputs()
        poses, disps, intrinsics, disps_sens, targets, weights, eta, ii, jj = inputs
        original_disps = disps.clone()
        original_intrinsics = intrinsics.clone()
        depth_active = torch.ones(poses.shape[0], device=poses.device)
        depth_active[1] = 0.0

        slam_ext.ba_extended(
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
        self.assertGreater((intrinsics[:2] - original_intrinsics[:2]).abs().max().item(), 0.0)
        torch.testing.assert_close(disps[1], original_disps[1], atol=0.0, rtol=0.0)

    def test_extended_returns_kernel_energy_when_requested(self):
        inputs = _make_ba_inputs()
        poses, disps, intrinsics, disps_sens, targets, weights, eta, ii, jj = inputs
        depth_active = torch.ones(poses.shape[0], device=poses.device)

        _, _, energy = slam_ext.ba_extended(
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


if __name__ == "__main__":
    unittest.main()
