import os
import unittest

try:
    import torch

    from vipe.ext import slam_ext
    from vipe.ext.lietorch import SE3
    from vipe.slam.ba.terms import DenseDepthFlowTerm
    from vipe.utils.cameras import CameraType
except ImportError:  # pragma: no cover - lets discovery work without the runtime env
    torch = None
    slam_ext = None
    SE3 = None
    DenseDepthFlowTerm = None
    CameraType = None


def _has_cuda_slam_ext() -> bool:
    return torch is not None and torch.cuda.is_available() and slam_ext is not None


@unittest.skipUnless(_has_cuda_slam_ext(), "CUDA slam extension is required")
class BAFusedTermsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        self._old_fused_term = os.environ.get("VIPE_ENABLE_BA_FUSED_TERM")

    def tearDown(self):
        if self._old_fused_term is None:
            os.environ.pop("VIPE_ENABLE_BA_FUSED_TERM", None)
        else:
            os.environ["VIPE_ENABLE_BA_FUSED_TERM"] = self._old_fused_term

    def test_dense_depth_flow_fused_term_matches_reference(self):
        device = torch.device("cuda")
        n_frames, height, width, n_terms = 4, 4, 5, 3

        poses = torch.zeros(n_frames, 7, device=device)
        poses[:, 6] = 1.0
        poses[:, 0] = torch.tensor([0.0, 0.02, 0.05, 0.09], device=device)
        poses[:, 1] = torch.tensor([0.0, -0.01, 0.015, 0.02], device=device)
        poses[:, 5] = torch.tensor([0.0, 0.01, -0.02, 0.015], device=device)
        poses[:, 6] = (1 - poses[:, 5] ** 2).sqrt()

        dense_disp = 0.7 + 0.1 * torch.rand(n_frames, height * width, device=device)
        intrinsics = torch.tensor([[80.0, 82.0, 32.0, 31.0]], device=device)
        rig = SE3.Identity(1, device=device)

        pose_i = torch.tensor([1, 2, 3], device=device)
        pose_j = torch.tensor([0, 1, 2], device=device)
        rig_i = torch.zeros(n_terms, device=device, dtype=torch.long)
        rig_j = torch.zeros(n_terms, device=device, dtype=torch.long)
        dense_disp_i = pose_i.clone()

        target = torch.rand(n_terms, height * width, 2, device=device) * 6.0
        weight = torch.rand(n_terms, height * width, 2, device=device)

        term = DenseDepthFlowTerm(
            pose_i,
            pose_j,
            rig_i,
            rig_j,
            dense_disp_i,
            target,
            weight,
            None,
            8.0,
            rig,
            (height, width),
            CameraType.PINHOLE,
        )
        variables = {
            "pose": SE3(poses.clone()),
            "dense_disp": dense_disp.clone(),
            "intrinsics": intrinsics.clone(),
        }

        os.environ.pop("VIPE_ENABLE_BA_FUSED_TERM", None)
        reference = term.forward(variables, jacobian=True, active_group_names={"pose", "dense_disp", "intrinsics"})
        os.environ["VIPE_ENABLE_BA_FUSED_TERM"] = "1"
        fused = term.forward(variables, jacobian=True, active_group_names={"pose", "dense_disp", "intrinsics"})

        torch.testing.assert_close(fused.r, reference.r, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(fused.w, reference.w, atol=0.0, rtol=0.0)
        for group_name in reference.J:
            torch.testing.assert_close(fused.J[group_name].i_inds, reference.J[group_name].i_inds)
            torch.testing.assert_close(fused.J[group_name].j_inds, reference.J[group_name].j_inds)
            torch.testing.assert_close(fused.J[group_name].data, reference.J[group_name].data, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
