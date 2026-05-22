# mypy: ignore-errors

from __future__ import annotations

import unittest

import scipy  # noqa: F401
import torch
from omegaconf import OmegaConf

from vipe.ext import slam_ext
from vipe.slam.components.buffer import GraphBuffer
from vipe.slam.components.sparse_tracks import DummySparseTracks
from vipe.utils.cameras import CameraType


def _require_cuda_ba_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fused BA parity tests")
    if not hasattr(slam_ext, "ba_extended"):
        raise RuntimeError("slam_ext.ba_extended is required for fused BA parity tests")


class FixtureSparseTracks(DummySparseTracks):
    def __init__(self, n_views: int, enabled: bool) -> None:
        super().__init__(n_views)
        self.enabled = enabled

    def compute_dense_disp_target_weight(
        self,
        source_view_inds: torch.Tensor,
        source_frame_inds: torch.Tensor,
        target_view_inds: torch.Tensor,
        target_frame_inds: torch.Tensor,
        image_size: tuple[int, int],
        dense_disp_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del source_view_inds, target_view_inds, image_size
        disp_h, disp_w = dense_disp_size
        yy, xx = torch.meshgrid(
            torch.arange(disp_h, device=source_frame_inds.device, dtype=torch.float32),
            torch.arange(disp_w, device=source_frame_inds.device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1)
        dt = (target_frame_inds - source_frame_inds).to(torch.float32).view(-1, 1, 1, 1)
        target = coords.unsqueeze(0) + torch.cat([0.03 * dt, -0.02 * dt], dim=-1)
        weight = torch.full_like(target, 0.07)
        weight = weight * (source_frame_inds != target_frame_inds).to(torch.float32).view(-1, 1, 1, 1)
        return target, weight


def _ba_config(fused: bool):
    return OmegaConf.create(
        {
            "dense_disp_alpha": 0.001,
            "fused": fused,
            "intrinsics_damping_scale": 1.0,
            "robust_kernel": None,
            "robust_kernel_threshold": 3.0,
            "gnc_mu_init": 1.0,
            "gnc_mu_step": 3.0,
            "gnc_mu_max": 1.0e6,
            "gnc_n_mu_steps": 4,
            "gnc_gn_iters_per_mu": 6,
        }
    )


def _make_video(fused: bool, sparse_tracks_enabled: bool):
    torch.manual_seed(20260519)
    device = torch.device("cuda")
    height, width, n_frames = 32, 40, 5
    video = GraphBuffer(
        height=height,
        width=width,
        n_views=1,
        buffer_size=6,
        init_disp=0.8,
        cross_view_idx=None,
        ba_config=_ba_config(fused),
        sparse_tracks=FixtureSparseTracks(n_views=1, enabled=sparse_tracks_enabled),
        camera_type=CameraType.PINHOLE,
        device=device,
    )
    video.n_frames = n_frames
    video.tstamp[:n_frames] = torch.arange(n_frames, device=device)
    video.intrinsics[0, :4] = torch.tensor([38.0, 38.0, 19.5, 15.5], device=device)

    video.poses[:n_frames, 0] = torch.linspace(0.0, 0.08, n_frames, device=device)
    video.poses[:n_frames, 1] = torch.tensor([0.0, 0.01, -0.005, 0.015, 0.02], device=device)
    video.poses[:n_frames, 2] = torch.linspace(0.0, 0.025, n_frames, device=device)

    disp_h, disp_w = height // 8, width // 8
    yy, xx = torch.meshgrid(
        torch.arange(disp_h, device=device, dtype=torch.float32),
        torch.arange(disp_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    base_disp = 0.72 + 0.015 * xx + 0.02 * yy
    for frame_idx in range(n_frames):
        video.disps[frame_idx, 0] = base_disp + 0.025 * frame_idx

    video.disps_sens[2, 0] = video.disps[2, 0] * 1.015
    video.disps_sens[4, 0] = video.disps[4, 0] * 0.985
    return video


def _make_real_run_shaped_ba_inputs(video):
    device = video.device
    disp_h, disp_w = video.height // 8, video.width // 8
    ii = torch.tensor([1, 2, 3, 4, 0, 3], device=device, dtype=torch.long)
    jj = torch.tensor([0, 0, 1, 2, 2, 3], device=device, dtype=torch.long)

    coords, _ = video.reproject_dense_disp(ii, jj)
    target = coords.reshape(ii.shape[0], disp_h * disp_w, 2).clone()

    yy, xx = torch.meshgrid(
        torch.arange(disp_h, device=device, dtype=torch.float32),
        torch.arange(disp_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    spatial_delta = torch.stack(
        [
            0.01 * torch.sin(0.7 * xx + 0.2 * yy),
            -0.012 * torch.cos(0.3 * xx - 0.5 * yy),
        ],
        dim=-1,
    ).reshape(1, disp_h * disp_w, 2)
    edge_delta = torch.linspace(-0.015, 0.02, ii.shape[0], device=device).view(-1, 1, 1)
    target = target + spatial_delta + torch.cat([edge_delta, -0.5 * edge_delta], dim=-1)

    weight = 0.2 + 0.03 * (xx + yy).reshape(1, disp_h * disp_w, 1)
    weight = weight.repeat(ii.shape[0], 1, 2).contiguous()
    weight[-1].zero_()  # Degenerate self-edge: present in real graphs, but contributes nothing here.

    disp_damping = 0.08 + 0.01 * torch.arange(6, device=device, dtype=torch.float32).view(-1, 1, 1)
    disp_damping = disp_damping + 0.002 * xx.unsqueeze(0) + 0.001 * yy.unsqueeze(0)
    return target.contiguous(), weight.contiguous(), disp_damping.contiguous(), ii, jj


def _run_generic_bundle_adjustment(
    video, ba_inputs, *, motion_only: bool, limited_disp: bool, optimize_intrinsics: bool
):
    target, weight, disp_damping, ii, jj = ba_inputs
    video.bundle_adjustment(
        target=target.clone(),
        weight=weight.clone(),
        disp_damping=disp_damping.clone(),
        ii=ii.clone(),
        jj=jj.clone(),
        t0=1,
        t1=5,
        n_iters=1,
        pose_damping=1e-3,
        pose_ep=0.1,
        motion_only=motion_only,
        limited_disp=limited_disp,
        optimize_intrinsics=optimize_intrinsics,
        optimize_rig_rotation=False,
        verbose=False,
    )


def _run_fused_bundle_adjustment(video, ba_inputs, *, motion_only: bool, limited_disp: bool, optimize_intrinsics: bool):
    target, weight, disp_damping, ii, jj = ba_inputs
    if not video.ba_config.fused:
        return False

    video._fused_ba(
        target=target.clone(),
        weight=weight.clone(),
        disp_damping=disp_damping.clone(),
        ii=ii.clone(),
        jj=jj.clone(),
        t0=1,
        t1=5,
        n_iters=1,
        pose_damping=1e-3,
        pose_ep=0.1,
        motion_only=motion_only,
        limited_disp=limited_disp,
        optimize_intrinsics=optimize_intrinsics,
        optimize_rig_rotation=False,
        weight_dense_disp=0.001,
        weight_tracks=0.001,
        verbose=False,
    )
    video.disps.clamp_(min=0.001)
    return True


class GraphBufferFusedBATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda_ba_runtime()

    def assert_graph_buffers_close(self, actual, expected) -> None:
        torch.testing.assert_close(actual.poses[:5], expected.poses[:5], atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(actual.disps[:5], expected.disps[:5], atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(actual.intrinsics, expected.intrinsics, atol=2e-4, rtol=2e-4)

    def test_fused_ba_matches_generic_solver_with_intrinsics_limited_disp_and_sparse_tracks(self):
        generic = _make_video(fused=False, sparse_tracks_enabled=True)
        fused = _make_video(fused=True, sparse_tracks_enabled=True)
        ba_inputs = _make_real_run_shaped_ba_inputs(generic)

        _run_generic_bundle_adjustment(
            generic,
            ba_inputs,
            motion_only=False,
            limited_disp=True,
            optimize_intrinsics=True,
        )
        used_fused_ba = _run_fused_bundle_adjustment(
            fused,
            ba_inputs,
            motion_only=False,
            limited_disp=True,
            optimize_intrinsics=True,
        )

        self.assertTrue(used_fused_ba)
        self.assert_graph_buffers_close(fused, generic)

    def test_fused_ba_matches_generic_solver_motion_only(self):
        generic = _make_video(fused=False, sparse_tracks_enabled=False)
        fused = _make_video(fused=True, sparse_tracks_enabled=False)
        ba_inputs = _make_real_run_shaped_ba_inputs(generic)
        original_fused_disps = fused.disps.clone()

        _run_generic_bundle_adjustment(
            generic,
            ba_inputs,
            motion_only=True,
            limited_disp=False,
            optimize_intrinsics=False,
        )
        used_fused_ba = _run_fused_bundle_adjustment(
            fused,
            ba_inputs,
            motion_only=True,
            limited_disp=False,
            optimize_intrinsics=False,
        )

        self.assertTrue(used_fused_ba)
        torch.testing.assert_close(fused.disps, original_fused_disps, atol=0.0, rtol=0.0)
        self.assert_graph_buffers_close(fused, generic)

    def test_fused_ba_logs_energy_when_verbose(self):
        fused = _make_video(fused=True, sparse_tracks_enabled=False)
        ba_inputs = _make_real_run_shaped_ba_inputs(fused)
        target, weight, disp_damping, ii, jj = ba_inputs

        with self.assertLogs("vipe.slam.components.buffer", level="INFO") as logs:
            fused.bundle_adjustment(
                target=target.clone(),
                weight=weight.clone(),
                disp_damping=disp_damping.clone(),
                ii=ii.clone(),
                jj=jj.clone(),
                t0=1,
                t1=5,
                n_iters=1,
                pose_damping=1e-3,
                pose_ep=0.1,
                motion_only=False,
                limited_disp=False,
                optimize_intrinsics=False,
                optimize_rig_rotation=False,
                verbose=True,
            )

        self.assertRegex(logs.output[-1], r"BA iters = 1, energy: [0-9.eE+-]+ -> [0-9.eE+-]+")

    def test_fused_ba_errors_for_unsupported_layout_instead_of_falling_back(self):
        fused = _make_video(fused=True, sparse_tracks_enabled=False)
        ba_inputs = _make_real_run_shaped_ba_inputs(fused)
        target, weight, disp_damping, ii, jj = ba_inputs

        with self.assertRaisesRegex(RuntimeError, "does not support optimizing rig rotation"):
            fused.bundle_adjustment(
                target=target.clone(),
                weight=weight.clone(),
                disp_damping=disp_damping.clone(),
                ii=ii.clone(),
                jj=jj.clone(),
                t0=1,
                t1=5,
                n_iters=1,
                pose_damping=1e-3,
                pose_ep=0.1,
                motion_only=False,
                limited_disp=False,
                optimize_intrinsics=False,
                optimize_rig_rotation=True,
                verbose=False,
            )

    def test_fused_ba_errors_for_robust_kernel_instead_of_falling_back(self):
        fused = _make_video(fused=True, sparse_tracks_enabled=False)
        fused.ba_config.robust_kernel = "huber"
        ba_inputs = _make_real_run_shaped_ba_inputs(fused)
        target, weight, disp_damping, ii, jj = ba_inputs

        with self.assertRaisesRegex(RuntimeError, "robust kernels are not supported"):
            fused.bundle_adjustment(
                target=target.clone(),
                weight=weight.clone(),
                disp_damping=disp_damping.clone(),
                ii=ii.clone(),
                jj=jj.clone(),
                t0=1,
                t1=5,
                n_iters=1,
                pose_damping=1e-3,
                pose_ep=0.1,
                motion_only=False,
                limited_disp=False,
                optimize_intrinsics=False,
                optimize_rig_rotation=False,
                verbose=False,
            )

    def test_hydra_config_gate_disables_fused_ba_without_env_vars(self):
        disabled = _make_video(fused=False, sparse_tracks_enabled=False)
        ba_inputs = _make_real_run_shaped_ba_inputs(disabled)

        used_fused_ba = _run_fused_bundle_adjustment(
            disabled,
            ba_inputs,
            motion_only=False,
            limited_disp=False,
            optimize_intrinsics=False,
        )

        self.assertFalse(used_fused_ba)


if __name__ == "__main__":
    unittest.main()
