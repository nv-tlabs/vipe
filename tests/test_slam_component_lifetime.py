from __future__ import annotations

import torch
from omegaconf import OmegaConf

from vipe.slam import system
from vipe.slam.networks import droid_net
from vipe.utils.cameras import CameraType


class _FakeRig:
    def __init__(self) -> None:
        self.data = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    def to(self, device: torch.device) -> "_FakeRig":
        self.data = self.data.to(device)
        return self


def _minimal_slam_config():
    return OmegaConf.create(
        {
            "visualize": False,
            "sparse_tracks": {"name": "dummy"},
            "n_views": 1,
            "height": 8,
            "width": 8,
            "buffer": 4,
            "init_disp": 1.0,
            "cross_view_idx": None,
            "ba": {},
            "camera_type": CameraType.PINHOLE,
            "filter_thresh": 16.0,
            "cross_view": False,
            "warmup": 2,
            "beta": 0.3,
            "frontend_nms": 1,
            "keyframe_thresh": 2.5,
            "frontend_window": 4,
            "frontend_thresh": 16.0,
            "frontend_radius": 1,
            "has_init_pose": False,
            "infill_chunk_size": 2,
            "keyframe_depth": None,
        },
        flags={"allow_objects": True},
    )


def test_shared_droid_net_reuses_frozen_eval_model(monkeypatch) -> None:
    load_calls = 0

    def fake_load_weights(self) -> None:
        nonlocal load_calls
        load_calls += 1

    droid_net.clear_shared_droid_net_cache()
    monkeypatch.setattr(droid_net.DroidNet, "load_weights", fake_load_weights)

    try:
        first = droid_net.get_shared_droid_net(torch.device("cpu"))
        second = droid_net.get_shared_droid_net(torch.device("cpu"))
    finally:
        droid_net.clear_shared_droid_net_cache()

    assert first is second
    assert load_calls == 1
    assert not first.training
    assert all(not parameter.requires_grad for parameter in first.parameters())
    assert first.image_mean.data_ptr() == second.image_mean.data_ptr()
    assert first.image_std.data_ptr() == second.image_std.data_ptr()


def test_slam_component_build_reuses_only_droid_net(monkeypatch) -> None:
    droid_net.clear_shared_droid_net_cache()
    monkeypatch.setattr(droid_net.DroidNet, "load_weights", lambda self: None)

    try:
        first = system.SLAMSystem(torch.device("cpu"), _minimal_slam_config())
        first.rig = _FakeRig()
        first._build_components()

        second = system.SLAMSystem(torch.device("cpu"), _minimal_slam_config())
        second.rig = _FakeRig()
        second._build_components()
    finally:
        droid_net.clear_shared_droid_net_cache()

    assert first.droid_net is second.droid_net
    assert first.sparse_tracks is not second.sparse_tracks
    assert first.buffer is not second.buffer
    assert first.motion_filter is not second.motion_filter
    assert first.frontend is not second.frontend
    assert first.frontend.graph is not second.frontend.graph
    assert first.backend is not second.backend
    assert first.inner_filler is not second.inner_filler
    assert first.buffer.poses.data_ptr() != second.buffer.poses.data_ptr()
    assert first.buffer.cross_view_idx.data_ptr() != second.buffer.cross_view_idx.data_ptr()
