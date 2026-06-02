from __future__ import annotations

import torch
from omegaconf import OmegaConf

from vipe.slam import system
from vipe.streams.base import VideoFrame


class _FakeBuffer:
    def __init__(self) -> None:
        self.n_frames = 0
        self.tstamp = torch.zeros(2, dtype=torch.int)
        self.images = torch.zeros(2, 1, 3, 8, 8)
        self.masks = torch.zeros(2, 1, 1, 1, dtype=torch.bool)
        self.fmaps = torch.zeros(2, 1, 128, 1, 1)
        self.nets = torch.zeros(2, 1, 128, 1, 1)
        self.inps = torch.zeros(2, 1, 128, 1, 1)
        self.intrinsics = torch.zeros(1, 4)
        self.disps_sens = torch.zeros(2, 1, 1, 1)
        self.poses = torch.zeros(2, 7)
        self.updated_depth_frame_idx: int | None = None

    def update_disps_sens(self, metric_depth, frame_idx: int | None) -> None:
        self.updated_depth_frame_idx = frame_idx


class _UnexpectedEncoderNet:
    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        raise AssertionError("feature encoder should not run for precomputed pass1 keyframes")

    def encode_context(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("context encoder should not run for precomputed pass1 keyframes")


def test_add_keyframe_reuses_pass1_motion_filter_precompute() -> None:
    slam = system.SLAMSystem(
        torch.device("cpu"),
        OmegaConf.create({"visualize": False, "optimize_intrinsics": False}),
    )
    slam.buffer = _FakeBuffer()
    slam.droid_net = _UnexpectedEncoderNet()
    slam.metric_depth = None
    images = torch.ones(1, 3, 8, 8)
    fmap = torch.full((1, 128, 1, 1), 3.0)
    net = torch.full((1, 128, 1, 1), 5.0)
    inp = torch.full((1, 128, 1, 1), 7.0)
    frame = VideoFrame(
        raw_frame_idx=0,
        rgb=torch.zeros(8, 8, 3),
        intrinsics=torch.tensor([1.0, 1.0, 4.0, 4.0]),
    )

    slam._add_keyframe(
        12,
        images,
        None,
        [frame],
        phase=1,
        precomputed_fmap=fmap,
        precomputed_context=(net, inp),
    )

    assert slam.buffer.n_frames == 1
    assert slam.buffer.tstamp[0].item() == 12
    assert torch.equal(slam.buffer.fmaps[0], fmap)
    assert torch.equal(slam.buffer.nets[0], net)
    assert torch.equal(slam.buffer.inps[0], inp)
    assert slam.buffer.updated_depth_frame_idx == 0
