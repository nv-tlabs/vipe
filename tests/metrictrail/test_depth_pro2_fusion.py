# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vipe.pipeline import processors
from vipe.priors.depth import depthpro2 as depth_pro2
from vipe.priors.depth.base import DepthEstimationResult
from vipe.streams.base import VideoFrame
from vipe.utils.cameras import CameraType

HW = (20, 30)


def scene_depth():
    return 2.0 + torch.linspace(0, 1, HW[0])[:, None] + torch.linspace(0, 1, HW[1])[None]


class TemporalNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.precision = torch.bfloat16
        self.fov = None
        self.streams = []
        self.offset = 0.0

    def stream(self, **settings):
        history = []

        def step(rgb):
            history.append((rgb.clone(), self.precision))
            depth = (2.0 / scene_depth() + 0.1 + self.offset * len(history)).reciprocal()
            depth[0, 0] = float("nan")
            return [{"depth": depth}]

        stream = SimpleNamespace(step=step, history=history, settings=settings)
        self.streams.append(stream)
        return stream


def frames(count=3, focal=30.0):
    for index in range(count):
        frame = VideoFrame(
            raw_frame_idx=10 + index * 3,
            rgb=torch.full((*HW, 3), 0.1 * (index + 1)),
            intrinsics=torch.tensor([focal, focal, HW[1] / 2, HW[0] / 2]),
            camera_type=CameraType.PINHOLE,
            mask=torch.ones(HW, dtype=torch.bool),
            instance=torch.zeros(HW, dtype=torch.uint8),
            instance_phrases={1: VideoFrame.SKY_PROMPT},
        )
        frame.mask[2:4, 2:4] = False
        frame.instance[5:7, 5:7] = 1
        yield frame


@pytest.fixture
def build(monkeypatch, tmp_path):
    network = TemporalNetwork()
    weights = depth_pro2.DepthPro2Weights(network, tmp_path)
    loaded, made, prompts, aligns = [], [], [], []

    def load(name, **kwargs):
        loaded.append(name)
        return weights

    class MetricPrompt:
        def __init__(self, role):
            self.role = role

        def estimate(self, request):
            prompts.append((self.role, request))
            return DepthEstimationResult(metric_depth=scene_depth())

    def metric_model(name):
        made.append(name)
        return MetricPrompt("metric")

    original_align = processors.align_inv_depth_to_depth

    def align(source, target, mask):
        aligns.append((source.clone(), target.clone(), mask.clone()))
        return original_align(source, target, mask)

    def no_vda(**kwargs):
        raise AssertionError("Depth Pro 2 fusion must not construct VDA")

    monkeypatch.setattr(VideoFrame, "cuda", lambda self: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(depth_pro2, "load_depth_pro2_weights", load)
    monkeypatch.setattr(processors, "make_depth_model", metric_model)
    monkeypatch.setattr(processors, "PriorDAModel", lambda: MetricPrompt("map"))
    monkeypatch.setattr(processors, "VideoDepthAnythingDepthModel", no_vda)
    monkeypatch.setattr(processors, "align_inv_depth_to_depth", align)

    def make(name="depthpro2-l", dense=True):
        slam_map = SimpleNamespace(project_map=lambda *args, **kwargs: scene_depth() if dense else torch.zeros(HW))
        slam = SimpleNamespace(
            slam_map=slam_map,
            trajectory=torch.zeros(3, 7),
            get_view_trajectory=lambda view: torch.zeros(3, 7),
        )
        model_name = "adaptive_unidepth-l" + (f"_{name}" if name is not None else "")
        processor = processors.AdaptiveDepthProcessor(slam, model=model_name)
        return processor

    return SimpleNamespace(make=make, network=network, loaded=loaded, made=made, prompts=prompts, aligns=aligns)


@pytest.mark.parametrize("name", ["depthpro2-l", "depthpro2-h"])
@pytest.mark.parametrize("dense", [False, True])
def test_fusion_uses_native_metric_prompts_and_affine_alignment(build, name, dense):
    processor = build.make(name, dense)
    output = list(processor.update_iterator(frames(), 0))
    assert build.loaded == [name] and build.made == ["unidepth-l"]
    assert [frame.raw_frame_idx for frame in output] == [10, 13, 16]
    assert [role for role, _ in build.prompts] == ["map" if dense else "metric"] * 3
    assert len(build.network.streams) == 1
    stream = build.network.streams[0]
    assert stream.settings == {"mode": "unbuffered", "window": 16, "f_px": 30.0}
    assert all(dtype == torch.float32 for _, dtype in stream.history)
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    for index, ((rgb, _), frame, (_, request), (_, _, mask)) in enumerate(
        zip(stream.history, output, build.prompts, build.aligns)
    ):
        torch.testing.assert_close(rgb[:, 0, 0], (0.1 * (index + 1) - mean) / std)
        expected_mask = frame.mask & ~frame.sky_mask
        expected_mask[0, 0] = False
        assert torch.equal(mask, expected_mask)
        if dense:
            torch.testing.assert_close(request.prompt_metric_depth, scene_depth() * frame.mask)
        expected = scene_depth()
        expected[0, 0] = 0
        torch.testing.assert_close(frame.metric_depth, expected, atol=1e-5, rtol=1e-5)
    assert processor.update_momentum == 0.99


def test_fusion_consumes_one_frame_at_a_time_and_replay_resets_history(build):
    processor = build.make()
    consumed = []

    def source():
        for frame in frames():
            consumed.append(frame.raw_frame_idx)
            yield frame

    iterator = processor.update_iterator(source(), 0)
    first = next(iterator)
    assert consumed == [10] and processor.temporal_depth_model.frames_seen == 1
    output = [first, *iterator]
    repeated = list(processor.update_iterator(frames(), 0))
    assert len(build.network.streams) == 2
    assert [len(stream.history) for stream in build.network.streams] == [3, 3]
    assert processor.temporal_depth_model.frames_seen == 3
    for left, right in zip(output, repeated):
        torch.testing.assert_close(left.metric_depth, right.metric_depth)


def test_fusion_rejects_focal_changes_and_can_restart_with_new_focal(build):
    processor = build.make()
    items = list(frames())
    items[1].intrinsics[0] = 31
    with pytest.raises(ValueError, match="stream focal/size changed"):
        list(processor.update_iterator(iter(items), 0))
    output = list(processor.update_iterator(frames(focal=32), 0))
    assert len(output) == 3
    assert build.network.streams[-1].settings["f_px"] == 32


def test_fusion_preserves_native_momentum_and_failed_fit_reuse(build, monkeypatch):
    processor = build.make()
    calls = []

    def fit(source, target, mask):
        calls.append(True)
        if len(calls) == 1:
            return source, source.new_tensor(0.5), source.new_tensor(-0.05)
        if len(calls) == 2:
            return source, source.new_tensor(0.7), source.new_tensor(0.15)
        raise RuntimeError("synthetic alignment failure")

    monkeypatch.setattr(processors, "align_inv_depth_to_depth", fit)
    output = list(processor.update_iterator(frames(), 0))
    torch.testing.assert_close(processor.cache_scale_bias[0], torch.tensor(0.502))
    torch.testing.assert_close(processor.cache_scale_bias[1], torch.tensor(-0.048))
    torch.testing.assert_close(output[1].metric_depth, output[2].metric_depth)
    assert all(frame.metric_depth[0, 0] == 0 for frame in output)
    with pytest.raises(RuntimeError, match="synthetic alignment failure"):
        next(processor.update_iterator(frames(), 0))


def test_temporal_constructor_error_is_not_swallowed_by_model_name_parsing(build, monkeypatch):
    def invalid_weights(*args, **kwargs):
        raise ValueError("invalid Depth Pro 2 checkpoint")

    monkeypatch.setattr(depth_pro2, "load_depth_pro2_weights", invalid_weights)
    with pytest.raises(ValueError, match="invalid Depth Pro 2 checkpoint"):
        build.make()


def test_streamed_fusion_matches_precomputed_temporal_fusion(build, monkeypatch):
    build.network.offset = 0.01
    streamed = build.make()
    precomputed = build.make(name=None)
    inverse = torch.stack([2.0 / scene_depth() + 0.1 + 0.01 * (index + 1) for index in range(3)])
    inverse[:, 0, 0] = 0
    precomputed.video_depth_model = object()
    monkeypatch.setattr(precomputed, "_compute_video_da", lambda source: (inverse, list(source)))
    current = list(streamed.update_iterator(frames(), 0))
    frozen = list(precomputed.update_iterator(frames(), 0))
    for actual, reference in zip(current, frozen):
        reference.metric_depth[0, 0] = 0
        assert actual.raw_frame_idx == reference.raw_frame_idx
        torch.testing.assert_close(actual.metric_depth, reference.metric_depth, atol=1e-5, rtol=1e-5)
    for actual, reference in zip(streamed.cache_scale_bias, precomputed.cache_scale_bias):
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
