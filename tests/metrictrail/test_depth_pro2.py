# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from vipe import get_config_path
from vipe.metrictrail.depth import make_metrictrail_depth_model
from vipe.priors.depth import depthpro2 as depth_pro2
from vipe.priors.depth import make_depth_model
from vipe.priors.depth.base import DepthEstimationInput, DepthType
from vipe.utils.model_cache import ModelCache


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.precision = torch.bfloat16
        self.fov = SimpleNamespace(precision=torch.bfloat16, focal_length_px=self.focal)
        self.calls, self.streams, self.fov_calls = [], [], []
        self.output, self.fail = None, False

    def focal(self, rgb):
        self.fov_calls.append((rgb.dtype, self.precision, self.fov.precision))
        return torch.tensor(12.5)

    def infer(self, rgb, *, f_px):
        self.calls.append((rgb.clone(), f_px, self.precision))
        return self.output if self.output is not None else {"depth": torch.full(rgb.shape[-2:], 2.5)}

    def stream(self, **kwargs):
        history = []

        def step(rgb):
            if self.fail:
                raise RuntimeError("stream failure")
            history.append(rgb.clone())
            self.infer(rgb, f_px=kwargs["f_px"])
            return [{"depth": torch.full(rgb.shape[-2:], float(len(history)))}]

        stream = SimpleNamespace(step=step, history=history, settings=kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture
def providers(monkeypatch, tmp_path):
    model, cache, loads = FakeModel(), ModelCache(), []
    weights = depth_pro2.DepthPro2Weights(model, tmp_path)

    def load(name, **kwargs):
        loads.append(name)
        return weights

    monkeypatch.setattr(depth_pro2, "load_depth_pro2_weights", load)

    def build(stream=False):
        return make_metrictrail_depth_model("depthpro2-l" + ("-stream" if stream else ""), model_cache=cache)

    return model, build, loads


def source(batch=None, focal=12.5):
    shape = (2, 3, 3) if batch is None else (batch, 2, 3, 3)
    return DepthEstimationInput(rgb=torch.full(shape, 0.5), intrinsics=torch.tensor([focal, focal, 1.5, 1.0]))


def test_shared_provider_has_no_metrictrail_import_dependency():
    code = """
import importlib.abc
import sys

class BlockMetricTrail(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vipe.metrictrail' or fullname.startswith('vipe.metrictrail.'):
            raise AssertionError('shared depth provider must not import MetricTrail')

sys.meta_path.insert(0, BlockMetricTrail())
from vipe.priors.depth.depthpro2 import DepthPro2Model
assert DepthPro2Model.__module__ == 'vipe.priors.depth.depthpro2'
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_checkpoint_paths_preserve_defaults_and_override_priority(monkeypatch, tmp_path):
    monkeypatch.delenv("DEPTHPRO2_CHECKPOINT_DIR", raising=False)
    repo = Path(__file__).resolve().parents[2]
    assert depth_pro2.checkpoint_directory() == repo / "ckpt" / "depth-pro-2"
    monkeypatch.setenv("DEPTHPRO2_CHECKPOINT_DIR", str(tmp_path / "environment"))
    assert depth_pro2.checkpoint_directory() == tmp_path / "environment"
    assert depth_pro2.checkpoint_directory(tmp_path / "explicit") == tmp_path / "explicit"


@pytest.mark.parametrize("focal", [float("nan"), float("inf"), 0.0, -1.0])
def test_fov_rejects_invalid_focal_without_metrictrail_dependency(providers, focal):
    model, build, _ = providers
    model.fov.focal_length_px = lambda rgb: torch.tensor(focal)
    with pytest.raises(ValueError, match="invalid focal"):
        build().estimate_free_fov_k4(torch.full((1, 3, 2, 3), 0.5))


def test_stateless_calibration_is_independent_fp32(providers):
    model, build, _ = providers
    provider = build()
    assert torch.equal(provider.estimate(source(2)).metric_depth, torch.full((2, 2, 3), 2.5))
    assert len(model.calls) == 2 and not model.streams
    assert all(f == 12.5 and dtype == torch.float32 for _, f, dtype in model.calls)
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(model.calls[0][0][:, 0, 0], (0.5 - mean) / std)
    assert provider.depth_type == DepthType.MODEL_METRIC_DEPTH


def test_cache_shares_weights_not_causal_history(providers):
    model, build, loads = providers
    first, second, calibration = build(True), build(True), build()
    assert first.network is second.network is calibration.network is model
    assert loads == ["depthpro2-l"]
    assert torch.equal(first.estimate(source(2)).metric_depth[:, 0, 0], torch.tensor([1.0, 2.0]))
    assert second.estimate(source()).metric_depth[0, 0] == 1
    assert first._stream is not second._stream
    assert first._stream.settings == {"mode": "unbuffered", "window": 16, "f_px": 12.5}
    assert all(call[2] == torch.float16 for call in model.calls)
    first.reset_stream()
    assert first.frames_seen == 0 and first._stream is None
    assert first.estimate(source()).metric_depth[0, 0] == 1
    assert second.estimate(source()).metric_depth[0, 0] == 2


def test_fov_is_fp32_and_does_not_consume_history(providers):
    model, build, _ = providers
    provider = build(True)
    provider.estimate(source())
    stream = provider._stream
    with torch.autocast("cpu", dtype=torch.bfloat16):
        k4 = provider.estimate_free_fov_k4(torch.full((1, 3, 2, 3), 0.5))
    np.testing.assert_array_equal(k4, [12.5, 12.5, 1.5, 1.0])
    assert model.fov_calls == [(torch.float32, torch.float32, torch.float32)]
    assert provider.frames_seen == 1 and provider._stream is stream and len(model.calls) == 1
    assert model.precision == model.fov.precision == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_stream_rebinding_requires_reset_and_failure_clears_history(providers):
    model, build, _ = providers
    provider = build(True)
    provider.estimate(source())
    with pytest.raises(ValueError, match="reset_stream"):
        provider.estimate(source(focal=13.0))
    changed_size = source()
    changed_size.rgb = torch.full((3, 3, 3), 0.5)
    with pytest.raises(ValueError, match="reset_stream"):
        provider.estimate(changed_size)
    provider.reset_stream()
    provider.estimate(source(focal=13.0))
    model.fail = True
    with pytest.raises(RuntimeError, match="stream failure"):
        provider.estimate(source(focal=13.0))
    assert provider._stream is None and provider.frames_seen == 0
    assert model.precision == model.fov.precision == torch.bfloat16


def test_metric_depth_is_not_rescaled_and_invalid_pixels_are_zero(providers):
    model, build, _ = providers
    model.output = {"depth": torch.tensor([[0.2, 500.0, 0.1], [501.0, float("nan"), float("inf")]])}
    torch.testing.assert_close(
        build().estimate(source()).metric_depth, torch.tensor([[0.2, 500.0, 0.0], [0.0, 0.0, 0.0]])
    )


@pytest.mark.parametrize(
    "output,error",
    [({}, RuntimeError), ({"depth": torch.zeros(2, 3)}, RuntimeError), ({"depth": torch.ones(1, 3)}, ValueError)],
)
def test_missing_invalid_or_wrong_shape_depth_fails(providers, output, error):
    model, build, _ = providers
    model.output = output
    with pytest.raises(error):
        build().estimate(source())


@pytest.mark.parametrize("name", ["depthpro2-l", "depthpro2-h"])
def test_existing_config_overrides_change_only_depth_provider_names(name):
    overrides = ["pipeline=metrictrail", "streams.base_path=clip.mp4"]
    with initialize_config_dir(version_base=None, config_dir=str(get_config_path())):
        baseline = OmegaConf.to_container(compose(config_name="default", overrides=overrides), resolve=True)
        actual = OmegaConf.to_container(
            compose(
                config_name="default",
                overrides=overrides
                + [
                    f"pipeline.init.calibration.depth_model={name}",
                    f"pipeline.slam.keyframe_depth={name}-stream",
                ],
            ),
            resolve=True,
        )
    baseline["pipeline"]["init"]["calibration"]["depth_model"] = name
    baseline["pipeline"]["slam"]["keyframe_depth"] = name + "-stream"
    assert actual == baseline


@pytest.mark.parametrize("name", ["depthpro2-l", "depthpro2-h"])
def test_native_registry_supports_stateless_focal_replay(providers, name):
    model, _, loads = providers
    provider = make_depth_model(name)
    for focal in (12.5, 14.0, 12.5):
        result = provider.estimate(source(focal=focal))
        assert result.metric_depth.shape == (2, 3)
    assert loads == [name]
    assert provider.depth_type == DepthType.MODEL_METRIC_DEPTH
    assert not model.streams and provider._stream is None
    assert [(focal, dtype) for _, focal, dtype in model.calls] == [
        (12.5, torch.float32),
        (14.0, torch.float32),
        (12.5, torch.float32),
    ]


@pytest.mark.parametrize("name", ["depthpro2-l-stream", "depthpro2-h-stream"])
def test_native_registry_rejects_causal_history_for_keyframe_replay(name):
    with pytest.raises(ValueError, match="DROID keyframe replay"):
        make_depth_model(name)


@pytest.mark.parametrize("name", ["depthpro2-l", "depthpro2-h"])
def test_native_config_preserves_dynamic_masks_with_matching_fusion(name):
    from vipe.config import parse_typed_config

    overrides = [
        "pipeline=default",
        "streams.base_path=clip.mp4",
        f"pipeline.slam.keyframe_depth={name}",
        "pipeline.output.save_viz=false",
    ]
    cfg = parse_typed_config("default", overrides)
    assert cfg.pipeline.slam.keyframe_depth == name
    assert cfg.pipeline.init.instance is not None
    assert cfg.pipeline.post.depth_align_model == f"adaptive_unidepth-l_{name}"


@pytest.mark.parametrize("name", ["depthpro2-l", "depthpro2-h"])
def test_native_raw_config_selects_matching_fusion_for_depth_pro2(name):
    with initialize_config_dir(version_base=None, config_dir=str(get_config_path())):
        cfg = compose(
            config_name="default",
            overrides=["pipeline=default", "streams.base_path=clip.mp4", f"pipeline.slam.keyframe_depth={name}"],
        )
        assert OmegaConf.to_container(cfg, resolve=True)["pipeline"]["post"]["depth_align_model"] == (
            f"adaptive_unidepth-l_{name}"
        )
        cfg.pipeline.slam.keyframe_depth = "unidepth-l"
        assert cfg.pipeline.post.depth_align_model == "adaptive_unidepth-l_svda"


@pytest.mark.parametrize("name", ["unidepth-l", "moge2-l"])
def test_native_fusion_default_is_unchanged_for_other_priors(name):
    from vipe.config import parse_typed_config

    cfg = parse_typed_config(
        "default", ["pipeline=default", "streams.base_path=clip.mp4", f"pipeline.slam.keyframe_depth={name}"]
    )
    assert cfg.pipeline.post.depth_align_model == "adaptive_unidepth-l_svda"


@pytest.mark.parametrize("pipeline", ["default", "metrictrail"])
@pytest.mark.parametrize("name", ["depthpro2-l", "depthpro2-h"])
@pytest.mark.parametrize(
    "model",
    [
        "adaptive_unidepth-l_svda",
        "adaptive_unidepth-l_vda",
        "adaptive_unidepth-l_depthpro2-l",
        "adaptive_unidepth-l_depthpro2-h",
        "null",
    ],
)
def test_explicit_fusion_overrides_win_for_depth_pro2(pipeline, name, model):
    from vipe.config import parse_typed_config

    provider = name + "-stream" if pipeline == "metrictrail" else name
    cfg = parse_typed_config(
        "default",
        [
            f"pipeline={pipeline}",
            "streams.base_path=clip.mp4",
            f"pipeline.slam.keyframe_depth={provider}",
            f"pipeline.post.depth_align_model={model}",
        ],
    )
    assert cfg.pipeline.post.depth_align_model == (None if model == "null" else model)


@pytest.mark.parametrize("name", ["depthpro2-l", "depthpro2-h"])
def test_metrictrail_default_still_disables_fusion(name):
    from vipe.config import parse_typed_config

    cfg = parse_typed_config(
        "default",
        ["pipeline=metrictrail", "streams.base_path=clip.mp4", f"pipeline.slam.keyframe_depth={name}-stream"],
    )
    assert cfg.pipeline.post.depth_align_model is None


@pytest.mark.parametrize(
    "pipeline,model",
    [("no_vda", "adaptive_unidepth-l"), ("static_vda", "adaptive_unidepth-l_vda"), ("wide_angle", None)],
)
def test_explicit_pipeline_post_presets_are_preserved(pipeline, model):
    from vipe.config import parse_typed_config

    cfg = parse_typed_config(
        "default", [f"pipeline={pipeline}", "streams.base_path=clip.mp4", "pipeline.slam.keyframe_depth=depthpro2-l"]
    )
    assert cfg.pipeline.post.depth_align_model == model
