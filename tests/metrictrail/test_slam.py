# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DPVOSLAMSystem with a scripted DPVO and depth model; transport, seeds, FP32 recovery, RGB rule, map export,
runtime record (CPU)."""

from __future__ import annotations

import gc
import sys
import types
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from dpvo_fakes import FakeDPVO, FakeOps, too_few_keyframes_diagnostics
from metrictrail_fakes import FakeStream

import vipe.metrictrail.slam as slam_module
from vipe import get_config_path
from vipe.metrictrail.config import resolve_dpvo_buffer_size
from vipe.metrictrail.depth_producer import TRACKING_NOT_INITIALIZED, TRACKING_OK
from vipe.metrictrail.slam import (
    FP32_RECOVERY_POLICY,
    NONFINITE_RECOVERY_POLICY,
    DPVOSLAMSystem,
    NetworkSlot,
    TrackingAttempt,
    attempt_valid,
    bgr_uint8,
    canonical_rgb_chw,
    canonical_rgb_chw_deferred,
    case_seed,
    complete_dino_fp32,
    evict_cached_model,
    fp32_recovery_dpvo_config,
    seed_case_name,
    slam_map_from_keyframe_points,
    track_with_nonfinite_recovery,
)
from vipe.metrictrail.transport import ReflectPaddedDPVOTransport, dpvo_transport_padding, transport_hw
from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from vipe.utils.model_cache import ModelCache

K4 = np.asarray([30.0, 30.0, 20.0, 12.0], dtype=np.float32)


class FakeCfg(SimpleNamespace):
    """The parts of DPVO's yacs config the tracker and the FP32 recovery touch."""

    def clone(self) -> "FakeCfg":
        return FakeCfg(**vars(self))

    def defrost(self) -> None:
        pass

    def freeze(self) -> None:
        pass


def fake_dino_network() -> SimpleNamespace:
    """A stand-in for the DINO-DPVO network: matching and context fusion on, DINOv3 in bfloat16 autocast."""

    patchify = SimpleNamespace(
        dino=SimpleNamespace(amp_dtype=torch.bfloat16),
        fuse_matching=True,
        fuse_context=True,
        match_fusion=object(),
        context_fusion=object(),
    )
    return SimpleNamespace(
        patchify=patchify,
        _metrictrail_dino_enabled=True,
        _metrictrail_dino_matching=True,
        _metrictrail_dino_context=True,
    )


class FakeMetricDepth(DepthEstimationModel):
    def __init__(self) -> None:
        self.calls: list[DepthEstimationInput] = []

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        self.calls.append(src)
        assert src.rgb is not None
        depth = 1.0 + src.rgb.mean(dim=-1)
        depth[0, 0] = 1000.0  # outside the 0.2-500 m validity range
        return DepthEstimationResult(metric_depth=depth)


@pytest.fixture
def system(monkeypatch: pytest.MonkeyPatch):
    """A DPVOSLAMSystem on CPU whose DPVO, network, DPVO config and map export are fakes."""

    built: list[SimpleNamespace] = []
    events: list[str] = []
    script = {
        "nan_attempts": 0,
        "initialized": True,
        "diagnostics": None,
        "global_ba": False,
        # build indexes of the passes whose DPVO runs out of CUDA memory on its third frame
        "oom_attempts": set(),
        # False: the fake keeps no reference to the network (to observe that the recovery frees it)
        "keep_networks": True,
    }

    def make_dpvo(cfg, network, ht, wd, viz=False):
        index = len(built)
        events.append(f"dpvo{index}")
        dpvo = FakeDPVO(
            initialized=script["initialized"],
            diagnostics=script["diagnostics"],
            nan_rows=(2,) if index < script["nan_attempts"] else (),
            fail_at=2 if index in script["oom_attempts"] else None,
            fail_with=torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.30 GiB (fake)"),
        )
        dino = getattr(getattr(network, "patchify", None), "dino", None)
        built.append(
            SimpleNamespace(
                dpvo=dpvo,
                hw=(ht, wd),
                viz=viz,
                seed=torch.initial_seed(),
                cfg=cfg,
                network=network if script["keep_networks"] else None,
                dino_dtype=getattr(dino, "amp_dtype", None),
                tf32=(bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32)),
            )
        )
        return dpvo

    fake_module = types.ModuleType("vipe.metrictrail.dpvo.dpvo")
    fake_module.DPVO = type(
        "DPVO", (), {"__new__": lambda cls, *a, **k: make_dpvo(*a, **k), "__call__": FakeDPVO.__call__}
    )
    monkeypatch.setitem(sys.modules, "vipe.metrictrail.dpvo.dpvo", fake_module)

    def fake_resolve(config, frame_count=None):
        size = resolve_dpvo_buffer_size(config.buffer_size, frame_count)
        script["resolved_frame_counts"].append(frame_count)
        cfg = FakeCfg(
            METRIC_DEPTH_PRIOR_WARMUP=12,
            METRIC_GLOBAL_BA=script["global_ba"],  # metrictrail_gba / _gba_refine: global BA with loop edges
            BUFFER_SIZE=size,
            MIXED_PRECISION=True,
            LOOP_CLOSURE=script["global_ba"],
            CLASSIC_LOOP_CLOSURE=False,
            METRIC_DEPTH=True,
        )
        return cfg, {"BUFFER_SIZE": size}

    script["resolved_frame_counts"] = []
    monkeypatch.setattr(slam_module, "resolve_dpvo_config", fake_resolve)
    monkeypatch.setattr(slam_module, "dpvo_slam_map", lambda dpvo: None)
    depth = FakeMetricDepth()
    preset = yaml.safe_load((get_config_path() / "slam" / "metrictrail_local.yaml").read_text())
    preset["keyframe_depth"] = "fake-metric"
    instance = DPVOSLAMSystem(torch.device("cpu"), preset)
    instance.ops = FakeOps()  # type: ignore[assignment]

    def depth_model():
        events.append("depth_model")
        return depth

    monkeypatch.setattr(instance, "depth_model", depth_model)
    # The cached (canonical) DINO-DPVO network, and the networks the FP32 recovery loads anew from the "checkpoint".
    canonical = fake_dino_network()
    fresh: list[SimpleNamespace] = []

    def load_fresh() -> SimpleNamespace:
        events.append("fresh_network")
        fresh.append(fake_dino_network())
        return fresh[-1]

    def dpvo_network():
        events.append("dpvo_network")
        return canonical, {"dpvo": {"sha256": "fake"}}

    def evict() -> bool:
        events.append("evict")
        return True

    monkeypatch.setattr(instance, "dpvo_network", dpvo_network)
    monkeypatch.setattr(instance, "fresh_dpvo_network", load_fresh)
    monkeypatch.setattr(instance, "evict_dpvo_network", evict)
    return SimpleNamespace(
        system=instance, built=built, script=script, depth=depth, canonical=canonical, fresh=fresh, events=events
    )


@pytest.mark.parametrize("depth_queue", [0, 4])
def test_run_tracks_every_frame_with_the_shared_k(system, depth_queue: int) -> None:
    system.system.config.depth_queue = depth_queue
    stream = FakeStream(10, hw=(24, 40))
    output = system.system.run(stream, intrinsics=K4)
    dpvo = system.built[0].dpvo
    assert [call["tstamp"] for call in dpvo.calls] == list(range(10))
    assert all(torch.equal(call["intrinsics"], torch.from_numpy(K4)) for call in dpvo.calls)
    # 24x40 is divisible by 4: no transport padding.
    assert system.built[0].hw == (24, 40) and system.built[0].viz is False
    first = dpvo.calls[0]
    assert first["metric_mask"][0, 0].item() is False and first["metric_depth"][0, 0].item() == 1.0
    assert first["metric_mask"][1:, 1:].all()
    assert first["image"].dtype == torch.uint8 and first["image"].shape == (3, 24, 40)
    assert output.c2w.shape == (10, 4, 4) and output.c2w.dtype == np.float64
    assert output.record["tracking_status"] == TRACKING_OK
    assert output.record["case_seed"] == case_seed("clip", 1234)
    # A finite trajectory: one pass, no recovery, and no seed-restart record any more.
    assert "seed_restart" not in output.record and len(system.built) == 1 and system.fresh == []
    assert output.record["nonfinite_recovery"]["recovery_triggered"] is False
    assert output.record["execution"]["depth_queue"] == depth_queue
    np.testing.assert_array_equal(output.source_frame_ids, np.arange(10))
    np.testing.assert_array_equal(output.intrinsics.cpu().numpy(), K4[None])
    assert system.depth.calls[0].camera_type.name == "PINHOLE"


def test_padded_raster_goes_through_the_transport(system) -> None:
    output = system.system.run(FakeStream(4, hw=(26, 42)), intrinsics=K4)
    assert system.built[0].hw == (28, 44)
    call = system.built[0].dpvo.calls[0]
    assert call["image"].shape == (3, 28, 44)
    assert call["intrinsics"].tolist() == [30.0, 30.0, 21.0, 13.0]
    assert not call["metric_mask"][0].any() and not call["metric_mask"][:, 0].any()
    assert output.record["dpvo_transport"] == {
        "source_hw": [26, 42],
        "padding_ltrb": [1, 1, 1, 1],
        "transport_hw": [28, 44],
    }


# ------------------------------------------------------------ non-finite recovery (default on)


@pytest.fixture
def recovery(system):
    """``system`` as every preset configures it: slam.nonfinite_recovery is on by default."""

    assert system.system.config.nonfinite_recovery is True
    return system


def test_recovery_is_on_in_every_preset() -> None:
    from vipe.metrictrail.config import MetricTrailSLAMConfig

    assert MetricTrailSLAMConfig.model_fields["nonfinite_recovery"].default is True
    for preset in ("local", "gba", "gba_refine"):
        raw = yaml.safe_load((get_config_path() / "slam" / f"metrictrail_{preset}.yaml").read_text())
        assert raw["nonfinite_recovery"] is True
        # Every preset accepts it, a global-BA preset and an unseeded run included (no validator refuses them).
        assert MetricTrailSLAMConfig.model_validate({**raw, "seed": None}).nonfinite_recovery is True


def test_nonfinite_then_finite_is_recovered_in_fp32_at_the_fixed_seeds(recovery, tmp_path) -> None:
    recovery.script["nan_attempts"] = 1  # the canonical pass diverges, the first FP32 attempt does not
    depth_zip = tmp_path / "depth.zip"
    output = recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4, depth_output=depth_zip)
    base = case_seed("clip", 1234)
    built = recovery.built
    # Canonical pass: mixed precision, the cached network, its DINOv3 bfloat16 autocast, at the case seed.
    # Recovery: one freshly loaded network, DPVO MIXED_PRECISION off, DINOv3 float32, TF32 off, seed base + 0.
    assert [entry.seed for entry in built] == [base, base]
    assert [entry.cfg.MIXED_PRECISION for entry in built] == [True, False]
    assert [entry.cfg.LOOP_CLOSURE for entry in built] == [False, False]
    assert built[0].network is recovery.canonical and len(recovery.fresh) == 1 and built[1].network is recovery.fresh[0]
    assert [entry.dino_dtype for entry in built] == [torch.bfloat16, torch.float32]
    assert built[1].tf32 == (False, False)
    assert recovery.fresh[0].patchify.dino.amp_dtype == torch.bfloat16  # restored after the attempt
    assert recovery.canonical.patchify.dino.amp_dtype == torch.bfloat16
    # The accepted FP32 trajectory, and only its depth, are kept.
    assert np.isfinite(output.c2w).all() and "seed_restart" not in output.record
    assert output.depth_artifact == depth_zip and depth_zip.is_file()
    assert output.record["metric_depth_artifact"]["frames"] == 6
    assert sorted(path.name for path in tmp_path.iterdir()) == ["depth.zip"]
    record = output.record["nonfinite_recovery"]
    assert record["enabled"] is True and record["status"] == "success"
    assert (record["policy"], record["recovery_policy"]) == (NONFINITE_RECOVERY_POLICY, FP32_RECOVERY_POLICY)
    assert (
        record["recovery_triggered"] is True and record["trigger"] == "nonfinite_canonical_mixed_precision_trajectory"
    )
    assert record["base_seed"] == base and record["predeclared_seed_candidates"] == [base + i for i in range(4)]
    assert record["selected_seed"] == base and record["selected_mode"] == "complete_fp32_recovery"
    assert record["selected_attempt_index"] == 0 and record["recovery_scope"] == "precision_only"
    assert record["fp32_dpvo_cfg"] == {"MIXED_PRECISION": False}
    assert record["preset_dpvo_cfg_kept"] == {
        "LOOP_CLOSURE": False,
        "CLASSIC_LOOP_CLOSURE": False,
        "METRIC_GLOBAL_BA": False,
    }
    canonical = record["canonical_attempt"]
    assert canonical["all_pose_rows_finite"] is False and canonical["nonfinite_pose_row_count"] == 1
    assert canonical["first_nonfinite_pose_row"] == 2 and canonical["dinov3_checkpoint_amp_dtype"] == "torch.bfloat16"
    assert canonical["canonical_mixed_precision"] is True and canonical["seed"] == base
    [attempt] = record["attempts"]
    assert attempt["seed"] == base and attempt["all_pose_rows_finite"] is True
    assert attempt["numeric_precision"]["dinov3_backbone_dtype"] == "float32"
    assert record["selected_numeric_precision"] == attempt["numeric_precision"]
    assert record["fresh_exact_checkpoint_reload_before_attempts"] is True


def test_recovery_tries_the_fixed_seeds_in_order(recovery) -> None:
    recovery.script["nan_attempts"] = 3  # canonical, FP32 seed + 0 and FP32 seed + 1 diverge
    output = recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    base = case_seed("clip", 1234)
    assert [entry.seed for entry in recovery.built] == [base, base, base + 1, base + 2]
    assert len(recovery.fresh) == 1 and all(entry.network is recovery.fresh[0] for entry in recovery.built[1:])
    record = output.record["nonfinite_recovery"]
    # Every pass is recorded: the canonical one and each FP32 attempt up to the kept one.
    assert record["canonical_attempt"]["all_pose_rows_finite"] is False
    assert [item["all_pose_rows_finite"] for item in record["attempts"]] == [False, False, True]
    assert [item["seed"] for item in record["attempts"]] == [base, base + 1, base + 2]
    assert record["selected_seed"] == base + 2 and record["selected_attempt_index"] == 2


def test_a_finite_canonical_pass_is_kept_without_recovery(recovery, tmp_path) -> None:
    depth_zip = tmp_path / "depth.zip"
    output = recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4, depth_output=depth_zip)
    # No NaN, nothing changes: one pass, the cached network in the preset's precision, at the case seed.
    assert len(recovery.built) == 1 and recovery.fresh == []
    [entry] = recovery.built
    assert entry.network is recovery.canonical and entry.cfg.MIXED_PRECISION is True
    assert entry.dino_dtype == torch.bfloat16 and entry.seed == case_seed("clip", 1234)
    assert output.depth_artifact == depth_zip and output.record["metric_depth_artifact"]["frames"] == 6
    record = output.record["nonfinite_recovery"]
    assert record["recovery_triggered"] is False and record["selected_mode"] == "canonical_mixed_precision"
    assert record["trigger"] is None and record["selected_numeric_precision"] is None
    assert record["attempts"] == [] and record["selected_seed"] == case_seed("clip", 1234)


def test_recovery_fails_when_every_fp32_attempt_diverges(recovery, tmp_path) -> None:
    recovery.script["nan_attempts"] = 5
    with pytest.raises(RuntimeError, match="every predeclared FP32 recovery attempt"):
        recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4, depth_output=tmp_path / "depth.zip")
    assert len(recovery.built) == 5  # the canonical pass and the four fixed seeds, nothing more
    assert list(tmp_path.iterdir()) == []  # no depth of a non-finite pass is kept


def test_gba_preset_keeps_loop_closure_and_global_ba_during_recovery(recovery, monkeypatch) -> None:
    recovery.script["global_ba"] = True  # metrictrail_gba / metrictrail_gba_refine
    monkeypatch.setattr(slam_module, "gba_record", lambda *args: {})  # the fake DPVO reports no global-BA diagnostics
    recovery.script["nan_attempts"] = 1
    output = recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    canonical, fp32 = (entry.cfg for entry in recovery.built)
    # Only the precision changes: the FP32 pass keeps the proximity loop edges and the metric global BA.
    assert (canonical.MIXED_PRECISION, fp32.MIXED_PRECISION) == (True, False)
    assert canonical.LOOP_CLOSURE is fp32.LOOP_CLOSURE is True
    assert canonical.METRIC_GLOBAL_BA is fp32.METRIC_GLOBAL_BA is True
    assert {key for key in vars(canonical) if getattr(canonical, key) != getattr(fp32, key)} == {"MIXED_PRECISION"}
    record = output.record["nonfinite_recovery"]
    assert record["status"] == "success" and record["fp32_dpvo_cfg"] == {"MIXED_PRECISION": False}
    assert record["preset_dpvo_cfg_kept"] == {
        "LOOP_CLOSURE": True,
        "CLASSIC_LOOP_CLOSURE": False,
        "METRIC_GLOBAL_BA": True,
    }


def test_unseeded_run_recovers_with_unseeded_fp32_attempts(recovery, monkeypatch: pytest.MonkeyPatch) -> None:
    seeded: list[int] = []
    monkeypatch.setattr(slam_module, "reset_random_state", seeded.append)
    recovery.system.config.seed = None  # an unseeded run, e.g. the LaMaria command
    recovery.script["nan_attempts"] = 2
    output = recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    assert seeded == [] and len(recovery.built) == 3 and len(recovery.fresh) == 1
    record = output.record["nonfinite_recovery"]
    assert record["seeded"] is False and record["base_seed"] is None and record["selected_seed"] is None
    assert record["predeclared_seed_candidates"] == [None] * 4 and record["selected_attempt_index"] == 1
    assert [item["seed"] for item in record["attempts"]] == [None, None]


def test_recovery_off_fails_on_a_nonfinite_trajectory(system) -> None:
    system.system.config.nonfinite_recovery = False
    output = system.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    assert output.record["nonfinite_recovery"] == {"enabled": False}
    system.script["nan_attempts"] = 2  # the second pass (the next DPVO built) diverges
    with pytest.raises(ValueError, match="non-finite pose rows .*slam.nonfinite_recovery is off"):
        system.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    assert len(system.built) == 2 and system.fresh == []  # one pass each, no restart


class WeakDinoNetwork:
    """``fake_dino_network`` as a weak-referenceable object (to observe when the recovery frees it)."""

    def __init__(self) -> None:
        self.__dict__.update(vars(fake_dino_network()))


def test_recovery_releases_the_cached_network_before_loading_a_fresh_one(recovery) -> None:
    recovery.script["nan_attempts"] = 1
    output = recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    # The canonical network leaves the model cache before the fresh one is loaded: never two on the GPU.
    assert recovery.events == ["depth_model", "dpvo_network", "dpvo0", "evict", "fresh_network", "dpvo1"]
    released = output.record["nonfinite_recovery"]["canonical_network_released"]
    # The fixture's stand-in cannot be weak-referenced (and the fixture holds it): its freedom is not observed.
    assert released == {"evicted_from_model_cache": True, "freed": None}
    assert output.record["nonfinite_recovery"]["fresh_network_load_error"] is None


def test_recovery_frees_the_canonical_network_of_a_real_model_cache(recovery, monkeypatch) -> None:
    """With nothing else holding it, the cached network is evicted and collected before the fresh load."""

    recovery.script["nan_attempts"] = 1
    recovery.script["keep_networks"] = False  # the fake DPVO keeps no reference to its network
    system = recovery.system
    system.model_cache = ModelCache()
    key = "metrictrail/dino_dpvo/fake"
    monkeypatch.setattr(system, "dpvo_network_cache_key", lambda: key)
    monkeypatch.delattr(system, "evict_dpvo_network")  # the real eviction (DPVOSLAMSystem.evict_dpvo_network)
    cached: list[weakref.ref] = []

    def dpvo_network():
        network = system.model_cache.get(key, WeakDinoNetwork)
        cached.append(weakref.ref(network))
        return network, {"dpvo": {"sha256": "fake"}}

    seen: dict[str, bool] = {}

    def load_fresh() -> WeakDinoNetwork:
        gc.collect()
        seen["canonical_alive"] = cached[0]() is not None
        seen["still_cached"] = key in system.model_cache._models
        return WeakDinoNetwork()

    monkeypatch.setattr(system, "dpvo_network", dpvo_network)
    monkeypatch.setattr(system, "fresh_dpvo_network", load_fresh)
    output = system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    assert seen == {"canonical_alive": False, "still_cached": False}
    released = output.record["nonfinite_recovery"]["canonical_network_released"]
    assert released == {"evicted_from_model_cache": True, "freed": True}
    # The next sequence (finite) loads the network into the cache again.
    system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    assert key in system.model_cache._models and len(cached) == 2 and cached[1]() is not None


def test_evict_cached_model_removes_one_key() -> None:
    cache = ModelCache()
    cache.get("a", lambda: 1)
    cache.get("b", lambda: 2)
    assert evict_cached_model(cache, "a") is True and evict_cached_model(cache, "a") is False
    assert cache.get("b", lambda: 3) == 2 and cache.get("a", lambda: 4) == 4


def test_network_slot_refuses_use_after_release() -> None:
    slot = NetworkSlot(WeakDinoNetwork(), evict=lambda: False)
    assert slot.get() is not None
    assert slot.release() == {"evicted_from_model_cache": False, "freed": True}
    with pytest.raises(RuntimeError, match="released"):
        slot.get()


def test_out_of_memory_recovery_attempt_is_recorded_and_the_next_seed_tried(recovery, tmp_path) -> None:
    recovery.script["nan_attempts"] = 1
    recovery.script["oom_attempts"] = {1}  # the first FP32 attempt runs out of CUDA memory (on its third frame)
    depth_zip = tmp_path / "depth.zip"
    output = recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4, depth_output=depth_zip)
    base = case_seed("clip", 1234)
    assert [entry.seed for entry in recovery.built] == [base, base, base + 1] and len(recovery.fresh) == 1
    record = output.record["nonfinite_recovery"]
    oom, kept = record["attempts"]
    assert (oom["status"], oom["seed"], oom["all_pose_rows_finite"]) == ("cuda_out_of_memory", base, False)
    assert "out of memory" in oom["error"] and oom["max_memory_allocated_gib"] is None  # no CUDA here
    assert oom["numeric_precision"]["dinov3_backbone_dtype"] == "float32"
    assert (kept["status"], kept["seed"], kept["all_pose_rows_finite"]) == ("finite", base + 1, True)
    assert record["status"] == "success" and record["out_of_memory_attempts"] == 1
    assert record["selected_seed"] == base + 1 and record["selected_attempt_index"] == 1
    # The failed attempt's partial depth is gone; the kept attempt's depth is published.
    assert output.depth_artifact == depth_zip and sorted(path.name for path in tmp_path.iterdir()) == ["depth.zip"]
    assert recovery.fresh[0].patchify.dino.amp_dtype == torch.bfloat16  # restored after the failed attempt too


def test_recovery_fails_closed_when_every_fp32_attempt_runs_out_of_memory(recovery, tmp_path) -> None:
    recovery.script["nan_attempts"] = 1
    recovery.script["oom_attempts"] = {1, 2, 3, 4}
    with pytest.raises(RuntimeError, match="non-finite or ran out of CUDA memory") as caught:
        recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4, depth_output=tmp_path / "depth.zip")
    assert len(recovery.built) == 5 and list(tmp_path.iterdir()) == []
    assert "'out_of_memory_attempts': 4" in str(caught.value) and "'status': 'failed_closed'" in str(caught.value)


def test_recovery_fails_closed_when_the_fresh_network_does_not_fit() -> None:
    nonfinite = TrackingAttempt(seed=3, tracked={}, valid=False, bad_rows=[1], source_ids=[0, 1])

    def attempt(cfg, network, seed):
        return nonfinite

    def fresh():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")

    cfg = SimpleNamespace(
        MIXED_PRECISION=True, METRIC_DEPTH=True, clone=lambda: FakeCfg(MIXED_PRECISION=True, METRIC_DEPTH=True)
    )
    with pytest.raises(RuntimeError, match="ran out of CUDA memory") as caught:
        track_with_nonfinite_recovery(
            attempt,
            cfg=cfg,
            canonical_network=NetworkSlot(WeakDinoNetwork()),
            base_seed=3,
            fresh_network=fresh,
            name="clip",
        )
    assert "'fresh_network_load_error': 'CUDA out of memory (fake)'" in str(caught.value)
    assert "'attempts': []" in str(caught.value)


def test_runtime_block_times_the_tracking_stage_after_model_load(recovery, monkeypatch) -> None:
    from vipe.metrictrail.runtime_record import TrackingRuntime

    class Recording(TrackingRuntime):
        def start(self) -> None:
            recovery.events.append("runtime.start")
            super().start()

    monkeypatch.setattr(slam_module, "TrackingRuntime", Recording)
    recovery.script["nan_attempts"] = 1
    output = recovery.system.run(FakeStream(6, hw=(24, 40)), intrinsics=K4)
    # The models are loaded before the peak statistics are reset and the clock starts; no warm-up pass.
    assert recovery.events == [
        "depth_model",
        "dpvo_network",
        "runtime.start",
        "dpvo0",
        "evict",
        "fresh_network",
        "dpvo1",
    ]
    runtime = output.record["runtime"]
    assert runtime["frames"] == 6 and runtime["tracking_passes"] == 2 and runtime["warmup"] is False
    assert runtime["tracking_fps"] == pytest.approx(6 / runtime["tracking_wall_seconds"])
    # The recovery's network load is left out of the tracking wall clock.
    assert set(runtime["excluded_seconds"]) == {"fp32_recovery_network_load"}
    # CPU: no CUDA call, no memory figures.
    assert runtime["max_memory_allocated_gib"] is None and runtime["max_memory_reserved_gib"] is None
    assert runtime["attention_backend"]["moge2_dinov2"] in ("xformers", "torch_sdpa")
    assert runtime["attention_backend"]["dinov3_dino_dpvo"] == "torch_sdpa"
    finite = recovery.system.run(FakeStream(4, hw=(24, 40)), intrinsics=K4).record["runtime"]
    assert finite["tracking_passes"] == 1 and finite["excluded_seconds"] == {} and finite["frames"] == 4


def test_fp32_recovery_config_and_precision_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FakeCfg(MIXED_PRECISION=True, LOOP_CLOSURE=True, CLASSIC_LOOP_CLOSURE=False, METRIC_DEPTH=True, OTHER=3)
    fp32 = fp32_recovery_dpvo_config(cfg)
    # Precision only: the loop edges of a global-BA preset stay on.
    assert (fp32.MIXED_PRECISION, fp32.LOOP_CLOSURE, fp32.CLASSIC_LOOP_CLOSURE, fp32.OTHER) == (False, True, False, 3)
    assert cfg.MIXED_PRECISION is True  # a copy
    with pytest.raises(ValueError, match="metric-depth"):
        fp32_recovery_dpvo_config(FakeCfg(METRIC_DEPTH=False))
    network = fake_dino_network()
    before = (bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32))
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with complete_dino_fp32(network) as audit:
            assert network.patchify.dino.amp_dtype == torch.float32
            assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert audit["dpvo_autocast_mixed_precision"] is False and audit["checkpoint_parameters_modified"] is False
        assert network.patchify.dino.amp_dtype == torch.bfloat16 and torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before
    with pytest.raises(RuntimeError, match="DINO-DPVO"):
        with complete_dino_fp32(SimpleNamespace()):
            pass
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    with pytest.raises(RuntimeError, match="TF32"):
        with complete_dino_fp32(network):
            pass


@pytest.mark.parametrize("preset", ["local", "gba", "gba_refine"])
def test_fp32_recovery_config_on_dpvos_own_config(preset: str) -> None:
    pytest.importorskip("yacs")
    from vipe.metrictrail.config import MetricTrailSLAMConfig
    from vipe.metrictrail.slam import resolve_dpvo_config

    raw = yaml.safe_load((get_config_path() / "slam" / f"metrictrail_{preset}.yaml").read_text())
    cfg, _ = resolve_dpvo_config(MetricTrailSLAMConfig.model_validate(raw), frame_count=100)
    fp32 = fp32_recovery_dpvo_config(cfg)
    changed = {key for key in cfg.keys() if cfg[key] != fp32[key]}
    assert changed == {"MIXED_PRECISION"} and cfg.MIXED_PRECISION is True and fp32.is_frozen()
    # The global-BA presets keep their proximity loop edges; local has none.
    assert fp32.LOOP_CLOSURE is fp32.METRIC_GLOBAL_BA is (preset != "local")
    assert fp32.CLASSIC_LOOP_CLOSURE is False


def test_not_initialised_sequence_returns_a_labelled_trajectory(system) -> None:
    system.script["initialized"] = False
    system.script["diagnostics"] = too_few_keyframes_diagnostics(2)
    output = system.system.run(FakeStream(5, hw=(24, 40)), intrinsics=K4)
    assert output.tracking_status == TRACKING_NOT_INITIALIZED
    assert output.record["metric_prior_status"] == "not_applicable_too_few_keyframes"


def test_frames_carrying_another_k_are_refused(system) -> None:
    class WithK(FakeStream):
        def __iter__(self):
            for frame in super().__iter__():
                frame.intrinsics = torch.tensor([1.0, 1.0, 1.0, 1.0])
                yield frame

    with pytest.raises(ValueError, match="not the shared"):
        system.system.run(WithK(3, hw=(24, 40)), intrinsics=K4)


def test_transport_padding_table() -> None:
    assert dpvo_transport_padding((280, 504)) == (0, 0, 0, 0)
    assert dpvo_transport_padding((288, 512)) == (0, 0, 0, 0)
    assert dpvo_transport_padding((388, 518)) == (1, 1, 0, 0)
    assert dpvo_transport_padding((434, 574)) == (1, 1, 1, 1)
    assert dpvo_transport_padding((378, 518)) == (1, 1, 1, 1)
    assert transport_hw((434, 574), (1, 1, 1, 1)) == (436, 576)


def test_transport_shifts_k_and_masks_the_border() -> None:
    received = {}

    def slam(tstamp, image, intrinsics, metric_depth, metric_mask):
        received.update(image=image, intrinsics=intrinsics, depth=metric_depth, mask=metric_mask)

    transport = ReflectPaddedDPVOTransport(slam, (434, 574), (1, 1, 1, 1))
    image = torch.rand(3, 434, 574)
    transport(
        0,
        image,
        torch.tensor([480.0, 480.0, 287.0, 217.0]),
        metric_depth=torch.rand(434, 574) + 1,
        metric_mask=torch.ones(434, 574, dtype=torch.bool),
    )
    assert received["intrinsics"].tolist() == [480.0, 480.0, 288.0, 218.0]
    assert received["image"].shape == (3, 436, 576) and torch.equal(received["image"][:, 1:-1, 1:-1], image)
    assert not received["mask"][[0, -1]].any() and not received["mask"][:, [0, -1]].any()
    assert received["mask"][1:-1, 1:-1].all()


def test_case_seed_rule() -> None:
    assert case_seed("02", 1234) == 1236
    import hashlib

    assert case_seed("desk", 1234) == 1234 + int.from_bytes(hashlib.sha256(b"desk").digest()[:4], "little")
    # The bundled demo (819 frames) is seeded as case "demo".
    assert case_seed("demo", 1234) == 1817287676 != case_seed("metrictrail_demo", 1234)


def test_seed_name_is_the_case_name(system) -> None:
    # A case is named after the file stem with spaces replaced by "_".
    assert seed_case_name("my clip") == "my_clip" and seed_case_name("02") == "02"
    assert seed_case_name("metrictrail_demo", "demo") == "demo"
    output = system.system.run(FakeStream(4, hw=(24, 40)), intrinsics=K4, name="my clip")
    assert (output.record["seed_name"], output.record["case_seed"]) == ("my_clip", case_seed("my_clip", 1234))
    system.system.config.seed_name = "demo"
    output = system.system.run(FakeStream(4, hw=(24, 40)), intrinsics=K4, name="metrictrail_demo")
    assert (output.record["seed_name"], output.record["case_seed"]) == ("demo", 1817287676)


def test_host_frames_have_their_8bit_check_on_the_tracking_device() -> None:
    # A frame on the host, tracked on another device ("meta" stands in for CUDA): the deferred check
    # must live on the tracking device, where _prepare stacks it with the depth and K checks.
    rgb = torch.arange(24, dtype=torch.uint8).reshape(2, 4, 3).float() / 255.0
    out, not_8bit = canonical_rgb_chw_deferred(
        rgb, normalization="metrictrail", device=torch.device("meta"), pinned=False
    )
    assert out.device.type == "meta" and not_8bit is not None and not_8bit.device.type == "meta"
    _, on_host = canonical_rgb_chw_deferred(rgb, normalization="metrictrail", device=torch.device("cpu"), pinned=False)
    assert on_host is not None and int(on_host) == 0


def test_saved_depth_is_built_before_dpvo_may_write_the_shared_mask() -> None:
    """Without dynamic masks the provider mask is the tensor DPVO receives; DPVO ANDs into it once ``ready``
    is recorded. The saved depth map must already be built by then (the depth writer reads it afterwards)."""

    from dpvo_fakes import FakeOps as BaseOps

    from vipe.metrictrail.depth import MetricDepthPrediction, MetricTrailMoGe2

    hw = (24, 40)
    valid = torch.ones(hw, dtype=torch.bool)
    valid[0, 0] = False

    class SharedMaskMoGe(MetricTrailMoGe2):
        def __init__(self) -> None:  # no weights
            pass

        def predict_metric(self, rgb_bchw: torch.Tensor, *, focal_px: float) -> MetricDepthPrediction:
            return MetricDepthPrediction(depth=torch.full((1, *hw), 2.0), valid=valid[None], fov_x_degrees=None)

    class ReadyHookOps(BaseOps):
        """Clears the mask in place right after ``ready`` is recorded, as DPVO's in-place AND would on the GPU."""

        def __init__(self) -> None:
            super().__init__()
            self.created = 0

        def event(self):
            event = super().event()
            self.created += 1
            if self.created == 2:  # _prepare creates depth_start, then ready
                record = event.record

                def record_then_write(stream=None) -> None:
                    record(stream)
                    valid.zero_()

                event.record = record_then_write  # type: ignore[method-assign]
            return event

    class Writer:
        def __init__(self) -> None:
            self.maps: dict[int, torch.Tensor] = {}

        def submit(self, index: int, depth: torch.Tensor) -> None:
            self.maps[index] = depth.clone()

    preset = yaml.safe_load((get_config_path() / "slam" / "metrictrail_local.yaml").read_text())
    system = DPVOSLAMSystem(torch.device("cpu"), preset)
    system.ops = ReadyHookOps()  # type: ignore[assignment]
    writer = Writer()
    frame = next(iter(FakeStream(1, hw=hw)))
    prepared = system._prepare(
        0,
        frame,
        False,
        depth_model=SharedMaskMoGe(),
        k4=K4,
        k_device=torch.from_numpy(K4.copy()),
        image_hw=hw,
        source_ids=[],
        depth_writer=writer,  # type: ignore[arg-type]
    )
    assert prepared.metric_mask.data_ptr() == valid.data_ptr() and not prepared.metric_mask.any()  # DPVO wrote it
    saved = writer.maps[0]
    assert saved[0, 0].item() == 0.0 and bool((saved.flatten()[1:] == 2.0).all())  # the mask as the provider gave it


def test_canonical_rgb_equals_loader_values() -> None:
    u8 = torch.arange(256, dtype=torch.uint8).repeat(3, 1).T.reshape(16, 16, 3)
    vipe_rgb = u8.float() / 255.0  # how ViPE's streams normalise
    rgb = canonical_rgb_chw(vipe_rgb, normalization="metrictrail", device=torch.device("cpu"), pinned=False)
    reference = np.ascontiguousarray(u8.numpy().transpose(2, 0, 1), dtype=np.float32)
    reference *= 1.0 / 255.0  # reference MetricTrail's loaders
    assert np.array_equal(rgb.numpy(), reference)
    assert not np.array_equal(vipe_rgb.permute(2, 0, 1).numpy(), reference)
    assert torch.equal(bgr_uint8(rgb), u8.permute(2, 0, 1)[[2, 1, 0]])
    with pytest.raises(ValueError, match="not 8-bit"):
        canonical_rgb_chw(torch.rand(4, 4, 3), normalization="metrictrail", device=torch.device("cpu"), pinned=False)
    stream_rgb = canonical_rgb_chw(vipe_rgb, normalization="stream", device=torch.device("cpu"), pinned=False)
    assert torch.equal(stream_rgb, vipe_rgb.permute(2, 0, 1))


def test_attempt_validity() -> None:
    poses = np.zeros((3, 7))
    poses[:, 6] = 1
    assert attempt_valid({"poses7_xyzw": poses, "timestamps": np.arange(3)}, count=3) == (True, [])
    poses[1, 0] = np.nan
    assert attempt_valid({"poses7_xyzw": poses, "timestamps": np.arange(3)}, count=3) == (False, [1])
    with pytest.raises(ValueError):
        attempt_valid({"poses7_xyzw": poses, "timestamps": np.arange(1, 4)}, count=3)


def test_patch_points_become_a_vipe_slam_map() -> None:
    xyz = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)
    rgb = torch.rand(2, 3, 3)
    valid = torch.tensor([[True, False, True], [True, True, True]])
    slam_map = slam_map_from_keyframe_points(xyz, rgb, valid, np.asarray([0, 7]))
    assert slam_map.dense_disp_frame_inds == [0, 7]
    assert slam_map.dense_disp_packinfo.tolist() == [[[0, 2]], [[2, 3]]]
    kf0_xyz, _ = slam_map.get_dense_disp_pcd(0)
    assert torch.equal(kf0_xyz, xyz[0][valid[0]])


def test_global_ba_record_labels_skips_only_without_global_ba() -> None:
    from vipe.metrictrail.slam import gba_record

    diagnostics = {
        "metric_global_ba": True,
        "global_ba_calls": 0,
        "loop_edges_added": 0,
        "global_ba_calls_with_metric_prior": 0,
        "normalize_called": False,
    }
    tracked = {"metric_prior_diagnostics": diagnostics, "tracking_status": TRACKING_NOT_INITIALIZED}
    block = gba_record({"METRIC_GLOBAL_BA": True}, tracked, "gba")["dpvo_metric_global_ba"]
    assert block["skipped_reason"] == TRACKING_NOT_INITIALIZED and block["tracking_status"] == TRACKING_NOT_INITIALIZED
    diagnostics["global_ba_calls"] = 2  # gba_refine's final passes ran inside terminate()
    assert gba_record({}, tracked, "gba_refine")["dpvo_metric_global_ba"]["skipped_reason"] is None
    with pytest.raises(RuntimeError, match="normalize"):
        gba_record({}, {"metric_prior_diagnostics": {**diagnostics, "normalize_called": True}}, "gba")
