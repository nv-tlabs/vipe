# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pipeline=metrictrail through ViPE's own config entry point, and MetricTrailPipeline.run with fakes (CPU)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from metrictrail_fakes import FakeStream
from omegaconf import DictConfig

import vipe.metrictrail.focal_refine.refine as refine_module
from vipe.config import DefaultPipelineConfig, parse_typed_config
from vipe.config.pipeline import PoseOnlyPipelineConfig
from vipe.metrictrail.config import MetricTrailPipelineConfig
from vipe.metrictrail.depth_producer import TRACKING_OK
from vipe.metrictrail.depth_writer import AsyncDepthWriter
from vipe.metrictrail.pipeline import MetricTrailPipeline, SharedIntrinsicsProcessor, record_path
from vipe.metrictrail.raster import plan_raster, raster_uint8
from vipe.metrictrail.slam import DPVOSLAMOutput
from vipe.metrictrail.streams import TruncatedSequenceError, tracking_stride_stream
from vipe.pipeline import make_pipeline
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils import io

PRESETS = {
    "metrictrail_local": ("local", False),
    "metrictrail_gba": ("gba", True),
    "metrictrail_gba_refine": ("gba_refine", True),
}


def _overrides(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "pipeline=metrictrail",
        f"streams.base_path={tmp_path / 'clip.mp4'}",
        f"pipeline.output.path={tmp_path / 'out'}",
        *extra,
    ]


@pytest.mark.parametrize("slam_preset", list(PRESETS))
@pytest.mark.parametrize("focal_refine", [False, True])
def test_presets_compose_and_build_the_pipeline(tmp_path: Path, slam_preset: str, focal_refine: bool) -> None:
    extra = [f"slam@pipeline.slam={slam_preset}", f"pipeline.focal_refine.enabled={str(focal_refine).lower()}"]
    config = parse_typed_config("default", _overrides(tmp_path, *extra))
    pipeline_config = config.pipeline
    assert isinstance(pipeline_config, MetricTrailPipelineConfig)
    name, global_ba = PRESETS[slam_preset]
    assert pipeline_config.slam.preset == name
    assert pipeline_config.slam.dpvo.METRIC_GLOBAL_BA is global_ba
    assert pipeline_config.focal_refine.enabled is focal_refine
    assert (pipeline_config.slam.depth_fp16, pipeline_config.slam.depth_queue) == (True, 4)
    assert pipeline_config.init.intrinsics == "sharedk" and pipeline_config.init.calibration.budget == "robust128"
    # Full sequence by default for this pipeline only.
    assert config.streams.frame_end == -1
    pipeline = make_pipeline(pipeline_config)
    assert isinstance(pipeline, MetricTrailPipeline)
    assert pipeline.config == pipeline_config
    assert pipeline.post_slam_hooks == []


def test_default_preset_is_local_and_defaults_match_the_reference(tmp_path: Path) -> None:
    config = parse_typed_config("default", _overrides(tmp_path)).pipeline
    assert isinstance(config, MetricTrailPipelineConfig)
    assert config.slam.preset == "local" and config.focal_refine.enabled is False
    assert config.focal_refine.seed == 7 and config.focal_refine.budget == 800
    assert config.output.save_artifacts is True
    # Defaults: FP16 depth, depth_queue 4, DPVO buffer auto, TF32 off.
    assert (config.slam.depth_fp16, config.slam.depth_queue, config.slam.buffer_size) == (True, 4, "auto")
    assert config.slam.tf32 is False
    # In-the-wild calibration policy.
    calibration = config.init.calibration
    assert (calibration.fallback, calibration.allow_unstable, calibration.fallback_to_initializer) == (
        "provider_focal",
        False,
        False,
    )
    assert calibration.terminal_min_accepted_keyframes == 16 and calibration.budget == "robust128"
    # In-the-wild raster: cover-fit to 280x504, one frame in k above 1500 frames.
    raster = config.init.input_raster
    assert (raster.policy, raster.height, raster.width, raster.max_frames) == ("cover_crop", 280, 504, 1500)
    assert set(type(raster).model_fields) == {"policy", "height", "width", "max_frames", "resampler"}
    assert raster.resampler == "bicubic_antialias"  # in-the-wild: antialiased resize
    # A non-finite trajectory is re-tracked in FP32 (precision only), by default.
    assert config.slam.nonfinite_recovery is True
    # MoGe-2 depth is saved by default, asynchronously.
    assert config.output.save_metric_depth == "async"


def test_raster_and_depth_options_and_benchmark_calibration_policies(tmp_path: Path) -> None:
    config = parse_typed_config(
        "default",
        _overrides(
            tmp_path,
            "pipeline.init.input_raster.policy=unchanged",
            "pipeline.output.save_metric_depth=off",
            "pipeline.slam.buffer_size=32768",
            "pipeline.init.calibration.fallback=none",
            "pipeline.init.calibration.allow_unstable=true",
            "pipeline.init.calibration.fallback_to_initializer=true",
        ),
    ).pipeline
    assert config.init.input_raster.policy == "unchanged" and config.output.save_metric_depth == "off"
    assert config.slam.buffer_size == 32768 and config.init.calibration.fallback == "none"
    assert config.init.tracking_stride == 1
    # The README's 7-Scenes options: the benchmark calibration policy (allow_unstable on top of the
    # default provider_focal), a plain resize to a raster that is not divisible by 4, and the tracking stride.
    seven = parse_typed_config(
        "default",
        _overrides(
            tmp_path,
            "pipeline.init.calibration.allow_unstable=true",
            "pipeline.init.input_raster.policy=resize",
            "pipeline.init.input_raster.height=434",
            "pipeline.init.input_raster.width=574",
            "pipeline.init.tracking_stride=5",
        ),
    ).pipeline.init
    assert (seven.calibration.fallback, seven.calibration.allow_unstable) == ("provider_focal", True)
    assert seven.input_raster.policy == "resize"
    assert (seven.input_raster.height, seven.input_raster.width, seven.tracking_stride) == (434, 574, 5)
    # The FP32 recovery of a non-finite trajectory is on for every preset, a global-BA one and an unseeded run
    # included (the Oxford command needs no flag); it can be turned off.
    for extra in (
        [],
        ["slam@pipeline.slam=metrictrail_gba"],
        ["slam@pipeline.slam=metrictrail_gba_refine"],
        ["pipeline.slam.seed=null"],
        ["pipeline.slam.dpvo.MIXED_PRECISION=false"],
    ):
        assert parse_typed_config("default", _overrides(tmp_path, *extra)).pipeline.slam.nonfinite_recovery is True
    off = parse_typed_config("default", _overrides(tmp_path, "pipeline.slam.nonfinite_recovery=false")).pipeline
    assert off.slam.nonfinite_recovery is False
    for bad in (
        ["pipeline.init.input_raster.width=3"],
        ["pipeline.init.input_raster.policy=stretch"],
        ["pipeline.init.input_raster.resampler=pil"],  # removed: ViPE's bilinear resize is the only one
        ["pipeline.init.tracking_stride=0"],
        ["pipeline.slam.buffer_size=0"],
        ["pipeline.slam.buffer_size=big"],
        ["pipeline.output.save_metric_depth=sync"],
        ["pipeline.init.calibration.fallback_to_initializer=true"],  # provider_focal replaces it
    ):
        with pytest.raises(Exception):
            parse_typed_config("default", _overrides(tmp_path, *bad))


def test_command_line_overrides_and_cli_switches(tmp_path: Path) -> None:
    config = parse_typed_config(
        "default",
        _overrides(
            tmp_path,
            "streams.frame_end=64",
            "pipeline.slam.depth_queue=0",
            "+pipeline.slam.dpvo.GBA_SCHEDULE_K=8",
            "slam@pipeline.slam=metrictrail_gba",
            # what `vipe infer -v` adds
            "pipeline.output.save_viz=true",
            "pipeline.slam.visualize=true",
        ),
    )
    assert config.streams.frame_end == 64
    assert config.pipeline.slam.depth_queue == 0 and config.pipeline.slam.dpvo.GBA_SCHEDULE_K == 8


def test_other_pipelines_keep_their_stream_defaults(tmp_path: Path) -> None:
    for name, kind in (("default", DefaultPipelineConfig), ("pose_only", PoseOnlyPipelineConfig)):
        config = parse_typed_config(
            "default", [f"pipeline={name}", f"streams.base_path={tmp_path}", f"pipeline.output.path={tmp_path}"]
        )
        assert isinstance(config.pipeline, kind) and config.streams.frame_end == 1000


@pytest.mark.parametrize(
    "overrides",
    [
        ["pipeline=default", "slam@pipeline.slam=metrictrail_gba"],
        ["pipeline=metrictrail", "slam@pipeline.slam=default"],
        ["pipeline=metrictrail", "pipeline.init.intrinsics=gt"],
    ],
)
def test_incompatible_combinations_are_refused(tmp_path: Path, overrides: list[str]) -> None:
    with pytest.raises(Exception):
        parse_typed_config("default", [f"streams.base_path={tmp_path}", *overrides])


# ---------------------------------------------------------------------- run()


def _frames(tmp_path: Path, count: int) -> Path:
    directory = tmp_path / "clip"
    directory.mkdir()
    for index in range(count):
        cv2.imwrite(str(directory / f"{index:06d}.png"), np.full((8, 12, 3), index, dtype=np.uint8))
    return directory


def _output(count: int, k4: np.ndarray, status: str = TRACKING_OK, shift: float = 0.0) -> DPVOSLAMOutput:
    c2w = np.tile(np.eye(4), (count, 1, 1))
    c2w[:, 0, 3] = np.arange(count) + shift
    return DPVOSLAMOutput(
        trajectory=None,  # type: ignore[arg-type]
        intrinsics=torch.from_numpy(np.asarray(k4, dtype=np.float32))[None],
        rig=None,
        slam_map=None,
        c2w=c2w,
        source_frame_ids=np.arange(count),
        record={"tracking_status": status, "runtime_seconds": 1.5, "dpvo_transport": {"padding_ltrb": [0, 0, 0, 0]}},
    )


class FakeSlam:
    """Tracks nothing; with ``depth_output`` it writes depth ``run number + 1`` m through the async writer."""

    def __init__(self) -> None:
        self.runs: list[np.ndarray] = []
        self.depth_outputs: list[Path | None] = []
        self.frames: list[list] = []

    def run(self, stream, *, intrinsics, name, depth_output=None):
        frames = list(stream)
        for frame in frames:
            assert torch.equal(frame.intrinsics, torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)))
        self.runs.append(np.asarray(intrinsics).copy())
        self.depth_outputs.append(depth_output)
        self.frames.append(frames)
        output = _output(len(frames), intrinsics, shift=float(len(self.runs) - 1))
        output.record["runtime"] = {"tracking_wall_seconds": float(len(self.runs)), "frames": len(frames)}
        if depth_output is not None:
            writer = AsyncDepthWriter(depth_output)
            for index, frame in enumerate(frames):
                writer.submit(index, torch.full(frame.rgb.shape[:2], float(len(self.runs))))
            output.record["metric_depth_artifact"] = {**writer.finish(expected_frames=len(frames)), "path": "x"}
            output.depth_artifact = Path(depth_output)
        return output


def _pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str, raster: str = "unchanged"
) -> tuple[MetricTrailPipeline, FakeSlam]:
    config = parse_typed_config(
        "default", _overrides(tmp_path, f"pipeline.init.input_raster.policy={raster}", *extra)
    ).pipeline
    pipeline = make_pipeline(config)
    assert isinstance(pipeline, MetricTrailPipeline)
    pipeline.device = torch.device("cpu")
    slam = FakeSlam()
    k4 = np.asarray([10.0, 10.0, 6.0, 4.0], dtype=np.float32)
    calibration = {
        "calibrated_K_fx_fy_cx_cy": k4.tolist(),
        "K_before_final_px": [10.0],
        "K_after_final24_px": [10.1],
        "selection_status": "stable_gt_free",
        "calibration_status": "stable_gt_free",
        "calibration_fallback": {"policy": "provider_focal", "applied": False, "reason": None},
    }
    monkeypatch.setattr(pipeline, "calibrate", lambda stream: SharedIntrinsicsProcessorWithRecord(k4, calibration))
    monkeypatch.setattr(pipeline, "make_slam", lambda: slam)
    return pipeline, slam


class SharedIntrinsicsProcessorWithRecord(SharedIntrinsicsProcessor):
    def __init__(self, k4: np.ndarray, record: dict) -> None:
        super().__init__(k4)
        self.record = record


def test_run_writes_vipe_artifacts_and_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch)
    pipeline.run(FakeStream(6, hw=(8, 12)))
    assert len(slam.runs) == 1
    artifact = io.ArtifactPath(tmp_path / "out", "clip")
    pose = np.load(artifact.pose_path)
    assert pose["data"].dtype == np.float64 and pose["data"].shape == (6, 4, 4)
    np.testing.assert_array_equal(pose["inds"], np.arange(6))
    assert float(pose["runtime"]) == 1.5 and pose["keyframe_ids"].shape == (0,)
    intrinsics = np.load(artifact.intrinsics_path)
    np.testing.assert_array_equal(intrinsics["data"], np.tile([10.0, 10.0, 6.0, 4.0], (6, 1)))
    assert artifact.camera_type_path.read_text().splitlines()[0] == "0: PINHOLE"
    assert pickle.loads(artifact.meta_info_path.read_bytes()) == {"ba_residual": 0.0}
    record = json.loads(record_path(artifact).read_text())
    assert record["sensor_K_or_GT_read"] is False and record["focal_refine"] == {"enabled": False}
    assert record["input_coverage"]["tail_truncated"] is False
    assert record["input_raster"]["policy"] == "unchanged" and record["input_raster"]["prepared_hw"] == [8, 12]
    assert record["calibration_status"] == "stable_gt_free" and record["calibration_fallback_reason"] is None
    assert record["nonfinite_recovery"] == {"enabled": False}
    # The tracking stage's runtime block (DPVOSLAMSystem's), labelled with its pass.
    assert record["runtime"] == {
        "tracking_wall_seconds": 1.0,
        "frames": 6,
        "tracking_pass": "stage_a",
        "stage_a": None,
    }
    # The metric depth of the tracking pass, in ViPE's depth artifact format.
    depth = list(io.read_depth_artifacts(artifact.depth_path))
    assert [index for index, _ in depth] == list(range(6)) and all(bool((map_ == 1.0).all()) for _, map_ in depth)
    assert record["metric_depth"]["saved"] is True and record["metric_depth"]["pass"] == "stage_a"
    assert record["metric_depth"]["artifact"] == "depth/clip.zip" and record["metric_depth"]["frames"] == 6
    assert not (tmp_path / "out" / "metrictrail" / "clip" / "depth_stage_a.zip").exists()  # committed, not copied
    # The tracking record names the published zip, not the staging file it was moved from.
    assert record["tracking"]["metric_depth_artifact"]["path"] == "depth/clip.zip"
    assert record["tracking"]["metric_depth_artifact"]["frames"] == 6
    # ViPE's artifact discovery finds the sequence.
    assert [path.artifact_name for path in io.ArtifactPath.glob_artifacts(tmp_path / "out")] == ["clip"]


def test_nonfinite_recovery_is_configured_and_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch)
    assert pipeline.slam_cfg.nonfinite_recovery is True  # the default; what make_slam hands DPVOSLAMSystem
    recovered = {"enabled": True, "recovery_triggered": True, "selected_seed": 7}
    tracked = slam.run

    def run(stream, **kwargs):
        output = tracked(stream, **kwargs)
        output.record["nonfinite_recovery"] = recovered
        return output

    slam.run = run  # type: ignore[method-assign]
    pipeline.run(FakeStream(4, hw=(8, 12)))
    record = json.loads(record_path(io.ArtifactPath(tmp_path / "out", "clip")).read_text())
    assert record["nonfinite_recovery"] == recovered and record["tracking"]["nonfinite_recovery"] == recovered
    assert record["pipeline"]["slam"]["nonfinite_recovery"] is True


def test_truncated_input_is_refused_before_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "calibrate", lambda stream: pytest.fail("calibrated a truncated input"))
    with pytest.raises(TruncatedSequenceError):
        pipeline.run(FrameDirStream(_frames(tmp_path, 6), seek_range=range(0, 4, 1)))
    assert slam.runs == []


@pytest.mark.parametrize("accepted", [False, True])
def test_focal_refinement_rebinds_and_retracks_only_when_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch, "pipeline.focal_refine.enabled=true")
    seen = {}

    def fake_refinement(config, **kwargs):
        seen.update(kwargs)
        frames = list(kwargs["frames"]())
        assert frames[1].dtype == np.float32 and frames[1].shape == (3, 8, 12)
        return {"enabled": True, "entered": True, "accepted": accepted, "focal": 12.5}

    monkeypatch.setattr(refine_module, "run_focal_refinement", fake_refinement)
    pipeline.run(FakeStream(5, hw=(8, 12)))
    assert seen["tracking_status"] == TRACKING_OK and seen["stage_a_k4"].tolist() == [10.0, 10.0, 6.0, 4.0]
    record = json.loads(record_path(io.ArtifactPath(tmp_path / "out", "clip")).read_text())
    if accepted:
        assert [run.tolist() for run in slam.runs] == [[10.0, 10.0, 6.0, 4.0], [12.5, 12.5, 6.0, 4.0]]
        assert record["bound_K_fx_fy_cx_cy"] == [12.5, 12.5, 6.0, 4.0] and record["stage_a_tracking"] is not None
        # Stage A's depth zip was deleted: its record names no file.
        assert record["stage_a_tracking"]["metric_depth_artifact"]["path"] is None
    else:
        assert len(slam.runs) == 1 and record["bound_K_fx_fy_cx_cy"] == [10.0, 10.0, 6.0, 4.0]
    pose = np.load(io.ArtifactPath(tmp_path / "out", "clip").pose_path)["data"]
    assert pose[0, 0, 3] == (1.0 if accepted else 0.0)
    # The saved depth is the final tracking pass's: the re-track's after an accepted stage B.
    staging = tmp_path / "out" / "metrictrail" / "clip"
    if accepted:
        assert slam.depth_outputs == [staging / "depth_stage_a.zip", staging / "depth_retrack.zip"]
    else:
        assert slam.depth_outputs == [staging / "depth_stage_a.zip"]
    depth = list(io.read_depth_artifacts(io.ArtifactPath(tmp_path / "out", "clip").depth_path))
    assert len(depth) == 5 and all(bool((map_ == (2.0 if accepted else 1.0)).all()) for _, map_ in depth)
    assert record["metric_depth"]["pass"] == ("focal_refine_retrack" if accepted else "stage_a")
    assert record["tracking"]["metric_depth_artifact"]["path"] == "depth/clip.zip"
    # The run record's runtime is the final tracking pass's; stage A's is kept beside a re-track's.
    runtime = record["runtime"]
    assert runtime["tracking_pass"] == ("focal_refine_retrack" if accepted else "stage_a")
    assert runtime["tracking_wall_seconds"] == (2.0 if accepted else 1.0) and runtime["frames"] == 5
    assert runtime["stage_a"] == ({"tracking_wall_seconds": 1.0, "frames": 5} if accepted else None)
    assert not list(staging.glob("depth_*.zip*"))  # the superseded stage-A depth is gone


def test_post_slam_hooks_run_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, _ = _pipeline(tmp_path, monkeypatch)
    calls = []

    def first(pipe, stream, final, record):
        calls.append("first")
        record["hook"] = "first"
        return None

    def second(pipe, stream, final, record):
        calls.append("second")
        return _output(len(final.c2w), final.intrinsics[0].numpy(), shift=100.0)

    pipeline.post_slam_hooks += [first, second]
    pipeline.run(FakeStream(3, hw=(8, 12)))
    assert calls == ["first", "second"]
    artifact = io.ArtifactPath(tmp_path / "out", "clip")
    assert np.load(artifact.pose_path)["data"][0, 0, 3] == 100.0
    assert json.loads(record_path(artifact).read_text())["hook"] == "first"


def test_payload_mode_returns_the_tracking_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch)
    pipeline.return_payload = True
    output = pipeline.run(FakeStream(3, hw=(8, 12)))
    assert isinstance(output.payload, DPVOSLAMOutput)
    assert not io.ArtifactPath(tmp_path / "out", "clip").pose_path.exists()
    assert slam.depth_outputs == [None]


def test_save_metric_depth_off_is_pose_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch, "pipeline.output.save_metric_depth=off")
    pipeline.run(FakeStream(4, hw=(8, 12)))
    artifact = io.ArtifactPath(tmp_path / "out", "clip")
    assert slam.depth_outputs == [None]
    assert artifact.pose_path.is_file() and not artifact.depth_path.exists()
    assert not (tmp_path / "out" / "depth").exists() and not (tmp_path / "out" / "metrictrail").exists()
    record = json.loads(record_path(artifact).read_text())
    assert record["metric_depth"] == {"mode": "off", "saved": False, "reason": "off"}


def test_failed_run_leaves_no_depth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch)

    def failing_hook(pipe, stream, final, record):
        raise RuntimeError("hook failed")

    pipeline.post_slam_hooks.append(failing_hook)
    with pytest.raises(RuntimeError, match="hook failed"):
        pipeline.run(FakeStream(4, hw=(8, 12)))
    assert slam.depth_outputs[0] is not None
    assert not list((tmp_path / "out").rglob("*.zip*"))


def test_default_raster_prepares_in_the_wild_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch, raster="cover_crop")
    stream = FakeStream(20, hw=(72, 128))
    pipeline.run(stream)
    frames = slam.frames[0]
    assert len(frames) == 20 and tuple(frames[0].rgb.shape) == (280, 504, 3)
    record = json.loads(record_path(io.ArtifactPath(tmp_path / "out", "clip")).read_text())
    raster = record["input_raster"]
    assert raster["policy"] == "cover_crop" and raster["interpolation"]["name"] == "torch_bicubic_antialias"
    assert raster["crop_left_top"] == [0, 2] and raster["resized_hw"] == [284, 504] and raster["frame_stride"] == 1
    assert raster["prepared_hw"] == [280, 504] and raster["frames"] == 20
    depth = list(io.read_depth_artifacts(io.ArtifactPath(tmp_path / "out", "clip").depth_path))
    assert tuple(depth[0][1].shape) == (280, 504)


def test_calibration_fallback_is_repeated_in_the_run_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, _ = _pipeline(tmp_path, monkeypatch)
    k4 = np.asarray([12.0, 12.0, 6.0, 4.0], dtype=np.float32)
    calibration = {
        "calibrated_K_fx_fy_cx_cy": k4.tolist(),
        "selection_status": "fallback_provider_focal",
        "calibration_status": "fallback_provider_focal",
        "calibration_fallback": {"policy": "provider_focal", "applied": True, "reason": "not_converged"},
    }
    monkeypatch.setattr(pipeline, "calibrate", lambda stream: SharedIntrinsicsProcessorWithRecord(k4, calibration))
    pipeline.run(FakeStream(3, hw=(8, 12)))
    record = json.loads(record_path(io.ArtifactPath(tmp_path / "out", "clip")).read_text())
    assert record["calibration_status"] == "fallback_provider_focal"
    assert record["calibration_fallback_reason"] == "not_converged"
    assert record["bound_K_fx_fy_cx_cy"] == k4.tolist()


def test_pipeline_takes_hydra_sections(tmp_path: Path) -> None:
    config = parse_typed_config("default", _overrides(tmp_path)).pipeline
    sections = config.to_dictconfig()
    assert isinstance(sections.init, DictConfig)
    del sections.instance
    pipeline = MetricTrailPipeline(**sections)
    assert pipeline.config.slam.preset == "local"
    _ = SimpleNamespace  # keep the import for readers of the fakes above


# ------------------------------------------------------------ tracking stride


class SourceIdSlam(FakeSlam):
    """FakeSlam whose output carries the tracked frames' raw_frame_idx, as DPVOSLAMSystem's does."""

    def run(self, stream, *, intrinsics, name, depth_output=None):
        output = super().run(stream, intrinsics=intrinsics, name=name, depth_output=depth_output)
        output.source_frame_ids = np.asarray([frame.raw_frame_idx for frame in self.frames[-1]], dtype=np.int64)
        return output


def _strided_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str, raster: str = "unchanged"
) -> tuple[MetricTrailPipeline, SourceIdSlam, list]:
    pipeline, _ = _pipeline(tmp_path, monkeypatch, "pipeline.init.tracking_stride=5", *extra, raster=raster)
    slam = SourceIdSlam()
    calibrated: list = []
    k4 = np.asarray([10.0, 10.0, 6.0, 4.0], dtype=np.float32)
    record = {"calibration_status": "stable_gt_free", "calibration_fallback": {"applied": False, "reason": None}}

    def calibrate(stream):
        calibrated.extend((frame.raw_frame_idx, frame.rgb.clone()) for frame in stream)
        return SharedIntrinsicsProcessorWithRecord(k4, record)

    monkeypatch.setattr(pipeline, "calibrate", calibrate)
    monkeypatch.setattr(pipeline, "make_slam", lambda: slam)
    return pipeline, slam, calibrated


def test_tracking_stride_calibrates_on_every_frame_and_tracks_every_kth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, slam, calibrated = _strided_pipeline(tmp_path, monkeypatch, "pipeline.focal_refine.enabled=true")
    refined = []

    def fake_refinement(config, **kwargs):
        refined.append(len(list(kwargs["frames"]())))
        return {"enabled": True, "entered": False, "accepted": False}

    monkeypatch.setattr(refine_module, "run_focal_refinement", fake_refinement)
    pipeline.run(FakeStream(23, hw=(8, 12), step=2))  # raw frame ids 0, 2, ..., 44
    # SharedK reads all 23 frames; DPVO and focal refinement see frames 0, 5, 10, 15, 20 (ordinals).
    assert [source_id for source_id, _ in calibrated] == [2 * index for index in range(23)]
    tracked = slam.frames[0]
    assert [frame.raw_frame_idx for frame in tracked] == [0, 10, 20, 30, 40] and refined == [5]
    for frame in tracked:
        assert torch.equal(frame.rgb, calibrated[frame.raw_frame_idx // 2][1])
    artifact = io.ArtifactPath(tmp_path / "out", "clip")
    pose = np.load(artifact.pose_path)
    np.testing.assert_array_equal(pose["source_frame_ids"], [0, 10, 20, 30, 40])
    np.testing.assert_array_equal(pose["inds"], np.arange(5))
    assert np.load(artifact.intrinsics_path)["data"].shape == (5, 4)
    assert len(list(io.read_depth_artifacts(artifact.depth_path))) == 5
    record = json.loads(record_path(artifact).read_text())
    assert record["tracking_stride"] == {
        "stride": 5,
        "calibration_frames": 23,
        "tracked_frames": 5,
        "source_frame_ids": [0, 10, 20, 30, 40],
    }


def test_seven_scenes_clip_is_resized_once_then_strided(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The README's 7-Scenes command on a stride-1 raw clip: 480x640 color frames (the depth and
    # pose files are not in the frame directory), read-time plain resize to 434x574, tracking stride 5.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)  # FrameDirStream calls .cuda()
    directory = tmp_path / "chess__seq-03"
    directory.mkdir()
    ramp = np.add.outer(np.arange(480), np.arange(640)).astype(np.float64)
    for index in range(17):
        image = np.stack([ramp / 4.0, ramp[::-1] / 5.0, np.full_like(ramp, 7.0 * index)], axis=-1)
        cv2.imwrite(str(directory / f"frame-{index:06d}.color.png"), (image % 256).astype(np.uint8))
    pipeline, slam, calibrated = _strided_pipeline(
        tmp_path,
        monkeypatch,
        "pipeline.init.input_raster.height=434",
        "pipeline.init.input_raster.width=574",
        raster="resize",
    )
    pipeline.run(FrameDirStream(directory))
    assert [source_id for source_id, _ in calibrated] == list(range(17))
    tracked = slam.frames[0]
    assert [frame.raw_frame_idx for frame in tracked] == [0, 5, 10, 15]
    # One read-time resize (the default bicubic with antialias, stretched to 434x574 without a crop, as for
    # 7-Scenes) feeds both: tracked frame 5 is calibration frame 5, pixel for pixel, so the
    # K SharedK calibrates is the K of the tracked frames.
    plan = plan_raster((480, 640), 17, 30.0, target_hw=(434, 574), policy="resize", resampler="bicubic_antialias")
    raw = cv2.cvtColor(cv2.imread(str(directory / "frame-000005.color.png")), cv2.COLOR_BGR2RGB)
    expected = torch.from_numpy(raster_uint8(raw, plan)).float() / 255.0
    assert torch.equal(tracked[1].rgb, calibrated[5][1]) and torch.equal(tracked[1].rgb, expected)
    for frame in tracked:
        assert torch.equal(frame.rgb, calibrated[frame.raw_frame_idx][1])
    record = json.loads(record_path(io.ArtifactPath(tmp_path / "out", "chess__seq-03")).read_text())
    raster = record["input_raster"]
    assert raster["policy"] == "resize" and raster["interpolation"]["name"] == "torch_bicubic_antialias"
    assert (raster["resized_hw"], raster["crop_left_top"], raster["prepared_hw"]) == ([434, 574], [0, 0], [434, 574])
    assert raster["frames"] == 17 and raster["frame_stride"] == 1
    assert record["tracking_stride"]["source_frame_ids"] == [0, 5, 10, 15]
    assert record["tracking_stride"]["calibration_frames"] == 17


def test_strided_stream_length_rate_and_ids() -> None:
    stream = FakeStream(23, hw=(8, 12), step=3)
    assert tracking_stride_stream(stream, 1) is stream
    strided = tracking_stride_stream(stream, 5)
    assert len(strided) == 5 and strided.fps() == 6.0 and strided.name() == "clip"
    assert [frame.raw_frame_idx for frame in strided] == [0, 15, 30, 45, 60]
    assert len(tracking_stride_stream(FakeStream(25, hw=(8, 12)), 5)) == 5
    with pytest.raises(ValueError, match="stride"):
        tracking_stride_stream(stream, 0)
