from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig
from pydantic import ValidationError

from vipe import get_config_path
from vipe.config import (
    DefaultPipelineConfig,
    FrameDirStreamListConfig,
    PanoramaPipelineConfig,
    RawMP4StreamListConfig,
    ViPEConfig,
    parse_typed_config,
)
from vipe.pipeline import make_pipeline
from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.streams.base import StreamList
from vipe.streams.frame_dir_stream import FrameDirStreamList
from vipe.streams.raw_mp4_stream import RawMP4StreamList


def _base_overrides(tmp_path: Path, pipeline: str = "default") -> list[str]:
    return [
        f"pipeline={pipeline}",
        f"streams.base_path={tmp_path / 'input.mp4'}",
        f"pipeline.output.path={tmp_path / 'out'}",
    ]


@pytest.mark.parametrize(
    ("pipeline", "pipeline_type"),
    [
        ("default", DefaultPipelineConfig),
        ("dav3", DefaultPipelineConfig),
        ("lyra", DefaultPipelineConfig),
        ("no_vda", DefaultPipelineConfig),
        ("static_vda", DefaultPipelineConfig),
        ("wide_angle", DefaultPipelineConfig),
        ("panorama", PanoramaPipelineConfig),
    ],
)
def test_parse_typed_config_pipeline_presets(tmp_path: Path, pipeline: str, pipeline_type: type) -> None:
    config = parse_typed_config("default", _base_overrides(tmp_path, pipeline=pipeline))

    assert isinstance(config, ViPEConfig)
    assert isinstance(config.pipeline, pipeline_type)
    assert isinstance(config.streams, RawMP4StreamListConfig)
    assert isinstance(config.pipeline.to_dictconfig(), DictConfig)


def test_parse_typed_config_frame_dir_stream(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    config = parse_typed_config(
        "default",
        [
            "pipeline=default",
            "streams=frame_dir_stream",
            f"streams.base_path={frame_dir}",
            f"pipeline.output.path={tmp_path / 'out'}",
        ],
    )

    assert isinstance(config.streams, FrameDirStreamListConfig)


def test_parse_typed_config_accepts_config_path_from_repo_root(tmp_path: Path) -> None:
    config = parse_typed_config("configs/default.yaml", _base_overrides(tmp_path))

    assert isinstance(config, ViPEConfig)
    assert isinstance(config.pipeline, DefaultPipelineConfig)


def test_config_source_tree_is_root_level_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "configs" / "default.yaml").is_file()
    assert not (repo_root / "vipe" / "configs").exists()
    assert get_config_path() == repo_root / "configs"


def test_typed_config_stream_factory_compatibility(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.touch()
    config = parse_typed_config("default", _base_overrides(tmp_path))

    stream_list = StreamList.make(config.streams)

    assert isinstance(stream_list, RawMP4StreamList)
    assert len(stream_list) == 1


def test_typed_config_frame_dir_factory_compatibility(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    config = parse_typed_config(
        "default",
        [
            "pipeline=default",
            "streams=frame_dir_stream",
            f"streams.base_path={frame_dir}",
            f"pipeline.output.path={tmp_path / 'out'}",
        ],
    )

    stream_list = StreamList.make(config.streams)

    assert isinstance(stream_list, FrameDirStreamList)
    assert len(stream_list) == 1


def test_typed_config_pipeline_factory_compatibility(tmp_path: Path) -> None:
    config = parse_typed_config("default", _base_overrides(tmp_path))

    pipeline = make_pipeline(config.pipeline)

    assert isinstance(pipeline, DefaultAnnotationPipeline)
    assert isinstance(pipeline.slam_cfg, DictConfig)
    assert Path(pipeline.out_cfg.path) == tmp_path / "out"


def test_parse_typed_config_accepts_fused_ba_override(tmp_path: Path) -> None:
    config = parse_typed_config(
        "default",
        [*_base_overrides(tmp_path), "pipeline.slam.ba.fused=true"],
    )

    assert isinstance(config.pipeline, DefaultPipelineConfig)
    assert config.pipeline.slam.ba.fused is True


@pytest.mark.parametrize(
    "override",
    [
        "streams.frame_skip=0",
        "pipeline.slam.sparse_tracks.name=bad",
        "pipeline.output.viz_downsample=0",
        "+pipeline.output.unexpected=1",
    ],
)
def test_parse_typed_config_rejects_invalid_values(tmp_path: Path, override: str) -> None:
    with pytest.raises(ValidationError):
        parse_typed_config("default", [*_base_overrides(tmp_path), override])
