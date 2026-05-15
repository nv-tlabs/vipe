# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
from pydantic.fields import FieldInfo

from vipe.config import parse_typed_config
from vipe.config.base_schema import BaseConfigSchema
from vipe.config.pipeline import (
    DefaultInitConfig,
    DefaultPipelineConfig,
    InstanceInitConfig,
    OutputConfig,
    PanoramaInitConfig,
    PanoramaPipelineConfig,
    PostConfig,
    VirtualCameraConfig,
)
from vipe.config.slam import BAConfig, SLAMConfig, SparseTracksConfig
from vipe.config.streams import BaseStreamListConfig, FrameDirStreamListConfig, RawMP4StreamListConfig
from vipe.config.vipe import ViPEConfig

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "reference" / "configuration.md"
CONFIG_ROOT = REPO_ROOT / "configs"

MODEL_SECTIONS: list[tuple[str, list[type[BaseConfigSchema]]]] = [
    ("Top-Level Config", [ViPEConfig]),
    ("Input Streams", [BaseStreamListConfig, RawMP4StreamListConfig, FrameDirStreamListConfig]),
    (
        "Pipelines",
        [
            InstanceInitConfig,
            DefaultInitConfig,
            PanoramaInitConfig,
            VirtualCameraConfig,
            PostConfig,
            OutputConfig,
            DefaultPipelineConfig,
            PanoramaPipelineConfig,
        ],
    ),
    ("SLAM", [SLAMConfig, BAConfig, SparseTracksConfig]),
]

PIPELINE_PURPOSES = {
    "default": "Default pipeline for pinhole videos.",
    "dav3": "Default pipeline using Depth Anything 3 for keyframe and multiview depth.",
    "lyra": "Configuration used for Lyra-style results, with MoGe keyframe depth and VDA alignment.",
    "no_vda": "Default pipeline without Video Depth Anything alignment.",
    "static_vda": "Default pipeline without instance segmentation, using static VDA alignment.",
    "wide_angle": "Default pipeline configured for wide-angle or fisheye input.",
    "panorama": "Panorama pipeline that projects 360-degree frames into virtual perspective views.",
}
PIPELINE_PRESET_ORDER = ["default", "dav3", "lyra", "no_vda", "static_vda", "wide_angle", "panorama"]


def _escape_table_cell(value: str) -> str:
    return value.replace("\n", "<br>").replace("|", "\\|")


def _format_code(value: Any) -> str:
    if value is None:
        return "`null`"
    if value is True:
        return "`true`"
    if value is False:
        return "`false`"
    if isinstance(value, str):
        return f"`{value}`"
    try:
        return f"`{json.dumps(value, sort_keys=True)}`"
    except TypeError:
        return f"`{value!r}`"


def _format_literal(args: tuple[Any, ...]) -> str:
    return " | ".join(_format_code(arg) for arg in args)


def _format_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        return _format_type(args[0])
    if origin is Literal:
        return _format_literal(args)
    if origin in (list, tuple):
        inner = ", ".join(_format_type(arg) for arg in args) if args else "Any"
        return f"{origin.__name__}[{inner}]"
    if origin in (types.UnionType, Union):
        return " | ".join(_format_type(arg) for arg in args)
    if annotation is None or annotation is type(None):
        return "null"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _constraint_parts(property_schema: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    constraints = [
        ("minimum", ">="),
        ("exclusiveMinimum", ">"),
        ("maximum", "<="),
        ("exclusiveMaximum", "<"),
        ("minLength", "min length"),
        ("maxLength", "max length"),
        ("minItems", "min items"),
        ("maxItems", "max items"),
    ]
    for key, label in constraints:
        if key in property_schema:
            parts.append(f"{label} {property_schema[key]}")
    if "const" in property_schema:
        parts.append(f"fixed {_format_code(property_schema['const'])}")
    if "enum" in property_schema:
        parts.append(f"choices {_format_literal(tuple(property_schema['enum']))}")
    return parts


def _field_default(field: FieldInfo) -> str:
    if field.is_required():
        return "required"
    return _format_code(field.default)


def _model_table(model: type[BaseConfigSchema]) -> str:
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    lines = [
        f"### {model.__name__}",
        "",
        (model.__doc__ or "").strip(),
        "",
        "| Field | Type | Default | Constraints | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for field_name, field in model.model_fields.items():
        property_schema = properties.get(field_name, {})
        constraints = ", ".join(_constraint_parts(property_schema))
        description = field.description or property_schema.get("description") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{field_name}`",
                    _escape_table_cell(_format_type(field.annotation)),
                    _field_default(field),
                    _escape_table_cell(constraints or "-"),
                    _escape_table_cell(description),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _load_config_yaml(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected mapping in {path}")
    return cast(dict[str, Any], loaded)


def _select(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _pipeline_preset_table() -> str:
    lines = [
        "## Pipeline Presets",
        "",
        "These are the pipeline values accepted by `pipeline=...` in Hydra overrides and by `vipe infer --pipeline`.",
        "",
        "| Preset | Purpose | Pipeline Class | Camera | Keyframe Depth | Depth Post-Processing |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    available = {path.stem: path for path in (CONFIG_ROOT / "pipeline").glob("*.yaml")}
    ordered_names = [name for name in PIPELINE_PRESET_ORDER if name in available]
    ordered_names.extend(sorted(set(available) - set(ordered_names)))
    for name in ordered_names:
        config = parse_typed_config(
            "default",
            [
                f"pipeline={name}",
                "streams.base_path=DOC_INPUT",
                "pipeline.output.path=DOC_OUTPUT",
            ],
        )
        pipeline = config.pipeline
        dumped = pipeline.model_dump()
        camera = _select(dumped, "init.camera_type") or "panorama"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    PIPELINE_PURPOSES.get(name, ""),
                    f"`{pipeline.instance.rsplit('.', 1)[-1]}`",
                    _format_code(camera),
                    _format_code(_select(dumped, "slam.keyframe_depth")),
                    _format_code(_select(dumped, "post.depth_align_model")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _stream_preset_table() -> str:
    lines = [
        "## Stream Presets",
        "",
        "Use `streams=raw_mp4_stream` for videos and `streams=frame_dir_stream` for directories of frames.",
        "",
        "| Preset | Implementation | `frame_start` | `frame_end` | `frame_skip` | `cached` |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for path in sorted((CONFIG_ROOT / "streams").glob("*.yaml")):
        data = _load_config_yaml(path)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{path.stem}`",
                    f"`{str(data['instance']).rsplit('.', 1)[-1]}`",
                    _format_code(data["frame_start"]),
                    _format_code(data["frame_end"]),
                    _format_code(data["frame_skip"]),
                    _format_code(data["cached"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render() -> str:
    lines = [
        "# Configuration Reference",
        "",
        "<!-- This file is generated by scripts/generate_config_docs.py. Do not edit it by hand. -->",
        "",
        "ViPE uses Hydra YAML presets for composition and Pydantic models for runtime validation. "
        "The tables below are generated from the Pydantic config models, so field descriptions, "
        "required values, and numeric constraints stay aligned with the code.",
        "",
        "Common override examples:",
        "",
        "```bash",
        "uv run vipe infer assets/examples/dog-example.mp4 --pipeline dav3",
        "uv run python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO.mp4",
        "uv run python run.py pipeline=default streams.base_path=YOUR_VIDEO.mp4 pipeline.post.depth_align_model=null",
        "```",
        "",
        _pipeline_preset_table(),
        "",
        _stream_preset_table(),
        "",
    ]
    for section_name, models in MODEL_SECTIONS:
        lines.extend([f"## {section_name}", ""])
        for model in models:
            lines.extend([_model_table(model), ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the ViPE configuration reference page.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if the generated file is out of date.")
    args = parser.parse_args()

    rendered = _render()
    output = args.output

    if args.check:
        if not output.exists():
            print(f"{output} does not exist. Run scripts/generate_config_docs.py.", file=sys.stderr)
            return 1
        current = output.read_text()
        if current != rendered:
            print(f"{output} is out of date. Run scripts/generate_config_docs.py.", file=sys.stderr)
            return 1
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
