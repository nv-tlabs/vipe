# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known (GT/sensor) intrinsics: the optional replacement for RGB-only SharedK.

This is the calibrated mode (``intrinsics_mode: calibrated_gt``). The K is given
in source-image pixels and mapped onto the input raster with one of:

- ``cover_crop`` (KITTI 280x504):
  ``s = max(W/w, H/h)``, the resized size rounded, an integer centre crop
  ``(left, top)``; ``K' = [fx s, fy s, cx s - left, cy s - top]``.
- ``cover_affine`` (LaMARia): the same ``s`` with the sub-pixel crop
  ``((w s - W) / 2, (h s - H) / 2)``.
- ``stretch`` (TUM 388x518): ``K' = [fx W/w, fy H/h, cx W/w, cy H/h]``, which is
  also what ViPE's ``VideoFrame.resize`` does to frame intrinsics (both in float32 arithmetic).

The result is cast to float32 by :func:`validate_k4` and is the one K both
the depth model (field of view from ``fx`` and the input width) and DPVO consume.
DPVO's reflect-pad transport (:mod:`vipe.metrictrail.transport`) then shifts the
principal point by the left/top padding at run time; the record states that
transport K too, but nothing here applies it to the frames.

:class:`KnownIntrinsicsProcessor` mirrors step 2's ``SharedIntrinsicsProcessor``:
it resolves the K at construction and writes it into every frame. The record
(``.record``) states ``intrinsics_source: known`` with the original and mapped K.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vipe.config.base_schema import config_to_primitive
from vipe.streams.base import FrameAttribute, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType

from ..calibration.processor import shared_device_k
from ..transport import dpvo_transport_padding, transport_hw
from .config import KnownIntrinsicsConfig, MetricTrailOptionalConfig, RasterMap

__all__ = [
    "INTRINSICS_SOURCE",
    "RECORD_SCHEMA",
    "KnownIntrinsicsConfig",
    "KnownIntrinsics",
    "KnownIntrinsicsProcessor",
    "ParsedIntrinsics",
    "cover_crop_geometry",
    "dpvo_transport_intrinsics",
    "map_intrinsics",
    "parse_intrinsics_file",
    "resolve_known_intrinsics",
    "validate_k4",
    "with_known_intrinsics",
]

INTRINSICS_SOURCE = "known"
RECORD_SCHEMA = "metrictrail_known_intrinsics_v1"
#: The record's ``intrinsics_mode`` for a known (calibrated) K.
CALIBRATED_MODE = "calibrated_gt"

# npz members holding K, in lookup order (the first one present is used).
NPZ_K_MEMBERS = ("intrinsics", "K", "input_intrinsics")
# Raster members (height, width) a K file may carry.
RASTER_KEYS = ("input_hw", "image_hw", "k_raster_hw", "hw")


def validate_k4(value: Any) -> np.ndarray:
    """Float32 ``[fx,fy,cx,cy]``, checked finite with positive focal."""

    result = np.asarray(value, dtype=np.float32)
    if result.shape != (4,) or not np.isfinite(result).all() or np.any(result[:2] <= 0):
        raise ValueError(f"K must be finite [fx,fy,cx,cy] with positive focal: {result}")
    return result


def _k4_from_any(value: Any, *, what: str) -> np.ndarray:
    """``[fx,fy,cx,cy]`` (float64) from a 4-vector, a 3x3 matrix or an (N,4) table (first row)."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, 3):
        return np.asarray([array[0, 0], array[1, 1], array[0, 2], array[1, 2]], dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 4:
        return np.asarray(array[0], dtype=np.float64)
    if array.shape == (4,):
        return array
    raise ValueError(f"{what}: expected [fx,fy,cx,cy] or a 3x3 K, got shape {array.shape}")


def _hw(value: Any, *, what: str) -> tuple[int, int]:
    items = [int(item) for item in np.asarray(value).reshape(-1)]
    if len(items) != 2 or min(items) <= 0:
        raise ValueError(f"{what}: expected (height, width), got {items}")
    return (items[0], items[1])


# ---------------------------------------------------------------- file parsing


@dataclass(frozen=True)
class ParsedIntrinsics:
    """A K read from a file: ``k4`` (float64) in the pixels of ``raster_hw`` (None: not stated)."""

    k4: np.ndarray
    raster_hw: tuple[int, int] | None
    field: str


def _parse_txt(path: Path) -> ParsedIntrinsics:
    numbers: list[float] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            numbers.extend(float(token) for token in re.split(r"[\s,;]+", line) if token)
    if len(numbers) == 4:
        return ParsedIntrinsics(np.asarray(numbers, dtype=np.float64), None, "fx_fy_cx_cy")
    if len(numbers) == 9:
        return ParsedIntrinsics(_k4_from_any(np.reshape(numbers, (3, 3)), what=str(path)), None, "K_3x3")
    raise ValueError(f"{path}: expected 4 numbers (fx fy cx cy) or 9 (a 3x3 K), found {len(numbers)}")


def _json_raster(payload: Mapping[str, Any], *, what: str) -> tuple[int, int] | None:
    for key in RASTER_KEYS:
        if key in payload:
            return _hw(payload[key], what=f"{what}:{key}")
    if "height" in payload and "width" in payload:
        return _hw([payload["height"], payload["width"]], what=what)
    return None


def _parse_json(path: Path) -> ParsedIntrinsics:
    payload = json.loads(path.read_text())
    what = str(path)
    if isinstance(payload, list):
        return ParsedIntrinsics(_k4_from_any(payload, what=what), None, "array")
    if not isinstance(payload, dict):
        raise ValueError(f"{what}: expected a JSON object or array")
    raster = _json_raster(payload, what=what)
    if all(key in payload for key in ("fx", "fy", "cx", "cy")):
        k4 = [payload[key] for key in ("fx", "fy", "cx", "cy")]
        return ParsedIntrinsics(np.asarray(k4, dtype=np.float64), raster, "fx_fy_cx_cy")
    for key in ("K_fx_fy_cx_cy", "intrinsics", "K"):
        if key in payload:
            return ParsedIntrinsics(_k4_from_any(payload[key], what=f"{what}:{key}"), raster, key)
    geometry = payload.get("geometry")
    if isinstance(geometry, dict) and "K" in geometry:  # prepared manifest (VBR)
        matrix = np.asarray(geometry["K"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"{what}: geometry.K must be 3x3")
        return ParsedIntrinsics(_k4_from_any(matrix, what=what), _json_raster(geometry, what=what), "geometry.K")
    cam0 = payload.get("cam0")
    if isinstance(cam0, dict) and "params" in cam0:  # pinhole calibration (LaMARia)
        params = np.asarray(cam0["params"], dtype=np.float64)
        if params.shape != (4,):
            raise ValueError(f"{what}: cam0.params must be [fx,fy,cx,cy]")
        resolution = cam0.get("resolution")
        if resolution is not None and not isinstance(resolution, dict):
            raise TypeError(f"{what}: cam0 resolution must be a width/height object")
        cam_raster = (int(resolution["height"]), int(resolution["width"])) if isinstance(resolution, dict) else raster
        return ParsedIntrinsics(params, cam_raster, "cam0.params")
    raise ValueError(f"{what}: no K found (keys fx/fy/cx/cy, K_fx_fy_cx_cy, intrinsics, K, geometry.K or cam0.params)")


def _parse_npz(path: Path) -> ParsedIntrinsics:
    with np.load(path, allow_pickle=False) as archive:
        for key in NPZ_K_MEMBERS:
            if key in archive.files:
                value = np.asarray(archive[key], dtype=np.float64)
                break
        else:
            raise KeyError(f"{path} has no K member (one of {NPZ_K_MEMBERS})")
        raster = None
        for raster_key in RASTER_KEYS:
            if raster_key in archive.files:
                raster = _hw(archive[raster_key], what=f"{path}:{raster_key}")
                break
    return ParsedIntrinsics(_k4_from_any(value, what=f"{path}:{key}"), raster, key)


def parse_intrinsics_file(path: str | Path) -> ParsedIntrinsics:
    """Read ``[fx,fy,cx,cy]`` (and its raster, when stated) from a .txt, .json or .npz file."""

    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _parse_npz(path)
    if suffix == ".json":
        return _parse_json(path)
    if suffix in (".txt", ".k", ""):
        return _parse_txt(path)
    raise ValueError(f"unsupported intrinsics file type {suffix!r}: use .txt, .json or .npz")


def _file_identity(path: Path) -> dict[str, Any]:
    """Resolved path, size and sha256 of a file."""

    resolved = path.expanduser().resolve()
    data = resolved.read_bytes()
    return {"path": str(resolved), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


# ------------------------------------------------------------ raster mapping


def cover_crop_geometry(source_hw: Sequence[int], target_hw: Sequence[int]) -> dict[str, Any]:
    """Cover-fit geometry: one scale, rounded size, integer centre crop."""

    raw_height, raw_width = (int(source_hw[0]), int(source_hw[1]))
    if raw_height <= 0 or raw_width <= 0:
        raise ValueError(f"raw image dimensions must be positive, got {tuple(source_hw)}")
    target_height, target_width = (int(target_hw[0]), int(target_hw[1]))
    scale = max(target_width / raw_width, target_height / raw_height)
    resized_width = int(round(raw_width * scale))
    resized_height = int(round(raw_height * scale))
    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    if resized_width < target_width or resized_height < target_height:
        raise AssertionError("cover-fit resize cannot be smaller than target crop")
    return {
        "raw_hw": (raw_height, raw_width),
        "scale": float(scale),
        "resized_wh": (resized_width, resized_height),
        "crop_left_top": (left, top),
    }


def map_intrinsics(
    k4: Any,
    source_hw: Sequence[int],
    target_hw: Sequence[int],
    raster_map: RasterMap | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map a source-pixel K onto ``target_hw``; returns float64 ``[fx,fy,cx,cy]`` and the geometry used."""

    k = np.asarray(k4, dtype=np.float64).reshape(4).copy()
    source = (int(source_hw[0]), int(source_hw[1]))
    target = (int(target_hw[0]), int(target_hw[1]))
    if source == target:
        return k, {"raster_map": "identity", "source_hw": list(source), "target_hw": list(target)}
    if raster_map is None:
        raise ValueError(
            f"known K raster {source} differs from the input raster {target}: set raster_map to cover_crop, "
            "cover_affine or stretch"
        )
    raw_h, raw_w = source
    out_h, out_w = target
    geometry: dict[str, Any] = {"raster_map": raster_map, "source_hw": list(source), "target_hw": list(target)}
    if raster_map == "cover_crop":
        cover = cover_crop_geometry(source, target)
        scale = cover["scale"]
        left, top = cover["crop_left_top"]
        k[:2] *= scale
        k[2] = k[2] * scale - left
        k[3] = k[3] * scale - top
        geometry.update(scale=scale, resized_wh=list(cover["resized_wh"]), crop_left_top=[float(left), float(top)])
    elif raster_map == "cover_affine":
        scale = max(out_w / raw_w, out_h / raw_h)
        crop_left = (raw_w * scale - out_w) / 2.0
        crop_top = (raw_h * scale - out_h) / 2.0
        k[:2] *= scale
        k[2] = k[2] * scale - crop_left
        k[3] = k[3] * scale - crop_top
        geometry.update(scale=float(scale), crop_left_top=[float(crop_left), float(crop_top)])
    elif raster_map == "stretch":
        # float32, exactly as ViPE's VideoFrame.resize scales frame intrinsics (and as
        # the prepared TUM 388x518 K was computed).
        scale_x = out_w / raw_w
        scale_y = out_h / raw_h
        k32 = torch.from_numpy(k.astype(np.float32))
        k32[0:4:2] *= scale_x
        k32[1:4:2] *= scale_y
        k = k32.numpy().astype(np.float64)
        geometry.update(scale_xy=[float(scale_x), float(scale_y)], arithmetic="float32")
    else:
        raise ValueError(f"unknown raster_map {raster_map!r}")
    return k, geometry


def dpvo_transport_intrinsics(k4: Any, input_hw: Sequence[int]) -> dict[str, Any]:
    """The K DPVO sees after step 2's reflect-pad transport (principal point + left/top padding)."""

    padding = dpvo_transport_padding(input_hw)
    shifted = np.asarray(k4, dtype=np.float32).reshape(4).copy()
    shifted[2] += float(padding[0])
    shifted[3] += float(padding[2])
    return {
        "source_hw": [int(input_hw[0]), int(input_hw[1])],
        "padding_ltrb": list(padding),
        "transport_hw": list(transport_hw(input_hw, padding)),
        "K_fx_fy_cx_cy": shifted.tolist(),
    }


# ------------------------------------------------------------------ resolver


@dataclass
class KnownIntrinsics:
    """The resolved K at the input raster (float32) and its run-record entry."""

    k4: np.ndarray
    record: dict[str, Any]


def _as_config(config: KnownIntrinsicsConfig | Mapping[str, Any]) -> KnownIntrinsicsConfig:
    if isinstance(config, KnownIntrinsicsConfig):
        return config
    return KnownIntrinsicsConfig.model_validate(config_to_primitive(config))


def _frame_k4(frame: VideoFrame) -> np.ndarray:
    if frame.intrinsics is None:
        raise ValueError(f"frame {frame.raw_frame_idx} carries no intrinsics")
    if frame.camera_type is not None and frame.camera_type != CameraType.PINHOLE:
        raise ValueError(f"frame {frame.raw_frame_idx} is {frame.camera_type}; MetricTrail tracks pinhole video")
    return frame.intrinsics.detach().to(device="cpu", dtype=torch.float64).numpy()[:4].copy()


def resolve_known_intrinsics(
    config: KnownIntrinsicsConfig | Mapping[str, Any],
    video_stream: VideoStream,
    *,
    prepared_from: tuple[Sequence[int], RasterMap] | None = None,
) -> KnownIntrinsics:
    """The known K mapped onto ``video_stream``'s raster, with its run-record entry.

    ``prepared_from`` = ``(source_hw, raster_map)``: the pipeline made ``video_stream``'s
    raster itself from a ``source_hw`` source (``pipeline.init.input_raster``:
    ``cover_crop``, or ``resize`` as ``stretch``). A K given without its own raster (no ``source_hw``, no raster
    member in the file) is then in that source's pixels, and a K whose raster is that
    source is mapped with the pipeline's own ``raster_map`` (another ``raster_map`` is
    refused). A K for a third raster (e.g. the sensor the video was cropped or scaled
    from) is mapped onto the video's raster with its ``raster_map`` and from there with
    the pipeline's. Without ``prepared_from`` the input raster is the source raster.
    """

    cfg = _as_config(config)
    input_hw = (int(video_stream.frame_size()[0]), int(video_stream.frame_size()[1]))
    # The raster a K without its own raster refers to: the input's, or the source the pipeline prepared it from.
    default_hw: tuple[int, int] = input_hw
    prepared_by: RasterMap | None = None
    if prepared_from is not None:
        default_hw = (int(prepared_from[0][0]), int(prepared_from[0][1]))
        prepared_by = prepared_from[1]
    stream_carries_k = FrameAttribute.INTRINSICS in video_stream.attributes()
    artifact: dict[str, Any] | None = None
    field: str
    if cfg.from_stream:
        if not stream_carries_k:
            raise ValueError(f"from_stream: stream {video_stream.name()!r} provides no intrinsics")
        origin = "stream"
        frames = iter(video_stream)
        try:
            first = next(frames)
        except StopIteration:
            raise ValueError(f"from_stream: stream {video_stream.name()!r} has no frames") from None
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                close()  # release the reader; the SLAM stage iterates the stream again
        original = _frame_k4(first)
        source_hw = input_hw
        field = "frame.intrinsics"
        default_hw, prepared_by = input_hw, None  # stream intrinsics are already at the input raster
    elif cfg.path is not None:
        origin = "file"
        path = Path(cfg.path).expanduser()
        parsed = parse_intrinsics_file(path)
        original = parsed.k4
        if cfg.source_hw is not None and parsed.raster_hw is not None and tuple(cfg.source_hw) != parsed.raster_hw:
            raise ValueError(f"{path}: K raster {parsed.raster_hw} != configured source_hw {tuple(cfg.source_hw)}")
        source_hw = cfg.source_hw if cfg.source_hw is not None else (parsed.raster_hw or default_hw)
        field = parsed.field
        artifact = {**_file_identity(path), "field": field}
    else:
        origin = "config"
        original = np.asarray([cfg.fx, cfg.fy, cfg.cx, cfg.cy], dtype=np.float64)
        source_hw = cfg.source_hw if cfg.source_hw is not None else default_hw
        field = "fx_fy_cx_cy"
    validate_k4(original)
    raster_map = cfg.raster_map
    k_raster = (int(source_hw[0]), int(source_hw[1]))
    if prepared_by is not None and k_raster == default_hw != input_hw:
        # The pipeline made the input raster from this source: map K the same way.
        if raster_map is not None and raster_map != prepared_by:
            raise ValueError(
                f"known K: the input raster {input_hw} was made from {default_hw} by {prepared_by} "
                f"(pipeline.init.input_raster), not by raster_map={raster_map}"
            )
        raster_map = prepared_by
    if prepared_by is not None and default_hw != input_hw and k_raster not in (default_hw, input_hw):
        # A third raster: raster_map says how the video was made from the K's raster, and the
        # pipeline made the input raster from the video; both maps apply, in that order.
        if raster_map is None:
            raise ValueError(
                f"known K raster {k_raster} differs from the video raster {default_hw} the pipeline prepared "
                f"its {input_hw} input from: set raster_map (how the video was made from the K's raster)"
            )
        to_video, to_video_geometry = map_intrinsics(original, k_raster, default_hw, raster_map)
        mapped, to_input_geometry = map_intrinsics(to_video, default_hw, input_hw, prepared_by)
        geometry = {
            "raster_map": f"{raster_map}+{prepared_by}",
            "source_hw": list(k_raster),
            "via_hw": list(default_hw),
            "target_hw": list(input_hw),
            "steps": [to_video_geometry, to_input_geometry],
        }
    else:
        mapped, geometry = map_intrinsics(original, source_hw, input_hw, raster_map)
    if prepared_by is not None:
        geometry["input_raster_prepared_from_hw"] = list(default_hw)
    k4 = validate_k4(mapped)
    record: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "intrinsics_source": INTRINSICS_SOURCE,
        "intrinsics_mode": CALIBRATED_MODE,
        "origin": origin,
        "field": field,
        "artifact": artifact,
        "original_K_fx_fy_cx_cy": [float(value) for value in original],
        "original_hw": [int(source_hw[0]), int(source_hw[1])],
        "raster_mapping": geometry,
        "input_hw": list(input_hw),
        "K_fx_fy_cx_cy": k4.tolist(),
        "bound_K_fx_fy_cx_cy": k4.tolist(),
        "dpvo_transport": dpvo_transport_intrinsics(k4, input_hw),
        "fov_x_degrees": float(math.degrees(2.0 * math.atan(float(input_hw[1]) / (2.0 * float(k4[0]))))),
        "depth_provider_K_equals_dpvo_K": True,
        "depth_provider_consumes": "fov_x_derived_from_exact_same_K",
        "dpvo_consumes": "exact_K_fx_fy_cx_cy",
        "stream_intrinsics_present": stream_carries_k,
        "stream_intrinsics_overridden": bool(stream_carries_k and origin != "stream"),
        "sensor_K_or_GT_read": True,
        "sensor_K_read_for_inference": True,
        "config": cfg.model_dump(),
    }
    return KnownIntrinsics(k4=k4, record=record)


# ----------------------------------------------------------------- processor


class KnownIntrinsicsProcessor(StreamProcessor):
    """Init stage: resolve the known K at construction, then assign it to every frame.

    Same interface as step 2's ``SharedKIntrinsicsProcessor`` (``.k4``, ``.record``),
    so the pipeline can bind either. With ``from_stream`` each frame's own K is
    checked against the shared one (``stream_tolerance_px``) before it is replaced
    by the exact shared float32 K that DPVO and the depth model use.
    """

    def __init__(
        self,
        video_stream: VideoStream,
        config: KnownIntrinsicsConfig | Mapping[str, Any],
        *,
        prepared_from: tuple[Sequence[int], RasterMap] | None = None,
    ) -> None:
        super().__init__()
        self.config = _as_config(config)
        resolved = resolve_known_intrinsics(self.config, video_stream, prepared_from=prepared_from)
        self.k4 = resolved.k4
        self.record = resolved.record
        self._reference = np.asarray(self.record["original_K_fx_fy_cx_cy"], dtype=np.float64)
        self._device_k: dict[torch.device, torch.Tensor] = {}

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        if self.config.from_stream:
            deviation = float(np.max(np.abs(_frame_k4(frame) - self._reference)))
            if deviation > self.config.stream_tolerance_px:
                raise ValueError(
                    f"frame {frame.raw_frame_idx}: stream K differs from the first frame's by {deviation:.3g} px; "
                    "MetricTrail tracks one shared K"
                )
        # Cloned on the device from one copy per device: no per-frame host copy of K.
        frame.intrinsics = shared_device_k(self._device_k, self.k4, frame.rgb.device)
        frame.camera_type = CameraType.PINHOLE
        return frame


def with_known_intrinsics(
    video_stream: VideoStream,
    config: MetricTrailOptionalConfig | KnownIntrinsicsConfig | Mapping[str, Any] | None,
    *,
    prepared_from: tuple[Sequence[int], RasterMap] | None = None,
) -> KnownIntrinsicsProcessor | None:
    """The known-K processor for ``config``, or None (SharedK) when no known intrinsics are configured.

    Accepts the whole optional block or just its ``intrinsics`` entry; ``prepared_from``
    as in :func:`resolve_known_intrinsics`.
    """

    if config is not None and not isinstance(config, (MetricTrailOptionalConfig, KnownIntrinsicsConfig)):
        primitive = config_to_primitive(config)
        if "intrinsics" in primitive or not set(primitive) & set(KnownIntrinsicsConfig.model_fields):
            config = MetricTrailOptionalConfig.model_validate(primitive)
        else:
            config = KnownIntrinsicsConfig.model_validate(primitive)
    if isinstance(config, MetricTrailOptionalConfig):
        config = config.intrinsics
    if config is None:
        return None
    return KnownIntrinsicsProcessor(video_stream, config, prepared_from=prepared_from)
