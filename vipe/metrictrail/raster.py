# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input raster: the read-time resize of :mod:`vipe.metrictrail`, as a stream wrapper.

MetricTrail never tracks a video at its native raster. An in-the-wild video is
prepared by

1. scaling it to *cover* ``height x width`` (280 x 504), then centre-cropping it to
   exactly that raster, so the aspect ratio is changed by cropping, never by squashing;
2. keeping the video's own frame rate unless it has more than ``max_frames`` (1500)
   frames, in which case one frame in ``k`` is kept with the smallest ``k`` that
   fits.

:func:`prepare_input_raster` applies ``pipeline.init.input_raster`` and returns a
:class:`PreparedRasterStream`; ViPE's own streams are not modified. Policies:

- ``cover_crop`` (default): that geometry and frame selection (:func:`plan_raster`):
  one scale that covers the target, the resized size rounded, an integer centre crop.
- ``resize``: a plain resize to ``height x width`` (independent x and y scales, no crop),
  with the same frame selection. 7-Scenes' 480 x 640 frames are read this way at 434 x 574.
- ``unchanged``: the stream as delivered (frames that were already prepared, e.g.
  benchmark frame directories).

Pixels (``resampler``): ``bicubic_antialias`` (the default, for every resized input) is
``torch.nn.functional.interpolate`` with ``mode="bicubic"``, ``align_corners=False``,
``antialias=True``: a proper low-pass downscale, closest to an ffmpeg Lanczos downscale
(PSNR 44-46 dB against Lanczos frames on example videos, vs 39-40 dB without
antialiasing). Without it, a large downscale (e.g. 1920 -> 504 wide) aliases, and DPVO
can lose track for a frame. ``bilinear`` is ViPE's own bilinear
resize, :meth:`vipe.streams.base.VideoFrame.resize` itself (``mode="bilinear"``,
``align_corners`` default (false), no antialias); it reproduces the validation numbers of the
benchmarks resized on read (Wild-SLAM, 7-Scenes, LaMaria). Either is applied to the frame as
ViPE's stream decoded it
(``uint8 / 255``; a video through OpenCV, as ``RawMp4Stream``), followed by the integer
crop (:func:`raster_uint8`, on the host, so the prepared bytes do not depend on the GPU
and the point-cloud export reproduces them). The prepared frame is rounded to 8 bit, the
precision the tracker and SharedK re-quantise every frame to (``rgb_normalization:
metrictrail``). There is no other interpolation in :mod:`vipe.metrictrail`.

Rotation: the frames are planned and resized as the stream delivers them. ViPE's
``RawMp4Stream`` decodes with ``cv2.VideoCapture`` at OpenCV's defaults, which apply a
clip's display rotation and report the displayed raster, so a portrait phone clip stored
landscape is cover-cropped as portrait (cropped, never squashed); the record's
``video_orientation`` states what the decoder did
(:func:`vipe.metrictrail.streams.video_orientation`).

Frame selection when ``k > 1``: one frame in ``k`` is kept as ffmpeg's ``fps`` filter
keeps it, which for a constant-rate video is source frame ``ceil(k * (j + 1/2)) - 1`` as
output ``j`` while that frame exists (frames 0, 2, 4, ... for ``k = 2``; 1, 4, 7, ... for
``k = 3``; 1, 5, 9, ... for ``k = 4``), checked against ffmpeg on synthetic constant-rate
clips (:func:`fps_filter_source_indices`).

The prepared frames are held in host memory as uint8 (at most ``max_frames`` frames:
about 0.42 MB each at 280 x 504), so the calibration, tracking and focal-refinement
passes read them without decoding again, and ``len()`` is exact before any work.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np
import torch

from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream

from .streams import video_orientation

logger = logging.getLogger(__name__)

RASTER_SCHEMA = "metrictrail_input_raster_v2"
#: Default prepared raster (height, width) and frame cap.
DEFAULT_HW = (280, 504)
DEFAULT_MAX_FRAMES = 1500
#: Fewer prepared frames than this are refused ("the pipeline needs at least 16").
MIN_PREPARED_FRAMES = 16
#: Largest distance (in 1/255 units) of ``rgb * 255`` from an integer for 8-bit input.
UINT8_TOLERANCE = 1.0e-3

#: The policies that resize; ``unchanged`` keeps the stream.
RESIZE_POLICIES: tuple[str, ...] = ("cover_crop", "resize")
#: Where each policy's geometry comes from (run record ``reference``).
POLICY_REFERENCE = {
    "cover_crop": "in-the-wild video preparation (cover-fit, integer centre crop, frame selection)",
    "resize": "7-Scenes loaders (plain resize, no crop)",
}
#: The resamplers (``input_raster.resampler``; run record ``interpolation``).
INTERPOLATIONS: dict[str, dict[str, Any]] = {
    "bicubic_antialias": {
        "name": "torch_bicubic_antialias",
        "call": "torch.nn.functional.interpolate",
        "mode": "bicubic",
        "align_corners": False,
        "antialias": True,
        "input": "uint8 / 255 as decoded by the ViPE stream",
        "output": "clamped to [0, 1], rounded to uint8",
    },
    "bilinear": {
        "name": "vipe_bilinear",
        "call": "vipe.streams.base.VideoFrame.resize",
        "mode": "bilinear",
        "align_corners": False,
        "antialias": False,
        "input": "uint8 / 255 as decoded by the ViPE stream",
        "output": "rounded to uint8",
    },
}
RESAMPLERS: tuple[str, ...] = tuple(INTERPOLATIONS)
_RESAMPLER_OF_NAME = {value["name"]: key for key, value in INTERPOLATIONS.items()}
#: A plan or run record without a resampler is ViPE's bilinear (the only one before ``resampler`` existed).
LEGACY_RESAMPLER = "bilinear"
INTERPOLATION = INTERPOLATIONS[LEGACY_RESAMPLER]


@dataclass(frozen=True)
class RasterPlan:
    """One input's raster geometry and frame stride."""

    policy: str
    source_hw: tuple[int, int]
    target_hw: tuple[int, int]
    #: ``cover_crop``'s one scale ``max(W/w, H/h)``; None for ``resize`` (independent x and y scales).
    scale: float | None
    resized_hw: tuple[int, int]
    crop_left_top: tuple[int, int]
    source_frames: int
    max_frames: int
    frame_stride: int
    source_fps: float
    prepared_fps: float
    resampler: str = LEGACY_RESAMPLER

    @property
    def resize(self) -> bool:
        return self.resized_hw != self.source_hw

    def record(self) -> dict[str, Any]:
        # The resampler is recorded as the record's ``interpolation`` (see resampler_of_record); a
        # ``resampler`` key meant the ffmpeg resampling of the first raster schema.
        value = {key: item for key, item in asdict(self).items() if key != "resampler"}
        return {key: list(item) if isinstance(item, tuple) else item for key, item in value.items()}


def plan_raster(
    source_hw: tuple[int, int],
    source_frames: int,
    source_fps: float,
    *,
    target_hw: tuple[int, int] = DEFAULT_HW,
    max_frames: int = DEFAULT_MAX_FRAMES,
    policy: str = "cover_crop",
    resampler: str = LEGACY_RESAMPLER,
) -> RasterPlan:
    """Resize, crop offsets and frame stride of ``policy`` (pixels resized with ``resampler``).

    ``cover_crop``: the cover-fit scale ``max(W/w, H/h)``, the resized size rounded and an
    integer centre crop. ``resize``: the target raster itself, no crop. Both use the smallest
    frame stride that keeps at most ``max_frames`` frames.
    """

    height_out, width_out = int(target_hw[0]), int(target_hw[1])
    # Multiples of 4 are not required: DPVO reflect-pads any other raster.
    if height_out < 1 or width_out < 1:
        raise ValueError(f"invalid prepared raster {target_hw}")
    if int(max_frames) < 1:
        raise ValueError("max_frames must be >= 1")
    if resampler not in RESAMPLERS:
        raise ValueError(f"unknown resampler {resampler!r}; choose one of {RESAMPLERS}")
    source_height, source_width = int(source_hw[0]), int(source_hw[1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"invalid source raster {source_hw}")
    scale: float | None
    if policy == "cover_crop":
        # --- one cover-fit scale, rounded size, integer centre crop ---
        scale = max(width_out / source_width, height_out / source_height)
        width = int(round(source_width * scale))
        height = int(round(source_height * scale))
        off_x, off_y = (width - width_out) // 2, (height - height_out) // 2
        # ---
    elif policy == "resize":
        scale, width, height, off_x, off_y = None, width_out, height_out, 0, 0
    else:
        raise ValueError(f"unknown input raster policy {policy!r}")
    # --- smallest stride that keeps at most max_frames frames ---
    stride = max(1, -(-int(source_frames) // int(max_frames)))
    out_fps = float(source_fps) / stride
    # ---
    return RasterPlan(
        policy=policy,
        source_hw=(source_height, source_width),
        target_hw=(height_out, width_out),
        scale=scale,
        resized_hw=(height, width),
        crop_left_top=(off_x, off_y),
        source_frames=int(source_frames),
        max_frames=int(max_frames),
        frame_stride=stride,
        source_fps=float(source_fps),
        prepared_fps=out_fps,
        resampler=resampler,
    )


def fps_filter_source_indices(source_frames: int, stride: int) -> list[int]:
    """Source frames that ffmpeg's ``fps`` filter at ``fps / stride`` keeps (constant rate).

    With ``round=near`` (the default) input frame ``i`` lands on output tick
    ``round(i / stride)`` (halves away from zero) and output ``j`` is the last input
    frame on a tick ``<= j``, i.e. ``ceil(stride * (j + 1/2)) - 1``; outputs whose
    frame does not exist are not emitted. ``stride == 1`` keeps every frame.
    """

    count, step = int(source_frames), int(stride)
    if step < 1:
        raise ValueError("stride must be >= 1")
    if step == 1:
        return list(range(count))
    kept: list[int] = []
    j = 0
    while True:
        index = (step * (2 * j + 1) + 1) // 2 - 1
        if index > count - 1:
            return kept
        kept.append(index)
        j += 1


# ------------------------------------------------------------------ pixels


def vipe_bilinear_resize(rgb: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """(H,W,3) float RGB resized to ``size_hw`` by ViPE's own :meth:`VideoFrame.resize` (bilinear)."""

    return VideoFrame(raw_frame_idx=0, rgb=rgb).resize((int(size_hw[0]), int(size_hw[1]))).rgb


def resampler_of_record(input_raster: dict[str, Any]) -> str:
    """The resampler a run record's ``input_raster`` used (its ``interpolation``; ``bilinear`` without one)."""

    name = (input_raster.get("interpolation") or {}).get("name")
    if name is None:
        return LEGACY_RESAMPLER
    if name not in _RESAMPLER_OF_NAME:
        raise ValueError(f"unknown input raster interpolation {name!r}")
    return _RESAMPLER_OF_NAME[name]


def resize_rgb(rgb: torch.Tensor, size_hw: tuple[int, int], resampler: str) -> torch.Tensor:
    """(H,W,3) float RGB resized to ``size_hw`` with ``resampler`` (see the module docstring)."""

    if resampler == "bilinear":
        return vipe_bilinear_resize(rgb, size_hw)
    if resampler == "bicubic_antialias":
        value = torch.nn.functional.interpolate(
            rgb.permute(2, 0, 1)[None],
            size=(int(size_hw[0]), int(size_hw[1])),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return value[0].permute(1, 2, 0).clamp_(0.0, 1.0)
    raise ValueError(f"unknown resampler {resampler!r}")


def raster_frame(rgb: torch.Tensor, plan: RasterPlan) -> torch.Tensor:
    """(H,W,3) float RGB in [0, 1] (ViPE's stream convention) -> the prepared (h,w,3) uint8 frame.

    The ``plan.resampler`` resize to ``plan.resized_hw`` (none when it is the source raster),
    the integer crop ``plan.crop_left_top``, then rounding to 8 bit; on ``rgb``'s device.
    """

    if tuple(rgb.shape) != (*plan.source_hw, 3):
        raise ValueError(f"frame is {tuple(rgb.shape)}, expected {(*plan.source_hw, 3)}")
    value = rgb.float()
    if plan.resize:
        value = resize_rgb(value, plan.resized_hw, plan.resampler)
    left, top = plan.crop_left_top
    height, width = plan.target_hw
    value = value[top : top + height, left : left + width]
    return torch.round(value * 255.0).clamp_(0.0, 255.0).to(torch.uint8).contiguous()


def raster_uint8(rgb_u8: np.ndarray, plan: RasterPlan) -> np.ndarray:
    """(H,W,3) uint8 as decoded -> the prepared frame (``uint8 / 255`` as ViPE's streams, :func:`raster_frame`)."""

    image = np.ascontiguousarray(rgb_u8)
    if image.shape[:2] != plan.source_hw or image.dtype != np.uint8:
        raise ValueError(f"frame is {image.shape} {image.dtype}, expected {plan.source_hw} uint8")
    return raster_frame(torch.from_numpy(image).float() / 255.0, plan).numpy()


# ------------------------------------------------------------------ stream


class PreparedRasterStream(VideoStream):
    """The prepared frames (uint8, host memory) as a ViPE stream of RGB-only frames.

    Each frame is ``uint8 / 255`` in float32, (H,W,3), on ``device``, the way ViPE's own
    streams normalise; ``raw_frame_idx`` is the frame's index in the source stream.
    On CUDA the frames sit in page-locked memory and are copied with
    ``non_blocking=True`` on the current stream, so reading a frame never
    synchronises the host.
    """

    def __init__(
        self,
        frames: np.ndarray,
        source_frame_ids: list[int],
        *,
        name: str,
        fps: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"prepared frames must be (N,H,W,3) uint8, got {frames.shape} {frames.dtype}")
        if len(source_frame_ids) != len(frames):
            raise ValueError("one source frame id per prepared frame")
        self.device = torch.device(device)
        host = torch.from_numpy(np.ascontiguousarray(frames))
        if self.device.type == "cuda":
            host = host.pin_memory()
        self._frames = host
        self.source_frame_ids = [int(item) for item in source_frame_ids]
        self._name = name
        self._fps = float(fps)

    def frame_size(self) -> tuple[int, int]:
        return (int(self._frames.shape[1]), int(self._frames.shape[2]))

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def attributes(self) -> set[FrameAttribute]:
        return set()

    def uint8_frame(self, index: int) -> torch.Tensor:
        return self._frames[index]

    def __iter__(self) -> Iterator[VideoFrame]:
        non_blocking = self.device.type == "cuda"
        for index, source_id in enumerate(self.source_frame_ids):
            rgb = self._frames[index].to(self.device, non_blocking=non_blocking).float() / 255.0
            yield VideoFrame(raw_frame_idx=source_id, rgb=rgb)


def collect_frames(frames: Iterator[np.ndarray], hw: tuple[int, int], *, capacity: int) -> np.ndarray:
    """Stack (H,W,3) uint8 frames into one array, preallocated for ``capacity`` frames (grown if needed)."""

    buffer = np.empty((max(int(capacity), 1), int(hw[0]), int(hw[1]), 3), dtype=np.uint8)
    count = 0
    for frame in frames:
        if count == len(buffer):
            grown = np.empty((2 * len(buffer), *buffer.shape[1:]), dtype=np.uint8)
            grown[:count] = buffer[:count]
            buffer = grown
        buffer[count] = frame
        count += 1
    return buffer[:count]  # a leading slice: contiguous, no copy


def _stream_uint8(frame: VideoFrame, policy: str) -> np.ndarray:
    """A ViPE frame's RGB as (H,W,3) uint8 on the host; refuses frames that carry K or are not 8-bit."""

    if frame.intrinsics is not None:
        raise ValueError(
            f"frame {frame.raw_frame_idx} carries intrinsics; input_raster.policy={policy} applies to RGB-only "
            "input (set pipeline.init.input_raster.policy=unchanged for prepared inputs)"
        )
    scaled = frame.rgb.detach().float() * 255.0
    quantized = torch.round(scaled)
    if (
        float((scaled - quantized).abs().max().item()) > UINT8_TOLERANCE
        or float(quantized.min().item()) < 0.0
        or float(quantized.max().item()) > 255.0
    ):
        raise ValueError(f"frame {frame.raw_frame_idx} is not 8-bit RGB; the input raster needs 8-bit input")
    return quantized.to(torch.uint8).cpu().numpy()


def prepare_input_raster(
    stream: VideoStream,
    config: Any,
    *,
    device: torch.device | None = None,
) -> tuple[VideoStream, dict[str, Any]]:
    """Apply ``pipeline.init.input_raster`` to ``stream``; returns the stream to process and its record.

    ``config`` has ``policy`` (``cover_crop`` | ``resize`` | ``unchanged``), ``height``,
    ``width``, ``max_frames`` and ``resampler`` (``bilinear`` when the config has none).
    """

    policy = str(config.policy)
    source_hw = (int(stream.frame_size()[0]), int(stream.frame_size()[1]))
    if policy == "unchanged":
        return stream, {
            "schema": RASTER_SCHEMA,
            "policy": "unchanged",
            "source_hw": list(source_hw),
            "prepared_hw": list(source_hw),
            "frames": len(stream),
            "frame_stride": 1,
            "interpolation": None,
            "video_orientation": video_orientation(stream),
        }
    if policy not in RESIZE_POLICIES:
        raise ValueError(f"unknown input raster policy {policy!r}")
    if FrameAttribute.INTRINSICS in stream.attributes():
        raise ValueError(
            f"stream {stream.name()!r} provides intrinsics; input_raster.policy={policy} applies to RGB-only input "
            "(set pipeline.init.input_raster.policy=unchanged for prepared inputs)"
        )
    target_hw = (int(config.height), int(config.width))
    device = (
        torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    plan = plan_raster(
        source_hw,
        len(stream),
        float(stream.fps()),
        target_hw=target_hw,
        max_frames=int(config.max_frames),
        policy=policy,
        resampler=str(getattr(config, "resampler", LEGACY_RESAMPLER)),
    )
    keep = fps_filter_source_indices(len(stream), plan.frame_stride)
    wanted = set(keep)
    source_ids: list[int] = []

    def prepared_frames() -> Iterator[np.ndarray]:
        for ordinal, frame in enumerate(stream):
            if ordinal in wanted:
                source_ids.append(int(frame.raw_frame_idx))
                # The stream's 8-bit frame (its RGB is uint8 / 255) through raster_uint8 on the host:
                # the point-cloud export's colours run the same call, so they are these bytes.
                yield raster_uint8(_stream_uint8(frame, policy), plan)

    frames = collect_frames(prepared_frames(), plan.target_hw, capacity=len(keep))
    if len(frames) != len(keep):
        raise ValueError(f"stream {stream.name()!r} yielded fewer frames than its length {len(stream)}")
    if len(frames) < MIN_PREPARED_FRAMES:
        raise ValueError(
            f"only {len(frames)} frames prepared from {stream.name()!r}; the pipeline needs at least "
            f"{MIN_PREPARED_FRAMES}"
        )
    prepared = PreparedRasterStream(frames, source_ids, name=stream.name(), fps=plan.prepared_fps, device=device)
    record = {
        "schema": RASTER_SCHEMA,
        "reference": POLICY_REFERENCE[policy],
        **plan.record(),
        "prepared_hw": list(plan.target_hw),
        "frames": len(prepared),
        "stream_frames": len(stream),
        "source_frame_count_from": "stream length",
        "source_frame_ids_rule": "constant_rate_fps_filter_rule",
        "source_frame_ids_head": source_ids[:4],
        "dropped_attributes": sorted(item.value for item in stream.attributes()),
        "interpolation": {**INTERPOLATIONS[plan.resampler], "torch": str(torch.__version__)},
        # A video file's display rotation and whether the decoder applied it (the plan is on the decoded raster).
        "video_orientation": video_orientation(stream),
        "host_bytes": int(frames.nbytes),
    }
    logger.info(
        "%s: input raster %s %s -> %s (resized %s, crop left/top %s, %s), stride %d, %d frames",
        stream.name(),
        policy,
        list(plan.source_hw),
        list(plan.target_hw),
        list(plan.resized_hw),
        list(plan.crop_left_top),
        plan.resampler,
        plan.frame_stride,
        len(prepared),
    )
    return prepared, record
