# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-sequence guarantee for MetricTrail inputs.

MetricTrail processes the whole input sequence by default. ViPE's stock
``raw_mp4_stream`` preset stops at ``frame_end: 1000``, so a video run through
it is silently cut after 1000 frames. :func:`sequence_coverage` reports whether
a ViPE file stream ends before its source does, and
:func:`require_full_sequence` refuses (or loudly warns about) such a stream.

Only the tail counts as truncation. A non-zero ``frame_start`` or a
``frame_skip`` > 1 is an explicit choice and is reported but not refused.

:func:`full_sequence_stream` also fixes the frame count before any work starts
(:func:`count_stream_frames`). An mp4 container may declare more frames than its
decoder delivers (VFR or edited phone clips), and ViPE's ``RawMp4Stream`` takes
its ``len()`` from that declaration. ViPE's own pipelines learn the real count
from their frame cache; MetricTrail streams its input instead, so it counts the
decodable frames once up front and tracks exactly those (the shortfall is
recorded), instead of failing after the whole sequence has been tracked. A
stream with no frame, or one that later yields fewer frames than were counted,
is refused with :class:`ShortStreamError` at that point.

:func:`tracking_stride_stream` is ``pipeline.init.tracking_stride``: DPVO tracks every k-th
frame while SharedK calibrates on all of them (7-Scenes tracks every fifth frame).
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Literal

from vipe.streams.base import CachedVideoStream, ProcessedVideoStream, VideoFrame, VideoStream

logger = logging.getLogger(__name__)

FULL_SEQUENCE_HINT = (
    "MetricTrail processes the full sequence: set streams.frame_end=-1 (ViPE's raw_mp4_stream preset "
    "defaults to frame_end: 1000)"
)


class TruncatedSequenceError(ValueError):
    """The input stream ends before its source does."""


class ShortStreamError(ValueError):
    """The input stream yields fewer frames than it declares (and than MetricTrail counted)."""


def _unwrap(stream: VideoStream) -> VideoStream:
    while isinstance(stream, ProcessedVideoStream):
        stream = stream.stream
    return stream


def _source_frame_count(stream: VideoStream) -> tuple[int | None, str]:
    """Total frames of the stream's source, and how that number was obtained."""

    frame_files = getattr(stream, "frame_files", None)
    if isinstance(frame_files, list):
        return len(frame_files), "frame_directory_listing"
    path = getattr(stream, "path", None)
    if isinstance(path, (str, Path)) and Path(path).suffix.lower() == ".mp4":
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        # The same container metadata ViPE's RawMp4Stream uses to resolve frame_end=-1.
        return (count if count > 0 else None), "video_container_frame_count"
    return None, "unknown_stream_type"


def video_orientation(stream: VideoStream) -> dict[str, Any] | None:
    """How the video file behind ``stream`` is oriented when ViPE decodes it; None if no video file is visible.

    ViPE's ``RawMp4Stream`` decodes with ``cv2.VideoCapture`` at OpenCV's defaults. OpenCV
    then applies the container's display rotation to every decoded frame
    (``CAP_PROP_ORIENTATION_AUTO``, on by default) and reports the displayed raster as the
    frame size, so a portrait phone clip stored landscape arrives as portrait frames.
    MetricTrail takes the frames exactly as the stream delivers them; this records what the
    decoder did (``orientation_meta_deg`` is OpenCV's ``CAP_PROP_ORIENTATION_META``).
    A cached stream (``vipe infer``) or a frame directory gives None.
    """

    base = _unwrap(stream)
    path = getattr(base, "path", None)
    if not isinstance(path, (str, Path)) or not Path(path).is_file():
        return None
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        meta_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        auto_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        rotation = float(capture.get(meta_property)) if meta_property is not None else None
        applied = bool(capture.get(auto_property)) if auto_property is not None else None
        decoded_hw = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))]
    finally:
        capture.release()
    transposed = bool(applied) and rotation is not None and int(round(rotation)) % 180 == 90
    return {
        "decoder": "cv2.VideoCapture with OpenCV's defaults (ViPE RawMp4Stream)",
        "opencv_version": str(cv2.__version__),
        "orientation_meta_deg": rotation,
        "rotation_applied_by_decoder": applied,
        "decoded_hw": decoded_hw,
        "stored_hw": decoded_hw[::-1] if transposed else decoded_hw,
    }


def sequence_coverage(stream: VideoStream, *, source_decoded_until: int | None = None) -> dict[str, Any]:
    """How much of its source a ViPE file stream covers (works through ``ProcessedVideoStream``).

    ``source_decoded_until``: the number of leading source frames the decoder delivered
    when asked for every frame below the stream's end (:func:`count_stream_frames`). If
    it stops short of that end, the source ends there, whatever its container declares.
    """

    base = _unwrap(stream)
    total, source = _source_frame_count(base)
    start = getattr(base, "start", None)
    end = getattr(base, "end", None)
    step = getattr(base, "step", None)
    if source_decoded_until is not None and isinstance(end, int) and source_decoded_until < end:
        total, source, end = source_decoded_until, "decoded_frame_count", source_decoded_until
    known = isinstance(total, int) and isinstance(end, int)
    truncated = isinstance(total, int) and isinstance(end, int) and end < total
    return {
        "stream": type(base).__name__,
        "name": stream.name(),
        "frames_in_stream": len(stream),
        "source_frames": total,
        "source_frame_count_from": source,
        "frame_start": start,
        "frame_end_effective": end,
        "frame_skip": step,
        "coverage_known": known,
        "tail_truncated": truncated,
        "dropped_tail_frames": total - end if isinstance(total, int) and isinstance(end, int) and truncated else 0,
    }


def require_full_sequence(
    stream: VideoStream,
    *,
    on_truncation: Literal["error", "warn"] = "error",
    source_decoded_until: int | None = None,
) -> dict[str, Any]:
    """Refuse (``error``) or warn about a stream whose tail was cut; returns :func:`sequence_coverage`."""

    coverage = sequence_coverage(stream, source_decoded_until=source_decoded_until)
    if coverage["tail_truncated"]:
        message = (
            f"input {coverage['name']!r} stops at frame {coverage['frame_end_effective']} of "
            f"{coverage['source_frames']} ({coverage['dropped_tail_frames']} trailing frames dropped). "
            f"{FULL_SEQUENCE_HINT}"
        )
        if on_truncation == "error":
            raise TruncatedSequenceError(message)
        logger.warning("!!! TRUNCATED INPUT !!! %s", message)
    elif not coverage["coverage_known"]:
        logger.info("MetricTrail cannot verify full-sequence coverage of %s", coverage["stream"])
    return coverage


# ----------------------------------------------------------------- frame count


def _decoded_mp4_frames(path: str | Path, stop: int) -> int:
    """How many of the first ``stop`` frames of ``path`` OpenCV decodes (``RawMp4Stream``'s reader)."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0
        count = 0
        # grab() decodes without the colour conversion of read(); RawMp4Stream stops
        # at the first frame read() cannot deliver, as this loop does.
        while count < stop and capture.grab():
            count += 1
    finally:
        capture.release()
    return count


def count_stream_frames(stream: VideoStream) -> dict[str, Any]:
    """The number of frames ``stream`` will actually yield, established before any processing.

    - frame directory: its file listing (exact, no read);
    - mp4 (``RawMp4Stream``, also inside ``ProcessedVideoStream``): one decode pass with
      OpenCV's ``grab`` over the frames below the stream's end (no colour conversion, no
      device copy);
    - a fully cached stream (``vipe infer`` caches its input): the cache's length;
    - anything else: one pass over the stream, stopping at its declared length.
    """

    declared = len(stream)
    base = _unwrap(stream)
    path = getattr(base, "path", None)
    seek: list[Any] = [getattr(base, name, None) for name in ("start", "end", "step")]
    decoded_until: int | None = None
    if isinstance(getattr(base, "frame_files", None), list):
        frames, source = declared, "frame_directory_listing"
    elif (
        isinstance(path, (str, Path)) and Path(path).suffix.lower() == ".mp4" and all(isinstance(v, int) for v in seek)
    ):
        start, end, step = seek
        decoded_until = _decoded_mp4_frames(path, end)
        frames, source = min(declared, len(range(start, decoded_until, step))), "mp4_decode_pass"
    elif isinstance(base, CachedVideoStream) and base.iterator is None:
        frames, source = declared, "frame_cache"
    else:
        frames, source = 0, "stream_pass"
        if declared > 0:
            for _ in stream:
                frames += 1
                if frames >= declared:
                    break
    return {
        "declared_frames": declared,
        "frames": frames,
        "frame_count_from": source,
        "declared_minus_actual": declared - frames,
        "source_decoded_until": decoded_until,
    }


class ExactLengthStream(ProcessedVideoStream):
    """``stream`` with its counted frame count as ``len()``; yields exactly that many frames.

    A pass that ends early raises :class:`ShortStreamError` at the missing frame (an
    input that changed between passes), so no stage silently sees fewer frames.
    """

    def __init__(self, stream: VideoStream, frames: int) -> None:
        super().__init__(stream, [])
        self._frames = int(frames)

    def __len__(self) -> int:
        return self._frames

    def __iter__(self) -> Any:
        return self._iterate()

    def _iterate(self) -> Any:
        iterator = iter(self.stream)
        for index in range(self._frames):
            try:
                frame: VideoFrame = next(iterator)
            except StopIteration:
                raise ShortStreamError(
                    f"input {self.name()!r} ended after {index} frames, but {self._frames} were counted before "
                    "processing (the input changed while MetricTrail was reading it)"
                ) from None
            yield frame


def full_sequence_stream(
    stream: VideoStream,
    *,
    on_truncation: Literal["error", "warn"] = "error",
) -> tuple[VideoStream, dict[str, Any]]:
    """The input as MetricTrail processes it, plus its coverage record; refuses cut or empty inputs.

    The frame count is fixed first (:func:`count_stream_frames`), then the tail
    truncation is checked (:func:`require_full_sequence`). A stream that declares more
    frames than it yields is returned as an :class:`ExactLengthStream` with the counted
    length, and the coverage record says so (``frame_count``).
    """

    counted = count_stream_frames(stream)
    coverage = require_full_sequence(
        stream, on_truncation=on_truncation, source_decoded_until=counted["source_decoded_until"]
    )
    coverage["frame_count"] = counted
    if counted["frames"] <= 0:
        raise ShortStreamError(
            f"input {stream.name()!r} has no decodable frame (it declares {counted['declared_frames']})"
        )
    if counted["frames"] != counted["declared_frames"]:
        logger.warning(
            "%s declares %d frames but yields %d (%s); MetricTrail tracks the %d it yields",
            stream.name(),
            counted["declared_frames"],
            counted["frames"],
            counted["frame_count_from"],
            counted["frames"],
        )
        stream = ExactLengthStream(stream, counted["frames"])
    coverage["frames_in_stream"] = len(stream)
    return stream, coverage


class StridedStream(ProcessedVideoStream):
    """Frames ``0, k, 2k, ...`` of ``stream`` with their own ``raw_frame_idx`` (the source frame ids)."""

    def __init__(self, stream: VideoStream, stride: int) -> None:
        super().__init__(stream, [])
        if int(stride) < 1:
            raise ValueError(f"tracking stride must be >= 1, got {stride}")
        self.stride = int(stride)

    def __len__(self) -> int:
        return -(-len(self.stream) // self.stride)

    def fps(self) -> float:
        return self.stream.fps() / self.stride

    def __iter__(self) -> Any:
        return itertools.islice(iter(self.stream), 0, None, self.stride)


def tracking_stride_stream(stream: VideoStream, stride: int) -> VideoStream:
    """The frames DPVO tracks: every ``stride``-th frame of ``stream`` (``stream`` itself for 1).

    The pipeline calibrates SharedK on every frame of ``stream`` and tracks this view,
    starting at frame 0; the pose artifact's ``source_frame_ids`` are the tracked frames'
    ``raw_frame_idx`` (0, 5, 10, ... for a frame directory at stride 5, as 7-Scenes runs).
    """

    return stream if int(stride) == 1 else StridedStream(stream, stride)
