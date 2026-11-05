# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous writer of the per-frame metric depth in ViPE's depth artifact format.

``pipeline.output.save_metric_depth: async`` (the default) saves the MoGe-2 metric
depth DPVO tracked with, one map per frame, as ViPE's ``depth/<name>.zip``: one
``{frame:05d}.exr`` per frame with a single HALF channel ``Z`` (metres; 0 = no valid
depth), ZIP-deflated, exactly as :func:`vipe.utils.io.save_depth_artifacts` writes it
and :func:`vipe.utils.io.read_depth_artifacts` reads it.

The write stays off the tracking path:

- :meth:`AsyncDepthWriter.submit` (called by the depth producer, on its CUDA stream)
  copies the map ``non_blocking`` into a page-locked host buffer, records a CUDA event
  behind the copy and queues the pair; it never waits for the copy or the disk (it
  only blocks when ``queue_size`` frames are already waiting);
- one writer thread waits on each event and appends the frame to the zip; nothing is
  flushed per frame;
- a writer error is re-raised in the caller (next :meth:`submit`, or
  :meth:`finish`); the writer thread is joined on every path (:meth:`finish`,
  :meth:`abort`);
- the zip is written as ``<path>.partial`` and renamed to ``<path>`` only by
  :meth:`finish`, after every frame was stored; :meth:`abort` deletes it.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEPTH_ARTIFACT_FORMAT = "vipe_depth_zip_exr_half_z"
#: Frames (pinned depth maps) waiting for the writer before :meth:`AsyncDepthWriter.submit` blocks.
DEPTH_WRITER_QUEUE = 64


def write_depth_entry(archive: zipfile.ZipFile, frame_idx: int, metric_depth: np.ndarray) -> None:
    """One frame as ViPE's ``save_depth_artifacts`` writes it (HALF ``Z`` channel EXR, ``{frame:05d}.exr``)."""

    import Imath
    import OpenEXR

    height, width = metric_depth.shape
    header = OpenEXR.Header(width, height)
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
    with tempfile.NamedTemporaryFile(suffix=".exr") as handle:
        exr = OpenEXR.OutputFile(handle.name, header)
        exr.writePixels({"Z": metric_depth.astype(np.float16).tobytes()})
        exr.close()
        archive.write(handle.name, f"{frame_idx:05d}.exr")


class DepthWriterError(RuntimeError):
    """The background depth writer failed; the cause is chained."""


class AsyncDepthWriter:
    """Per-frame (H,W) float32 metric depth -> ViPE's depth zip, written by one background thread."""

    def __init__(self, path: str | Path, *, queue_size: int = DEPTH_WRITER_QUEUE) -> None:
        if int(queue_size) < 1:
            raise ValueError("queue_size must be >= 1")
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.queue_size = int(queue_size)
        self.submitted = 0
        self.written = 0
        self.hw: tuple[int, int] | None = None
        self._jobs: queue.Queue[Any] | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._error_reported = False
        self._state = "open"  # open -> finished | aborted
        self._submit_seconds = 0.0
        self._writer_seconds = 0.0

    # ------------------------------------------------------------ producer side

    def submit(self, index: int, depth: torch.Tensor) -> None:
        """Queue frame ``index`` (consecutive from 0): ``depth`` is (H,W), metres, 0 where invalid."""

        started = time.perf_counter()
        if self._state != "open":
            raise RuntimeError(f"depth writer is {self._state}")
        self._raise_error()  # stop early once the writer has failed
        if int(index) != self.submitted:
            raise RuntimeError(f"depth writer got frame {index}, expected {self.submitted}")
        if depth.ndim != 2:
            raise ValueError(f"metric depth must be (H,W), got {tuple(depth.shape)}")
        hw = (int(depth.shape[0]), int(depth.shape[1]))
        if self.hw is None:
            self.hw = hw
        elif hw != self.hw:
            raise ValueError(f"metric depth raster changed from {self.hw} to {hw}")
        ready: Any = None
        if depth.is_cuda:
            host = torch.empty(hw, dtype=torch.float32, pin_memory=True)
            # On the caller's (producer's) stream: the allocator may recycle the GPU
            # source only after this copy (stream order).
            host.copy_(depth.detach(), non_blocking=True)
            # blocking: the writer sleeps on it instead of spinning a core.
            ready = torch.cuda.Event(blocking=True)
            ready.record(torch.cuda.current_stream(depth.device))
        else:
            host = depth.detach().to(torch.float32).clone()
        if self._thread is None:
            self.partial.parent.mkdir(parents=True, exist_ok=True)
            self._jobs = queue.Queue(maxsize=self.queue_size)
            self._thread = threading.Thread(
                target=self._loop, args=(self._jobs,), name="metrictrail-depth-writer", daemon=True
            )
            self._thread.start()
        assert self._jobs is not None
        self._jobs.put((int(index), host, ready))
        self.submitted += 1
        self._submit_seconds += time.perf_counter() - started

    # -------------------------------------------------------------- writer side

    def _loop(self, jobs: queue.Queue[Any]) -> None:
        archive: zipfile.ZipFile | None = None
        failed = False
        try:
            archive = zipfile.ZipFile(self.partial, "w", zipfile.ZIP_DEFLATED)
        except BaseException as error:  # re-raised in the owning thread
            self._error = error
            failed = True
        expected = 0
        while True:
            job = jobs.get()
            if job is None:
                break
            if failed:
                # Keep draining so that a producer blocked on put() is released.
                del job
                continue
            started = time.perf_counter()
            try:
                index, host, ready = job
                if index != expected:
                    raise RuntimeError(f"depth writer got frame {index}, expected {expected}")
                if ready is not None:
                    ready.synchronize()
                assert archive is not None
                write_depth_entry(archive, index, host.numpy())
                expected += 1
                self.written = expected
            except BaseException as error:  # re-raised in the owning thread
                self._error = error
                failed = True
            finally:
                del job
                self._writer_seconds += time.perf_counter() - started
        if archive is not None:
            try:
                archive.close()
            except BaseException as error:
                if self._error is None:
                    self._error = error

    # --------------------------------------------------------------- lifecycle

    def _raise_error(self) -> None:
        error = self._error
        if error is not None and not self._error_reported:
            self._error_reported = True
            raise DepthWriterError(f"metric depth writer failed for {self.path}") from error

    def _join(self) -> None:
        thread, jobs = self._thread, self._jobs
        self._thread = self._jobs = None
        if thread is not None:
            assert jobs is not None
            jobs.put(None)  # the writer keeps consuming, even after an error
            thread.join()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def finish(self, *, expected_frames: int | None = None) -> dict[str, Any]:
        """Drain and join the writer, re-raise its error, then publish ``<path>``; returns the record."""

        if self._state != "open":
            raise RuntimeError(f"depth writer is {self._state}")
        try:
            self._join()
            self._raise_error()
            if expected_frames is not None and self.submitted != int(expected_frames):
                raise RuntimeError(f"depth writer received {self.submitted} of {expected_frames} frames")
            if self.written != self.submitted:
                raise RuntimeError(f"depth writer stored {self.written} of {self.submitted} frames")
            if self.submitted == 0:
                raise RuntimeError("depth writer received no frame")
            os.replace(self.partial, self.path)
        except BaseException:
            self._discard()
            self._state = "aborted"
            raise
        self._state = "finished"
        return self.record()

    def abort(self) -> None:
        """Join the writer and delete the partial file. Never raises the writer's own error (it was
        either reported already or is superseded by the error that caused the abort)."""

        if self._state != "open":
            return
        self._state = "aborted"
        try:
            self._join()
        finally:
            self._error_reported = True
            self._discard()

    def _discard(self) -> None:
        try:
            self.partial.unlink()
        except FileNotFoundError:
            pass

    def record(self) -> dict[str, Any]:
        return {
            "format": DEPTH_ARTIFACT_FORMAT,
            "frames": self.written,
            "raster_hw": list(self.hw) if self.hw is not None else None,
            "unit": "metre",
            "invalid_value": 0.0,
            "stored_dtype": "float16",
            "writer": "async_pinned_copy_writer_thread",
            "producer_submit_seconds": self._submit_seconds,
            "writer_thread_seconds": self._writer_seconds,
            "queue_size": self.queue_size,
        }
