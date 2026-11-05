# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViPE's dynamic-object masking with DPVO as the SLAM backend (``pipeline.init.instance``).

ViPE (``vipe/pipeline/default.py``, ``vipe/slam``):

1. init stage: ``TrackAnythingProcessor`` sets ``frame.instance`` and
   ``frame.mask = erode((instance == 0) | sky, 5)`` (True = static, usable pixel);
2. SLAM: ``frame.mask`` is bilinearly resized to the network's feature grid and
   ``> 0.9`` is valid (``SLAMSystem._precompute_features``); the flow weights of
   invalid source pixels are zeroed (``FactorGraph``), so dynamic pixels add
   nothing to BA;
3. the exported ``SLAMMap`` leaves those pixels out (``GraphBuffer``);
4. post: the masks are output artifacts and steer the depth alignment.

DPVO equivalent, without touching the vendored DPVO:

1. the same ``TrackAnythingProcessor`` in MetricTrail's init stage, run once per
   video (:class:`InstanceMasks` records its outputs and replays them for
   re-tracking and for the outputs);
2. DPVO's BA only sees patches, so dynamic pixels are left out by never placing
   a patch centroid on them: the mask goes through the DPVO transport, is
   bilinearly resized to the patchifier's 1/4 grid and ``> 0.9`` is valid (ViPE's
   rule), and the DINO patchifier rejection-samples DPVO's own uniform draws
   until every centroid is valid (:func:`sample_valid_centroids`). A frame is
   tracked unmasked (unrestricted centroids, depth mask unchanged) and counted
   as a fallback when it has fewer valid interior cells than patches per frame,
   or when its static cells carry so little metric depth that DPVO's metric
   API would reject the masked frame and abort the run (a static region that is
   only sky, for example; see :meth:`DynamicMaskTracking.prepare`);
3. the metric-depth prior mask of a restricted frame is ANDed with the mask;
4. patch points on invalid cells are left out of the exported ``SLAMMap``.
"""

from __future__ import annotations

import math
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from vipe.streams.base import FrameAttribute, StreamProcessor, VideoFrame, VideoStream
from vipe.utils import io

#: ViPE's validity threshold on the bilinearly resized mask (``vipe/slam/system.py``).
VALID_MASK_THRESHOLD = 0.9
#: Candidates drawn per round are at least this many times the centroids still needed.
CANDIDATE_FACTOR = 4
#: Rejection rounds before giving up (each round expects twice the missing centroids).
MAX_REJECTION_ROUNDS = 64
#: DPVO's feature stride: the patchifier samples centroids on the 1/4 raster.
DPVO_FEATURE_STRIDE = 4
#: A frame stays restricted only while the chance that none of its centroids carries a metric
#: prior (DPVO then rejects the frame and the run aborts) is at most this, or no higher than
#: without masking.
METRIC_PRIOR_MISS_TOLERANCE = 1e-6
FALLBACK_TOO_FEW_STATIC_CELLS = "too_few_static_cells"
FALLBACK_METRIC_PRIOR = "static_cells_lack_metric_depth"


class InstanceMasks(StreamProcessor):
    """ViPE's ``TrackAnythingProcessor``, run on the first pass only; later passes replay its outputs.

    The first pass (the tracking pass) runs the processor and keeps
    ``frame.instance``, ``frame.instance_phrases`` and ``frame.mask`` of every
    frame on the host. Any later pass over the same video (an FP32 recovery pass, the
    focal-refinement re-track, the output stream) gets exactly those values back.
    """

    def __init__(self, track_anything: StreamProcessor) -> None:
        self.track_anything = track_anything
        self.instances: list[torch.Tensor | None] = []
        self.phrases: list[dict[int, str] | None] = []
        self.masks: list[torch.Tensor | None] = []
        self.started = False
        self.complete = False

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return self.track_anything.update_attributes(previous_attributes)

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("InstanceMasks works on the whole iterator")

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if self.complete:
            return self._replay(previous_iterator)
        if self.started:
            raise RuntimeError("the TrackAnything pass was interrupted and cannot be resumed")
        self.started = True
        return self._track(self.track_anything.update_iterator(previous_iterator, pass_idx))

    def _track(self, iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        for frame_idx, frame in enumerate(iterator):
            if frame_idx != len(self.masks):
                raise AssertionError("TrackAnything frame order drifted")
            self.instances.append(None if frame.instance is None else frame.instance.detach().cpu())
            self.phrases.append(None if frame.instance_phrases is None else dict(frame.instance_phrases))
            self.masks.append(None if frame.mask is None else frame.mask.detach().cpu())
            yield frame
        self.complete = True

    def _replay(self, iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        count = 0
        for frame_idx, frame in enumerate(iterator):
            if frame_idx >= len(self.masks):
                raise ValueError(f"replayed stream is longer than the {len(self.masks)} tracked frames")
            device = frame.rgb.device
            instance, mask, phrases = self.instances[frame_idx], self.masks[frame_idx], self.phrases[frame_idx]
            frame.instance = None if instance is None else instance.to(device)
            frame.instance_phrases = None if phrases is None else dict(phrases)
            frame.mask = None if mask is None else mask.to(device)
            count += 1
            yield frame
        if count != len(self.masks):
            raise ValueError(f"replayed stream has {count} frames, TrackAnything tracked {len(self.masks)}")


def save_instance_artifacts(artifact_path: io.ArtifactPath, stream: VideoStream) -> None:
    """ViPE's mask artifacts (``mask/<n>.zip`` of PNG instances, ``mask/<n>.txt``), as ``io.save_artifacts`` writes them."""

    instance_list = [
        (frame_idx, frame_data.instance)
        for frame_idx, frame_data in enumerate(stream)
        if frame_data.instance is not None
    ]
    if len(instance_list) > 0:
        artifact_path.mask_path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(artifact_path.mask_path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, instance in instance_list:
                _, mask_buffer = cv2.imencode(".png", instance.cpu().numpy().astype(np.uint8))
                z.writestr(f"{frame_idx:05d}.png", mask_buffer.tobytes())

    instance_phrases_combined: dict[int, str] = {}
    for frame_data in stream:
        if frame_data.instance_phrases is not None:
            instance_phrases_combined.update(frame_data.instance_phrases)
    if len(instance_phrases_combined) > 0:
        artifact_path.mask_phrase_path.parent.mkdir(exist_ok=True, parents=True)
        with artifact_path.mask_phrase_path.open("w") as f:
            for idx, phrase in instance_phrases_combined.items():
                f.write(f"{idx}: {phrase}\n")


def static_pixels(mask: torch.Tensor) -> torch.Tensor:
    """ViPE's ``frame.mask`` as a bool tensor (True = static, usable)."""

    return mask if mask.dtype == torch.bool else mask > 0.5


def centroid_valid_grid(mask: torch.Tensor, padding_ltrb: Sequence[int], grid_hw: tuple[int, int]) -> torch.Tensor:
    """ViPE's SLAM validity rule on DPVO's centroid grid.

    ``mask`` (H,W) on the input raster goes through the DPVO transport (reflected
    like the image when the raster is padded), is resized bilinearly to the
    patchifier's grid ``grid_hw`` and is valid where ``> 0.9``, exactly as
    ``SLAMSystem._precompute_features`` does on ViPE's 1/8 feature grid.
    """

    value = mask[None, None].float()
    if any(int(item) for item in padding_ltrb):
        value = torch_f.pad(value, tuple(int(item) for item in padding_ltrb), mode="reflect")
    return torch_f.interpolate(value, grid_hw, mode="bilinear")[0, 0] > VALID_MASK_THRESHOLD


def valid_interior_cells(valid: torch.Tensor) -> int:
    """Valid cells where DPVO may put a centroid (``x`` in [1, w-2], ``y`` in [1, h-2])."""

    height, width = valid.shape[-2:]
    return int(valid[..., 1 : height - 1, 1 : width - 1].sum().item())


def metric_prior_cells(metric_mask: torch.Tensor, padding_ltrb: Sequence[int]) -> torch.Tensor:
    """(h,w) bool on DPVO's centroid grid: the cells whose metric depth DPVO reads as valid.

    DPVO reads a patch's metric prior at transport pixel ``(4y+1, 4x+1)``
    (``prepare_metric_disparity``: ``valid[1::4, 1::4]``), and the transport pads
    the metric mask with False.
    """

    value = metric_mask[None, None].bool()
    if any(int(item) for item in padding_ltrb):
        value = torch_f.pad(value, tuple(int(item) for item in padding_ltrb), mode="constant", value=False)
    return value[0, 0, 1::DPVO_FEATURE_STRIDE, 1::DPVO_FEATURE_STRIDE]


def metric_prior_miss_probability(candidates: torch.Tensor, prior: torch.Tensor, count: int) -> float:
    """Chance that ``count`` uniform centroid draws over the interior ``candidates`` all miss ``prior`` cells.

    That is the chance that DPVO's metric API rejects the frame ("no valid
    sampled DPVO patches"); exact for DPVO's RANDOM centroids.
    """

    height, width = candidates.shape[-2:]
    inside = candidates[1 : height - 1, 1 : width - 1]
    total = int(inside.sum().item())
    if total == 0:
        return 1.0
    hits = int((inside & prior[1 : height - 1, 1 : width - 1]).sum().item())
    return (1.0 - hits / total) ** int(count)


def sample_valid_centroids(
    valid: torch.Tensor, frames: int, count: int, *, max_rounds: int = MAX_REJECTION_ROUNDS
) -> tuple[torch.Tensor, torch.Tensor]:
    """DPVO's RANDOM centroid draw restricted to ``valid`` (h,w) cells, by rejection sampling.

    Candidates are drawn as DPVO draws them (``torch.randint(1, w-1)`` for x, then
    ``torch.randint(1, h-1)`` for y, on ``valid``'s device), in rounds of extra
    candidates; accepted ones keep their draw order and the first ``count`` per
    frame are returned as (frames, count) int64 ``x``, ``y``. Uniform over the
    valid interior cells, like DPVO's draw over all of them.
    """

    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError(f"centroid validity must be a (h,w) bool grid, got {tuple(valid.shape)} {valid.dtype}")
    height, width = int(valid.shape[0]), int(valid.shape[1])
    available = valid_interior_cells(valid)
    if available == 0:
        raise ValueError("no valid centroid position")
    accept_rate = available / float((height - 2) * (width - 2))
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for _ in range(int(frames)):
        kept_x: list[torch.Tensor] = []
        kept_y: list[torch.Tensor] = []
        found = 0
        for _round in range(int(max_rounds)):
            draws = max(CANDIDATE_FACTOR * count, math.ceil(2.0 * (count - found) / accept_rate))
            x = torch.randint(1, width - 1, size=[draws], device=valid.device)
            y = torch.randint(1, height - 1, size=[draws], device=valid.device)
            keep = valid[y, x]
            kept_x.append(x[keep])
            kept_y.append(y[keep])
            found += int(keep.sum().item())
            if found >= count:
                break
        else:
            raise RuntimeError(f"rejection sampling found {found}/{count} valid centroids in {max_rounds} rounds")
        xs.append(torch.cat(kept_x)[:count])
        ys.append(torch.cat(kept_y)[:count])
    return torch.stack(xs), torch.stack(ys)


@dataclass
class FrameMaskRecord:
    """Per-frame mask state: the centroid grid (host), whether centroids were restricted, the dynamic fraction."""

    grid: torch.Tensor
    restricted: bool
    dynamic_fraction: float
    valid_cells: int
    fallback_reason: str | None = None


class DynamicMaskTracking:
    """Per tracking attempt: ViPE masks -> DPVO centroid grids, metric-mask AND, map exclusion, record."""

    def __init__(
        self,
        *,
        image_hw: tuple[int, int],
        padding_ltrb: Sequence[int],
        transport_hw: tuple[int, int],
        patches_per_frame: int,
    ) -> None:
        if transport_hw[0] % DPVO_FEATURE_STRIDE or transport_hw[1] % DPVO_FEATURE_STRIDE:
            raise ValueError(f"DPVO raster {transport_hw} is not divisible by {DPVO_FEATURE_STRIDE}")
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.padding_ltrb = tuple(int(item) for item in padding_ltrb)
        self.grid_hw = (int(transport_hw[0]) // DPVO_FEATURE_STRIDE, int(transport_hw[1]) // DPVO_FEATURE_STRIDE)
        self.patches_per_frame = int(patches_per_frame)
        self.frames: dict[int, FrameMaskRecord] = {}
        self.pending: dict[int, torch.Tensor | None] = {}
        self.map_points_excluded = 0

    def prepare(self, index: int, frame: VideoFrame, metric_mask: torch.Tensor) -> torch.Tensor:
        """Producer side: the frame's centroid grid, and its metric mask (ANDed with the static pixels).

        The frame falls back to unmasked tracking when it has fewer valid
        interior cells than patches per frame, or when the masked frame would
        make DPVO's metric API reject it: DPVO aborts the run when no sampled
        patch carries a metric prior, so a restricted frame must keep that
        chance at most :data:`METRIC_PRIOR_MISS_TOLERANCE`, or no higher than
        without masking (a static region that is only sky has no metric depth).
        """

        if frame.mask is None:
            raise ValueError(f"frame {frame.raw_frame_idx} carries no ViPE mask (init.instance is on)")
        static = static_pixels(frame.mask).to(device=metric_mask.device)
        if tuple(static.shape) != self.image_hw or tuple(metric_mask.shape) != self.image_hw:
            raise ValueError(f"mask {tuple(static.shape)} / metric mask {tuple(metric_mask.shape)} != {self.image_hw}")
        grid = centroid_valid_grid(static, self.padding_ltrb, self.grid_hw)
        valid_cells = valid_interior_cells(grid)
        masked = (metric_mask & static).contiguous()
        reason: str | None = None
        if valid_cells < self.patches_per_frame:
            reason = FALLBACK_TOO_FEW_STATIC_CELLS
        else:
            masked_miss = metric_prior_miss_probability(
                grid, metric_prior_cells(masked, self.padding_ltrb), self.patches_per_frame
            )
            if masked_miss > METRIC_PRIOR_MISS_TOLERANCE:
                unmasked_miss = metric_prior_miss_probability(
                    torch.ones_like(grid), metric_prior_cells(metric_mask, self.padding_ltrb), self.patches_per_frame
                )
                if masked_miss > unmasked_miss:
                    reason = FALLBACK_METRIC_PRIOR
        restricted = reason is None
        self.frames[int(index)] = FrameMaskRecord(
            grid=grid.cpu(),
            restricted=restricted,
            dynamic_fraction=1.0 - float(static.float().mean().item()),
            valid_cells=valid_cells,
            fallback_reason=reason,
        )
        self.pending[int(index)] = grid if restricted else None
        return masked if restricted else metric_mask

    def transport(self, slam: Any, patchifier: Any, ops: Any) -> CentroidMaskedTransport:
        if not hasattr(patchifier, "centroid_valid"):
            raise TypeError(f"{type(patchifier).__name__} cannot restrict DPVO patch centroids")
        return CentroidMaskedTransport(slam, patchifier, self, ops)

    def take(self, index: int) -> torch.Tensor | None:
        if int(index) not in self.pending:
            raise KeyError(f"no prepared mask for DPVO frame {index}")
        return self.pending.pop(int(index))

    def static_patches(self, ordinals: np.ndarray, centres: torch.Tensor) -> torch.Tensor:
        """(n,M) bool: the patch centres (n,M,2 xy on the 1/4 grid) of keyframes ``ordinals`` lie on valid cells."""

        height, width = self.grid_hw
        xy = centres.detach().float().cpu().round().long()
        x = xy[..., 0].clamp(0, width - 1)
        y = xy[..., 1].clamp(0, height - 1)
        static = torch.stack([self.frames[int(ordinal)].grid[y[row], x[row]] for row, ordinal in enumerate(ordinals)])
        self.map_points_excluded += int((~static).sum().item())
        return static

    def record(self) -> dict[str, Any]:
        ordered = [self.frames[index] for index in sorted(self.frames)]
        fractions = [item.dynamic_fraction for item in ordered]
        fallback = [index for index in sorted(self.frames) if not self.frames[index].restricted]
        return {
            "enabled": True,
            "source": "vipe_track_anything_frame_mask",
            "validity_rule": "bilinear resize of frame.mask to the DPVO 1/4 grid, > 0.9 (ViPE's SLAM rule)",
            "centroid_grid_hw": list(self.grid_hw),
            "min_valid_centroid_cells": self.patches_per_frame,
            "frames": len(ordered),
            "dynamic_pixel_fraction_mean": float(np.mean(fractions)) if fractions else None,
            "dynamic_pixel_fraction_max": float(np.max(fractions)) if fractions else None,
            "dynamic_pixel_fractions": fractions,
            "fallback_count": len(fallback),
            "fallback_frames": fallback,
            "fallback_reasons": {
                reason: [index for index in fallback if self.frames[index].fallback_reason == reason]
                for reason in (FALLBACK_TOO_FEW_STATIC_CELLS, FALLBACK_METRIC_PRIOR)
            },
            "metric_prior_miss_tolerance": METRIC_PRIOR_MISS_TOLERANCE,
            "metric_mask_excludes_dynamic": True,
            "slam_map_points_excluded": self.map_points_excluded,
        }


class CentroidMaskedTransport:
    """Hands each frame's centroid grid to the DINO patchifier for the one DPVO call of that frame."""

    def __init__(self, slam: Any, patchifier: Any, state: DynamicMaskTracking, ops: Any) -> None:
        self.slam = slam
        self.patchifier = patchifier
        self.state = state
        self.ops = ops

    def __call__(self, timestamp: int, image: torch.Tensor, intrinsics: torch.Tensor, **kwargs: Any) -> Any:
        grid = self.state.take(int(timestamp))
        if grid is not None:
            # The grid may come from the producer's CUDA stream (see depth_producer).
            self.ops.record_stream(grid, self.ops.current_stream())
        self.patchifier.centroid_valid = grid
        try:
            return self.slam(timestamp, image, intrinsics, **kwargs)
        finally:
            self.patchifier.centroid_valid = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.slam, name)
