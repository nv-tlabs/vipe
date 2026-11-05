# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViPE's dynamic-object masking and VDA depth fusion with the DPVO backend (CPU).

Off (the default) is bit-identical to the pipeline without these features; on,
the masks follow ViPE's rules (TrackAnything in the init stage, bilinear > 0.9
validity, no BA contribution, no metric prior, no map point on a dynamic pixel)
and ``post.depth_align_model`` runs ViPE's own ``AdaptiveDepthProcessor``.
"""

from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from dpvo_fakes import FakeDPVO, FakeOps
from dynamic_fakes import (
    CpuTorch,
    MaskedStream,
    MatrixSE3,
    altcorr,
    dpvo_images,
    dynamic_box,
    make_patchifier,
)
from metrictrail_fakes import FakeStream
from pydantic import ValidationError

import vipe.metrictrail.dynamic as dynamic_module
import vipe.metrictrail.focal_refine.refine as refine_module
import vipe.metrictrail.slam as slam_module
import vipe.pipeline.processors as processors_module
import vipe.slam.interface as interface_module
from vipe import get_config_path
from vipe.config import parse_typed_config
from vipe.metrictrail.config import MetricTrailInstanceConfig, MetricTrailPipelineConfig, MetricTrailPostConfig
from vipe.metrictrail.dino_fusion import resize_dino_to_dpvo_raster
from vipe.metrictrail.dynamic import (
    DynamicMaskTracking,
    InstanceMasks,
    centroid_valid_grid,
    sample_valid_centroids,
)
from vipe.metrictrail.pipeline import MetricTrailPipeline, SharedIntrinsicsProcessor, record_path
from vipe.metrictrail.slam import DPVOSLAMOutput, DPVOSLAMSystem, slam_map_from_keyframe_points
from vipe.pipeline import make_pipeline
from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from vipe.streams.base import FrameAttribute, StreamProcessor, VideoFrame
from vipe.utils import io

K4 = np.asarray([30.0, 30.0, 20.0, 12.0], dtype=np.float32)


# ------------------------------------------------------------ patchifier (off)


def reference_forward_1cdc24c(
    self, images, patches_per_image=80, disps=None, centroid_sel_strat="RANDOM", return_color=False
):
    """``DinoAugmentedPatchifier.forward`` at 1cdc24c (before masking existed), verbatim but for ``super()``."""

    base = super(type(self), self).forward(
        images,
        patches_per_image=patches_per_image,
        disps=disps,
        centroid_sel_strat=centroid_sel_strat,
        return_color=return_color,
    )
    if return_color:
        fmap, base_gmap, imap, patches, index, color = base
    else:
        fmap, base_gmap, imap, patches, index = base
    # gmap must be resampled after matching features are augmented.
    del base_gmap

    batch, frames, _, height, width = fmap.shape
    dino_raw = self.dino(images)
    dino_low = self.dino_projection(dino_raw.reshape(-1, dino_raw.shape[2], dino_raw.shape[3], dino_raw.shape[4]))
    dino_low = resize_dino_to_dpvo_raster(
        dino_low,
        original_hw=(images.shape[-2], images.shape[-1]),
        target_hw=(height, width),
        foundation_patch_size=self.dino.patch_size,
    ).reshape(batch, frames, -1, height, width)
    split = list(dino_low.split(self.adapter_dim, dim=2))
    dino_match = split.pop(0) if self.fuse_matching else None
    dino_context = split.pop(0) if self.fuse_context else None

    if self.match_fusion is not None:
        assert dino_match is not None
        match_delta = self.match_fusion(dino_match.reshape(batch * frames, -1, height, width)).reshape(
            batch, frames, 128, height, width
        )
        fmap = fmap + match_delta

    center = self.patch_size // 2
    coordinates = self._snapshot_patch_coordinates(
        patches,
        frames=frames,
        patches_per_image=patches_per_image,
        center=center,
    )
    gmap = altcorr.patchify(fmap[0], coordinates, center).view(batch, -1, 128, self.patch_size, self.patch_size)
    if self.context_fusion is not None:
        assert dino_context is not None
        dino_patches = altcorr.patchify(dino_context[0], coordinates, 0).view(batch, -1, self.adapter_dim)
        context = imap[..., 0, 0]
        context_delta = self.context_fusion(dino_patches)
        imap = (context + context_delta).unsqueeze(-1).unsqueeze(-1)

    if return_color:
        return fmap, gmap, imap, patches, index, color
    return fmap, gmap, imap, patches, index


def _run(patchifier, call, *, seed: int, strategy: str, color: bool, count: int = 7):
    images = dpvo_images()
    disps = torch.rand(1, 1, 8, 12, generator=torch.Generator().manual_seed(5))
    torch.manual_seed(seed)
    with torch.no_grad():
        out = call(
            patchifier, images, patches_per_image=count, disps=disps, centroid_sel_strat=strategy, return_color=color
        )
    return out, torch.get_rng_state()


@pytest.mark.parametrize("strategy", ["RANDOM", "GRADIENT_BIAS"])
@pytest.mark.parametrize("color", [False, True])
def test_off_patchifier_is_bit_identical_to_the_base(strategy: str, color: bool) -> None:
    patchifier = make_patchifier()
    assert patchifier.centroid_valid is None
    new, new_rng = _run(patchifier, type(patchifier).__call__, seed=1234, strategy=strategy, color=color)
    ref, ref_rng = _run(patchifier, reference_forward_1cdc24c, seed=1234, strategy=strategy, color=color)
    assert len(new) == len(ref) and all(torch.equal(a, b) for a, b in zip(new, ref))
    assert torch.equal(new_rng, ref_rng)  # same torch RNG consumption
    if strategy == "RANDOM":
        # The centroids are DPVO's own two draws.
        torch.manual_seed(1234)
        x = torch.randint(1, 12 - 1, size=[1, 7])
        y = torch.randint(1, 8 - 1, size=[1, 7])
        assert torch.equal(new[3][0, :, :2, 1, 1], torch.stack([x, y], dim=-1).float()[0])


# --------------------------------------------------------- patchifier (masked)


def test_vipe_validity_rule_on_the_dpvo_grid() -> None:
    mask = torch.ones(24, 40, dtype=torch.bool)
    mask[5:13, 9:22] = False
    grid = centroid_valid_grid(mask, (0, 0, 0, 0), (6, 10))
    # ViPE's SLAMSystem._precompute_features, on DPVO's 1/4 grid instead of ViPE's 1/8.
    vipe = F.interpolate(mask[None, None].float(), (6, 10), mode="bilinear")[0, 0] > 0.9
    assert torch.equal(grid, vipe)
    bilinear = F.interpolate(mask[None, None].float(), (6, 10), mode="bilinear")[0, 0]
    partial = (bilinear > 0.0) & (bilinear < 1.0)
    assert partial.any() and not grid[partial & (bilinear <= 0.9)].any()
    # Padded raster: the mask goes through the transport like the image (reflect), then to the 1/4 grid.
    padded = centroid_valid_grid(mask[:22, :38], (1, 1, 1, 1), (6, 10))
    transported = F.pad(mask[:22, :38][None, None].float(), (1, 1, 1, 1), mode="reflect")
    assert torch.equal(padded, F.interpolate(transported, (6, 10), mode="bilinear")[0, 0] > 0.9)


def test_masked_patchifier_samples_only_valid_centroids() -> None:
    patchifier = make_patchifier()
    valid = torch.zeros(8, 12, dtype=torch.bool)
    valid[2:5, 3:6] = True
    patchifier.centroid_valid = valid
    for strategy in ("RANDOM", "GRADIENT_BIAS"):
        torch.manual_seed(3)
        with torch.no_grad():
            fmap, gmap, imap, patches, index, color = patchifier(
                dpvo_images(), patches_per_image=20, centroid_sel_strat=strategy, return_color=True
            )
        xy = patches[0, :, :2, 1, 1].long()
        assert xy.shape == (20, 2) and bool(valid[xy[:, 1], xy[:, 0]].all())
        assert gmap.shape == (1, 20, 128, 3, 3) and imap.shape == (1, 20, 16, 1, 1) and color.shape == (1, 20, 3)
        assert torch.equal(index, torch.zeros(20, dtype=torch.long))


def test_masked_forward_is_the_vendored_forward_for_the_same_centroids(monkeypatch: pytest.MonkeyPatch) -> None:
    patchifier = make_patchifier()
    base, _ = _run(patchifier, type(patchifier).__call__, seed=11, strategy="RANDOM", color=True)

    def dpvo_draw(valid, frames, count):
        # DPVO's own draw (what the base forward consumed after seed 11).
        return torch.randint(1, 12 - 1, size=[frames, count]), torch.randint(1, 8 - 1, size=[frames, count])

    monkeypatch.setattr(dynamic_module, "sample_valid_centroids", dpvo_draw)
    patchifier.centroid_valid = torch.ones(8, 12, dtype=torch.bool)
    masked, _ = _run(patchifier, type(patchifier).__call__, seed=11, strategy="RANDOM", color=True)
    assert all(torch.equal(a, b) for a, b in zip(base, masked))


def test_rejection_sampling_is_uniform_over_the_valid_interior() -> None:
    valid = torch.zeros(8, 12, dtype=torch.bool)
    valid[0, :] = True  # border row: never a DPVO centroid position
    valid[3, 2:9] = True
    valid[6, 10] = True
    torch.manual_seed(0)
    x, y = sample_valid_centroids(valid, 2, 2000)
    assert x.shape == y.shape == (2, 2000) and x.dtype == torch.int64
    assert bool(valid[y, x].all()) and bool((y >= 1).all())
    counts = torch.bincount((y * 12 + x).reshape(-1), minlength=96).float()
    hit = counts[counts > 0]
    assert len(hit) == 8 and (hit.max() / hit.min()) < 1.5
    with pytest.raises(ValueError, match="no valid centroid"):
        sample_valid_centroids(torch.zeros(8, 12, dtype=torch.bool), 1, 5)


# ------------------------------------------------------------- tracker (masked)


class CapturingDPVO(FakeDPVO):
    """FakeDPVO that also records the centroid grid the patchifier holds during each call."""

    def __init__(self, network: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.network = network
        self.grids: list[torch.Tensor | None] = []

    def __call__(self, tstamp, image, intrinsics, metric_depth=None, metric_mask=None):
        grid = self.network.patchify.centroid_valid
        self.grids.append(None if grid is None else grid.clone())
        return super().__call__(tstamp, image, intrinsics, metric_depth=metric_depth, metric_mask=metric_mask)


class FakeMetricDepth(DepthEstimationModel):
    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        assert src.rgb is not None
        depth = 1.0 + src.rgb.mean(dim=-1)
        depth[0, 0] = 1000.0  # outside the 0.2-500 m validity range
        return DepthEstimationResult(metric_depth=depth)


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch):
    built: list[CapturingDPVO] = []
    network = SimpleNamespace(patchify=SimpleNamespace(centroid_valid=None))

    def make_dpvo(cfg, net, ht, wd, viz=False):
        dpvo = CapturingDPVO(net)
        built.append(dpvo)
        return dpvo

    fake_module = types.ModuleType("vipe.metrictrail.dpvo.dpvo")
    fake_module.DPVO = type(
        "DPVO", (), {"__new__": lambda cls, *a, **k: make_dpvo(*a, **k), "__call__": FakeDPVO.__call__}
    )
    monkeypatch.setitem(sys.modules, "vipe.metrictrail.dpvo.dpvo", fake_module)
    cfg = SimpleNamespace(
        METRIC_DEPTH_PRIOR_WARMUP=12,
        METRIC_GLOBAL_BA=False,
        PATCHES_PER_FRAME=8,
        BUFFER_SIZE=8192,
        MIXED_PRECISION=True,
    )
    monkeypatch.setattr(slam_module, "resolve_dpvo_config", lambda config, **_: (cfg, {"BUFFER_SIZE": 8192}))
    maps: list[Any] = []
    monkeypatch.setattr(slam_module, "dpvo_slam_map", lambda dpvo, patch_static=None: maps.append(patch_static))
    preset = yaml.safe_load((get_config_path() / "slam" / "metrictrail_local.yaml").read_text())
    preset["keyframe_depth"] = "fake-metric"
    system = DPVOSLAMSystem(torch.device("cpu"), preset)
    system.ops = FakeOps()  # type: ignore[assignment]
    depth = FakeMetricDepth()
    monkeypatch.setattr(system, "depth_model", lambda: depth)
    monkeypatch.setattr(system, "dpvo_network", lambda: (network, {"dpvo": {"sha256": "fake"}}))
    return SimpleNamespace(system=system, built=built, network=network, maps=maps)


def _mostly_dynamic_at(fallback: int):
    def mask_fn(hw: tuple[int, int], index: int) -> torch.Tensor:
        if index == fallback:
            mask = torch.zeros(hw, dtype=torch.bool)
            mask[:6, :8] = True  # a few static pixels: fewer valid cells than patches per frame
            return mask
        return dynamic_box(hw, index)

    return mask_fn


@pytest.mark.parametrize("depth_queue", [0, 4])
def test_masked_tracking_restricts_centroids_and_the_metric_mask(tracker, depth_queue: int) -> None:
    tracker.system.config.depth_queue = depth_queue
    stream = MaskedStream(6, hw=(24, 40), mask_fn=_mostly_dynamic_at(2))
    output = tracker.system.run(stream, intrinsics=K4)
    dpvo = tracker.built[0]
    assert [call["tstamp"] for call in dpvo.calls] == list(range(6))
    record = output.record["dynamic_mask"]
    for index, call in enumerate(dpvo.calls):
        static = stream.mask_fn((24, 40), index)
        depth_valid = torch.ones(24, 40, dtype=torch.bool)
        depth_valid[0, 0] = False
        grid = F.interpolate(static[None, None].float(), (6, 10), mode="bilinear")[0, 0] > 0.9
        if index == 2:
            # Fallback: unrestricted centroids and the depth mask as without masking.
            assert dpvo.grids[index] is None
            assert torch.equal(call["metric_mask"], depth_valid)
        else:
            assert torch.equal(dpvo.grids[index], grid)
            assert torch.equal(call["metric_mask"], depth_valid & static)
        assert record["dynamic_pixel_fractions"][index] == pytest.approx(1.0 - static.float().mean().item())
    # The grid is handed in for one call only.
    assert tracker.network.patchify.centroid_valid is None
    assert record["enabled"] is True and record["fallback_count"] == 1 and record["fallback_frames"] == [2]
    assert record["centroid_grid_hw"] == [6, 10] and record["min_valid_centroid_cells"] == 8
    assert tracker.maps[0] is not None  # the map export gets the static-patch filter
    if depth_queue:
        # Grids made on the producer stream are protected on DPVO's stream.
        # Image, depth and mask of every frame, plus the grid of the 5 restricted frames.
        assert len(tracker.system.ops.record_stream_calls) == 3 * 6 + 5


def test_masked_tracking_on_a_padded_raster(tracker) -> None:
    stream = MaskedStream(3, hw=(26, 42))
    output = tracker.system.run(stream, intrinsics=K4)
    call = tracker.built[0].calls[1]
    static = stream.mask_fn((26, 42), 1)
    grid = centroid_valid_grid(static, (1, 1, 1, 1), (7, 11))
    assert torch.equal(tracker.built[0].grids[1], grid)
    assert call["metric_mask"].shape == (28, 44) and not call["metric_mask"][0].any()
    depth_valid = torch.ones(26, 42, dtype=torch.bool)
    depth_valid[0, 0] = False
    assert torch.equal(call["metric_mask"][1:-1, 1:-1], depth_valid & static)
    assert output.record["dynamic_mask"]["centroid_grid_hw"] == [7, 11]


def test_off_tracking_hands_no_grid_and_records_off(tracker) -> None:
    output = tracker.system.run(FakeStream(4, hw=(24, 40)), intrinsics=K4)
    assert tracker.built[0].grids == [None] * 4
    assert output.record["dynamic_mask"] == {"enabled": False}
    assert tracker.maps == [None]  # dpvo_slam_map(dpvo) as before


def test_frames_without_a_mask_are_refused_when_masking_is_on(tracker) -> None:
    class Gappy(MaskedStream):
        def __iter__(self):
            for index, frame in enumerate(super().__iter__()):
                if index == 1:
                    frame.mask = None
                yield frame

    with pytest.raises(ValueError, match="carries no ViPE mask"):
        tracker.system.run(Gappy(3, hw=(24, 40)), intrinsics=K4)


# ------------------------------------------------------------------- map export


def test_masked_patch_points_are_excluded_from_the_slam_map(monkeypatch: pytest.MonkeyPatch) -> None:
    def point_cloud(poses, patches, intrinsics, ix):
        x, y, d = patches.permute(0, 1, 3, 4, 2).unbind(-1)
        return torch.stack([x, y, torch.ones_like(x), d], dim=-1)  # xyz = (x, y, 1) / d

    monkeypatch.setitem(sys.modules, "vipe.metrictrail.dpvo.projective_ops", SimpleNamespace(point_cloud=point_cloud))
    monkeypatch.setitem(sys.modules, "vipe.metrictrail.dpvo.lietorch", SimpleNamespace(SE3=lambda data: data))
    n, patches_per_frame = 2, 4
    centres = torch.tensor([[2, 2], [5, 3], [8, 4], [3, 1]] * n, dtype=torch.float32)
    patches = torch.zeros(4 * patches_per_frame, 3, 3, 3)  # DPVO's buffer is larger than the m live patches
    patches[: n * patches_per_frame, 0] = centres[:, 0, None, None]
    patches[: n * patches_per_frame, 1] = centres[:, 1, None, None]
    patches[: n * patches_per_frame, 2] = 0.5
    dpvo = SimpleNamespace(
        n=n,
        m=n * patches_per_frame,
        M=patches_per_frame,
        poses=torch.zeros(n, 7),
        patches=patches[None],
        intrinsics=torch.zeros(1, n, 4),
        ix=torch.zeros(4 * patches_per_frame, dtype=torch.long),
        pg=SimpleNamespace(colors_=torch.full((n, patches_per_frame, 3), 255, dtype=torch.uint8), tstamps_=[0, 3]),
    )
    state = DynamicMaskTracking(
        image_hw=(24, 40), padding_ltrb=(0, 0, 0, 0), transport_hw=(24, 40), patches_per_frame=1
    )
    static = torch.ones(24, 40, dtype=torch.bool)
    for index in range(4):
        mask = static.clone()
        if index == 3:
            mask[8:16, 16:28] = False  # covers grid cells (2..3, 4..6): the centre (5, 3) of keyframe 3
        state.prepare(index, VideoFrame(raw_frame_idx=index, rgb=torch.zeros(24, 40, 3), mask=mask), static.clone())
    assert not bool(state.frames[3].grid[3, 5]) and bool(state.frames[0].grid[3, 5])

    full = slam_module.dpvo_slam_map(dpvo)
    masked = slam_module.dpvo_slam_map(dpvo, state.static_patches)
    assert full.dense_disp_packinfo.tolist() == [[[0, 4]], [[4, 4]]]
    assert masked.dense_disp_packinfo.tolist() == [[[0, 4]], [[4, 3]]]
    assert masked.dense_disp_frame_inds == [0, 3]
    xyz, _ = masked.get_dense_disp_pcd(1)
    assert torch.equal(xyz, torch.tensor([[2.0, 2.0, 1.0], [8.0, 4.0, 1.0], [3.0, 1.0, 1.0]]) / 0.5)
    assert state.record()["slam_map_points_excluded"] == 1


def test_fallback_frames_are_counted_and_left_unmasked() -> None:
    state = DynamicMaskTracking(
        image_hw=(24, 40), padding_ltrb=(0, 0, 0, 0), transport_hw=(24, 40), patches_per_frame=8
    )
    depth_valid = torch.ones(24, 40, dtype=torch.bool)
    for index, mask in enumerate((dynamic_box((24, 40), 0), _mostly_dynamic_at(1)((24, 40), 1))):
        out = state.prepare(index, VideoFrame(raw_frame_idx=index, rgb=torch.zeros(24, 40, 3), mask=mask), depth_valid)
        assert torch.equal(out, depth_valid & mask if index == 0 else depth_valid)
    assert state.take(0) is not None and state.take(1) is None
    record = state.record()
    assert record["fallback_count"] == 1 and record["fallback_frames"] == [1] and record["frames"] == 2
    assert record["fallback_reasons"] == {"too_few_static_cells": [1], "static_cells_lack_metric_depth": []}
    assert state.frames[1].valid_cells < 8 <= state.frames[0].valid_cells


def _sky_only_static(hw: tuple[int, int], index: int) -> torch.Tensor:
    """ViPE mask whose static pixels are only the sky rows (add_sky): a dynamic object fills the rest."""

    mask = torch.zeros(hw, dtype=torch.bool)
    mask[:10] = True
    return mask


def test_static_cells_without_metric_depth_fall_back_instead_of_aborting() -> None:
    # DPVO rejects a frame in which no sampled patch has a metric prior (and the run aborts), so a
    # masked frame must not leave its centroids only where there is no metric depth.
    state = DynamicMaskTracking(
        image_hw=(24, 40), padding_ltrb=(0, 0, 0, 0), transport_hw=(24, 40), patches_per_frame=8
    )
    no_sky_depth = torch.ones(24, 40, dtype=torch.bool)
    no_sky_depth[:10] = False
    band_depth = torch.zeros(24, 40, dtype=torch.bool)
    band_depth[12:16] = True  # DPVO samples row 13 (grid row 3) only
    bottom_object = torch.ones(24, 40, dtype=torch.bool)
    bottom_object[16:] = False
    cases = [
        # (static mask, metric mask, restricted): sky-only static region -> fallback (masked: certain rejection)
        (_sky_only_static((24, 40), 0), no_sky_depth, False),
        # a dynamic box in a scene with metric depth -> masked as before
        (dynamic_box((24, 40), 1), no_sky_depth, True),
        # sparse depth already without masking; masking makes a rejection less likely -> masked
        (bottom_object, band_depth, True),
    ]
    for index, (mask, metric_mask, restricted) in enumerate(cases):
        frame = VideoFrame(raw_frame_idx=index, rgb=torch.zeros(24, 40, 3), mask=mask)
        out = state.prepare(index, frame, metric_mask)
        assert state.frames[index].valid_cells >= 8
        assert torch.equal(out, metric_mask & mask if restricted else metric_mask)
        assert (state.take(index) is not None) is restricted
    record = state.record()
    assert record["fallback_frames"] == [0]
    assert record["fallback_reasons"] == {"too_few_static_cells": [], "static_cells_lack_metric_depth": [0]}


def test_masked_tracking_of_a_sky_only_static_frame_completes(tracker, monkeypatch: pytest.MonkeyPatch) -> None:
    class NoSkyDepth(FakeMetricDepth):
        def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
            depth = super().estimate(src).metric_depth
            assert depth is not None
            depth[:10] = float("inf")  # no metric depth on the sky (MoGe-2's mask)
            return DepthEstimationResult(metric_depth=depth)

    depth = NoSkyDepth()
    monkeypatch.setattr(tracker.system, "depth_model", lambda: depth)

    def mask_fn(hw: tuple[int, int], index: int) -> torch.Tensor:
        return _sky_only_static(hw, index) if index == 1 else dynamic_box(hw, index)

    output = tracker.system.run(MaskedStream(4, hw=(24, 40), mask_fn=mask_fn), intrinsics=K4)
    dpvo = tracker.built[0]
    assert [dpvo.grids[index] is None for index in range(4)] == [False, True, False, False]
    depth_valid = torch.ones(24, 40, dtype=torch.bool)
    depth_valid[:10] = False
    assert torch.equal(dpvo.calls[1]["metric_mask"], depth_valid)  # unmasked: DPVO keeps a metric prior
    record = output.record["dynamic_mask"]
    assert record["fallback_frames"] == [1] and record["fallback_reasons"]["static_cells_lack_metric_depth"] == [1]


# --------------------------------------------------------- init stage (pipeline)


class FakeTrackAnything(StreamProcessor):
    """ViPE's TrackAnythingProcessor interface: a person box (id 1) and sky rows (id 2); ViPE's mask rule."""

    built: list[FakeTrackAnything] = []

    def __init__(self, mask_phrases, add_sky, sam_run_gap=30, mask_expand=5, model_cache=None) -> None:
        self.args = {"phrases": list(mask_phrases), "add_sky": add_sky, "sam_run_gap": sam_run_gap}
        self.model_cache = model_cache
        self.processed: list[int] = []
        FakeTrackAnything.built.append(self)

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert frame.intrinsics is not None, "SharedK must run before TrackAnything (init-stage order)"
        self.processed.append(frame_idx)
        height, width = frame.size()
        instance = torch.zeros(height, width, dtype=torch.uint8)
        instance[:2] = 2
        instance[height // 3 : 2 * height // 3, width // 4 + frame_idx : width // 2 + frame_idx] = 1
        frame.instance = instance
        frame.instance_phrases = {1: "person", 2: "sky"}
        frame.mask = (instance == 0) | (instance == 2)
        return frame


class SharedKWithRecord(SharedIntrinsicsProcessor):
    def __init__(self, k4: np.ndarray) -> None:
        super().__init__(k4)
        self.record = {"calibrated_K_fx_fy_cx_cy": np.asarray(k4).tolist()}


def _world_points(count_hw: tuple[int, int], image_hw: tuple[int, int], k4: np.ndarray, z: float) -> torch.Tensor:
    """Points that frame 0 (identity pose) sees at pixel centres of a regular grid, depth ``z``."""

    fx, fy, cx, cy = (float(item) for item in k4)
    vs = torch.linspace(2, image_hw[0] - 3, count_hw[0]).round() + 0.5
    us = torch.linspace(2, image_hw[1] - 3, count_hw[1]).round() + 0.5
    v, u = torch.meshgrid(vs, us, indexing="ij")
    return torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, torch.full_like(u, z)], dim=-1).reshape(-1, 3)


class FakeMaskSlam:
    """DPVOSLAMSystem.run stand-in: checks the init-stage frames, returns a DPVO-shaped output."""

    def __init__(self, image_hw: tuple[int, int], k4: np.ndarray) -> None:
        self.image_hw = image_hw
        self.k4 = k4
        self.seen_masks: list[list[torch.Tensor]] = []
        self.dense_map: list[bool] = []

    def run(self, stream, *, intrinsics, name, depth_output=None, dense_map=False):
        # dense_map: the pipeline asks for the dense keyframe map exactly when depth fusion is on
        # (vipe.metrictrail.fused_map); this fake's patch map stands for either.
        self.dense_map.append(dense_map)
        masks = []
        for index, frame in enumerate(stream):
            assert frame.intrinsics is not None and frame.mask is not None and frame.instance is not None
            masks.append(frame.mask.clone())
        self.seen_masks.append(masks)
        count = len(masks)
        c2w = np.tile(np.eye(4), (count, 1, 1))
        c2w[:, 0, 3] = 0.05 * np.arange(count)
        points = _world_points((10, 12), self.image_hw, self.k4, z=2.5)
        keyframes = [0, 2, 4]
        # Keyframe k holds a third of the points (DPVO's patches of that keyframe).
        xyz = torch.stack([points[k::3][: len(points) // 3] for k in range(len(keyframes))])
        slam_map = slam_map_from_keyframe_points(
            xyz, torch.rand(*xyz.shape), torch.ones(xyz.shape[:2], dtype=torch.bool), np.asarray(keyframes)
        )
        return DPVOSLAMOutput(
            trajectory=MatrixSE3(torch.from_numpy(c2w).float()),  # type: ignore[arg-type]
            intrinsics=torch.from_numpy(np.asarray(intrinsics, dtype=np.float32))[None],
            rig=MatrixSE3(torch.eye(4)[None]),  # type: ignore[arg-type]
            slam_map=slam_map,
            c2w=c2w,
            source_frame_ids=np.arange(count),
            record={
                "tracking_status": "tracked",
                "runtime_seconds": 1.0,
                "dynamic_mask": {"enabled": True, "fallback_count": 0, "dynamic_pixel_fraction_mean": 0.1},
            },
        )


IMAGE_HW = (40, 60)
PIPELINE_K4 = np.asarray([50.0, 50.0, 30.0, 20.0], dtype=np.float32)


@pytest.fixture
def cpu_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """ViPE's cached output stream and AdaptiveDepthProcessor move frames to CUDA; keep them on CPU here."""

    monkeypatch.setattr(VideoFrame, "cuda", lambda self: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(interface_module, "torch", CpuTorch())


def _pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> tuple[MetricTrailPipeline, FakeMaskSlam]:
    overrides = [
        "pipeline=metrictrail",
        f"streams.base_path={tmp_path / 'clip.mp4'}",
        f"pipeline.output.path={tmp_path / 'out'}",
        # Synthetic pre-prepared frames, no metric depth written by the fake tracker.
        "pipeline.init.input_raster.policy=unchanged",
        "pipeline.output.save_metric_depth=off",
        *extra,
    ]
    pipeline = make_pipeline(parse_typed_config("default", overrides).pipeline)
    assert isinstance(pipeline, MetricTrailPipeline)
    pipeline.device = torch.device("cpu")
    slam = FakeMaskSlam(IMAGE_HW, PIPELINE_K4)
    monkeypatch.setattr(pipeline, "calibrate", lambda stream: SharedKWithRecord(PIPELINE_K4))
    monkeypatch.setattr(pipeline, "make_slam", lambda: slam)
    FakeTrackAnything.built = []
    monkeypatch.setattr(processors_module, "TrackAnythingProcessor", FakeTrackAnything)
    return pipeline, slam


@pytest.mark.parametrize("accepted", [False, True])
def test_init_stage_runs_track_anything_once_for_slam_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_frames: None, accepted: bool
) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch, "pipeline.init.instance={}", "pipeline.focal_refine.enabled=true")
    monkeypatch.setattr(
        refine_module, "run_focal_refinement", lambda config, **kwargs: {"accepted": accepted, "focal": 55.0}
    )
    pipeline.run(FakeStream(5, hw=IMAGE_HW))
    [track] = FakeTrackAnything.built
    # ViPE's own arguments (init.instance defaults = ViPE's default.yaml), one run over the video.
    assert track.args == {
        "phrases": ["person", "animal", "vehicle", "ball", "balloon", "gun", "pet", "car", "bus"],
        "add_sky": True,
        "sam_run_gap": 60,
    }
    assert track.model_cache is pipeline.model_cache
    assert track.processed == list(range(5))
    # SLAM saw the masks on every frame (the re-track replays the same ones).
    assert len(slam.seen_masks) == (2 if accepted else 1)
    for masks in slam.seen_masks[1:]:
        assert all(torch.equal(a, b) for a, b in zip(masks, slam.seen_masks[0]))
    artifact = io.ArtifactPath(tmp_path / "out", "clip")
    with zipfile.ZipFile(artifact.mask_path) as archive:
        assert archive.namelist() == [f"{index:05d}.png" for index in range(5)]
        first = cv2.imdecode(np.frombuffer(archive.read("00001.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    assert int(first.max()) == 2 and int((first == 1).sum()) > 0
    assert artifact.mask_phrase_path.read_text() == "1: person\n2: sky\n"
    assert not artifact.depth_path.exists()  # post.depth_align_model is null: pose-only
    record = json.loads(record_path(artifact).read_text())
    assert record["dynamic_mask"]["enabled"] is True and record["depth_align_model"] is None
    assert record["pipeline"]["init"]["instance"]["kf_gap_sec"] == 2.0


def test_instance_masks_replay_and_refuse_an_interrupted_pass() -> None:
    FakeTrackAnything.built = []
    masks = InstanceMasks(FakeTrackAnything(["person"], add_sky=True))
    stream = FakeStream(3, hw=(12, 16))

    def frames():
        for frame in stream:
            frame.intrinsics = torch.ones(4)
            yield frame

    first = list(masks.update_iterator(frames(), 0))
    again = list(masks.update_iterator(frames(), 0))
    assert FakeTrackAnything.built[0].processed == [0, 1, 2]
    assert all(torch.equal(a.mask, b.mask) and torch.equal(a.instance, b.instance) for a, b in zip(first, again))
    assert again[1].instance_phrases == {1: "person", 2: "sky"}
    broken = InstanceMasks(FakeTrackAnything(["person"], add_sky=True))
    iterator = broken.update_iterator(frames(), 0)
    next(iterator)
    with pytest.raises(RuntimeError, match="interrupted"):
        broken.update_iterator(frames(), 0)


# ------------------------------------------------------------- depth fusion (VDA)


def _true_depth(hw: tuple[int, int]) -> torch.Tensor:
    return 2.0 + torch.linspace(0.0, 1.0, hw[1])[None].expand(hw).clone() + 0.01 * torch.arange(hw[0])[:, None]


class FakePromptModel(DepthEstimationModel):
    prompts: list[torch.Tensor] = []

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        assert src.prompt_metric_depth is not None and src.rgb is not None
        FakePromptModel.prompts.append(src.prompt_metric_depth.clone())
        return DepthEstimationResult(metric_depth=_true_depth(tuple(src.rgb.shape[:2])))


class FakeVDA(DepthEstimationModel):
    built: list[str] = []

    def __init__(self, model: str = "vits") -> None:
        FakeVDA.built.append(model)

    @property
    def depth_type(self) -> DepthType:
        return DepthType.AFFINE_DISP

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        assert src.video_frame_list is not None
        inv = torch.stack([2.0 / _true_depth(frame.shape[:2]) + 0.1 for frame in src.video_frame_list])
        return DepthEstimationResult(relative_inv_depth=inv)


class FakeMetric(DepthEstimationModel):
    calls = 0

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        FakeMetric.calls += 1
        raise AssertionError("the DPVO map covers the frames; ViPE must take the SLAM-prompted path")


def _expected_prompt(c2w: np.ndarray, points: torch.Tensor, k4: np.ndarray, hw: tuple[int, int]) -> torch.Tensor:
    """ViPE's ``project_map`` (no infill) recomputed with numpy for frame pose ``c2w``."""

    w2c = np.linalg.inv(c2w)
    cam = points.double().numpy() @ w2c[:3, :3].T + w2c[:3, 3]
    u = k4[0] * cam[:, 0] / cam[:, 2] + k4[2]
    v = k4[1] * cam[:, 1] / cam[:, 2] + k4[3]
    keep = (u > 0) & (u < hw[1]) & (v > 0) & (v < hw[0]) & (cam[:, 2] > 0)
    out = torch.zeros(hw)
    out[torch.from_numpy(np.floor(v[keep])).long(), torch.from_numpy(np.floor(u[keep])).long()] = torch.from_numpy(
        cam[keep, 2]
    ).float()
    return out


@pytest.mark.parametrize("masked", [False, True])
def test_vipe_adaptive_depth_runs_on_the_dpvo_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_frames: None, masked: bool
) -> None:
    extra = ["pipeline.post.depth_align_model=adaptive_unidepth-l_svda"]
    if masked:
        extra.append("pipeline.init.instance={}")
    pipeline, slam = _pipeline(tmp_path, monkeypatch, *extra)
    if not masked:
        slam.run = _unmasked(slam.run)  # type: ignore[method-assign]
    FakePromptModel.prompts, FakeVDA.built, FakeMetric.calls = [], [], 0
    made: list[str] = []
    monkeypatch.setattr(processors_module, "PriorDAModel", FakePromptModel)
    monkeypatch.setattr(processors_module, "VideoDepthAnythingDepthModel", FakeVDA)
    monkeypatch.setattr(processors_module, "make_depth_model", lambda name: made.append(name) or FakeMetric())
    count = 7
    pipeline.run(FakeStream(count, hw=IMAGE_HW))
    assert slam.dense_map == [True]  # depth fusion: the dense keyframe map was requested

    # ViPE's own parse of adaptive_unidepth-l_svda.
    assert made == ["unidepth-l"] and FakeVDA.built == ["vits"] and FakeMetric.calls == 0
    assert len(FakePromptModel.prompts) == count
    points = _world_points((10, 12), IMAGE_HW, PIPELINE_K4, z=2.5)
    used = torch.cat([points[k::3][: len(points) // 3] for k in range(3)])
    for index, prompt in enumerate(FakePromptModel.prompts):
        c2w = np.eye(4)
        c2w[0, 3] = 0.05 * index
        expected = _expected_prompt(c2w, used, PIPELINE_K4, IMAGE_HW)
        if masked:
            expected = expected * slam.seen_masks[0][index].float()  # ViPE: depth_map * frame.mask
        assert int((prompt > 0).sum()) > 50
        torch.testing.assert_close(prompt, expected, rtol=1e-5, atol=1e-5)

    artifact = io.ArtifactPath(tmp_path / "out", "clip")
    depths = list(io.read_depth_artifacts(artifact.depth_path))
    assert [index for index, _ in depths] == list(range(count))
    for _, depth in depths:
        # VDA aligned to the SLAM-prompted metric depth: back to the scene depth (half-precision EXR).
        torch.testing.assert_close(depth.float(), _true_depth(IMAGE_HW), rtol=2e-3, atol=2e-3)
    assert artifact.pose_path.is_file() and record_path(artifact).is_file()
    assert json.loads(record_path(artifact).read_text())["depth_align_model"] == "adaptive_unidepth-l_svda"
    assert artifact.mask_path.exists() is masked


def _unmasked(run):
    """FakeMaskSlam.run for a stream without init-stage masks (masking off)."""

    def wrapped(stream, *, intrinsics, name, depth_output=None, dense_map=False):
        frames = list(stream)
        assert all(frame.mask is None and frame.instance is None for frame in frames)

        class Replay:
            def __iter__(self):
                for frame in frames:
                    yield VideoFrame(
                        raw_frame_idx=frame.raw_frame_idx,
                        rgb=frame.rgb,
                        intrinsics=frame.intrinsics,
                        mask=torch.ones(frame.size(), dtype=torch.bool),
                        instance=torch.zeros(frame.size(), dtype=torch.uint8),
                    )

        return run(Replay(), intrinsics=intrinsics, name=name, dense_map=dense_map)

    return wrapped


def test_dpvo_output_without_a_map_is_refused_for_depth_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_frames: None
) -> None:
    pipeline, slam = _pipeline(tmp_path, monkeypatch, "pipeline.post.depth_align_model=adaptive_unidepth-l")
    original = _unmasked(slam.run)

    def no_map(stream, *, intrinsics, name, depth_output=None, dense_map=False):
        assert dense_map  # depth fusion asks for the dense keyframe map
        output = original(stream, intrinsics=intrinsics, name=name)
        output.slam_map = None
        return output

    slam.run = no_map  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="patch map"):
        pipeline.run(FakeStream(3, hw=IMAGE_HW))


# ------------------------------------------------------------------- config


def _vipe_preset(name: str) -> dict:
    return yaml.safe_load((get_config_path() / "pipeline" / f"{name}.yaml").read_text())


def test_defaults_are_off(tmp_path: Path) -> None:
    config = parse_typed_config(
        "default", ["pipeline=metrictrail", f"streams.base_path={tmp_path}", f"pipeline.output.path={tmp_path}"]
    ).pipeline
    assert isinstance(config, MetricTrailPipelineConfig)
    assert config.init.instance is None and config.post.depth_align_model is None


def test_instance_switch_takes_vipe_fields_and_defaults(tmp_path: Path) -> None:
    base = ["pipeline=metrictrail", f"streams.base_path={tmp_path}", f"pipeline.output.path={tmp_path}"]
    vipe_instance = _vipe_preset("default")["init"]["instance"]
    config = parse_typed_config("default", [*base, "pipeline.init.instance={}"]).pipeline
    assert config.init.instance.model_dump() == vipe_instance
    custom = parse_typed_config(
        "default", [*base, "pipeline.init.instance={kf_gap_sec:1.0,phrases:[person,car],add_sky:false}"]
    ).pipeline
    assert custom.init.instance == MetricTrailInstanceConfig(kf_gap_sec=1.0, phrases=["person", "car"], add_sky=False)


def test_depth_align_model_accepts_vipe_adaptive_recipes_and_refuses_mvd(tmp_path: Path) -> None:
    vipe_values = {
        name: _vipe_preset(name)["post"]["depth_align_model"]
        for name in ("default", "no_vda", "static_vda", "lyra", "dav3")
    }
    assert vipe_values == {
        "default": "adaptive_unidepth-l_svda",
        "no_vda": "adaptive_unidepth-l",
        "static_vda": "adaptive_unidepth-l_vda",
        "lyra": "adaptive_moge_vda",
        "dav3": "mvd_dav3",
    }
    for name, value in vipe_values.items():
        if name == "dav3":
            with pytest.raises(ValidationError, match="mvd_.*not supported"):
                MetricTrailPostConfig(depth_align_model=value)
        else:
            assert MetricTrailPostConfig(depth_align_model=value).depth_align_model == value
    for bad in ("unidepth-l", "adaptive", "adaptive_unidepth-l_foo", "adaptive__vda", "dap"):
        with pytest.raises(ValidationError, match="adaptive_<metric>"):
            MetricTrailPostConfig(depth_align_model=bad)
    with pytest.raises(Exception, match="not supported"):
        parse_typed_config(
            "default",
            [
                "pipeline=metrictrail",
                f"streams.base_path={tmp_path}",
                f"pipeline.output.path={tmp_path}",
                "pipeline.post.depth_align_model=mvd_dav3",
            ],
        )
