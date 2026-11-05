# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage B: refine the shared focal from revisit pairs.

``run_stage_b`` takes its paths from the caller, and this module imports
nothing from ViPE so that ``stage_b_cli.py`` can run it in a separate
interpreter that only has RoMa's dependencies.

Stage A estimates the focal from a bundle adjustment whose frames are
consecutive.  When the camera translates forward and barely rotates, a focal
error and a depth/translation error cancel almost exactly, so the adjustment is
happy with the wrong focal and the trajectory drifts.  That is the failure on
VBR, where stage A lands 7.3% high and the local ATE is 33.9 m.

Revisit pairs break the degeneracy: the same place seen after a large heading
change gives an epipolar geometry that a wrong focal cannot explain away.  The
pairs come from stage A's own trajectory -- drifted, but good enough to say
which frames are near each other -- and the fit uses only fundamental matrices
recovered from RGB.

The fit does not assume where the principal point is.  On VBR it is 17.3 px from
the middle of the image, and under forward motion a shift of the centre and a
change of focal produce nearly the same bearings, so the cost has a long shallow
valley in the (focal, centre) plane rather than an isolated minimum.  Slicing
that valley at an assumed centre returns the focal at the slice, displaced along
the valley; minimising over a range of plausible centres at each focal projects
onto the valley floor instead.  Recovering the centre itself is not possible from
these pairs and is not attempted -- it is profiled out, not estimated.

What makes the projection well posed is pairs whose geometry differs, since
their valleys then run in different directions and intersect.  That is why the
heading band admits a wide range rather than isolating one kind of pair.

Everything here is RGB-only.  Sensor intrinsics and ground-truth poses are never
read; the caller supplies stage A's focal and stage A's poses.  Nothing is drawn
from a global random stream: pairs are selected, and the matcher is seeded, from
a hash of the frame indices, so the result does not depend on a seed.

The estimate is accepted only if it passes checks computed from the pairs
themselves: the profile minimum must be interior to the search grid, the
bootstrap must not rail, the answer must not change much when the centre is
assumed rather than profiled out, and the refinement must not move implausibly
far from stage A.  A rejected estimate falls back to stage A, so stage B can help
but not hurt.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class StageBSettings:
    enabled: bool = False
    budget: int = 800
    radius_m: float = 15.0
    min_frame_gap: int = 200
    trajectory_stride: int = 5
    candidate_band_deg: tuple[float, float] = (25.0, 140.0)
    coarse_resolution: int = 560
    # Matching runs once, at the coarse resolution.  A refused fit keeps stage
    # A's focal; there is no second, upsampled attempt.
    upsample_predictions: bool = False
    samples_per_pair: int = 3000
    max_correspondences: int = 150
    min_inliers: int = 100
    ransac_threshold_px: float = 0.6
    grid_relative: tuple[float, float] = (0.60, 1.30)
    max_rail_fraction: float = 0.10
    max_deviation_pct: float = 25.0
    # The two readings of the cost -- centre assumed, centre profiled out -- agree
    # to within 1.3% on the five VBR sequences where the pair geometry pins the
    # focal on its own, and disagree by 26-44% on the one where it does not.  A
    # threshold anywhere between those two populations gives the same decisions.
    max_estimator_disagreement_pct: float = 15.0
    centre_halfwidth_px: float = 40.0
    centre_grid_points: int = 41
    # The cost is a mean over each pair, so correspondences past this buy nothing
    # the focal grid can resolve, and the profile is two-dimensional now.
    profile_points_per_pair: int = 60
    bootstrap_draws: int = 400
    # Wide enough that the pairs entering the fit differ in geometry.  A narrow
    # band admits only one kind of pair, whose focal/centre trade-offs are then
    # nearly parallel and do not intersect.
    fit_band_deg: tuple[float, float] = (25.0, 140.0)
    # Whether a sequence enters stage B is decided per sequence from stage A's
    # own output; see routing.py for the thresholds and what they were measured
    # against.  Setting route_automatically False falls back to `enabled` alone,
    # which is what the ablation configs want.
    route_automatically: bool = True
    # Two readings of the same mined supply.  The count is what stage B can
    # actually spend -- it never uses more than `budget` pairs -- and the
    # per-frame density catches short sequences where every frame trivially
    # counts as a revisit.  A sequence must clear both.
    gate_revisit_count: int = 1000
    gate_revisit_per_frame: float = 0.15
    gate_focal_drift_pct: float = 0.20
    gate_focal_drift_max_pct: float = 2.5


def _pair_rank(pair: tuple[int, int]) -> int:
    """A stable, uniform order on pairs that depends only on the pair itself."""
    digest = hashlib.sha256(f"{pair[0]}:{pair[1]}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def revisit_candidates(c2w: np.ndarray, settings: StageBSettings) -> list[tuple[int, int]]:
    """Frame pairs that stage A's trajectory places near each other.

    The trajectory is drifted, so this is a proposal, not a fact; a false
    proposal costs one match and is then diluted by the rest.
    """
    xyz, forward = c2w[:, :3, 3], c2w[:, :3, 2]
    index = np.arange(0, len(xyz), settings.trajectory_stride)
    points = xyz[index]
    low, high = settings.candidate_band_deg
    pairs: list[tuple[int, int]] = []
    for offset, i in enumerate(index):
        near = index[np.linalg.norm(points - points[offset], axis=1) < settings.radius_m]
        for j in near[near - i > settings.min_frame_gap]:
            cosine = float(np.dot(forward[i], forward[j]))
            heading = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if low <= heading <= high:
                pairs.append((int(i), int(j)))
    if len(pairs) > settings.budget:
        # Spending the budget by position in this list makes the choice depend
        # on how long the list happens to be.  Stage A's focal repeats to within
        # 0.4 percentage points but its trajectory does not, so the candidate
        # count moves by a few hundred between runs, and an even spread over it
        # then lands on almost disjoint pairs: measured across four independent
        # runs the selected sets overlapped 1-13% where the candidate sets they
        # came from overlapped 87-96%.  That is where stage B's run-to-run
        # spread comes from, not the matcher or the fit.
        #
        # Rank each pair by a hash of its own frame indices instead.  A pair is
        # then kept or dropped on its own account, so the selection inherits
        # whatever stability the candidate set has.
        pairs = sorted(sorted(pairs, key=_pair_rank)[: settings.budget])
    return pairs


def load_matcher(roma_repo: Path | None, roma_checkpoints: Path, device: Any, settings: StageBSettings) -> Any:
    import sys

    import torch

    if roma_repo is not None and str(roma_repo) not in sys.path:
        sys.path.insert(0, str(roma_repo))
    torch.set_float32_matmul_precision("highest")
    from romatch import roma_outdoor  # type: ignore[import-not-found]

    return roma_outdoor(
        device=device,
        weights=torch.load(roma_checkpoints / "roma_outdoor.pth", map_location=device),
        dinov2_weights=torch.load(roma_checkpoints / "dinov2_vitl14_pretrain.pth", map_location=device),
        coarse_res=settings.coarse_resolution,
        upsample_res=settings.coarse_resolution,
        # upsample_res == coarse_res makes the refinement pass re-run the same
        # resolution, which costs 40% of stage B and buys nothing.
        upsample_preds=settings.upsample_predictions,
        use_custom_corr=False,
    )


def fundamental_matrices(
    matcher: Any,
    images: Sequence[np.ndarray],
    pairs: Sequence[tuple[int, int]],
    hw: tuple[int, int],
    settings: StageBSettings,
    device: Any,
) -> dict[str, np.ndarray]:
    """RoMa correspondences and a fundamental matrix for each proposed pair.

    Images arrive as CHW arrays because that is what every dataset loader here
    produces; RoMa accepts PIL, which keeps this independent of whether the
    frames are on disk as files or packed in an archive.
    """
    import cv2
    import torch
    from PIL import Image

    def as_image(frame: np.ndarray) -> Image.Image:
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        return Image.fromarray(array[:, :, :3])

    height, width = hw
    matrices, points_a, points_b, counts, kept = [], [], [], [], []
    for i, j in pairs:
        # The matcher draws its correspondences with torch.multinomial, off the
        # global generator.  Seeding that once per run makes every pair's sample
        # depend on the whole history of pairs before it, so a single pair that
        # differs between two runs re-randomises all the rest -- which is why
        # making the pair sets 90% identical did not make the fits agree.
        # Seeding from the pair itself makes a pair's correspondences a function
        # of that pair alone, so shared pairs match identically across runs.
        torch.manual_seed(_pair_rank((i, j)) % (2**31))
        try:
            with torch.inference_mode():
                warp, certainty = matcher.match(as_image(images[i]), as_image(images[j]), device=device)
                sample, _ = matcher.sample(warp, certainty, num=settings.samples_per_pair)
                pa, pb = matcher.to_pixel_coordinates(sample, height, width, height, width)
            pa = pa.detach().cpu().numpy().astype(np.float32)
            pb = pb.detach().cpu().numpy().astype(np.float32)
            matrix, mask = cv2.findFundamentalMat(
                pa,
                pb,
                cv2.USAC_MAGSAC,
                ransacReprojThreshold=settings.ransac_threshold_px,
                confidence=0.9999,
                maxIters=40000,
            )
        except cv2.error:
            continue
        except Exception:
            continue
        if matrix is None or matrix.shape != (3, 3):
            continue
        inlier = mask.ravel() > 0
        if inlier.sum() < settings.min_inliers:
            continue
        a, b = pa[inlier], pb[inlier]
        if len(a) > settings.max_correspondences:
            take = np.linspace(0, len(a) - 1, settings.max_correspondences).astype(int)
            a, b = a[take], b[take]
        matrices.append(matrix.astype(np.float64))
        points_a.append(a.astype(np.float64))
        points_b.append(b.astype(np.float64))
        counts.append(len(a))
        kept.append((i, j))
    if not matrices:
        return {}
    return {
        "F": np.stack(matrices),
        "xa": np.concatenate(points_a),
        "xb": np.concatenate(points_b),
        "counts": np.asarray(counts),
        "pairs": np.asarray(kept),
    }


def pair_headings(bundle: dict[str, np.ndarray], focal: float, hw: tuple[int, int]) -> np.ndarray:
    """Relative rotation of each pair, measured under stage A's focal.

    Only the band edges depend on this, and those are coarse, so an approximate
    focal is sufficient -- which is what lets the band be applied before the
    focal is known.
    """
    import cv2

    height, width = hw
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    F, xa, xb, counts = (bundle["F"], bundle["xa"], bundle["xb"], bundle["counts"])
    headings = np.full(len(F), np.nan)
    offset = 0
    for p in range(len(F)):
        a = xa[offset : offset + counts[p]].astype(np.float32)
        b = xb[offset : offset + counts[p]].astype(np.float32)
        offset += counts[p]
        try:
            E = K.T @ F[p] @ K
            U, S, Vt = np.linalg.svd(E)
            mean = (S[0] + S[1]) / 2
            _, R, _, _ = cv2.recoverPose(U @ np.diag([mean, mean, 0.0]) @ Vt, a, b, K)
            headings[p] = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
        except Exception:
            pass
    return headings


def principal_point_grid(hw: tuple[int, int], settings: StageBSettings) -> np.ndarray:
    """Principal-point columns (x) the fit is allowed to consider.

    Wide enough to contain any centre a normal camera could have, and no wider:
    the point is to stop assuming a value, not to estimate one.
    """
    return np.linspace(
        hw[1] / 2 - settings.centre_halfwidth_px, hw[1] / 2 + settings.centre_halfwidth_px, settings.centre_grid_points
    )


def _profile_cost_torch(F, A, B, owner, counts, hw, grid, columns, device):
    """The same arithmetic as profile_cost, on the GPU stage B already holds.

    Profiling the centre turns one curve into a surface, and the rank-2
    projection needs a 3x3 SVD for every (pair, focal, centre): several million
    of them, which is minutes of numpy and seconds here.  The matcher has been
    freed by this point, so the device is idle.
    """
    import torch

    dev = torch.device(device)
    Ft = torch.as_tensor(np.ascontiguousarray(F), dtype=torch.float64, device=dev)
    At = torch.as_tensor(np.ascontiguousarray(A), dtype=torch.float64, device=dev)
    Bt = torch.as_tensor(np.ascontiguousarray(B), dtype=torch.float64, device=dev)
    own = torch.as_tensor(owner, dtype=torch.long, device=dev)
    cnt = torch.as_tensor(counts, dtype=torch.float64, device=dev)
    height = float(hw[0])
    cx = torch.as_tensor(np.asarray(columns), dtype=torch.float64, device=dev)
    C, P = len(columns), len(F)

    K = torch.zeros((C, 3, 3), dtype=torch.float64, device=dev)
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = cx, height / 2, 1.0
    ends = torch.as_tensor(np.cumsum(counts) - 1, dtype=torch.long, device=dev)
    out = torch.empty((P, len(grid), C), dtype=torch.float64, device=dev)
    with torch.no_grad():
        for column, focal in enumerate(grid):
            K[:, 0, 0] = K[:, 1, 1] = float(focal)
            Kt = K.transpose(1, 2)
            E = torch.einsum("cij,pjk,ckl->cpil", Kt, Ft, K)
            U, S, Vh = torch.linalg.svd(E)
            mean = (S[..., 0] + S[..., 1]) / 2
            D = torch.zeros_like(E)
            D[..., 0, 0] = mean
            D[..., 1, 1] = mean
            Ki = torch.linalg.inv(K)
            rank2 = torch.einsum("cji,cpjk,ckl->cpil", Ki, U @ D @ Vh, Ki)
            rank2 = rank2 / rank2.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-30)
            per = rank2[:, own]
            FA = torch.einsum("cnij,nj->cni", per, At)
            FtB = torch.einsum("cnji,nj->cni", per, Bt)
            num = torch.einsum("ni,cni->cn", Bt, FA) ** 2
            den = FA[..., 0] ** 2 + FA[..., 1] ** 2 + FtB[..., 0] ** 2 + FtB[..., 1] ** 2
            residual = (num / den.clamp_min(1e-12)).clamp_max(4.0)
            # Sum per pair by differencing a cumulative sum rather than
            # index_add_, which uses atomics on CUDA and so would put a
            # run-to-run nondeterminism back into a result that is otherwise
            # a pure function of the frames.
            running = torch.cumsum(residual, dim=1)
            totals = running[:, ends]
            totals = totals - torch.nn.functional.pad(totals[:, :-1], (1, 0))
            out[:, column, :] = (totals / cnt).transpose(0, 1)
    return out.cpu().numpy()


def profile_cost(
    bundle: dict[str, np.ndarray],
    mask: np.ndarray,
    grid: np.ndarray,
    hw: tuple[int, int],
    centres: np.ndarray | None = None,
    chunk: int = 8,
    device: Any = None,
) -> np.ndarray:
    """Per-pair Sampson cost of the nearest rank-2 essential matrix.

    With `centres` the cost is returned per (pair, focal, centre); without it the
    centre is the image middle and the last axis is dropped, which is what the
    heading band and the diagnostics want.
    """
    height, width = hw
    columns = np.asarray([width / 2.0]) if centres is None else np.asarray(centres)
    F = bundle["F"][mask]
    counts = bundle["counts"][mask]
    per_point = np.repeat(mask, bundle["counts"])
    A = np.hstack([bundle["xa"][per_point], np.ones((counts.sum(), 1))])
    B = np.hstack([bundle["xb"][per_point], np.ones((counts.sum(), 1))])
    owner = np.repeat(np.arange(len(F)), counts)
    if device is not None:
        cost = _profile_cost_torch(F, A, B, owner, counts, hw, grid, columns, device)
        return cost[:, :, 0] if centres is None else cost
    cost = np.empty((len(F), len(grid), len(columns)), dtype=np.float64)
    for column, focal in enumerate(grid):
        # The centre axis is evaluated in chunks: it is small, and one value at a
        # time would spend most of the run in Python call overhead.
        for start in range(0, len(columns), chunk):
            block = columns[start : start + chunk]
            K = np.zeros((len(block), 3, 3))
            K[:, 0, 0] = K[:, 1, 1] = focal
            K[:, 0, 2] = block
            K[:, 1, 2] = height / 2
            K[:, 2, 2] = 1.0
            E = np.einsum("cji,pjk,ckl->cpil", K, F, K)
            U, S, Vt = np.linalg.svd(E)
            D = np.zeros_like(E)
            mean = (S[..., 0] + S[..., 1]) / 2
            D[..., 0, 0] = mean
            D[..., 1, 1] = mean
            K_inv = np.linalg.inv(K)
            rank2 = np.einsum("cji,cpjk,ckl->cpil", K_inv, U @ D @ Vt, K_inv)
            rank2 /= np.maximum(np.abs(rank2).max(axis=(2, 3), keepdims=True), 1e-30)
            per = rank2[:, owner]
            FA = np.einsum("cnij,nj->cni", per, A)
            FtB = np.einsum("cnji,nj->cni", per, B)
            numerator = np.einsum("ni,cni->cn", B, FA) ** 2
            denominator = FA[..., 0] ** 2 + FA[..., 1] ** 2 + FtB[..., 0] ** 2 + FtB[..., 1] ** 2
            residual = np.minimum(numerator / np.maximum(denominator, 1e-12), 4.0)
            for c in range(len(block)):
                cost[:, column, start + c] = np.bincount(owner, weights=residual[c], minlength=len(F)) / counts
    return cost[:, :, 0] if centres is None else cost


def _summarise(cost: np.ndarray, grid: np.ndarray, low: float, high: float, draws: int) -> dict[str, Any] | None:
    """Focal and bootstrap interval from a (pair, focal, centre) cost block.

    The centre axis is minimised over, so a block of one column is the fit that
    assumes the image middle and the full block is the fit that does not.
    """
    if len(cost) < 25:
        return None

    def fit(rows: np.ndarray) -> float:
        return float(grid[int(np.argmin(cost[rows].mean(axis=0).min(axis=1)))])

    focal = fit(np.arange(len(cost)))
    rng = np.random.default_rng(0)
    draw = np.array([fit(rng.integers(0, len(cost), len(cost))) for _ in range(draws)])
    railed = (draw <= low * 1.02) | (draw >= high * 0.98)
    interior = draw[~railed]
    # A railed resample means the profile had no interior minimum for that
    # bootstrap sample; including those turns the percentile into a statement
    # about the grid rather than the data.
    source = interior if len(interior) >= 40 else draw
    ci = [float(np.percentile(source, 2.5)), float(np.percentile(source, 97.5))]
    return {
        "focal": focal,
        "ci": ci,
        "ci_width_pct": 100.0 * (ci[1] - ci[0]) / focal,
        "rail_fraction": float(railed.mean()),
        "pairs": int(len(cost)),
        "at_grid_edge": bool(focal <= low * 1.02 or focal >= high * 0.98),
    }


def _thin(bundle: dict[str, np.ndarray], mask: np.ndarray, per_pair: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Keep at most `per_pair` correspondences of each selected pair.

    The cost is a mean over each pair before anything else looks at it, so the
    extra correspondences buy resolution the focal grid does not have, and the
    profile is evaluated over a two-dimensional grid where they are not free.
    """
    counts = bundle["counts"]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    rows, kept = [], []
    for p in np.flatnonzero(mask):
        index = np.arange(offsets[p], offsets[p + 1])
        if len(index) > per_pair:
            index = index[np.linspace(0, len(index) - 1, per_pair).astype(int)]
        rows.append(index)
        kept.append(len(index))
    if not rows:
        return {}, np.zeros(0, dtype=bool)
    pick = np.concatenate(rows)
    thinned = {"F": bundle["F"][mask], "xa": bundle["xa"][pick], "xb": bundle["xb"][pick], "counts": np.asarray(kept)}
    return thinned, np.ones(len(thinned["F"]), dtype=bool)


def refine_focal(
    bundle: dict[str, np.ndarray],
    stage_a_focal: float,
    hw: tuple[int, int],
    settings: StageBSettings,
    device: Any = None,
) -> dict[str, Any]:
    """Fit the focal over several heading bands and decide whether to keep it."""
    if not bundle:
        return {"accepted": False, "focal": stage_a_focal, "reject_reasons": ["no usable pairs"], "bands": {}}
    headings = pair_headings(bundle, stage_a_focal, hw)
    low = settings.grid_relative[0] * stage_a_focal
    high = settings.grid_relative[1] * stage_a_focal
    # Anchoring the grid to stage A bounds the refinement; an unbounded grid lets
    # a monotone profile slide to the boundary and return a nonsense focal.
    grid = np.exp(np.linspace(math.log(low), math.log(high), 401))

    lo, hi = settings.fit_band_deg
    band = np.isfinite(headings) & (headings >= lo) & (headings <= hi)
    if band.sum() < 25:
        return {
            "accepted": False,
            "focal": stage_a_focal,
            "bands": {},
            "reject_reasons": [f"only {int(band.sum())} pairs in the fit band"],
        }

    thinned, mask = _thin(bundle, band, settings.profile_points_per_pair)
    centres = principal_point_grid(hw, settings)
    cost = profile_cost(thinned, mask, grid, hw, centres=centres, device=device)
    middle = int(np.argmin(np.abs(centres - hw[1] / 2)))

    # Two readings of the same cost: the slice that assumes the centre is in the
    # middle of the image, and the profile that does not assume it at all.  The
    # second is the answer; the first is only there to say how much the centre
    # matters for this pair set.
    pinned = _summarise(cost[:, :, middle : middle + 1], grid, low, high, settings.bootstrap_draws)
    profiled = _summarise(cost, grid, low, high, settings.bootstrap_draws)
    if pinned is None or profiled is None:
        return {"accepted": False, "focal": stage_a_focal, "bands": {}, "reject_reasons": ["too few pairs to fit"]}

    disagreement = 100.0 * abs(profiled["focal"] - pinned["focal"]) / pinned["focal"]
    deviation = 100.0 * abs(profiled["focal"] - stage_a_focal) / stage_a_focal
    reasons: list[str] = []
    if profiled["at_grid_edge"]:
        reasons.append("profile minimum sits at the grid edge")
    if profiled["rail_fraction"] > settings.max_rail_fraction:
        reasons.append(f"bootstrap railed on {profiled['rail_fraction']:.1%} of draws")
    if disagreement > settings.max_estimator_disagreement_pct:
        reasons.append(f"centre changes the focal by {disagreement:.1f}%")
    if deviation > settings.max_deviation_pct:
        reasons.append(f"moves {deviation:.1f}% from stage A")
    return {
        "accepted": not reasons,
        "focal": profiled["focal"] if not reasons else stage_a_focal,
        "stage_b_focal": profiled["focal"],
        "stage_a_focal": stage_a_focal,
        "band": [lo, hi],
        "ci": profiled["ci"],
        "ci_width_pct": profiled["ci_width_pct"],
        "rail_fraction": profiled["rail_fraction"],
        "deviation_pct": deviation,
        "disagreement_pct": disagreement,
        "centre_pinned_focal": pinned["focal"],
        "centre_pinned_ci_width_pct": pinned["ci_width_pct"],
        "pairs_in_band": profiled["pairs"],
        "reject_reasons": reasons,
        "bands": {},
    }


def settings_from_config(config: Any) -> StageBSettings:
    """``StageBSettings`` from a mapping or an object with the same field names.

    The MetricTrail ViPE pipeline passes its ``focal_refine`` config, whose
    stage-B fields have exactly these names and defaults.
    """
    raw = (
        config if isinstance(config, dict) else (config.model_dump() if hasattr(config, "model_dump") else vars(config))
    )
    known = {f.name for f in StageBSettings.__dataclass_fields__.values()}
    values: dict[str, Any] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
    return StageBSettings(**values)


def _as_uint8(frame: np.ndarray) -> np.ndarray:
    """Frames as uint8 RGB, whatever the loader produced.

    Mirrors what ``fundamental_matrices`` does when it builds a PIL image, so
    staging in this form changes no pixel the matcher ever sees.
    """
    if frame.dtype == np.uint8:
        return frame
    scaled = frame * 255.0 if float(frame.max()) <= 1.0 else frame
    return np.clip(scaled, 0, 255).astype(np.uint8)


def run_stage_b(
    settings: StageBSettings,
    provisional_focal: float,
    provisional_npz: Path,
    frames: Iterable[np.ndarray],
    hw: tuple[int, int],
    device: Any,
    workdir: Path,
    *,
    seed: int,
    interpreter: str,
    roma_repo: Path | None,
    roma_checkpoints: Path,
) -> dict[str, Any]:
    """Refine the shared focal from revisit pairs, in a separate process.

    RoMa's dependencies need not be present in the tracking environment, so the
    fit runs under ``interpreter`` (the caller resolves it; default: the running
    interpreter) as ``stage_b_cli.py``, which imports only this module. A fresh
    process also keeps the tracking process's precision flags (the SharedK
    calibration turns TF32 off process-wide) out of the matcher. Frames (CHW, any
    dtype) are staged to disk once as uint8;
    the child sees only those, stage A's trajectory, and stage A's focal.

    Every call restages: the workdir is keyed by the stream name only, so frames
    or a result left there by an earlier (failed or different) run are never
    reused, and the staged frames are deleted whatever the outcome.
    """
    import time

    started = time.perf_counter()
    workdir.mkdir(parents=True, exist_ok=True)
    images_npz = workdir / "frames.npz"
    settings_json = workdir / "settings.json"
    out_json = workdir / "stage_b.json"

    # Nothing from an earlier run in this workdir may be mistaken for this one's
    # (it is keyed by the stream name only: a rerun into the same output, another
    # clip of the same name, another resolution).
    staging = images_npz.with_suffix(".partial.npz")
    for stale in (images_npz, staging, out_json):
        stale.unlink(missing_ok=True)
    try:
        return _stage_and_fit(
            settings,
            provisional_focal,
            provisional_npz,
            frames,
            device,
            seed=seed,
            interpreter=interpreter,
            roma_repo=roma_repo,
            roma_checkpoints=roma_checkpoints,
            images_npz=images_npz,
            staging=staging,
            settings_json=settings_json,
            out_json=out_json,
            started=started,
        )
    finally:
        # The staged frames are pure scratch and the largest thing this stage writes.
        images_npz.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)


def _stage_and_fit(
    settings: StageBSettings,
    provisional_focal: float,
    provisional_npz: Path,
    frames: Iterable[np.ndarray],
    device: Any,
    *,
    seed: int,
    interpreter: str,
    roma_repo: Path | None,
    roma_checkpoints: Path,
    images_npz: Path,
    staging: Path,
    settings_json: Path,
    out_json: Path,
    started: float,
) -> dict[str, Any]:
    import dataclasses
    import json
    import os
    import subprocess
    import time

    # Staged as uint8, which is what the matcher wants anyway.  Written to a
    # temporary name first so a kill mid-write cannot leave a truncated archive
    # behind under the final name.  (The suffix must end in .npz: np.savez
    # appends it otherwise, and the rename would then look for a file that was
    # never written.)
    stacked = np.stack([_as_uint8(np.asarray(frame)) for frame in frames])
    np.savez(staging, images=stacked)
    del stacked
    staging.replace(images_npz)
    settings_json.write_text(json.dumps(dataclasses.asdict(settings)))

    here = Path(__file__).resolve().parent
    command = [
        str(interpreter),
        str(here / "stage_b_cli.py"),
        "--images-npz",
        str(images_npz),
        "--provisional-npz",
        str(provisional_npz),
        "--focal",
        f"{provisional_focal:.6f}",
        "--settings-json",
        str(settings_json),
        "--seed",
        str(seed),
        "--roma-checkpoints",
        str(roma_checkpoints),
        "--device",
        str(device),
        "--out",
        str(out_json),
    ]
    if roma_repo is not None:
        command += ["--roma-repo", str(roma_repo)]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(here),
        env={**dict(os.environ), "PYTHONPATH": str(here), "PYTHONNOUSERSITE": "1"},
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0 or not out_json.is_file():
        return {
            "accepted": False,
            "focal": provisional_focal,
            "seconds": elapsed,
            "reject_reasons": [f"stage B subprocess failed rc={proc.returncode}"],
            "stderr": proc.stderr[-1500:],
        }
    outcome = json.loads(out_json.read_text())
    outcome["seconds"] = elapsed
    # Recorded because the focal depends on it: the matcher samples its
    # correspondences with torch.multinomial, so the seed is part of what
    # produced this number and a rerun without it is not the same experiment.
    outcome["seed"] = int(seed)
    return outcome
