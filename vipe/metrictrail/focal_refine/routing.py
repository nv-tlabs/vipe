# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide, per sequence and without ground truth, whether stage B is warranted.

``route`` takes stage A's trajectory and SharedK record in memory.

Stage B costs ~300 s of GPU and a second tracking pass, and it is not free of
risk: run on a sequence whose focal is already right, its own accept gate does
not reliably reject.  Measured on Oxford-Spires, stage B accepted on every scene
it was offered -- christ-church-03 250.65 -> 251.01, keble-college-02
249.90 -> 256.59, christ-church-02 250.90 -> 247.44.  So a sequence routed into
stage B by mistake does not merely waste GPU time, it silently moves a correct
focal.  The routing therefore has to be right, not merely cheap.

Three gates, all read off artefacts stage A has already written:

  revisit count    revisit_candidates              >= 1000
  revisit supply   revisit_candidates / frame      >= 0.15
  focal drift      |f_24 - f_before| / f_final %   >= 0.20

The first two are one capability test read two ways.  Stage B works by mining
wide-baseline revisit pairs, so a sequence that never returns near its own path
cannot be refined by it whatever its focal error.  Together they block all of
KITTI.

The count gate was added after the density gate alone, at its original 0.35,
blocked VBR ciampino_train1 -- the sequence stage B helps most, 8.6% focal error
down to 0.3% and 74.13 m ATE down to 15.42 m.  Density alone is the wrong
question: stage B spends at most `budget` pairs however long the sequence runs,
and ciampino has 3501 of them, 4.4x the budget, spread over 18846 frames.  On
absolute count the two families separate 6.3x with no overlap -- KITTI peaks at
559 pairs (seq 00), VBR bottoms out at ciampino's 3501 -- against only 1.5x on
density, which is why the density threshold was easy to miscalibrate.  Density
is kept because it is what catches the opposite failure: a short indoor sequence
where every frame trivially counts as a revisit.

The second asks whether stage A's focal ever came to rest.  By the final-24
milestone all 128 accepted keyframes are in hand and no new data is arriving, so
a focal still moving at that point is one the bundle adjustment has no restoring
force on -- a flat cost in the focal direction, which is exactly the forward
motion degeneracy stage B exists to break.  This gate blocks all of
Oxford-Spires and LaMARia.

Neither gate is a clean physical law on its own.  Drift also fires on KITTI 02,
04, 06, 08 and 10, where stage A is fine and the movement is ordinary
convergence (seq 10 travels 293 -> 502 px and is a convergence failure, not a
degeneracy); the revisit gate is what removes those.  Conversely the revisit
gate assumes a scene scale comparable to its 15 m radius, so indoor sequences
that move only a few metres make every frame a "revisit" -- measured on TUM
fr3, sitting_halfsphere yields 9.2 candidates per frame, more than any VBR
sequence.  The two gates cover each other's failure mode; neither stands alone.

Measured over 33 sequences from VBR, KITTI, Oxford-Spires and LaMARia: zero
false positives, and all 7 VBR sequences routed in.  The count threshold sits
between KITTI's 559 and VBR's 3501, the density one between KITTI's 0.123 and
VBR's 0.186, and drift at the log-midpoint of 0.246 vs 0.159.

An earlier version of this file claimed the density gap was 0.607 vs 0.133 and
set the threshold at their log-midpoint, 0.35.  That measurement was wrong --
VBR's true minimum is ciampino's 0.186 -- and 0.35 sat above it, so the gate
excluded the sequence it should most have admitted.  The numbers above are
re-measured from the stage A trajectories this pipeline actually produces.

A third gate, the A-prime profile well depth, was evaluated and dropped: over
the same 33 sequences it blocked nothing the two gates above did not already
block, while costing an extra audit pass and a third fitted threshold.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .stage_b import StageBSettings, revisit_candidates


@dataclass(frozen=True)
class RoutingDecision:
    enter: bool
    revisit_count: int
    revisit_per_frame: float
    focal_drift_pct: float | None
    blocked_by: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        return {
            "schema": "metrictrail_stage_b_routing_v1",
            "enter_stage_b": self.enter,
            "revisit_count": self.revisit_count,
            "revisit_per_frame": self.revisit_per_frame,
            "focal_drift_pct": self.focal_drift_pct,
            "blocked_by": list(self.blocked_by),
            "uses_ground_truth": False,
        }


def revisit_supply(c2w: np.ndarray, settings: StageBSettings) -> tuple[int, float]:
    """Revisit candidates, as an absolute count and per frame.

    The budget exists to cap stage B's matching cost; here the true supply is
    what carries the signal, so it must not be truncated.

    Both numbers are returned because they answer different questions and the
    count is the one that nearly went missing: stage B consumes at most
    ``settings.budget`` pairs however long the sequence is, so a long sequence
    with ample absolute supply can still look starved per frame.  That is what
    happened to VBR ciampino_train1 -- 3501 pairs, 4.4x the budget, but 18846
    frames, so 0.186 per frame.
    """
    if len(c2w) == 0:
        return 0, 0.0
    uncapped = dataclasses.replace(settings, budget=len(c2w) ** 2)
    count = len(revisit_candidates(c2w, uncapped))
    return count, count / len(c2w)


def focal_drift(calibration: Mapping[str, Any] | Path) -> float | None:
    """How far stage A's focal still moved during its final refinement, in percent.

    ``calibration`` is the SharedK record (or a JSON file holding it). Returns
    None when the record lacks these fields rather than guessing, so a missing
    measurement blocks the sequence instead of silently admitting it.
    """
    if isinstance(calibration, Mapping):
        record = dict(calibration)
    else:
        try:
            record = json.loads(Path(calibration).read_text())
        except (OSError, ValueError):
            return None
    first = lambda key: (record.get(key) or [None])[0]  # noqa: E731
    before = first("K_before_final_px")
    after = first("K_after_final24_px")
    final = (record.get("calibrated_K_fx_fy_cx_cy") or [None])[0]
    if None in (before, after, final) or not final:
        return None
    return 100.0 * abs(after - before) / final


def route(c2w: np.ndarray, calibration: Mapping[str, Any] | Path, settings: StageBSettings) -> RoutingDecision:
    """Routing decision from stage A's camera-to-world trajectory and its SharedK record."""
    c2w = np.asarray(c2w, dtype=np.float64)

    count, supply = revisit_supply(c2w, settings)
    drift = focal_drift(calibration)

    blocked: list[str] = []
    if count < settings.gate_revisit_count:
        blocked.append("revisit_count")
    if supply < settings.gate_revisit_per_frame:
        blocked.append("revisit_supply")
    if drift is None or drift < settings.gate_focal_drift_pct:
        blocked.append("focal_drift")
    if drift is not None and drift > settings.gate_focal_drift_max_pct:
        blocked.append("focal_settled")
    return RoutingDecision(not blocked, count, supply, drift, tuple(blocked))
