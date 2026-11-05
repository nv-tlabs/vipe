# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage B entry point, run as its own process (``python stage_b_cli.py ...``).

``--roma-repo`` is optional (default: ``romatch`` must import).

RoMa pulls in dependencies the tracking environment does not carry (loguru
among them), so stage B executes under its own interpreter and hands the
outcome back as JSON.  Keeping it out of process also means a RoMa import
failure cannot take down a tracking run that has already committed.

The arguments are exactly what stage B is allowed to see: stage A's focal,
stage A's trajectory, and the frames.  No sensor K, no ground truth.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-npz", type=Path, required=True, help="CHW uint8 frames staged by the caller")
    ap.add_argument("--provisional-npz", type=Path, required=True)
    ap.add_argument("--focal", type=float, required=True)
    ap.add_argument("--settings-json", type=Path, required=True)
    ap.add_argument(
        "--roma-repo", type=Path, default=None, help="RoMa checkout to put on sys.path (default: romatch must import)"
    )
    ap.add_argument("--roma-checkpoints", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234, help="seeds the matcher's correspondence sampling")
    args = ap.parse_args(argv)

    import random

    import torch

    # Run by path in a separate interpreter: stage_b.py sits next to this file
    # (on sys.path) and imports nothing from ViPE.
    core = importlib.import_module(f"{__package__}.stage_b" if __package__ else "stage_b")
    StageBSettings, refine_focal = core.StageBSettings, core.refine_focal
    revisit_candidates, fundamental_matrices = core.revisit_candidates, core.fundamental_matrices
    load_matcher = core.load_matcher

    # RoMa draws its correspondences with torch.multinomial -- samples_per_pair
    # of them out of a dense warp, for every mined pair -- so without this the
    # focal stage B returns is not reproducible.  Measured on VBR: the same
    # stage A trajectory for pincio_train0 gave 597.4 px on one run and 517.8 on
    # another, +18.3% against +2.5% focal error, and the accept gate passed both
    # because a bad draw is confidently wrong rather than visibly uncertain.
    # The bootstrap in refine_focal already has its own generator; this covers
    # the matcher.  (OpenCV's MAGSAC was checked and is deterministic without
    # seeding, so it is not a source here.)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed % (2**32))
    random.seed(args.seed)

    settings = StageBSettings(**json.loads(args.settings_json.read_text()))
    with np.load(args.provisional_npz, allow_pickle=False) as archive:
        key = next(k for k in ("c2w", "predicted_c2w", "c2w_cam0_baseline", "c2w_cam0") if k in archive.files)
        c2w = np.asarray(archive[key], dtype=np.float64)
    with np.load(args.images_npz, allow_pickle=False) as archive:
        images = archive["images"]

    pairs = revisit_candidates(c2w, settings)
    usable = [(i, j) for i, j in pairs if max(i, j) < len(images)]
    if len(usable) < 25:
        outcome = {
            "accepted": False,
            "focal": args.focal,
            "candidates": len(usable),
            "reject_reasons": [f"only {len(usable)} revisit candidates"],
        }
        args.out.write_text(json.dumps(outcome, indent=1))
        return 0

    device = torch.device(args.device)
    hw = (int(images.shape[-2]), int(images.shape[-1]))
    matcher = load_matcher(args.roma_repo, args.roma_checkpoints, device, settings)
    bundle = fundamental_matrices(matcher, images, usable, hw, settings, device)
    del matcher
    torch.cuda.empty_cache()

    # Keep the correspondences the fit was made from.  Every accept rule worth
    # trying is a function of this bundle alone, so persisting it turns gate
    # design into a CPU replay over runs already paid for rather than another
    # pass over the GPU.  A few MB per sequence against ~330 s of matching.
    pairs_npz = args.out.with_name("pairs.npz")
    staging = pairs_npz.with_suffix(".partial.npz")
    # The bundle already carries the pairs it kept; `usable` is what was proposed.
    np.savez_compressed(
        staging,
        focal_stage_a=np.float64(args.focal),
        hw=np.asarray(hw, np.int32),
        proposed=np.asarray(usable, np.int32),
        **bundle,
    )
    staging.replace(pairs_npz)

    outcome = refine_focal(bundle, args.focal, hw, settings, device=device)
    outcome["candidates"] = len(usable)
    outcome["pairs_usable"] = int(len(bundle.get("F", [])))
    outcome["pairs_npz"] = str(pairs_npz)
    args.out.write_text(json.dumps(outcome, indent=1))
    print(json.dumps({k: v for k, v in outcome.items() if k != "bands"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
