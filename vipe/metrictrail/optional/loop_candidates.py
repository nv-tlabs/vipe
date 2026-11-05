# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert externally produced frame-pair CSV to the loop-candidate file format.

This module performs no image retrieval, model execution or tensor exchange.
It uses only the Python standard library and does not install any dependencies.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .loop_files import atomic_json

SCHEMA = "metrictrail_loop_candidates_v1"
MAX_CANDIDATES = 1000


def export_candidates(pairs: Path, output: Path, *, producer: str | None = None) -> dict:
    pairs, output = (path.expanduser().resolve() for path in (pairs, output))
    if output == pairs:
        raise ValueError("output must not replace the input CSV")
    candidates: list[dict[str, int | float]] = []
    seen: set[tuple[int, int]] = set()
    with pairs.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["src_frame", "dst_frame", "score"]:
            raise ValueError("CSV header must be src_frame,dst_frame,score")
        for line, row in enumerate(reader, start=2):
            if len(candidates) >= MAX_CANDIDATES:
                raise ValueError(f"at most {MAX_CANDIDATES} candidate pairs are supported")
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV line {line} must have exactly three fields")
            try:
                src, dst = int(row["src_frame"]), int(row["dst_frame"])
                score = float(row["score"])
            except ValueError as error:
                raise ValueError(f"invalid candidate on CSV line {line}") from error
            src, dst = max(src, dst), min(src, dst)
            if dst < 0 or src == dst or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid frame pair or score on CSV line {line}")
            if (src, dst) in seen:
                raise ValueError(f"duplicate frame pair on CSV line {line}")
            seen.add((src, dst))
            candidates.append({"src_frame": src, "dst_frame": dst, "score": score})
    document = {"schema": SCHEMA, "candidates": candidates}
    if producer is not None:
        if not isinstance(producer, str) or not producer.strip() or len(producer) > 1024:
            raise ValueError("producer must contain 1 to 1024 characters when supplied")
        document["producer"] = producer
    atomic_json(output, document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="external CSV with src_frame,dst_frame,score header")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer", help="optional name of the external producer")
    args = parser.parse_args(argv)
    export_candidates(args.pairs, args.output, producer=args.producer)
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
