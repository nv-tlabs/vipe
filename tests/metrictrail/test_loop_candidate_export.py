# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from vipe.metrictrail.optional.loop_candidates import export_candidates, main


def test_plain_csv_export_preserves_input(tmp_path):
    pairs, output = tmp_path / "pairs.csv", tmp_path / "loops.json"
    pairs.write_text("src_frame,dst_frame,score\n0,100,0.95\n")
    before = pairs.read_bytes()
    assert main(["--pairs", str(pairs), "--output", str(output), "--producer", "SALAD"]) == 0
    value = json.loads(output.read_text())
    assert value == {
        "schema": "metrictrail_loop_candidates_v1",
        "producer": "SALAD",
        "candidates": [{"src_frame": 100, "dst_frame": 0, "score": 0.95}],
    }
    assert before == pairs.read_bytes()


@pytest.mark.parametrize(
    "rows",
    [
        "1,0,nan\n",
        "1,0,inf\n",
        "1,0,1.1\n",
        "0,0,0.9\n",
        "1,-1,0.9\n",
        "1.5,0,0.9\n",
        "1,0,0.9\n0,1,0.9\n",
        "1,0\n",
        "1,0,0.9,extra\n",
    ],
)
def test_invalid_csv_is_rejected_before_output(tmp_path, rows):
    pairs, output = tmp_path / "pairs.csv", tmp_path / "loops.json"
    pairs.write_text("src_frame,dst_frame,score\n" + rows)
    with pytest.raises(ValueError):
        export_candidates(pairs, output)
    assert not output.exists()


def test_export_cannot_overwrite_input(tmp_path):
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("src_frame,dst_frame,score\n")
    with pytest.raises(ValueError):
        export_candidates(pairs, pairs)
    assert pairs.read_text() == "src_frame,dst_frame,score\n"


@pytest.mark.parametrize("producer", [" ", "x" * 1025])
def test_export_does_not_emit_unsupported_producer_metadata(tmp_path, producer):
    pairs, output = tmp_path / "pairs.csv", tmp_path / "loops.json"
    pairs.write_text("src_frame,dst_frame,score\n")
    with pytest.raises(ValueError, match="producer"):
        export_candidates(pairs, output, producer=producer)
    assert not output.exists()
