# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU candidate-file validation and ViPE pose artifact integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydantic
import pytest
from scipy.spatial.transform import Rotation

from vipe.metrictrail.optional import loop_closure as lc
from vipe.metrictrail.optional.loop_files import CANDIDATE_SCHEMA, MAX_CANDIDATES
from vipe.metrictrail.optional.loop_optimizer import (
    GraphSettings,
    optimize_identity_graph,
)


def files(tmp_path, *, frame_ids=None, poses=None, candidates=None):
    frame_ids = np.array([10, 20, 30, 40], dtype=np.int64) if frame_ids is None else frame_ids
    if poses is None:
        poses = np.tile(np.eye(4), (len(frame_ids), 1, 1))
        poses[:, 0, 3] = np.arange(len(frame_ids), dtype=np.float64)
    (tmp_path / "pose").mkdir(parents=True)
    baseline = tmp_path / "pose" / "sequence.npz"
    np.savez(baseline, data=poses, inds=frame_ids, note=np.asarray("retained metadata"))
    manifest = {
        "schema": CANDIDATE_SCHEMA,
        "producer": "external detector",
        "candidates": candidates or [],
    }
    candidate = tmp_path / "loops.json"
    candidate.write_text(json.dumps(manifest))
    (tmp_path / "vipe").mkdir()
    (tmp_path / "vipe" / "sequence_metrictrail.json").write_text(
        json.dumps(
            {
                "status": "success",
                "name": "sequence",
                "tracking_status": "tracked",
                "pipeline": {"slam": {"dpvo": {"METRIC_GLOBAL_BA": False, "LOOP_CLOSURE": False}}},
            }
        )
    )
    return (
        SimpleNamespace(output_root=tmp_path / "pose_lc"),
        "sequence",
        baseline,
        candidate,
    )


def run(values, **kwargs):
    _, name, baseline, candidate = values
    options = lc.LoopClosureOptions(enabled=True, candidates=str(candidate), **kwargs)
    return lc.maybe_run_loop_closure(options, baseline.parent.parent, name)


def rewrite_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_rotation_and_translation_constraints_are_consistent():
    pytest.importorskip("pypose")
    poses = np.tile(np.eye(4), (5, 1, 1))
    angles = np.array([0, 0.5, 1.0, 0.5, 0])
    poses[:, :3, :3] = Rotation.from_rotvec(angles[:, None] * [0, 0, 1]).as_matrix()
    poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]
    corrected, info = optimize_identity_graph(poses, [(4, 0)], GraphSettings())
    np.testing.assert_allclose(corrected, poses, rtol=0, atol=1e-6)
    assert info["final_cost"] < 1e-12


def test_pose_graph_reduces_nonzero_rotation_and_translation_closure_error():
    pytest.importorskip("pypose")
    progress = np.linspace(0.0, 1.0, 8)
    poses = np.tile(np.eye(4), (8, 1, 1))
    poses[:, :3, :3] = Rotation.from_rotvec((2 * np.pi * progress + 0.08 * progress)[:, None] * [0, 0, 1]).as_matrix()
    poses[:, 0, 3] = np.sin(2 * np.pi * progress) + 0.25 * progress
    poses[:, 1, 3] = 1 - np.cos(2 * np.pi * progress)
    corrected, info = optimize_identity_graph(poses, [(7, 0)], GraphSettings())
    assert np.linalg.norm(Rotation.from_matrix(corrected[-1, :3, :3]).as_rotvec()) < 0.08
    assert np.linalg.norm(corrected[-1, :3, 3]) < 0.25
    np.testing.assert_array_equal(corrected[0], poses[0])
    assert info["final_cost"] < info["initial_cost"]


@pytest.mark.parametrize(
    "row",
    [
        {"src_frame": 4, "dst_frame": 0, "score": 0.9},
        {"src_frame": True, "dst_frame": 10, "score": 0.9},
        {"src_frame": 40.0, "dst_frame": 10, "score": 0.9},
        {"src_frame": 10, "dst_frame": 40, "score": 0.9},
        {"src_frame": 40, "dst_frame": 40, "score": 0.9},
        {"src_frame": 40, "dst_frame": 10, "score": True},
        {"src_frame": 40, "dst_frame": 10, "score": float("nan")},
        {"src_frame": 40, "dst_frame": 10, "score": float("inf")},
        {"src_frame": 40, "dst_frame": 10, "score": -0.1},
        {"src_frame": 40, "dst_frame": 10, "score": 1.1},
        {"src_frame": 40, "dst_frame": 10},
        {"src_frame": 40, "dst_frame": 10, "score": 0.9, "transform": []},
    ],
)
def test_invalid_candidate_rows_never_create_output(tmp_path, row):
    values = files(tmp_path, candidates=[row])
    with pytest.raises(ValueError):
        run(values)
    assert not values[0].output_root.exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda obj: obj.update(schema="unsupported"),
        lambda obj: obj.update(unknown=True),
        lambda obj: obj.update(producer={"not": "text"}),
        lambda obj: obj.update(candidates={}),
        lambda obj: obj.update(candidates=[{"src_frame": 40, "dst_frame": 10, "score": 0.9}] * 2),
        lambda obj: obj.update(candidates=[{}] * (MAX_CANDIDATES + 1)),
    ],
)
def test_malformed_manifests(tmp_path, change):
    values = files(tmp_path)
    rewrite_json(values[3], change)
    with pytest.raises(ValueError):
        run(values)


@pytest.mark.parametrize(
    "frame_ids",
    [
        np.array([0.0, 1.0, 2.0]),
        np.array([False, True]),
        np.array([1, 1, 2]),
        np.array([2, 1, 0]),
        np.array([-1, 0, 1]),
        np.array([], dtype=np.int64),
    ],
)
def test_invalid_baseline_frame_inventory(tmp_path, frame_ids):
    values = files(tmp_path, frame_ids=frame_ids)
    with pytest.raises(ValueError):
        run(values)


def test_reject_nonfinite_and_reflection_poses(tmp_path):
    for label, bad in (("nan", float("nan")), ("reflection", -1.0)):
        directory = tmp_path / label
        directory.mkdir()
        poses = np.tile(np.eye(4), (4, 1, 1))
        poses[1, 0, 0] = bad
        values = files(directory, poses=poses)
        with pytest.raises(ValueError):
            run(values)


def test_defaults_and_explicit_candidates():
    assert lc.LoopClosureOptions().enabled is False
    with pytest.raises(pydantic.ValidationError, match="requires candidates"):
        lc.LoopClosureOptions(enabled=True)
    assert lc.LoopClosureOptions(enabled=True, candidates="pairs.json").candidates == "pairs.json"
    with pytest.raises(pydantic.ValidationError):
        lc.LoopClosureOptions(checkpoint="weights.ckpt")


def test_disabled_is_lazy(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vipe.metrictrail.optional.loop_trajectory", None)
    assert lc.maybe_run_loop_closure(None, tmp_path, "sequence") is None
    assert lc.maybe_run_loop_closure(lc.LoopClosureOptions(), tmp_path, "sequence") is None
    assert not (tmp_path / "pose_lc").exists()


def test_pose_round_trip_preserves_inputs_and_anchor(tmp_path):
    pytest.importorskip("pypose")
    values = files(
        tmp_path,
        candidates=[
            {"src_frame": 40, "dst_frame": 10, "score": 0.1},
            {"src_frame": 30, "dst_frame": 10, "score": 0.9},
        ],
    )
    original = [values[2].read_bytes(), values[3].read_bytes()]
    record = run(values)
    assert record["applied_loop_edges"] == 2
    assert record["lc_implementation"] == "external_candidates_pypose_identity_se3_pgo_v1"
    assert record["optimization"]["iterations"] <= 30
    assert record["optimization"]["settings"]["direct_inliers"] == 20
    assert record["optimization"]["settings"]["rigid_min_inliers"] == 24
    assert record["optimization"]["settings"]["lambda_init"] == 1.0e-6
    assert original == [values[2].read_bytes(), values[3].read_bytes()]
    edges = json.loads(Path(record["outputs"]["edges"]["path"]).read_text())["edges"]
    assert [(edge["src_pos"], edge["dst_pos"]) for edge in edges] == [(2, 0), (3, 0)]
    with (
        np.load(values[2]) as before,
        np.load(record["outputs"]["npz"]["path"]) as after,
    ):
        np.testing.assert_array_equal(after["data"][0], before["data"][0])
        np.testing.assert_array_equal(after["inds"], before["inds"])
        assert after["data"][-1, 0, 3] < before["data"][-1, 0, 3]
    assert record["optimization"]["final_cost"] < record["optimization"]["initial_cost"]
    with pytest.raises(FileExistsError):
        run(values)
    assert run(values, overwrite=True)["status"] == "success"


def test_empty_candidates_preserve_all_arrays(tmp_path):
    values = files(tmp_path)
    record = run(values)
    with (
        np.load(values[2]) as before,
        np.load(record["outputs"]["npz"]["path"]) as after,
    ):
        for key in before.files:
            np.testing.assert_array_equal(after[key], before[key])
            assert after[key].dtype == before[key].dtype


def test_legacy_minimum_edge_score(tmp_path):
    values = files(
        tmp_path,
        candidates=[
            {"src_frame": 40, "dst_frame": 10, "score": 0.024},
            {"src_frame": 30, "dst_frame": 10, "score": 0.025},
        ],
    )
    report = run(values)
    assert report["supplied_candidates"] == 2
    assert report["applied_loop_edges"] == 1
    edges = json.loads(Path(report["outputs"]["edges"]["path"]).read_text())["edges"]
    assert edges[0]["src_frame"] == 30


def test_duplicate_json_keys(tmp_path):
    values = files(tmp_path)
    values[3].write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        run(values)
    assert not values[0].output_root.exists()


@pytest.mark.parametrize("setting", ["METRIC_GLOBAL_BA", "LOOP_CLOSURE"])
def test_global_ba_input_rejected(tmp_path, setting):
    values = files(tmp_path)
    rewrite_json(
        tmp_path / "vipe" / "sequence_metrictrail.json",
        lambda record: record["pipeline"]["slam"]["dpvo"].update({setting: True}),
    )
    with pytest.raises(ValueError, match="local-only"):
        run(values)
    assert not values[0].output_root.exists()


def test_path_traversal_rejected(tmp_path):
    values = files(tmp_path)
    values = (values[0], "../sequence", values[2], values[3])
    with pytest.raises(ValueError, match="filename component"):
        run(values)


def test_empty_candidate_cli_runs_without_torch_or_pypose(tmp_path):
    values = files(tmp_path)
    script = """
import importlib.abc
import sys
class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "pypose", "faiss", "cv2"}:
            raise AssertionError("unexpected model/GPU dependency: " + name)
sys.meta_path.insert(0, Blocked())
from vipe.metrictrail.optional.loop_closure import main
main(sys.argv[1:])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            "--name",
            "sequence",
            "--candidates",
            str(values[3]),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]["status"] == "success"


@pytest.mark.parametrize(
    "change",
    [
        lambda obj: obj.update(status="failed"),
        lambda obj: obj.update(name="other"),
        lambda obj: obj.pop("tracking_status"),
        lambda obj: obj.pop("pipeline"),
    ],
)
def test_invalid_run_record_rejected(tmp_path, change):
    values = files(tmp_path)
    rewrite_json(tmp_path / "vipe" / "sequence_metrictrail.json", change)
    with pytest.raises(ValueError):
        run(values)
    assert not values[0].output_root.exists()


def test_missing_run_record_rejected(tmp_path):
    values = files(tmp_path)
    (tmp_path / "vipe" / "sequence_metrictrail.json").unlink()
    with pytest.raises(ValueError, match="run record"):
        run(values)


def test_stereo_record_and_uninitialized_skip(tmp_path):
    values = files(tmp_path)
    metadata = tmp_path / "vipe" / "sequence_metrictrail.json"
    metadata.write_text(
        json.dumps(
            {
                "status": "success",
                "case": "sequence",
                "num_frames": 4,
                "tracking_status": "tracked",
                "loop_closure": False,
                "global_ba": False,
            }
        )
    )
    assert run(values)["status"] == "success"
    rewrite_json(metadata, lambda obj: obj.update(tracking_status="not_initialized"))
    assert run(values, overwrite=True)["status"] == "skipped"
    assert not any(values[0].output_root.iterdir())
