"""``python -m metrictrail_eval.plot``: the ATE a figure shows is the ATE ``score`` reports.

Synthetic GT and predictions (the writers of ``test_vipe_output``) for every
dataset; the plot CLI and the score CLI run on the same prediction directory
and every per-sequence ATE and the mean must be identical, the figures and
the aligned-trajectory NPZ must exist, and the RMSE of the drawn points must
restate the ATE.  The default figure style (title = the sequence's name, no
vertical axis label, legend GT / prediction only in one row between the title
and the axes, no start markers) and the ``--details`` style are checked on the
drawn axes.  CPU only; run from the repository root: ``python -m pytest tests``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from metrictrail_eval import aligned, kitti, plot, registry, score, tum, vbr, wild_slam
from test_vipe_output import (  # helpers only (no test functions, which pytest would collect again)
    HAS_EVO,
    TRAJECTORY_CASES,
    estimate,
    gt_kitti,
    gt_tum,
    gt_vbr,
    gt_wild_slam,
    write_npz_prediction,
    write_vipe_output,
)

needs_evo = pytest.mark.skipif(not HAS_EVO, reason="evo not installed")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def run_score(tmp_path: Path, tag: str, argv: list[str]) -> dict:
    out = tmp_path / "scores" / f"{tag}.json"
    assert score.main([*argv, "--out", str(out)]) == 0
    return json.loads(out.read_text())


def run_plot(out: Path, argv: list[str], *, expect: int = 0) -> dict | None:
    assert plot.main([*argv, "--out", str(out), "--dpi", "40"]) == expect
    summary = out / "summary.json"
    return json.loads(summary.read_text()) if summary.is_file() else None


def assert_figures(summary: dict, cases: list[str]) -> None:
    run = summary["run"]
    out_dir = Path(run["out_dir"])
    assert summary["problems"] == []
    for case in cases:
        figure = run["per_case"][case]["figure"]
        png = Path(figure["png"])
        assert png == out_dir / f"{case}.png" and png.read_bytes()[:8] == PNG_MAGIC
        assert figure["drawn_rmse_abs_diff_m"] <= 1e-9 * max(1.0, figure["drawn_rmse_m"])
    assert (out_dir / "overview.png").read_bytes()[:8] == PNG_MAGIC
    with np.load(out_dir / "aligned_trajectories.npz") as archive:
        for case in cases:
            gt_xyz, pred_xyz = archive[f"{case}/gt_xyz"], archive[f"{case}/pred_xyz"]
            assert gt_xyz.shape == pred_xyz.shape == (len(archive[f"{case}/frame_ids"]), 3)
            rmse = math.sqrt(float(np.mean(np.sum((gt_xyz - pred_xyz) ** 2, axis=1))))
            assert rmse == pytest.approx(run["per_case"][case]["figure"]["drawn_rmse_m"], rel=1e-12)


def assert_same_ate(scored: dict, summary: dict) -> None:
    """Bit-identical per-sequence results (every field score writes) and mean."""

    (expected,), got = scored["runs"], summary["run"]
    assert expected["per_case"].keys() == got["per_case"].keys()
    for case, row in expected["per_case"].items():
        for key, value in row.items():
            assert got["per_case"][case][key] == value, (case, key)
        factor = plot.TO_METRES[summary["units"]] / plot.TO_METRES[summary["display_units"]]
        assert got["per_case"][case]["figure"]["ate_shown"] == row["ate"] * factor
    assert got["mean_ate"] == expected["mean_ate"] and got["num_cases"] == expected["num_cases"]
    for key in ("dataset", "stage", "protocol", "units", "gt_root", "cases", "pred_format"):
        assert summary[key] == scored[key], key


# --------------------------------------------------------------------------- every dataset, both formats


@pytest.mark.parametrize("pred_format", ["npz", "vipe"])
@pytest.mark.parametrize("dataset", sorted(TRAJECTORY_CASES))
def test_plot_ate_equals_score(
    dataset: str, pred_format: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, n, write_gt, evo = TRAJECTORY_CASES[dataset]
    if evo and not HAS_EVO:
        pytest.skip("evo not installed")
    gt_root = tmp_path / "gt"
    gt = write_gt(gt_root, case, n, monkeypatch)
    pred = estimate(gt, seed=11)
    if pred_format == "vipe":
        write_vipe_output(tmp_path / "pred", case, pred)
    else:
        write_npz_prediction(tmp_path / "pred", case, pred)
    argv = ["--dataset", dataset, "--gt-root", str(gt_root), "--cases", case, "--pred", str(tmp_path / "pred"),
            "--pred-format", pred_format]
    scored = run_score(tmp_path, "score", argv)
    summary = run_plot(tmp_path / "figures", [*argv, "--check-score", "--pdf",
                                              "--compare", str(tmp_path / "scores" / "score.json"), "--compare-tol", "0"])
    assert_same_ate(scored, summary)
    assert_figures(summary, [case])
    run = summary["run"]
    assert run["score_check"]["identical"] and run["score_check"]["differences"] == []
    assert run["compare"]["passed"] and run["compare"]["identical_cases"] == 1
    assert run["compare"]["mean_abs_diff"] == 0.0
    assert Path(run["per_case"][case]["figure"]["pdf"]).read_bytes()[:4] == b"%PDF"
    assert summary["plane"] == list(plot.GROUND_PLANES[dataset])
    assert summary["display_units"] == plot.DISPLAY_UNITS.get(dataset, "m")
    # Default style: the title is the sequence's name, the legend one row above the axes.
    assert summary["details"] is False and summary["legend"] == "top"
    assert run["per_case"][case]["figure"]["title"] == plot.case_title(dataset, case)
    assert run["per_case"][case]["figure"]["legend"] == "top"


@pytest.mark.parametrize("dataset", sorted(TRAJECTORY_CASES))
def test_evaluate_case_aligned_is_evaluate_case(
    dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plot entry point returns exactly evaluate_case's result; its points restate the ATE."""

    case, n, write_gt, evo = TRAJECTORY_CASES[dataset]
    if evo and not HAS_EVO:
        pytest.skip("evo not installed")
    gt = write_gt(tmp_path / "gt", case, n, monkeypatch)
    path = write_npz_prediction(tmp_path / "pred", case, estimate(gt, seed=5))
    module = registry.get(dataset)
    kwargs = {"protocol": module.DEFAULT_PROTOCOL, "stage": "local"}
    plain = module.evaluate_case(case, path, tmp_path / "gt", **kwargs)
    result, trajectory = module.evaluate_case_aligned(case, path, tmp_path / "gt", **kwargs)
    assert result == plain
    assert isinstance(trajectory, aligned.AlignedTrajectory)
    ate_m = plain["ate"] * plot.TO_METRES[module.UNITS]
    assert abs(trajectory.rmse() - ate_m) <= 1e-9 * max(1.0, ate_m)
    # The Sim(3) maps the raw prediction onto the drawn (aligned) points.
    with np.load(path) as archive:
        raw = np.asarray(archive["c2w"], dtype=np.float64)[trajectory.frame_ids, :3, 3]
    mapped = raw @ trajectory.sim3[:3, :3].T + trajectory.sim3[:3, 3]
    np.testing.assert_allclose(mapped, trajectory.pred_xyz, rtol=0, atol=1e-9 * max(1.0, float(np.abs(raw).max())))
    assert len(trajectory.gt_all_xyz) == len(gt)


# --------------------------------------------------------------------------- dataset specifics


@needs_evo
def test_tum_nan_gt_rows_are_not_drawn_as_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, n = "desk", 80
    gt = gt_tum(tmp_path / "gt", case, n, monkeypatch)
    write_npz_prediction(tmp_path / "pred", case, estimate(gt, seed=3))
    argv = ["--dataset", "tum", "--gt-root", str(tmp_path / "gt"), "--cases", case, "--pred", str(tmp_path / "pred")]
    scored = run_score(tmp_path, "score", argv)
    summary = run_plot(tmp_path / "figures", argv)
    assert_same_ate(scored, summary)
    figure = summary["run"]["per_case"][case]["figure"]
    valid = int(np.isfinite(gt.reshape(n, -1)).all(axis=1).sum())
    assert figure["frames_evaluated"] == figure["gt_frames_drawn"] == valid == n - 2
    assert figure["display_units"] == "cm" and figure["ate_shown"] == scored["runs"][0]["per_case"][case]["ate"] * 100
    with np.load(Path(summary["run"]["aligned_trajectories"])) as archive:
        assert np.isnan(archive[f"{case}/gt_all_xyz"][[3, 17]]).all()
        assert not np.isin([3, 17], archive[f"{case}/frame_ids"]).any()


@needs_evo
def test_wild_slam_cm_and_invalid_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, n = "ANYmal1", 70
    gt = gt_wild_slam(tmp_path / "gt", case, n, monkeypatch)
    write_vipe_output(tmp_path / "vipe", case, estimate(gt, seed=4))
    argv = ["--dataset", "wild_slam", "--gt-root", str(tmp_path / "gt"), "--cases", case,
            "--pred", str(tmp_path / "vipe"), "--pred-format", "vipe"]
    scored = run_score(tmp_path, "score", argv)
    summary = run_plot(tmp_path / "figures", argv)
    assert_same_ate(scored, summary)
    row = summary["run"]["per_case"][case]
    assert summary["units"] == wild_slam.UNITS == "cm" and row["figure"]["display_units"] == "cm"
    assert row["figure"]["ate_shown"] == row["ate"]  # already cm
    assert abs(row["figure"]["drawn_rmse_m"] * 100 - row["ate"]) < 1e-9
    assert row["figure"]["frames_evaluated"] == n - 3 and row["figure"]["alignment_backend"] == "evo"


@needs_evo
def test_vbr_sparse_subset_draws_full_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, n = "campus_train0", 150
    gt = gt_vbr(tmp_path / "gt", case, n, monkeypatch)
    keep = np.setdiff1d(np.arange(n), [5, 6, 40, 41, 42, 149])
    write_vipe_output(tmp_path / "vipe", case, estimate(gt, seed=12)[keep], inds=keep, record={"status": "success"})
    argv = ["--dataset", "vbr", "--gt-root", str(tmp_path / "gt"), "--cases", case,
            "--pred", str(tmp_path / "vipe"), "--pred-format", "vipe"]
    scored = run_score(tmp_path, "score", argv)
    summary = run_plot(tmp_path / "figures", argv)
    assert_same_ate(scored, summary)
    figure = summary["run"]["per_case"][case]["figure"]
    assert figure["frames_evaluated"] == len(keep) and figure["gt_frames_drawn"] == n
    with np.load(Path(summary["run"]["aligned_trajectories"])) as archive:
        np.testing.assert_array_equal(archive[f"{case}/frame_ids"], keep)
        np.testing.assert_array_equal(archive[f"{case}/gt_xyz"], gt[keep, :3, 3])
    assert vbr.NAME == summary["dataset"]


@needs_evo
def test_kitti_umeyama_protocol_and_plane(tmp_path: Path) -> None:
    """KITTI's historical ``umeyama`` layout (sim3_umeyama) and the --plane override."""

    case, n = "04", 90
    gt = gt_kitti(tmp_path / "gt", case, n)
    pred = estimate(gt, seed=8)
    (tmp_path / "hist" / "sequences" / case).mkdir(parents=True)
    np.savez(tmp_path / "hist" / "sequences" / case / "predictions.npz", c2w_cam0_baseline=pred, c2w_cam0=pred[::-1])
    argv = ["--dataset", "kitti", "--gt-root", str(tmp_path / "gt"), "--cases", case,
            "--pred", str(tmp_path / "hist"), "--protocol", "umeyama"]
    scored = run_score(tmp_path, "score", argv)
    summary = run_plot(tmp_path / "figures", [*argv, "--plane", "xy", "--units", "cm", "--label", "Ours"])
    assert_same_ate(scored, summary)
    figure = summary["run"]["per_case"][case]["figure"]
    assert summary["plane"] == ["x", "y"] and summary["label"] == "Ours"
    assert figure["alignment_backend"] == "umeyama"
    assert figure["ate_shown"] == scored["runs"][0]["per_case"][case]["ate"] * 100
    assert plot.GROUND_PLANES[kitti.NAME] == ("x", "z")


@needs_evo
def test_lc_stage_details_and_failed_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--stage LC reads pose_lc/; --details titles; a wrong record fails; --pred takes exactly one run."""

    case, n = "desk", 80
    gt = gt_tum(tmp_path / "gt", case, n, monkeypatch)
    root = tmp_path / "run"
    write_vipe_output(root, case, estimate(gt, seed=41), record={"status": "success"})
    write_vipe_output(root, case, estimate(gt, seed=51), pose_dir="pose_lc", record={"status": "success"})
    argv = ["--dataset", "tum", "--gt-root", str(tmp_path / "gt"), "--cases", case, "--stage", "LC",
            "--pred-format", "vipe", "--pred", str(root)]
    scored = run_score(tmp_path, "score", argv)
    record = tmp_path / "scores" / "score.json"
    summary = run_plot(tmp_path / "figures", [*argv, "--compare", str(record), "--compare-tol", "0", "--no-overview"])
    assert summary["stage"] == "LC" and summary["run"]["pred"] == scored["runs"][0]["pred"]
    assert summary["run"]["per_case"][case]["prediction"].endswith(f"pose_lc/{case}.npz")
    assert summary["run"]["compare"]["passed"] and "overview" not in summary["run"]
    assert {path.name for path in (tmp_path / "figures").iterdir()} == {
        f"{case}.png", "aligned_trajectories.npz", "summary.json"}  # no per-run sub-directories
    # Default style: no dataset and no "(after LC)" in the title.
    assert summary["details"] is False and summary["run"]["per_case"][case]["figure"]["title"] == "fr1/desk"
    assert_same_ate(scored, summary)
    # --details: the earlier title (dataset, sequence, stage); the same ATE.
    detailed = run_plot(tmp_path / "details", [*argv, "--details"])
    assert detailed["details"] is True and detailed["problems"] == []
    assert detailed["run"]["per_case"][case]["figure"]["title"] == "TUM RGB-D fr1/desk (after LC)"
    assert (tmp_path / "details" / "overview.png").read_bytes()[:8] == PNG_MAGIC
    assert_same_ate(scored, detailed)
    # A record that differs in the last bit fails --compare-tol 0.
    tampered = json.loads(record.read_text())
    tampered["runs"][0]["per_case"][case]["ate"] += 1e-12
    (tmp_path / "tampered.json").write_text(json.dumps(tampered))
    again = run_plot(tmp_path / "again", [*argv, "--compare", str(tmp_path / "tampered.json"), "--compare-tol", "0"],
                     expect=1)
    assert not again["run"]["compare"]["passed"] and again["problems"]
    assert tum.NAME == again["dataset"]
    # Exactly one --pred: a second directory, or --pred given twice, is an argparse error (exit 2), nothing is
    # written.
    for extra in ([str(root)], ["--pred", str(root)]):
        with pytest.raises(SystemExit) as error:
            plot.main([*argv, *extra, "--out", str(tmp_path / "two")])
        assert error.value.code == 2 and not (tmp_path / "two").exists()


@needs_evo
def test_default_out_and_legend_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --out: <project>/outputs/figures/<dataset>_<stage>[_vipe]; the legend spot is recorded
    (default: the row above the axes; --legend auto: a free corner; --legend outside: right of the axes)."""

    case, n = "04", 90
    gt = gt_kitti(tmp_path / "gt", case, n)
    write_vipe_output(tmp_path / "vipe", case, estimate(gt, seed=6))
    monkeypatch.setattr(plot, "REPO_ROOT", tmp_path / "project")
    argv = ["--dataset", "kitti", "--gt-root", str(tmp_path / "gt"), "--cases", case, "--pred", str(tmp_path / "vipe"),
            "--pred-format", "vipe", "--dpi", "40"]
    assert plot.main(argv) == 0
    summary = json.loads((tmp_path / "project" / "outputs" / "figures" / "kitti_local_vipe" / "summary.json").read_text())
    figure = summary["run"]["per_case"][case]["figure"]
    assert summary["legend"] == "top" and figure["legend"] == "top" and not figure["legend_covers_trajectory"]
    assert plot.main([*argv, "--legend", "auto", "--out", str(tmp_path / "auto")]) == 0
    summary = json.loads((tmp_path / "auto" / "summary.json").read_text())
    figure = summary["run"]["per_case"][case]["figure"]
    assert summary["legend"] == "auto" and figure["legend"] in plot.LEGEND_CORNERS
    assert not figure["legend_covers_trajectory"]
    assert plot.main([*argv, "--legend", "outside", "--out", str(tmp_path / "outside")]) == 0
    summary = json.loads((tmp_path / "outside" / "summary.json").read_text())
    assert summary["run"]["per_case"][case]["figure"]["legend"] == "outside right"


def _trajectory(xz: np.ndarray) -> aligned.AlignedTrajectory:
    xyz = np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]])
    ids = np.arange(len(xyz))
    return aligned.AlignedTrajectory(frame_ids=ids, gt_xyz=xyz, pred_xyz=xyz, gt_ids=ids, gt_all_xyz=xyz,
                                     sim3=np.eye(4), scale=1.0, backend="test")


def _spy_axes(monkeypatch: pytest.MonkeyPatch) -> list:
    """The axes save_figure draws on (captured when the legend is placed)."""

    seen: list = []
    original = plot.place_legend

    def spy(figure: object, axis: object, *args: object) -> dict:
        seen.append(axis)
        return original(figure, axis, *args)

    monkeypatch.setattr(plot, "place_legend", spy)
    return seen


@pytest.mark.parametrize("details", [False, True])
def test_figure_style(details: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: title as given (the sequence name), x label only, legend GT / prediction in one row between
    the title and the axes, no start markers.  --details: also the vertical axis label, the legend headed by
    the ATE text, and start markers."""

    seen = _spy_axes(monkeypatch)
    t = np.linspace(0.0, 1.0, 200)
    trajectory = _trajectory(np.column_stack([40.0 * t, 100.0 * t ** 2]))
    ate_text = "ATE 2.77 m\n4,541 frames" if details else None
    plot.save_figure(trajectory, ("x", "z"), [tmp_path / "style.png"], dpi=40, label="Prediction", title="00",
                     ate_text=ate_text, details=details)
    (axis,) = seen
    legend = axis.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert axis.get_title() == "00" and axis.get_xlabel() == "world x (m)"
    if details:
        assert labels == ["GT", "Prediction", "start"] and legend.get_title().get_text() == ate_text
        assert axis.get_ylabel() == "world z (m)"
        assert len(axis.get_lines()) == 5  # GT, prediction, two start markers, the legend's start handle
    else:
        assert labels == ["GT", "Prediction"] and legend.get_title().get_text() == ""
        assert axis.get_ylabel() == ""
        assert len(axis.get_lines()) == 2  # GT and prediction only
    assert (tmp_path / "style.png").read_bytes()[:8] == PNG_MAGIC
    # The legend row: above the axes, below the title (also with the two-line ATE heading), all entries in
    # one row, smaller than the title, narrower than the axes.
    figure = axis.figure
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    title_box = axis.title.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    axes_box = axis.get_window_extent(renderer)
    assert axes_box.y1 < legend_box.y0 < legend_box.y1 < title_box.y0
    entries = [text.get_window_extent(renderer) for text in legend.get_texts()]
    assert max(box.y0 for box in entries) < min(box.y1 for box in entries)  # one row
    assert all(box.x0 > axes_box.x0 and box.x1 < axes_box.x1 for box in entries)
    assert legend.get_texts()[0].get_fontsize() < axis.title.get_fontsize()
    assert legend_box.width < 0.6 * axes_box.width


@pytest.mark.parametrize("details", [False, True])
def test_legend_placement(details: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    common = {"dpi": 40, "label": "Prediction", "title": "test", "details": details,
              "ate_text": "ATE 1.00 m\n1,600 frames" if details else None}
    # The default row above the axes never covers the trajectory, whatever its shape.
    t = np.linspace(0.0, 1.0, 400)
    line = _trajectory(np.column_stack([0.0 * t, 100.0 * t]))
    assert plot.save_figure(line, ("x", "z"), [tmp_path / "top.png"], **common) == {
        "legend": "top", "frame_growth": 0.0, "legend_covers_trajectory": False}
    # --legend auto, without widening, so that the fallbacks do not depend on font metrics.
    monkeypatch.setattr(plot, "FRAME_GROWTH", (0.0,))
    # A straight line leaves the lower right corner of its square frame free: the reference spot, frame as is.
    assert plot.save_figure(line, ("x", "z"), [tmp_path / "line.png"], legend="auto", **common) == {
        "legend": "lower right", "frame_growth": 0.0, "legend_covers_trajectory": False}
    # A square ring along the frame covers every corner.
    ring = _trajectory(100.0 * np.concatenate([
        np.column_stack([t, 0 * t]), np.column_stack([1 + 0 * t, t]),
        np.column_stack([1 - t, 1 + 0 * t]), np.column_stack([0 * t, 1 - t]),
    ]))
    assert plot.save_figure(ring, ("x", "z"), [tmp_path / "auto.png"], legend="auto", **common)["legend"] == (
        "outside right")
    assert plot.save_figure(ring, ("x", "z"), [tmp_path / "inside.png"], legend="inside", **common) == {
        "legend": "lower right", "frame_growth": 0.0, "legend_covers_trajectory": True}
    assert plot.save_figure(line, ("x", "z"), [tmp_path / "out.png"], legend="outside", **common)["legend"] == (
        "outside right")
    # Widening the frame towards the lower right corner frees it.
    monkeypatch.setattr(plot, "FRAME_GROWTH", (0.0, 0.5))
    assert plot.save_figure(ring, ("x", "z"), [tmp_path / "wide.png"], legend="auto", **common) == {
        "legend": "lower right", "frame_growth": 0.5, "legend_covers_trajectory": False}
    # The frame is square around the trajectory's centre (half the largest span x 1.07).
    (x0, x1), (y0, y1) = plot._limits(np.column_stack([0.0 * t, 100.0 * t]))
    assert x1 - x0 == pytest.approx(107.0) and y1 - y0 == pytest.approx(107.0) and x0 + x1 == pytest.approx(0.0)


def test_overview_title() -> None:
    run = {"mean_ate": 0.04606, "num_cases": 9}
    assert plot.overview_title(tum, "local", "Prediction", run, "cm") == "TUM RGB-D: Prediction, mean ATE 4.61 cm (n=9)"
    run = {"mean_ate": 2.771322, "num_cases": 11}
    assert plot.overview_title(kitti, "LC", "Ours", run, "m") == "KITTI (after LC): Ours, mean ATE 2.77 m (n=11)"


def test_format_and_planes() -> None:
    assert plot.format_value(13.8834) == "13.88" and plot.format_value(123.45) == "123.5"
    assert plot.format_value(0.04606) == "0.0461" and plot.format_value(float("nan")) == "nan"
    assert set(plot.GROUND_PLANES) == set(registry.REGISTRY)
    for horizontal, vertical in plot.GROUND_PLANES.values():
        assert horizontal != vertical
    # Default titles: the sequence as the dataset names it; KITTI's bare numbers with the dataset.
    titles = {
        ("kitti", "00"): "KITTI 00",
        ("vbr", "campus_train0"): "campus_train0",
        ("tum", "desk"): "fr1/desk",
        ("tum", "360"): "fr1/360",
        ("7scenes", "chess__seq-03"): "chess/seq-03",
        ("wild_slam", "ANYmal1"): "ANYmal1",
        ("oxford", "keble-college-02"): "keble-college-02",
    }
    for (dataset, case), title in titles.items():
        assert plot.case_title(dataset, case) == title, (dataset, case)
    # --details: "<dataset> <sequence>" (no doubled dataset for KITTI).
    assert plot.sequence_name("kitti", "00") == "00" and plot.sequence_name("7scenes", "chess__seq-03") == "chess/seq-03"
