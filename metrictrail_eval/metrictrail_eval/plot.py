"""Plot each sequence's GT and Sim(3)-aligned prediction with the ATE that ``score`` reports.

    python -m metrictrail_eval.plot --dataset kitti --pred-format vipe --pred $VIPE/kitti --stage LC
    python -m metrictrail_eval.plot --dataset tum --pred $MT/outputs/tum_uncalibrated --out outputs/figures/tum

The prediction options are those of ``score`` (``--dataset``, ``--pred``,
``--pred-format npz|vipe``, ``--gt-root``, ``--stage local|LC``, ``--protocol``,
``--cases``, ``--vipe-name``, ``--work-dir``, ``--allow-partial``).  Every
sequence is scored by ``score.score_run`` itself: the same prediction reader,
frame matching, alignment, units and mean.  The only difference is that the
dataset module's ``evaluate_case_aligned`` is called, which runs the code of
``evaluate_case`` and returns the same result plus the aligned trajectory the
ATE was computed on (``aligned.py``).  Each figure re-derives the RMSE from the
drawn points and the run fails if it differs from the scored ATE (beyond
floating-point rounding).

Figures are top-down views of the dataset's ground plane, chosen from the GT
world frame's up axis (``--plane auto``): KITTI and 7-Scenes (y down) x-z,
Wild-SLAM MoCap (y up) z-x, VBR / Oxford-Spires / LaMaria / TUM (z up) x-y.
Positions are in metres; the ATE (``summary.json``, the overview, and the
figures with ``--details``) uses the dataset's table unit (``--units auto``:
cm for TUM, 7-Scenes and Wild-SLAM, m otherwise).

Default style (paper figures): the title is the sequence's own name as the
dataset names it (``campus_train0``, ``fr1/desk``, ``chess/seq-03``; KITTI's
bare numbers with the dataset, ``KITTI 00``; other datasets the sequence key
as is), the horizontal axis is labelled, the vertical one is not, and the
legend has two entries, ``GT`` and the prediction (``--label``, default
``Prediction``), in one small row between the title and the axes, so that it
never covers the plot (``--legend top``).  ``--details`` adds what the earlier
figures showed: ``<dataset> <sequence>`` and ``(after LC)`` in the title, the
vertical axis label, the ATE and the number of evaluated frames as the legend
title, and start markers.

Each figure is framed square around the trajectory like the paper figures.
``--legend auto`` puts the legend inside instead, in the first corner where it
covers no trajectory (lower right, upper right, lower left, upper left),
widening the frame towards that corner if needed, and only then to the right
of the axes (``inside`` never moves it out, ``outside`` always does).

``--pred`` is one run.  Written to ``--out`` (default
``<project>/outputs/figures/<dataset>_<stage>[_vipe]``): ``<case>.png`` (and
``<case>.pdf`` with ``--pdf``), ``overview.png`` (all sequences, titled with
the mean ATE), ``aligned_trajectories.npz`` (``<case>/{frame_ids, gt_xyz,
pred_xyz, gt_ids, gt_all_xyz, sim3}``) and ``summary.json``: the score result
of every sequence (as in ``score``'s JSON) plus the figure, the ATE shown and
the checks.  ``--check-score`` also scores the same directory with the plain
``score.score_run`` in the same call and requires identical results;
``--compare SCORE_JSON`` compares the ATEs with a JSON written by ``score``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import score, sim3_ate, vipe_output
from .aligned import AlignedTrajectory
from .common import REPO_ROOT, atomic_write_json
from .registry import get

SCHEMA = "metrictrail_eval_plot_v1"
AXES = {"x": 0, "y": 1, "z": 2}
#: Top-down view per dataset: (horizontal, vertical) world axes, right-handed seen from above.
GROUND_PLANES = {
    "kitti": ("x", "z"),  # camera frame of frame 0: y points down
    "7scenes": ("x", "z"),  # KinectFusion world: y points (roughly) down
    "wild_slam": ("z", "x"),  # MoCap world: y points up
    "vbr": ("x", "y"),  # z up
    "oxford": ("x", "y"),  # z up
    "lamaria": ("x", "y"),  # z up
    "tum": ("x", "y"),  # MoCap world: z up
}
PLANES = ("auto", "xy", "yx", "xz", "zx", "yz", "zy")
#: Unit of the ATE label per dataset (the tables' unit); the score JSON keeps the scorer's ``UNITS``.
DISPLAY_UNITS = {"tum": "cm", "7scenes": "cm", "wild_slam": "cm"}
TO_METRES = {"m": 1.0, "cm": 0.01}
TITLES = {
    "kitti": "KITTI",
    "vbr": "VBR",
    "oxford": "Oxford-Spires",
    "lamaria": "LaMaria",
    "tum": "TUM RGB-D",
    "7scenes": "7-Scenes",
    "wild_slam": "Wild-SLAM",
}
#: |drawn RMSE - scored ATE| allowed, relative to max(1 m, ATE): only floating-point rounding.
DRAWN_RMSE_RTOL = 1e-9

# Paper style of the earlier trajectory figures: GT grey and thick under a thinner green prediction.
GT_COLOR = "#777777"
PRED_COLOR = "#72C83D"
GT_LINE_WIDTH = 7.0
GT_ALPHA = 0.50
PRED_LINE_WIDTH = 3.2
FIGSIZE = (7.4, 7.0)
STYLE = {
    "font.size": 17,
    "axes.titlesize": 18,
    "axes.labelsize": 27,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 14,
}
LEGEND_STYLE = {
    "title_fontproperties": {"weight": "bold", "size": 16.0},
    "alignment": "left",
    "frameon": True,
    "framealpha": 0.92,
    "borderaxespad": 0.35,
    "borderpad": 0.4,
    "labelspacing": 0.25,
    "handlelength": 1.8,
    "handletextpad": 0.45,
}
#: Square frame around the trajectory, padded like the earlier KITTI figures (radius = half the largest span x 1.07).
FRAME_PADDING = 0.035
#: Legend spots tried in this order (the earlier figures: KITTI lower right, LaMaria upper right).
LEGEND_CORNERS = ("lower right", "upper right", "lower left", "upper left")
#: Widening of the square frame towards the legend's corner (fraction of its side) when no corner is free.
FRAME_GROWTH = (0.0, 0.1, 0.2)
LEGEND_MODES = ("top", "auto", "inside", "outside")
#: ``--legend top`` (default): one small row between the title and the axes, so it never covers a trajectory.
TOP_LEGEND_STYLE = {
    "frameon": False,
    "fontsize": 12,
    "title_fontsize": 12,
    "handlelength": 1.6,
    "handletextpad": 0.4,
    "columnspacing": 1.4,
    "borderaxespad": 0.15,
    "borderpad": 0.1,
}
#: Gap (points) from the top of the legend row to the title's baseline (room for descenders such as p, _).
TOP_LEGEND_TITLE_GAP = 7.0
OVERVIEW_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m metrictrail_eval.plot",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    score.add_prediction_arguments(parser)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output directory (default: <project>/outputs/figures/<dataset>_<stage>[_vipe])",
    )
    parser.add_argument("--label", default="Prediction", help="legend label of the prediction")
    parser.add_argument(
        "--details", action="store_true",
        help="the earlier detailed style: '<dataset> <sequence> (after LC)' title, vertical axis label, legend "
        "headed by the ATE and the number of evaluated frames, start markers (default: title = sequence name, "
        "no vertical axis label, legend GT / prediction only, no start markers)",
    )
    parser.add_argument(
        "--legend", choices=LEGEND_MODES, default="top",
        help="top (default): one small row between the title and the axes; auto: inside, in the first corner "
        "free of the trajectory (lower right, upper right, lower left, upper left; the square frame widened "
        "towards it if needed), else right of the axes; inside: never outside (lower right when no corner is "
        "free); outside: always right of the axes",
    )
    parser.add_argument(
        "--plane", choices=PLANES, default="auto",
        help="(horizontal, vertical) world axes; auto = the dataset's ground plane seen from above",
    )
    parser.add_argument(
        "--units", choices=("auto", "m", "cm"), default="auto",
        help="unit of the shown ATE (summary.json, overview, --details legend; auto: cm for tum / 7scenes / "
        "wild_slam, m otherwise)",
    )
    parser.add_argument("--pdf", action="store_true", help="also write <case>.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-overview", action="store_true", help="do not write overview.png")
    parser.add_argument(
        "--check-score", action="store_true",
        help="also score the same directory with score.score_run in this call and require identical results",
    )
    parser.add_argument(
        "--compare", type=Path, default=None, metavar="SCORE_JSON",
        help="compare every ATE with this JSON written by score (same dataset, stage and protocol)",
    )
    parser.add_argument(
        "--compare-tol", type=float, default=1e-9,
        help="--compare: largest allowed |ATE difference| in the score unit (default 1e-9)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


# --------------------------------------------------------------------------- formatting


def format_value(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def sequence_name(dataset: str, case: str) -> str:
    """The sequence as the dataset names it (``campus_train0``, ``fr1/desk``, ``chess/seq-03``, ``00``)."""

    if dataset == "7scenes":
        return case.replace("__", "/")
    if dataset == "tum":
        return f"fr1/{case}"
    return case


def case_title(dataset: str, case: str) -> str:
    """Default figure title: the sequence's name; KITTI's bare numbers as ``KITTI 00``."""

    name = sequence_name(dataset, case)
    return f"KITTI {name}" if dataset == "kitti" else name


def plane_axes(dataset: str, plane: str) -> tuple[str, str]:
    if plane == "auto":
        return GROUND_PLANES[dataset]
    return plane[0], plane[1]


# --------------------------------------------------------------------------- drawing


Limits = tuple[tuple[float, float], tuple[float, float]]


def _limits(points: np.ndarray, pad: float = FRAME_PADDING) -> Limits:
    """Square frame centred on the trajectory: half its largest span x (1 + 2 pad), as the earlier figures."""

    finite = points[np.isfinite(points).all(axis=1)]
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = 0.5 * max(float((hi - lo).max()), 1e-3) * (1.0 + 2.0 * pad)
    return (
        (float(centre[0] - radius), float(centre[0] + radius)),
        (float(centre[1] - radius), float(centre[1] + radius)),
    )


def _widened(limits: Limits, corner: str, growth: float) -> Limits:
    """The square frame widened by ``growth`` x its side towards ``corner`` (it stays square)."""

    (x0, x1), (y0, y1) = limits
    extra = growth * (x1 - x0)
    x0, x1 = (x0, x1 + extra) if "right" in corner else (x0 - extra, x1)
    y0, y1 = (y0 - extra, y1) if "lower" in corner else (y0, y1 + extra)
    return (x0, x1), (y0, y1)


def _set_limits(axis: Any, limits: Limits) -> None:
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])


def draw(
    axis: Any,
    trajectory: AlignedTrajectory,
    axes: tuple[str, str],
    *,
    label: str,
    title: str,
    size: float = 1.0,
    details: bool = False,
) -> Limits:
    """GT (grey, thick; frames without GT break the line) under the aligned prediction (green), equal axes
    in metres, square frame; the legend entries are GT and ``label`` only.  ``details``: also start markers
    (with a ``start`` legend entry) and the vertical axis label.  Returns the frame; the legend is placed
    by the caller."""

    h, v = AXES[axes[0]], AXES[axes[1]]
    gt = trajectory.gt_all_xyz[:, (h, v)]
    pred = trajectory.pred_xyz[:, (h, v)]
    limits = _limits(np.concatenate([gt, pred]))
    axis.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, linewidth=GT_LINE_WIDTH * size, alpha=GT_ALPHA,
              solid_capstyle="round", solid_joinstyle="round", label="GT", zorder=2)
    axis.plot(pred[:, 0], pred[:, 1], color=PRED_COLOR, linewidth=PRED_LINE_WIDTH * size, alpha=1.0,
              solid_capstyle="round", solid_joinstyle="round", label=label, zorder=10)
    if details:
        gt_start = gt[np.isfinite(gt).all(axis=1)][0]
        marker = {"marker": "o", "markeredgecolor": "black", "markeredgewidth": 1.2 * size, "linestyle": "none"}
        axis.plot(*gt_start, markersize=11.0 * size, markerfacecolor=GT_COLOR, zorder=12, **marker)
        axis.plot(*pred[0], markersize=8.0 * size, markerfacecolor=PRED_COLOR, zorder=13, **marker)
        axis.plot([], [], markersize=8.0 * size, markerfacecolor="white", label="start", **marker)
    _set_limits(axis, limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.55, alpha=0.25)
    axis.set_xlabel(f"world {axes[0]} (m)")
    if details:
        axis.set_ylabel(f"world {axes[1]} (m)")
    axis.set_title(title, fontweight="bold", pad=10)
    return limits


def _legend(axis: Any, loc: str, ate_text: str | None, **where: Any) -> Any:
    legend = axis.legend(loc=loc, title=ate_text, **where, **LEGEND_STYLE)
    legend.get_frame().set_edgecolor("#C9CED6")
    legend.set_zorder(20)
    return legend


def _covers_trajectory(figure: Any, axis: Any, legend: Any) -> bool:
    """Whether the legend box (padded by half the GT line width) contains a drawn trajectory point."""

    figure.draw_without_rendering()  # settles the layout and the legend's position
    box = legend.get_window_extent(figure.canvas.get_renderer()).padded(GT_LINE_WIDTH * figure.dpi / 72.0 / 2)
    for line in axis.get_lines():
        xy = np.asarray(line.get_xydata(), dtype=np.float64).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) and box.count_contains(axis.transData.transform(xy)) > 0:
            return True
    return False


def place_legend(figure: Any, axis: Any, limits: Limits, ate_text: str | None, mode: str) -> dict[str, Any]:
    """The legend (headed by ``ate_text`` unless None).  ``top``: one row of entries centred between the axes
    and the title, which is raised by the legend's measured height (an ATE heading adds lines).  Otherwise
    inside the axes like the paper figures: the first of ``LEGEND_CORNERS`` where it covers no trajectory
    point, with the square frame widened towards that corner by the first sufficient ``FRAME_GROWTH``.  When no
    spot is free: right of the axes (``auto``) or the lower right corner anyway (``inside``).  ``outside``:
    always right of the axes."""

    if mode == "top":
        labels = axis.get_legend_handles_labels()[1]
        legend = axis.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=max(1, len(labels)), title=ate_text,
            **TOP_LEGEND_STYLE,
        )
        legend.set_zorder(20)
        # Height of the legend's top above the axes in points (independent of the axes' final size).
        figure.draw_without_rendering()
        renderer = figure.canvas.get_renderer()
        rise = (legend.get_window_extent(renderer).y1 - axis.get_window_extent(renderer).y1) * 72.0 / figure.dpi
        axis.set_title(axis.get_title(), fontweight="bold", pad=rise + TOP_LEGEND_TITLE_GAP)
        return {"legend": "top", "frame_growth": 0.0, "legend_covers_trajectory": False}
    if mode != "outside":
        for growth in FRAME_GROWTH:
            for corner in LEGEND_CORNERS:
                _set_limits(axis, _widened(limits, corner, growth))
                legend = _legend(axis, corner, ate_text)
                if not _covers_trajectory(figure, axis, legend):
                    return {"legend": corner, "frame_growth": growth, "legend_covers_trajectory": False}
                legend.remove()
        _set_limits(axis, limits)
        if mode == "inside":
            _legend(axis, LEGEND_CORNERS[0], ate_text)
            return {"legend": LEGEND_CORNERS[0], "frame_growth": 0.0, "legend_covers_trajectory": True}
    # Right of the axes (constrained layout makes room; the equal-aspect box keeps its left edge).
    axis.set_anchor("W")
    _legend(axis, "upper left", ate_text, bbox_to_anchor=(1.02, 1.0))
    return {"legend": "outside right", "frame_growth": 0.0, "legend_covers_trajectory": False}


def save_figure(
    trajectory: AlignedTrajectory, axes: tuple[str, str], paths: Sequence[Path], *, dpi: int, label: str,
    title: str, ate_text: str | None = None, legend: str = "top", details: bool = False,
) -> dict[str, Any]:
    """One sequence's figure; returns where the legend went (``place_legend``).  ``ate_text`` heads the
    legend when given (``--details``); ``details`` also draws the start markers and the vertical axis label."""

    import matplotlib
    import matplotlib.style
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # matplotlib defaults (evo, imported by the scorers, switches the global style to seaborn's) + paper sizes.
    with matplotlib.style.context("default"), matplotlib.rc_context(STYLE):
        figure = Figure(figsize=FIGSIZE, layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        limits = draw(axis, trajectory, axes, label=label, title=title, details=details)
        placement = place_legend(figure, axis, limits, ate_text, legend)
        for path in paths:
            figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    return placement


def save_overview(
    panels: list[tuple[AlignedTrajectory, str, str]], axes: tuple[str, str], path: Path, *, label: str,
    title: str, dpi: int = 120, details: bool = False,
) -> None:
    """All sequences in one grid; every panel is titled with its sequence and ATE, the figure with the
    run's mean ATE.  ``details``: start markers and vertical axis labels as in the ``--details`` figures."""

    import matplotlib
    import matplotlib.style
    from matplotlib.figure import Figure

    count = len(panels)
    columns = min(count, 4 if count <= 12 else 5 if count <= 20 else 6)
    rows = math.ceil(count / columns)
    with matplotlib.style.context("default"), matplotlib.rc_context(OVERVIEW_STYLE):
        figure = Figure(figsize=(3.9 * columns, 3.8 * rows + 0.5), layout="constrained")
        for index, (trajectory, panel_title, ate_text) in enumerate(panels):
            axis = figure.add_subplot(rows, columns, index + 1)
            draw(axis, trajectory, axes, label=label, title=f"{panel_title}\n{ate_text}", size=0.45, details=details)
            if index == 0:
                _legend(axis, "best", None)
        figure.suptitle(title, fontweight="bold", fontsize=13)
        figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)


# --------------------------------------------------------------------------- checks


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def compare_results(plotted: dict[str, Any], scored: dict[str, Any]) -> dict[str, Any]:
    """Every per-sequence field of ``score.score_run`` must be identical in the plotted run."""

    differences: list[str] = []
    for case, expected in scored["per_case"].items():
        got = plotted["per_case"].get(case)
        if got is None:
            differences.append(f"{case}: not plotted")
            continue
        for key, value in expected.items():
            if not _same(got.get(key), value):
                differences.append(f"{case}.{key}: {got.get(key)!r} != {value!r}")
    for key in ("missing", "mean_ate", "num_cases"):
        if not _same(plotted.get(key), scored.get(key)):
            differences.append(f"{key}: {plotted.get(key)!r} != {scored.get(key)!r}")
    return {
        "identical": not differences,
        "differences": differences,
        "per_case_ate": {case: row["ate"] for case, row in scored["per_case"].items()},
        "mean_ate": scored.get("mean_ate"),
    }


def compare_with_json(
    run: dict[str, Any], path: Path, *, dataset: str, stage: str, protocol: str, tolerance: float
) -> dict[str, Any]:
    recorded = json.loads(path.read_text(encoding="utf-8"))
    for key, value in (("dataset", dataset), ("stage", stage), ("protocol", protocol), ("metric", "trajectory")):
        if recorded.get(key, value) != value:
            raise SystemExit(f"--compare {path}: {key} is {recorded.get(key)!r}, this run is {value!r}")
    candidates = [item for item in recorded.get("runs", []) if item.get("pred") == run["pred"]]
    note = "matched by pred path"
    if not candidates and len(recorded.get("runs", [])) == 1:
        candidates, note = recorded["runs"], "the JSON's only run (pred path differs)"
    if not candidates:
        raise SystemExit(f"--compare {path}: no run with pred {run['pred']}")
    reference = candidates[0]
    per_case: dict[str, Any] = {}
    worst = 0.0
    for case, row in run["per_case"].items():
        other = reference.get("per_case", {}).get(case)
        if other is None:
            per_case[case] = {"plotted": row["ate"], "recorded": None, "abs_diff": None, "identical": False}
            worst = math.inf
            continue
        diff = abs(row["ate"] - other["ate"])
        worst = max(worst, diff)
        per_case[case] = {"plotted": row["ate"], "recorded": other["ate"], "abs_diff": diff,
                          "identical": row["ate"] == other["ate"]}
    # The means are comparable only over the same sequences (e.g. not with --cases on a subset).
    same_cases = set(per_case) == set(reference.get("per_case", {}))
    mean_diff = (
        abs(run["mean_ate"] - reference["mean_ate"])
        if same_cases and reference.get("mean_ate") is not None else None
    )
    return {
        "score_json": str(path),
        "match": note,
        "tolerance": tolerance,
        "passed": worst <= tolerance and (mean_diff is None or mean_diff <= tolerance),
        "identical_cases": sum(1 for item in per_case.values() if item["identical"]),
        "cases": len(per_case),
        "recorded_cases": sorted(reference.get("per_case", {})),
        "max_abs_diff": worst,
        "mean_plotted": run["mean_ate"],
        "mean_recorded": reference.get("mean_ate"),
        "mean_abs_diff": mean_diff,
        "per_case": per_case,
    }


# --------------------------------------------------------------------------- main


def plot_run(
    module: Any, pred: Path, out_dir: Path, *, cases: Sequence[str], gt_root: Path, stage: str, protocol: str,
    allow_partial: bool, reader: score.PredictionReader, axes: tuple[str, str], display_units: str, label: str,
    pdf: bool, dpi: int, legend: str = "top", details: bool = False,
) -> tuple[dict[str, Any], list[str], list[tuple[AlignedTrajectory, str, str]]]:
    """Score one prediction directory through ``score.score_run`` and draw every sequence.

    Returns the run (``score_run``'s result + figure details), the problems and the overview panels."""

    out_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    panels: list[tuple[AlignedTrajectory, str, str]] = []
    to_display = TO_METRES[module.UNITS] / TO_METRES[display_units]
    suffix = " (after LC)" if stage == "LC" else ""

    def on_aligned(case: str, result: dict[str, Any], trajectory: AlignedTrajectory | None) -> None:
        if trajectory is None:
            problems.append(f"{case}: the scorer returned no aligned trajectory (too few frames)")
            result["figure"] = None
            return
        ate = float(result["ate"])
        drawn = trajectory.rmse()
        diff = abs(drawn - ate * TO_METRES[module.UNITS])
        if not diff <= DRAWN_RMSE_RTOL * max(1.0, ate * TO_METRES[module.UNITS]):
            problems.append(f"{case}: drawn RMSE {drawn!r} m differs from the scored ATE {ate!r} {module.UNITS}")
        shown = ate * to_display
        if details:
            ate_text = f"ATE {format_value(shown)} {display_units}\n{len(trajectory.frame_ids):,} frames"
            title = f"{TITLES[module.NAME]} {sequence_name(module.NAME, case)}{suffix}"
        else:
            ate_text, title = None, case_title(module.NAME, case)
        paths = [out_dir / f"{case}.png"] + ([out_dir / f"{case}.pdf"] if pdf else [])
        placement = save_figure(trajectory, axes, paths, dpi=dpi, label=label, ate_text=ate_text, title=title,
                                legend=legend, details=details)
        panels.append((trajectory, case_title(module.NAME, case), f"ATE {format_value(shown)} {display_units}"))
        for key, value in trajectory.arrays().items():
            arrays[f"{case}/{key}"] = value
        result["figure"] = {
            "png": str(paths[0]),
            "pdf": str(paths[1]) if pdf else None,
            "plane": list(axes),
            "title": title,
            **placement,
            "ate_shown": shown,
            "ate_shown_text": format_value(shown),
            "display_units": display_units,
            "frames_evaluated": int(len(trajectory.frame_ids)),
            "gt_frames_drawn": int(np.isfinite(trajectory.gt_all_xyz).all(axis=1).sum()),
            "alignment_backend": trajectory.backend,
            "sim3": trajectory.sim3.tolist(),
            "sim3_scale": trajectory.scale,
            "drawn_rmse_m": drawn,
            "drawn_rmse_abs_diff_m": diff,
        }

    run = score.score_run(
        module, pred, cases=cases, gt_root=gt_root, stage=stage, protocol=protocol,
        allow_partial=allow_partial, reader=reader, on_aligned=on_aligned,
    )
    run["out_dir"] = str(out_dir)
    if arrays:
        np.savez_compressed(out_dir / "aligned_trajectories.npz", **arrays)
        run["aligned_trajectories"] = str(out_dir / "aligned_trajectories.npz")
    run["mean_ate_shown"] = run["mean_ate"] * to_display
    figures = sum(1 for row in run["per_case"].values() if row.get("figure"))
    print(f"  wrote {figures} figure(s) to {out_dir}", flush=True)
    return run, problems, panels


def overview_title(module: Any, stage: str, label: str, run: dict[str, Any], display_units: str) -> str:
    """``<dataset>[ (after LC)]: <label>, mean ATE <value> <unit> (n=<sequences>)``."""

    to_display = TO_METRES[module.UNITS] / TO_METRES[display_units]
    suffix = " (after LC)" if stage == "LC" else ""
    return (f"{TITLES[module.NAME]}{suffix}: {label}, mean ATE {format_value(run['mean_ate'] * to_display)} "
            f"{display_units} (n={run['num_cases']})")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    module = get(args.dataset)
    protocol, gt_root = score.resolve_protocol_and_gt_root(args, module, points_metric=False)
    cases = score._select_cases(module, args.cases)
    reader = score._prediction_reader(args, module)
    axes = plane_axes(module.NAME, args.plane)
    display_units = DISPLAY_UNITS.get(module.NAME, "m") if args.units == "auto" else args.units
    vipe_format = reader.format == vipe_output.FORMAT
    out = (
        args.out.expanduser().resolve()
        if args.out is not None
        else REPO_ROOT / "outputs" / "figures" / f"{module.NAME}_{args.stage}{'_vipe' if vipe_format else ''}"
    )
    pred = args.pred.expanduser().resolve()
    fmt = f", {vipe_output.FORMAT}" if vipe_format else ""

    print(f"{module.NAME} [{args.stage}, trajectory, {protocol}{fmt}] {pred} -> {out}", flush=True)
    run, problems, panels = plot_run(
        module, pred, out, cases=cases, gt_root=gt_root, stage=args.stage, protocol=protocol,
        allow_partial=args.allow_partial, reader=reader, axes=axes, display_units=display_units,
        label=args.label, pdf=args.pdf, dpi=args.dpi, legend=args.legend, details=args.details,
    )
    if args.check_score:
        print(f"--check-score: score.score_run on {pred}", flush=True)
        scored = score.score_run(
            module, pred, cases=cases, gt_root=gt_root, stage=args.stage, protocol=protocol,
            allow_partial=args.allow_partial, reader=reader,
        )
        run["score_check"] = compare_results(run, scored)
        if not run["score_check"]["identical"]:
            problems += [f"score check: {item}" for item in run["score_check"]["differences"]]
        print(f"  score check: {'identical' if run['score_check']['identical'] else 'DIFFERENT'}", flush=True)
    if args.compare is not None:
        check = run["compare"] = compare_with_json(
            run, args.compare.expanduser().resolve(), dataset=module.NAME, stage=args.stage, protocol=protocol,
            tolerance=args.compare_tol,
        )
        mean_text = "not compared (other sequences)" if check["mean_abs_diff"] is None else (
            f"{check['mean_abs_diff']:.3g}")
        print(f"  compare with {check['score_json']}: {check['identical_cases']}/{check['cases']} identical, "
              f"max |diff| {check['max_abs_diff']:.3g} {module.UNITS}, mean |diff| {mean_text}"
              f" -> {'PASS' if check['passed'] else 'FAIL'}", flush=True)
        if not check["passed"]:
            problems.append(f"--compare {check['score_json']} failed")
    if not args.no_overview and panels:
        title = overview_title(module, args.stage, args.label, run, display_units)
        save_overview(panels, axes, out / "overview.png", label=args.label, title=title, details=args.details)
        run["overview"] = str(out / "overview.png")

    summary = {
        "schema": SCHEMA,
        "dataset": module.NAME,
        "stage": args.stage,
        "metric": "trajectory",
        "protocol": protocol,
        "units": module.UNITS,
        "display_units": display_units,
        "pred_format": reader.format,
        "gt_root": str(gt_root),
        "cases": list(cases),
        "plane": list(axes),
        "label": args.label,
        "legend": args.legend,
        "details": args.details,
        "run": run,
        "problems": problems,
        "evaluator": {**sim3_ate.backend_identity(), "python": platform.python_version(), "module": module.__name__},
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(out / "summary.json", summary)
    print(f"wrote {out / 'summary.json'}", flush=True)
    for item in problems:
        print(f"ERROR {item}", file=sys.stderr, flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
