"""Score MetricTrail predictions: per-sequence and mean Sim(3) ATE (default), or
point-cloud metrics (``--metric points``, Oxford-Spires and 7-Scenes).

    python -m metrictrail_eval.score --dataset kitti --pred outputs/kitti_uncalibrated --stage LC
    python -m metrictrail_eval.score --dataset 7scenes --metric points --pred outputs/sevenscenes_uncalibrated

``--pred`` is one MetricTrail run directory (``<stage>/inference/...``) or a
plain directory of ``<case>.npz`` predictions.
Paths given on the command line are relative to the working directory;
without ``--gt-root`` the dataset's entry in the package's ``datasets.json`` is used,
resolved against the repository root.  The result is printed and written as
JSON (``--out``, default ``<repo>/outputs/scores/<dataset>_<stage>[_points].json``).
For ``--metric points`` the depth comes from the prediction (``depth_m``) or
from the run's frontend depth cache (``frontend-cache/<dataset>/<case>`` above
the prediction file, or ``--depth-cache``); the GT root defaults to the
dataset's ``points_gt_root``.

``--pred-format vipe`` scores ViPE output directories instead
(``pose/<name>.npz``, ``pose_lc/<name>.npz`` for ``--stage LC``,
``intrinsics/<name>.npz``, ``depth/<name>.zip``; ``<name>`` = the sequence key
unless ``--vipe-name CASE=NAME``).  Each sequence is converted into the plain
prediction file (and, for points, the depth cache) that the dataset scorer reads,
in a scratch directory (``--work-dir``, default the system temp directory) that
is removed afterwards; the protocols run unchanged (``vipe_output``).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import logging
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from . import sim3_ate, vipe_output
from .common import REPO_ROOT, atomic_write_json, dataset_defaults, resolve_path
from .predictions import STAGES, is_run_dir
from .registry import REGISTRY, get

SCHEMA = "metrictrail_eval_score_v1"
METRICS = ("trajectory", "points")
PRED_FORMATS = ("npz", vipe_output.FORMAT)


@dataclass(frozen=True)
class CasePrediction:
    """One sequence's prediction file as the dataset scorer reads it."""

    path: Path
    depth_cache: Path | None
    fields: dict[str, Any]
    restore: Callable[[Mapping[str, Any]], dict[str, Any]] = dict


@dataclass(frozen=True)
class PredictionReader:
    """``--pred-format``: plain prediction files (``npz``) or ViPE output directories (``vipe``)."""

    format: str = "npz"
    vipe_names: Mapping[str, str] = field(default_factory=dict)
    work_dir: Path | None = None

    def layout(self, pred: Path, stage: str) -> str:
        if self.format == vipe_output.FORMAT:
            return "vipe_output"
        return "run_dir" if is_run_dir(pred, stage) else "prediction_dir"

    @contextlib.contextmanager
    def open(
        self, module: Any, pred: Path, *, stage: str, case: str, protocol: str,
        with_depth: bool, depth_cache: Path | None,
    ) -> Iterator[CasePrediction | None]:
        """The prediction of ``case`` under ``pred`` (None when it has none)."""

        if self.format != vipe_output.FORMAT:
            path = module.find_prediction(pred, stage, case, protocol)
            yield None if path is None else CasePrediction(path, depth_cache, {"prediction": str(path)})
            return
        found = vipe_output.locate(pred, stage, self.vipe_names.get(case, case))
        if found is None:
            yield None
            return
        with contextlib.ExitStack() as stack:
            try:
                converted = stack.enter_context(
                    vipe_output.materialize(found, case, with_depth=with_depth, work_dir=self.work_dir)
                )
            except vipe_output.VipeOutputError as error:
                raise SystemExit(f"{module.NAME}/{case}: {error}") from error
            if module.find_prediction(converted.directory, stage, case, protocol) != converted.prediction:
                raise SystemExit(
                    f"{module.NAME} protocol {protocol!r} reads its own historical layout, not a plain "
                    "prediction file; it is not available with --pred-format vipe"
                )
            yield CasePrediction(
                converted.prediction,
                converted.depth_cache,
                {"prediction": str(found.pose), "prediction_format": vipe_output.FORMAT,
                 "vipe": converted.provenance},
                converted.restore,
            )


NPZ_READER = PredictionReader()


class _OneRun(argparse.Action):
    """``--pred`` given twice is an error (argparse would otherwise keep the last one silently)."""

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any,
                 option_string: str | None = None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} takes one run directory; it was given more than once")
        setattr(namespace, self.dest, values)


def add_prediction_arguments(parser: argparse.ArgumentParser) -> None:
    """The dataset / prediction / GT / sequence options (shared with ``plot``)."""

    parser.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    parser.add_argument(
        "--pred", required=True, type=Path, action=_OneRun,
        help="run directory or prediction directory, or with --pred-format vipe a ViPE output directory",
    )
    parser.add_argument("--gt-root", type=Path, default=None, help="GT root (default: the package's datasets.json)")
    parser.add_argument("--stage", choices=STAGES, default="local")
    parser.add_argument("--protocol", default=None, help="trajectory protocol variant (see README)")
    parser.add_argument(
        "--pred-format", choices=PRED_FORMATS, default="npz",
        help="npz: MetricTrail run directory or prediction directory (default); vipe: ViPE output "
        "directory (pose/<name>.npz, pose_lc/ for --stage LC, intrinsics/, depth/<name>.zip)",
    )
    parser.add_argument(
        "--vipe-name", nargs="+", default=None, metavar="CASE=NAME",
        help="--pred-format vipe: ViPE artifact name of a sequence (default: the sequence key)",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="--pred-format vipe: scratch directory for the converted predictions (default: system temp)",
    )
    parser.add_argument("--cases", nargs="+", default=None, help="subset of sequences")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="skip sequences without a prediction instead of failing",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m metrictrail_eval.score",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_prediction_arguments(parser)
    parser.add_argument(
        "--metric", choices=METRICS, default="trajectory",
        help="trajectory ATE (default) or point-cloud metrics (oxford, 7scenes)",
    )
    parser.add_argument(
        "--depth-cache", type=Path, default=None,
        help="--metric points: root holding <case>/depth.f32 + mask.u8 (default: found from --pred)",
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON output path")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _select_cases(module: Any, requested: Sequence[str] | None) -> tuple[str, ...]:
    if not requested:
        return tuple(module.CASES)
    unknown = sorted(set(requested) - set(module.CASES))
    if unknown:
        raise SystemExit(f"unknown {module.NAME} sequences: {unknown}")
    chosen = set(requested)
    return tuple(case for case in module.CASES if case in chosen)


def _points_macro(per_case: dict[str, Any], cases: Sequence[str]) -> dict[str, float]:
    """Unweighted mean over the sequences (canonical order) of every numeric metric."""

    ordered = [case for case in cases if case in per_case]
    first = per_case[ordered[0]]
    return {
        key: float(np.mean([per_case[case][key] for case in ordered]))
        for key, value in first.items()
        if isinstance(value, float) and all(isinstance(per_case[c].get(key), float) for c in ordered)
    }


def score_points_run(
    module: Any,
    pred: Path,
    *,
    cases: Sequence[str],
    gt_root: Path,
    stage: str,
    protocol: str,
    allow_partial: bool,
    depth_cache: Path | None,
    reader: PredictionReader = NPZ_READER,
) -> dict[str, Any]:
    per_case: dict[str, Any] = {}
    missing: list[str] = []
    distance_key, f1_key = module.POINT_HEADLINE
    for case in cases:
        with reader.open(
            module, pred, stage=stage, case=case, protocol=protocol, with_depth=True, depth_cache=depth_cache
        ) as found:
            if found is None:
                if not allow_partial:
                    raise SystemExit(
                        f"{module.NAME}/{case}: no prediction under {pred} (stage {stage}); "
                        "use --allow-partial to skip"
                    )
                missing.append(case)
                continue
            result = module.evaluate_points_case(
                case, found.path, gt_root, protocol=protocol, stage=stage, depth_cache=found.depth_cache
            )
            per_case[case] = {**found.fields, **found.restore(result)}
        print(f"  {case:<24}{distance_key} {result[distance_key]:.6f} m  {f1_key} {result[f1_key]:.4f} %",
              flush=True)
    if not per_case:
        raise SystemExit(f"no {module.NAME} predictions found under {pred}")
    macro = _points_macro(per_case, cases)
    print(f"  {'mean':<24}{distance_key} {macro[distance_key]:.6f} m  {f1_key} {macro[f1_key]:.4f} %"
          f"  (n={len(per_case)})", flush=True)
    return {
        "pred": str(pred),
        "layout": reader.layout(pred, stage),
        "per_case": per_case,
        "missing": missing,
        "macro": macro,
        "num_cases": len(per_case),
    }


def score_run(
    module: Any,
    pred: Path,
    *,
    cases: Sequence[str],
    gt_root: Path,
    stage: str,
    protocol: str,
    allow_partial: bool,
    reader: PredictionReader = NPZ_READER,
    on_aligned: Callable[[str, dict[str, Any], Any], None] | None = None,
) -> dict[str, Any]:
    """Score every sequence of one prediction directory.

    ``on_aligned(case, per_case_result, aligned_trajectory)`` (``plot``) makes each
    sequence run through the dataset module's ``evaluate_case_aligned``, which is
    ``evaluate_case`` (same computation, same result) that also returns the
    aligned trajectory its ATE was computed on.
    """

    per_case: dict[str, Any] = {}
    missing: list[str] = []
    for case in cases:
        with reader.open(
            module, pred, stage=stage, case=case, protocol=protocol, with_depth=False, depth_cache=None
        ) as found:
            if found is None:
                if not allow_partial:
                    raise SystemExit(
                        f"{module.NAME}/{case}: no prediction under {pred} (stage {stage}); "
                        "use --allow-partial to skip"
                    )
                missing.append(case)
                continue
            if on_aligned is None:
                result = module.evaluate_case(case, found.path, gt_root, protocol=protocol, stage=stage)
            else:
                result, trajectory = module.evaluate_case_aligned(
                    case, found.path, gt_root, protocol=protocol, stage=stage
                )
            per_case[case] = {**found.fields, **found.restore(result)}
            if on_aligned is not None:
                on_aligned(case, per_case[case], trajectory)
        print(f"  {case:<24}{result['ate']:12.6f} {module.UNITS}", flush=True)
    if not per_case:
        raise SystemExit(f"no {module.NAME} predictions found under {pred}")
    mean = module.macro({case: per_case[case]["ate"] for case in per_case})
    print(f"  {'mean':<24}{mean:12.6f} {module.UNITS}  (n={len(per_case)})", flush=True)
    return {
        "pred": str(pred),
        "layout": reader.layout(pred, stage),
        "per_case": per_case,
        "missing": missing,
        "mean_ate": mean,
        "num_cases": len(per_case),
    }


def _version(module_name: str) -> str:
    try:
        return str(__import__(module_name).__version__)
    except Exception:  # noqa: BLE001 - reporting only
        return "unavailable"


def _prediction_reader(args: argparse.Namespace, module: Any) -> PredictionReader:
    if args.pred_format != vipe_output.FORMAT:
        for option, value in (("--vipe-name", args.vipe_name), ("--work-dir", args.work_dir)):
            if value is not None:
                raise SystemExit(f"{option} applies to --pred-format {vipe_output.FORMAT}")
        return NPZ_READER
    if getattr(args, "depth_cache", None) is not None:
        raise SystemExit("--depth-cache applies to --pred-format npz (ViPE depth is read from depth/<name>.zip)")
    names: dict[str, str] = {}
    for item in args.vipe_name or ():
        case, sep, name = item.partition("=")
        if not sep or not name or Path(name).name != name or case not in module.CASES:
            raise SystemExit(
                f"--vipe-name {item!r}: expected CASE=NAME with CASE one of the {module.NAME} sequences"
            )
        names[case] = name
    if not vipe_output.is_vipe_output(args.pred.expanduser()):
        raise SystemExit(f"{args.pred} is not a ViPE output directory (no pose/)")
    work_dir = args.work_dir.expanduser().resolve() if args.work_dir is not None else None
    return PredictionReader(vipe_output.FORMAT, names, work_dir)


def resolve_protocol_and_gt_root(args: argparse.Namespace, module: Any, *, points_metric: bool) -> tuple[str, Path]:
    """``--protocol`` (default: the dataset's) and ``--gt-root`` (default: ``datasets.json``)."""

    protocol = args.protocol or module.DEFAULT_PROTOCOL
    if protocol not in module.PROTOCOLS:
        raise SystemExit(f"{module.NAME}: protocol must be one of {module.PROTOCOLS}")
    if args.gt_root is not None:
        gt_root = args.gt_root.expanduser().resolve()
    else:
        defaults = dataset_defaults()["datasets"][module.NAME]
        gt_root = resolve_path(defaults["points_gt_root" if points_metric else "gt_root"])
    if not gt_root.exists():
        raise SystemExit(f"GT root does not exist: {gt_root}")
    return protocol, gt_root


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    module = get(args.dataset)
    points_metric = args.metric == "points"
    if points_metric and not hasattr(module, "evaluate_points_case"):
        raise SystemExit(f"--metric points is available for oxford and 7scenes, not {module.NAME}")
    protocol, gt_root = resolve_protocol_and_gt_root(args, module, points_metric=points_metric)
    cases = _select_cases(module, args.cases)
    depth_cache = args.depth_cache.expanduser().resolve() if args.depth_cache is not None else None
    reader = _prediction_reader(args, module)
    vipe_format = reader.format == vipe_output.FORMAT

    pred = args.pred.expanduser().resolve()
    label = module.POINT_PROTOCOL if points_metric else protocol
    fmt = f", {vipe_output.FORMAT}" if vipe_format else ""
    print(f"{module.NAME} [{args.stage}, {args.metric}, {label}{fmt}] {pred}", flush=True)
    if points_metric:
        run = score_points_run(
            module, pred, cases=cases, gt_root=gt_root, stage=args.stage,
            protocol=protocol, allow_partial=args.allow_partial, depth_cache=depth_cache,
            reader=reader,
        )
    else:
        run = score_run(
            module, pred, cases=cases, gt_root=gt_root, stage=args.stage,
            protocol=protocol, allow_partial=args.allow_partial, reader=reader,
        )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": module.NAME,
        "stage": args.stage,
        "metric": args.metric,
        "protocol": module.POINT_PROTOCOL if points_metric else protocol,
        "units": module.POINT_UNITS if points_metric else module.UNITS,
        "pred_format": reader.format,
        "gt_root": str(gt_root),
        "cases": list(cases),
        "runs": [run],  # the one --pred run, as a one-element list (the layout score JSON readers take)
        "evaluator": {
            **sim3_ate.backend_identity(),
            "python": platform.python_version(),
            "module": module.__name__,
        },
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if points_metric:
        result["evaluator"]["open3d_version"] = _version("open3d")
        result["evaluator"]["opencv_version"] = _version("cv2")
        result["evaluator"]["scipy_version"] = _version("scipy")

    out = (
        args.out.expanduser().resolve()
        if args.out is not None
        else REPO_ROOT / "outputs" / "scores"
        / f"{module.NAME}_{args.stage}{'_vipe' if vipe_format else ''}{'_points' if points_metric else ''}.json"
    )
    atomic_write_json(out, result)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
