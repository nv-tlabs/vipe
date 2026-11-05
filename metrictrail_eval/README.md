# MetricTrail-eval

Trajectory evaluation for MetricTrail predictions: Sim(3)-aligned ATE RMSE per
sequence and its mean, on KITTI odometry, VBR, Oxford-Spires (long, stride 1),
LaMaria, TUM RGB-D fr1, 7-Scenes and Wild-SLAM MoCap; plus the dense
point-cloud metrics (Chamfer distance, accuracy / completeness, precision /
recall / F1) of Oxford-Spires and 7-Scenes (`--metric points`).  Predictions
are MetricTrail prediction files or ViPE output directories
(`--pred-format vipe`).  Each dataset module
reproduces exactly how the method's historical numbers were computed (frame
matching, alignment, units, averaging); on the historical predictions it
returns the recorded per-sequence values.  It is kept apart from the MetricTrail
code so that ground truth is only ever opened here.

`python -m metrictrail_eval.plot` draws, per sequence, the GT and the
Sim(3)-aligned prediction that the scorer's ATE was computed on, by the same
code, and records that ATE (see [Visualizing ATE](#visualizing-ate)).


## Install

Python 3.10, `numpy==1.26.4`, `scipy`, `matplotlib`, `evo==1.36.5` (the
versions every record was produced with); nothing is compiled.  `--metric
points` also needs `open3d==0.19.0` and OpenCV (`pip install -e '.[points]'`);
the point records were produced with scipy 1.15.3, open3d 0.19.0 and OpenCV
5.0.0.  Point metrics on ViPE output directories also read ViPE's EXR depth
(`OpenEXR` and `Imath`, `pip install -e '.[points,vipe]'`).

Use an environment of its own (a venv or conda env), not ViPE's: the pins above would
downgrade the numpy of ViPE's environment.  The package does not import ViPE.

```bash
pip install -e .            # or just run from this directory
export PYTHONNOUSERSITE=1
python -m metrictrail_eval.selftest     # data-free check of the metric backends
python -m pytest -q tests               # CPU-only tests on synthetic GT (no data/ needed)
```

**Ground truth (`data/`).** Every scoring and plotting command reads the
dataset's GT from `--gt-root` or, without it, from its default root under
`data/` in this directory ([Data](#data)).  `data/` is not part of the
repository; link it once to the data directory of a MetricTrail checkout,
which has the same layout (see the MetricTrail README):

```bash
MT=/path/to/MetricTrail     # MetricTrail checkout: data/ = prepared benchmarks, outputs/<config>/ = its runs
ln -s $MT/data data         # once, in this directory (data is git-ignored)
```

Without `data/`, every command needs `--gt-root` (it stops with `GT root does
not exist: <this directory>/data/...` otherwise).  The Oxford-Spires points
GT (`data/Oxford-Spires/points_gt`, `--metric points`) is not part of the
MetricTrail data layout; link it there or pass it with `--gt-root`.

## Data

GT roots are command-line arguments (`--gt-root`); without one, the dataset's
entry in the package data `metrictrail_eval/datasets.json` is used, relative
to this directory (`data/`):

| `--dataset` | default `--gt-root` | GT read |
|---|---|---|
| `kitti` | `data/KITTI/prepared_280x504` | `<NN>.npz` `c2w_gt` (or the historical `<NN>/attempt-*/inputs/<NN>_<profile>.npz` tree, last attempt) |
| `vbr` | `data/VBR/prepared_280x504` | `<seq>/sequence.npz` `c2w_gt`, `frame_ids` (+ manifest / `.success` attestation) |
| `oxford` | `data/Oxford-Spires/prepared_long_378x518` | `<scene>/sequence.npz` `c2w_gt` (+ suite and scene attestation, `rgb_timing.npz`) |
| `lamaria` | `data/LaMaria` | `training/<seq>/ground_truth/pGT/<seq>.txt`, `pinhole_calibrations/<seq>.json` (`cam0.T_b_s` only), `asl_folder/<seq>/aria/cam0/data.csv` |
| `tum` | `data/TUM-RGBD/prepared_388x518` | `<scene>/sequence.npz` `c2w_gt`, `gt_valid` (+ manifest / `.success` / input contract) |
| `7scenes` | `data/7-Scenes/raw` | `<scene>/seq-NN/frame-XXXXXX.pose.txt` |
| `wild_slam` | `data/Wild-SLAM/prepared_native_720x1280` | `<seq>/sequence.npz` `c2w_gt`, `gt_valid` (+ manifest, `SUCCESS`) |

With `--metric points` the default GT root is the dataset's `points_gt_root`:

| `--dataset` | default `--gt-root` (points) | GT read |
|---|---|---|
| `oxford` | `data/Oxford-Spires/points_gt` | `pc/<scene>/{DONE.json, frame_ids.txt, poses_c2w.txt, depth/XXXXXX.npy}` (projected full-TLS depth of every 10th frame, native 1080x1440) and `processed_sequences/<scene>/{poses_c2w.txt, intrinsics.txt}` |
| `7scenes` | `data/7-Scenes/raw` | `<scene>/seq-NN/frame-XXXXXX.{pose.txt, depth.proj.png}` |

Please refer to the [Data preparation](https://github.com/Robbyant/lingbot-map/tree/main/benchmark#supported-datasets) of Lingbot-Map for detailed data preparation.

## Scoring

`python -m metrictrail_eval.score` scores one run of one dataset per call
(one `--pred`): one line per sequence, the mean, and a JSON with every
per-sequence value.  The dataset's
GT root is `--gt-root` or, without it, its `datasets.json` entry (under
`data/`, see [Data](#data)).  The ATE is in `m`, except Wild-SLAM in `cm` (the
tables also print TUM and 7-Scenes in cm: x100).

**1. MetricTrail predictions** (`--pred-format npz`, the default): `--pred` is
a MetricTrail run directory (`$MT/outputs/<config>`, written by a run of
`configs/<config>.json`, holding `<stage>/inference/`) or a plain directory
of `<case>.npz` predictions.

```bash
MT=/path/to/MetricTrail             # MetricTrail checkout (as in Install)
python -m metrictrail_eval.score --dataset kitti     --pred $MT/outputs/kitti_uncalibrated --stage local
python -m metrictrail_eval.score --dataset kitti     --pred $MT/outputs/kitti_uncalibrated --stage LC
python -m metrictrail_eval.score --dataset vbr       --pred $MT/outputs/vbr --stage local
python -m metrictrail_eval.score --dataset oxford    --pred $MT/outputs/oxford
python -m metrictrail_eval.score --dataset lamaria   --pred $MT/outputs/lamaria
python -m metrictrail_eval.score --dataset tum       --pred $MT/outputs/tum_uncalibrated
python -m metrictrail_eval.score --dataset 7scenes   --pred $MT/outputs/sevenscenes_uncalibrated
python -m metrictrail_eval.score --dataset wild_slam --pred $MT/outputs/wild_uncalibrated
```
**2. Before / after loop closure** (`--stage`, default `local`):

| `--stage` | MetricTrail run directory | ViPE output directory | datasets with loop closure |
|---|---|---|---|
| `local` (before LC) | `local/inference/<case>.npz` or `local/inference/predictions/<case>.npz` | `pose/<case>.npz` | all |
| `LC` (after LC) | `LC/inference/<case>.npz` (or `.../predictions/<case>.npz`) | `pose_lc/<case>.npz` | KITTI, VBR (configs with LC) |

A plain prediction directory has no stages (`--stage` only labels the
result).  For VBR, `pose/` is the final tracked pass (after stage B).

**3. Point clouds** (`--metric points`, Oxford-Spires and 7-Scenes): needs the
prediction's depth (`depth_m`, the run's frontend depth cache, or ViPE's
`depth/<case>.zip`) and the `points` extra.

```bash
python -m metrictrail_eval.score --dataset 7scenes --metric points --pred $MT/outputs/sevenscenes_uncalibrated
python -m metrictrail_eval.score --dataset oxford  --metric points --pred $MT/outputs/oxford
python -m metrictrail_eval.score --dataset 7scenes --metric points --pred-format vipe --pred $VIPE/sevenscenes
```

Options:

* `--cases` restricts the sequences; `--allow-partial` skips sequences that
  have no prediction (by default every sequence of the dataset is required).
* `--protocol` selects a dataset's trajectory protocol variant (only KITTI
  has two: `evo`, the default, and `umeyama`).
* `--depth-cache ROOT` (`--metric points`, npz): a root holding
  `<case>/{depth.f32, mask.u8, contract.json, SUCCESS}` when the run's
  frontend cache (`frontend-cache/<dataset>/<case>/` in the nearest ancestor
  of the prediction file) is elsewhere.
* `--work-dir` (vipe): scratch directory for the converted predictions
  (default: the system temp directory; an Oxford-Spires scene needs about
  4 GB for points).
* In a run directory, `<stage>/inference/<case>.npz` and
  `<stage>/inference/predictions/<case>.npz` both existing is an error.

Output: one line per sequence, the mean, and a JSON (`--out`, default
`outputs/scores/<dataset>_<stage>[_vipe][_points].json`) with every
per-sequence metric, the prediction and GT files used and the mean.

## Visualizing ATE

`python -m metrictrail_eval.plot` takes the same prediction options as the
scorer (`--dataset`, `--pred`, `--pred-format npz|vipe`, `--gt-root`,
`--stage local|LC`, `--protocol`, `--cases`, `--vipe-name`, `--work-dir`,
`--allow-partial`) plus `--out DIR`, and writes, for the one run given by
`--pred`, one figure per sequence: the GT trajectory and the Sim(3)-aligned
prediction, top-down, titled with the sequence's name; the ATE goes to
`summary.json` and the overview (and into the figure with `--details`).
Without `--out` the figures go to `outputs/figures/<dataset>_<stage>[_vipe]/`
under this directory.

```bash
VIPE=/path/to/vipe_results          # as in Scoring; $MT likewise
# KITTI uncalibrated, before and after loop closure (pose/ and pose_lc/)
python -m metrictrail_eval.plot --dataset kitti --pred-format vipe --pred $VIPE/kitti --stage local --out outputs/figures/kitti/pose
python -m metrictrail_eval.plot --dataset kitti --pred-format vipe --pred $VIPE/kitti --stage LC    --out outputs/figures/kitti/pose_lc
# VBR after loop closure, one sequence, also as PDF
python -m metrictrail_eval.plot --dataset vbr --pred-format vipe --pred $VIPE/vbr --stage LC --cases campus_train0 \
    --out outputs/figures/vbr/pose_lc --pdf
```

Figure (default): GT grey and thick (frames without GT break the line), the aligned prediction green and thinner (the evaluated frames), equal axes in metres with only the horizontal axis labelled (`world x (m)`), and as the title the sequence's own name as the dataset names it.

## Third-party code

`vendored/` holds two third-party evaluator files, used under their licences
(Apache-2.0; the licence texts are kept verbatim in `licenses/`, and the
copyright and modification notices inside the files are kept as they are).
