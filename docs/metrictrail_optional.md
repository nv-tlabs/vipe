# MetricTrail: optional inputs and outputs

`pipeline=metrictrail` runs RGB-only SharedK calibration, metric depth and
DINO-fused DPVO. The optional inputs and outputs below live in one place: the
config block `pipeline.optional` (code in `vipe/metrictrail/optional/`), plus the
stereo runner, which is a separate entry point (`vipe/metrictrail/stereo/`).

**All of them are off by default.** With everything off, the pipeline runs MetricTrail.

| Option | Kind | Switch | Default |
| --- | --- | --- | --- |
| Known intrinsics (GT/sensor K) | input | `pipeline.optional.intrinsics` | `null`: RGB-only SharedK |
| Loop closure | output | `python -m vipe.metrictrail.optional.loop_closure` with external candidate files | `false` |
| Stereo | input | separate runner `python -m vipe.metrictrail.stereo` | not used |

ViPE's own optional features keep ViPE's field names and are documented in [dynamic masks and depth fusion](metrictrail_dynamic_and_fusion.md):

| Option | Kind | Switch | Default |
| --- | --- | --- | --- |
| Dynamic masking | input | `pipeline.init.instance` | `null` (off) |
| Depth fusion | output | `pipeline.post.depth_align_model` | `null` (off) |

## Known intrinsics

A known pinhole K replaces the RGB-only SharedK calibration. The run then is MetricTrail's calibrated mode.

### Intrinsics Sources

Give exactly one source:

| Source | Config | Notes |
| --- | --- | --- |
| Explicit values | `fx`, `fy`, `cx`, `cy` | All four, in source-image pixels. |
| File | `path` | `.txt`, `.json` or `.npz`; see below. |
| Input stream | `from_stream: true` | The stream's frames carry intrinsics (already at the input raster). Every frame must match the first one within `stream_tolerance_px` (default 1e-3 px). MetricTrail tracks one shared K. |

Accepted file layouts:

- `.txt`: four numbers `fx fy cx cy`, or nine numbers (a 3x3 K). Separators can be
  whitespace, `,` or `;`. Text after `#` is ignored.
- `.json`:
  - `{"fx", "fy", "cx", "cy"}`;
  - `K_fx_fy_cx_cy` (4 values);
  - `intrinsics` or `K` (4 values or 3x3);
  - a bare 4-value or 3x3 array;
  - `geometry.K` (prepared VBR manifest);
  - `cam0.params` with `cam0.resolution {width, height}` (LaMARia pinhole calibration).

  The raster may be given as `input_hw`, `image_hw`, `k_raster_hw` or `hw` (`[height, width]`),
  or as `height` and `width`.
- `.npz`: the first member found among `intrinsics`, `K`, `input_intrinsics`. The member may be 4 values, a 3x3 K, or an (N, 4) table, of
  which the first row is used. The first of the members `input_hw`, `image_hw`,
  `k_raster_hw`, `hw` that is present gives the raster.

### Usage

```bash
# explicit K of the raw 480x640 TUM images; the frames were stretched to 388x518 beforehand
python run.py pipeline=metrictrail streams=frame_dir_stream streams.base_path=frames/ \
  pipeline.init.input_raster.policy=unchanged \
  'pipeline.optional.intrinsics={fx: 517.306408, fy: 516.469215, cx: 318.643040, cy: 255.313989, source_hw: [480, 640], raster_map: stretch}'

# in-the-wild video with its calibration (K in the video's own pixels): mapped through the default 280x504 cover-crop
python run.py pipeline=metrictrail streams.base_path=clip.mp4 \
  'pipeline.optional.intrinsics={path: calib/K.txt}'

# a stream whose frames carry intrinsics (already at the input raster)
python run.py pipeline=metrictrail ... pipeline.init.input_raster.policy=unchanged \
  'pipeline.optional.intrinsics={from_stream: true}'
```

## Loop closure

Optional loop closure reads a saved local trajectory and externally generated
candidate frame pairs, then adds identity SE(3) loop constraints to a pose graph.
It uses PyPose on CPU for SE(3) operations and autograd, and SciPy for sparse
linear solves. Retrieval is a separate step: ViPE consumes candidate JSON and
does not import SALAD or load retrieval checkpoints.

### Install the pose-graph dependency

In the **ViPE environment**, install the pinned optional dependency:

```bash
python -m pip install pypose==0.6.8
# Alternatively, when installing this checkout:
python -m pip install '.[metrictrail-lc]'
```

PyPose is imported only when loop-closure optimization is needed. Local
tracking does not require it. This dependency belongs in the ViPE environment,
not only in the separate SALAD environment described below.

### Install SALAD separately

To generate loop candidates with [SALAD](https://github.com/serizba/salad),
install it in a separate checkout and environment:

```bash
git clone https://github.com/serizba/salad.git /your/external-tools/salad
cd /your/external-tools/salad
conda env create -f environment.yml
conda activate salad
```

For the subsequent pose-graph step, install `pypose==0.6.8` in the **ViPE
environment**, as described above; installing it only in the SALAD environment
is not sufficient. PyPose is an optional Apache-2.0 dependency.

Follow the [upstream README](https://github.com/serizba/salad#setup) for inference
usage and [checkpoint downloads](https://github.com/serizba/salad#evaluation).
SALAD source is covered by its [GPLv3 license](https://github.com/serizba/salad/blob/main/LICENSE);
model weights are subject to their applicable terms.

**An external exporter is required.** Use an external SALAD retrieval workflow
to produce frame pairs and scores from the same ordered RGB frames used for
tracking. Upstream `eval.py` evaluates predefined benchmark datasets; it does
not export MetricTrail candidate files from an arbitrary image folder.
ViPE does not include a SALAD exporter or install or run SALAD automatically.
SALAD is optional; local tracking does not require it.

### Candidate files

Generate pairs for the same frames that produced `pose/<name>.npz` with an
external retrieval tool. Frame IDs must be values of that file's `inds` array.
Do not use image filenames, timestamps, or `source_frame_ids`.
The JSON format is:

```json
{
  "schema": "metrictrail_loop_candidates_v1",
  "producer": "SALAD",
  "candidates": [
    {"src_frame": 100, "dst_frame": 0, "score": 0.95}
  ]
}
```

Each pair uses two distinct frame IDs with `src_frame > dst_frame` and a finite
score in `[0, 1]`. Supply at most 1,000 unique pairs; the optimizer uses up to
300 pairs, ordered by score, retaining scores at least `0.025` as in the original
KITTI/VBR presets. An empty list preserves the baseline poses.
Candidate scores rank pairs; they do not supply relative camera transforms.
Use geometrically verified revisits with compatible camera orientations.

Alternatively, export a CSV with this header and one row per pair:

```csv
src_frame,dst_frame,score
100,0,0.95
```

Return to the ViPE checkout and environment, then convert the CSV to JSON:

```bash
python -m vipe.metrictrail.optional.loop_candidates \
  --pairs /path/to/clip-pairs.csv \
  --output /path/to/clip-loops.json \
  --producer SALAD
```

### Apply loop closure

After tracking with `metrictrail_local`, pass the candidate file:

```bash
python -m vipe.metrictrail.optional.loop_closure vipe_results/ --name clip \
  --candidates /path/to/clip-loops.json
```

Use `--preset vbr` for VBR weighting and `--overwrite` to replace existing
loop-closure artifacts. The command reads `pose/<name>.npz` and
`vipe/<name>_metrictrail.json`. The run record must identify a successful local
trajectory and match its name.
No RGB frames or model checkpoints are needed for this pose-graph step; only
external candidate generation needs retrieval images and weights.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Apply supplied candidate files after saving the local trajectory. |
| `candidates` | `null` | Candidate JSON for the saved trajectory. |
| `preset` | `kitti` | CPU pose-graph weighting: `kitti` or `vbr`. |
| `overwrite` | `false` | Replace existing loop-closure artifacts. |

Pipeline loop closure requires `metrictrail_local`,
`pipeline.output.save_artifacts=true`, and a candidate file.

### Outputs

- `pose_lc/<name>.npz`: corrected camera-to-world poses in ViPE format (`data`, `inds`).
- `pose_lc/<name>.json`: optimization report and input paths.
- `pose_lc/<name>.candidates.json` and `pose_lc/<name>.identity_edges.json`:
  accepted candidates and applied loop edges.

The original `pose/` artifacts and candidate file remain unchanged.
External retrieval software and weights are obtained separately under their own terms.

## Stereo runner

Stereo MetricTrail is stereo tracking : DINO-fused stereo DPVO with rectified-stereo
factors, a known K and a metric baseline, plus per-frame metric depth
(FP16, depth queue 4). It is a separate entry point, not a pipeline option and
not in ViPE's pipeline registry; `pipeline=metrictrail` never imports it.

```bash
python -m vipe.metrictrail.build --stereo          # once: the stereo DPVO kernels
python -m vipe.metrictrail.stereo --left left/ --right right/ \
  --intrinsics FX FY CX CY --baseline 0.54 --output vipe_results/ --name 00
```

- `--left`, `--right`: a video file or a frame directory each (ViPE's
  `RawMp4Stream` / `FrameDirStream`), rectified, with the same size and length.
  Two frame directories must pair by frame number.
- `--intrinsics`: the left camera's pinhole K, in pixels of the input frames.
  `--baseline`: the baseline in metres.
- `buffer_size: auto` (default) sizes DPVO's buffers.

Outputs, in the layout of `pipeline=metrictrail`: `pose/<name>.npz` (left camera,
plus `stereo_baseline_m`), `intrinsics/<name>.npz` and `_camera.txt`,
`vipe/<name>_info.pkl`, the run record `vipe/<name>_metrictrail.json`, and the
SLAM map with `save_slam_map`. Loop closure is not part of the runner; to run it
after tracking, the loop-closure CLI above accepts the runner's output directory
with the candidate JSON.
