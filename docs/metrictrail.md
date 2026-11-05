# MetricTrail pipeline

MetricTrail tracks a monocular RGB video with metric scale and without known intrinsics:

0. **Input raster**: by default the video is prepared with the geometry and frame selection
   of MetricTrail's in-the-wild tool (280x504 cover-crop, source frame rate).
1. **SharedK**: one pinhole K for the whole video, calibrated from RGB only on ViPE's
   keyframe SLAM (MoGe-2 metric depth, focal optimisation in the fused BA). Inputs that
   carry intrinsics are refused. If the calibration cannot be trusted, the depth model's
   own focal is used instead (in-the-wild policy, below). A known K
   (`pipeline.optional.intrinsics`, see [optional inputs and outputs](metrictrail_optional.md))
   replaces SharedK; focal refinement is then skipped.
2. **Tracking**: DPVO with DINOv3-fused patch features, fed MoGe-2 metric depth for every
   frame (FP16 autocast). The depth runs one bounded queue ahead of DPVO on its own CUDA
   stream (`depth_queue: 4`).
3. **Focal refinement** (optional, off by default): one RoMa revisit-pair fit of the
   focal ("stage B"), with GT-free routing and an accept gate.

The pipeline code and optional CPU loop closure live in `vipe/metrictrail/`
([Repository layout](#repository-layout)). ViPE
itself only registers the pipeline schema (`vipe/config/pipeline.py`) and lists it in the
configuration reference.

## Repository layout

| Folder | Contents | License |
| --- | --- | --- |
| `vipe/metrictrail/` | the pipeline: SharedK calibration, DPVO backend, focal refinement, stereo runner, point-cloud export, CPU loop closure | Apache-2.0 (vendored DPVO: MIT; LieTorch: BSD-3-Clause) |
| `configs/pipeline/metrictrail.yaml`, `configs/slam/metrictrail_*.yaml` | pipeline and tracker presets; benchmark commands in `configs/slam/README_metrictrail.md` | Apache-2.0 |
| `metrictrail_eval/` on the branch `metrictrail_eval` | evaluation (Sim(3) ATE, ATE plots, point-cloud metrics) of ViPE output directories, its own package `metrictrail-eval`; it does not import ViPE (`git worktree add ../vipe-metrictrail-eval metrictrail_eval`) | vendored files: `metrictrail_eval/metrictrail_eval/licenses/` |
| `tests/metrictrail/` | CPU tests of the pipeline | Apache-2.0 |

Loop closure consumes externally generated candidate files; see
[optional inputs and outputs](metrictrail_optional.md). The separate `metrictrail_eval/`
package is not part of the `vipe` wheel or sdist.
The evaluation lives on its own branch, `metrictrail_eval`; check it out next to this tree and run
it from its folder in an environment of its own (`metrictrail_eval/README.md` there).

## Optional inputs and outputs

Every option is **off by default**; with all of them off the pipeline runs
MetricTrail. A video resized on read goes through torch's antialiased bicubic resize, so
its pixels, and with them K and the trajectory, differ slightly from a run with ViPE's
bilinear resize (`pipeline.init.input_raster.resampler=bilinear`).

| Option | Switch | Page |
| --- | --- | --- |
| Known intrinsics (GT/sensor K, replaces SharedK) | `'pipeline.optional.intrinsics={path: K.txt}'` | [optional inputs and outputs](metrictrail_optional.md) |
| CPU loop closure from external candidate files (writes `pose_lc/`) | `python -m vipe.metrictrail.optional.loop_closure` | [optional inputs and outputs](metrictrail_optional.md) |
| Stereo (known K and baseline) | separate runner `python -m vipe.metrictrail.stereo` | [optional inputs and outputs](metrictrail_optional.md) |
| ViPE dynamic-object masks | `'pipeline.init.instance={}'` | [dynamic masks and depth fusion](metrictrail_dynamic_and_fusion.md) |
| ViPE depth fusion | `pipeline.post.depth_align_model=adaptive_unidepth-l_vda` (static scene) or `adaptive_unidepth-l_svda` (with the masks) | [dynamic masks and depth fusion](metrictrail_dynamic_and_fusion.md) |
| Fused point cloud (PLY) of a finished run | separate tool `python -m vipe.metrictrail.pointcloud` | [point-cloud export](metrictrail_pointcloud.md) |

The two ViPE features are independent and can be turned on separately or together: the
dynamic masks keep moving objects out of tracking (tracking only), depth fusion replaces
the per-frame MoGe-2 depth with ViPE's temporally smoothed depth (depth and point cloud
only; whether it helps depends on the video). As in ViPE's presets: a scene with moving
objects uses the masks with `adaptive_unidepth-l_svda` (`default`), a static scene uses
`adaptive_unidepth-l_vda` without masks (`static_vda`).

## Running

```bash
# Build DPVO's CUDA kernels once, after installing ViPE (see "Build" below).
python -m vipe.metrictrail.build

python run.py pipeline=metrictrail streams.base_path=clip.mp4       # in-the-wild video
vipe infer clip.mp4 -p metrictrail -o vipe_results/                 # same
# frames that are already prepared (e.g. benchmark frame directories): keep them as they are
python run.py pipeline=metrictrail streams=frame_dir_stream streams.base_path=frames/ \
    pipeline.init.input_raster.policy=unchanged
```

**Benchmarks.** The command, preset, input raster, focal refinement and loop closure of each
benchmark (TUM, Wild-SLAM, 7-Scenes, KITTI, Oxford-Spires, VBR, LaMaria) are in
`configs/slam/README_metrictrail.md`.

**Full sequences.** `pipeline=metrictrail` sets `streams.frame_end: -1`, so `run.py`
processes the whole video (the `raw_mp4_stream` preset stops at frame 1000 for every
other pipeline, unchanged). An explicit `streams.frame_end=N` on the command line still
wins; the pipeline then refuses the cut input unless
`pipeline.init.on_truncated_input=warn`. `vipe infer` always reads whole videos.

**Frame count.** Before calibration the pipeline fixes the number of frames it will
track: a frame directory by its listing, a video by one decode pass (OpenCV `grab`, no
colour conversion), a cached stream by its cache, any other stream by one read. An mp4
whose container declares more frames than its decoder delivers (variable-frame-rate or
edited phone clips) is tracked over the frames it delivers, and the run record states
both numbers (`input_coverage.frame_count`); an input without a frame is refused before
any model runs.

**Host memory.** `vipe infer` decodes and caches every frame of the input as float32 in
host RAM before the pipeline starts (ViPE's CLI does this for every pipeline except
`pose_only_long`; MetricTrail does not change the CLI): H x W x 12 bytes per frame, e.g.
11 MB at 1280x720, 25 MB at 1920x1080, 1.7 MB at 504x280. For long or high-resolution
inputs use `run.py`, which streams the input from disk on each pass:

```bash
python run.py pipeline=metrictrail streams.base_path=clip.mp4
python run.py pipeline=metrictrail streams=frame_dir_stream streams.base_path=frames/
```

`output.save_viz` (and `vipe infer -v`) caches the rendered output frames the same way.

**Example.** `assets/examples/metrictrail_demo.mp4` (1280x720, 30 fps, 819 frames):

```bash
python run.py pipeline=metrictrail streams.base_path=assets/examples/metrictrail_demo.mp4
python run.py pipeline=metrictrail streams.base_path=assets/examples/metrictrail_demo.mp4 \
    pipeline.slam.seed_name=demo
```

## Calibration policy

The default is MetricTrail's in-the-wild policy: robust128 budget, terminal
minimum 16 keyframes, `fallback: provider_focal`, `allow_unstable: false`,
`fallback_to_initializer: false`. When SharedK does not converge, runs out of frames, keeps
too few keyframes, or raises inside the calibration sweep, the depth model's own focal
(median of the first 32 frames) with the principal point at the image centre is bound instead.

### Reuse one calibration across comparisons

Set `pipeline.init.calibration_cache=/path/to/clip_sharedk.json` to create an
immutable RGB-only calibration artifact on the first run and reuse exactly its
selected K on later runs. This keeps mask-off/mask-on comparisons on the same
calibration and avoids repeating SharedK. The default is `null` (calibrate every
run). Select a separate path for each prepared video:

```bash
python run.py pipeline=metrictrail streams.base_path=clip.mp4 \
    pipeline.init.calibration_cache=/tmp/clip_sharedk.json pipeline.output.path=results/off
python run.py pipeline=metrictrail streams.base_path=clip.mp4 \
    pipeline.init.calibration_cache=/tmp/clip_sharedk.json pipeline.output.path=results/on \
    'pipeline.init.instance={}'
```

Before reuse, the pipeline reads every full-rate prepared RGB frame, checking
its normalized pixels, source frame ID, frame count and raster, plus the complete
calibration configuration and resolved calibration SLAM defaults. The scan is
streamed and does not retain the whole video. A mismatch or invalid artifact
raises before masking or tracking; an existing artifact is never overwritten or
silently recalibrated. Concurrent first writers fail rather than replace K.
Masking settings and tracking stride do not affect this RGB-only identity.

This is an explicitly **frozen calibration**, not automatic model-cache
invalidation: use a new artifact path after changing calibration code or model
weights. Reuse does not load the original models to revalidate their weights.
The artifact retains the full original calibration record, including provider
fallbacks and GT-free provenance; it cannot be combined with
`pipeline.optional.intrinsics` and never labels model-estimated K as sensor/GT K.

In the output JSON, `calibration` retains the original record and original
timings. `calibration_execution.current_run_wall_seconds` measures the current
invocation, including the RGB identity scan, creation or load/validation;
`mode` is `created` or `reused`. The artifact path and SHA256 bind the result to
the exact frozen calibration. With the default `null` path,
`calibration_execution` is null and existing calibration timing semantics apply.
Stage-B focal refinement, when enabled, still runs independently after stage A;
the frozen artifact fixes the stage-A K rather than the final refined K.

## One pipeline, three tracker presets

The tracker backend is a preset of ViPE's `slam` config group. Select it with the
package-qualified override `slam@pipeline.slam=...`; the shorter forms
`pipeline/slam=...` and `slam=...` are rejected by Hydra in ViPE's layout.

| preset | DPVO backend | use |
| --- | --- | --- |
| `metrictrail_local` (default) | fixed-lag local BA with the metric-depth prior | long outdoor sequences |
| `metrictrail_gba` | + proximity loop edges and metric-scale global BA (t0 = all) | short indoor / handheld sequences |
| `metrictrail_gba_refine` | + scheduled global BA, culled-frame interpolation, final re-measured global BA, scale latent, reverse edges, keyframe-hole guard | long handheld indoor sequences with revisits|

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_gba streams.base_path=clip.mp4
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_gba_refine ...
# DPVO keys a preset does not list need "+":
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_gba +pipeline.slam.dpvo.GBA_SCHEDULE_K=8 ...
```

## Focal refinement

```bash
python run.py pipeline=metrictrail pipeline.focal_refine.enabled=true \
    pipeline.focal_refine.roma_python=/path/to/roma/env/bin/python ...
```

Stage B is enabled per run by this config switch (for example for VBR); its
GT-free routing still decides per sequence whether to enter. With a known K it is
skipped and the run record states
`focal_refine: {enabled: false, requested: true, skipped_reason: known_intrinsics}`.

Flow: stage A tracks with the SharedK focal; GT-free routing on
stage A's trajectory and calibration decides whether to enter; stage B runs once (RoMa
at 560 px, seed 7) in a separate interpreter; its accept gate decides; only an accepted
focal is rebound (fx = fy) and the sequence re-tracked. RoMa (`romatch`) is an optional
dependency of that interpreter only (`focal_refine.roma_python`, default
`$VIPE_METRICTRAIL_ROMA_PYTHON`, else the running one; `focal_refine.roma_repo` for a
checkout). Checkpoints `roma_outdoor.pth` and `dinov2_vitl14_pretrain.pth` come from
`focal_refine.roma_checkpoints`, `$VIPE_METRICTRAIL_ROMA_CHECKPOINTS` or
`$TORCH_HOME/hub/metrictrail/roma`.
