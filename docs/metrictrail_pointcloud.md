# MetricTrail point-cloud export

An opt-in tool that fuses a finished ViPE run into one coloured point cloud and writes it
as a binary PLY. No pipeline calls it. It only reads the artifacts of a run that has
already finished.

```bash
python -m vipe.metrictrail.pointcloud vipe_results/ --name clip --video clip.mp4
python -m vipe.metrictrail.pointcloud vipe_results/ --name clip --frames frames/ \
    --output clip.ply --voxel-size 0.05 --max-depth 100
```

## Inputs

| from the output directory | |
| --- | --- |
| `pose/<name>.npz` | `data`: OpenCV camera-to-world `[N,4,4]`, `inds`; `source_frame_ids` if present |
| `intrinsics/<name>.npz` | `data`: `fx, fy, cx, cy` `[N,4]`, `inds`; `_camera.txt` must say `PINHOLE` |
| `depth/<name>.zip` | ViPE's metric depth: one half-float EXR per frame (channel `Z`), as `vipe.utils.io` writes it |
| `vipe/<name>_metrictrail.json` | the run record, if present: its `input_raster` entry |

`depth/<name>.zip` is required. A MetricTrail run writes it by default
(`output.save_metric_depth: async`); a run with `save_metric_depth: off` is pose-only and
cannot be exported. The frame count is the depth artifact's. A frame that has no pose,
intrinsics or depth is left out of the fusion.

**RGB** is the frames tracking used, taken in order from the run's input: `--video`
(decoded with OpenCV as ViPE's video stream, `RawMp4Stream`, does) or `--frames` (listed
and read as ViPE's frame-directory stream does). If the pose file records
`source_frame_ids`, artifact frame `i` is coloured from input frame `source_frame_ids[i]`,
so a run that kept one frame in k (`input_raster.frame_stride`) or tracked one prepared
frame in k is coloured from exactly the frames it tracked.
Otherwise it is input frame `i`, and if the input and the artifacts have different frame
counts, only the shorter count is fused and a warning is logged.

How a frame reaches the depth raster follows the run record's `input_raster`:

| `input_raster` | input | frames |
| --- | --- | --- |
| `policy: cover_crop` | `--video` or `--frames` | `vipe.metrictrail.raster.raster_uint8` with the recorded plan: the recorded resampler (default bicubic with antialias) to cover the raster, then the integer centre crop |
| `policy: resize` | `--video` or `--frames` | `vipe.metrictrail.raster.raster_uint8` with the recorded plan: the recorded resampler to the raster, no crop |
| `policy: unchanged`, or no run record | `--video` or `--frames` | as given |


## What is fused

For every `frame_stride`-th frame with a finite pose and intrinsics:

1. take every `pixel_stride`-th pixel in both directions whose depth is finite and lies in
   `[min_depth_m, max_depth_m]` (both bounds inclusive), and back-project it through
   the pinhole `fx, fy, cx, cy`;
2. if the frame yields more than `max_input_points / fused_frames` points, keep a seeded
   random subset of that size, so the point budget is spread evenly over the sequence;
3. **multi-view consistency** (skipped when `min_consistent` is 0): project the point into
   up to `consistency_max_views` frames within `consistency_window` of this one (nearest
   first), read that frame's depth at the rounded pixel, and bring that surface point back
   into this frame. The neighbour agrees when the round trip lands within
   `pixel_threshold` pixels and `relative_depth_threshold` in relative depth. The point
   is kept when at least `min_consistent` neighbours agree. This removes flying points at
   depth discontinuities, which no single depth map can identify;
4. colour the point with the input frame's RGB at that pixel, move it to world with the
   frame's camera-to-world pose, and add it to a voxel grid of size `voxel_size_m`. Each
   voxel becomes one point, at the mean position and the mean colour of its points.

**Frame.** By default the cloud is written in the z-up frame of the original MetricTrail
visualization: after fusion, ViPE's world frame is rotated so that the mean camera up
direction (minus the mean camera +Y of the OpenCV poses) becomes +Z. The rotation is
printed as `world_to_output_rotation`. `--world-frame` writes the cloud in ViPE's world
frame instead, the frame of `pose/<name>.npz`, unrotated.

## Options

Every setting can be changed on the command line. Options not given take the default, or
the `--filter` preset's value for the five filter options.

| option | default | `--filter strict` | what it does |
| --- | --- | --- | --- |
| `--frame-stride` | 1 | | fuse every N-th frame (the filter still compares against every frame) |
| `--pixel-stride` | 2 | | take every N-th pixel in both directions |
| `--voxel-size` | 0.02 | | voxel size in metres; each voxel becomes one point |
| `--min-depth` | 0.2 | | nearest depth kept, metres |
| `--max-depth` | 15 | | farthest depth kept, metres (the depth cap) |
| `--max-input-points` | 8000000 | | point budget before voxelization, spread evenly over the fused frames |
| `--filter` | `default` | | flying-point filter preset: `default` or `strict` |
| `--min-consistent` | 2 | 4 | neighbouring frames that must reproduce a point; 0 disables the filter |
| `--consistency-window` | 5 | 9 | neighbour search window, frames |
| `--consistency-max-views` | 6 | 10 | neighbouring frames checked per point, nearest first |
| `--pixel-threshold` | 1.0 px | 0.5 px | round-trip reprojection tolerance |
| `--relative-depth-threshold` | 0.05 | 0.02 | round-trip relative depth tolerance |
| `--world-frame` | off (z-up frame) | | write ViPE's world frame instead |
| `--output` | `<output_dir>/pointcloud/<name>.ply` | | PLY path |

The defaults are the settings of the original MetricTrail visualization of in-the-wild
clips. `--filter strict` is the stronger flying-point filter its figure of the demo clip
(`assets/examples/metrictrail_demo.mp4`) used: a point is kept only when at least 4 of up
to 10 neighbouring frames reproduce it within 0.5 px and 2 % depth. It removes more flying
points along depth discontinuities and keeps less of a fast-moving or distant scene.
Explicit filter options override the preset's values (`--filter strict --min-consistent 3`).

**Frame stride.** `--frame-stride N` fuses every N-th frame; the consistency filter still
compares each fused point against all frames. Use it when a subject moves through the
scene: with every frame fused, the subject appears once per frame along its path.

**Depth cap.** `--max-depth` drops every pixel farther than the cap (`--min-depth`
the nearer ones). The default 15 m suits indoor and street-level clips. A distant scene
(a coastline, a landscape) needs a larger cap, and usually a coarser `--voxel-size` to
stay within the point budget.

The example videos in `assets/examples/` (run each with
`python run.py pipeline=metrictrail streams.base_path=assets/examples/<name>.mp4 pipeline.output.path=vipe_results/<name>`):

```bash
# metrictrail_demo (indoor walk-through, 819 frames): the strict filter
python -m vipe.metrictrail.pointcloud vipe_results/metrictrail_demo --name metrictrail_demo \
    --video assets/examples/metrictrail_demo.mp4 --filter strict
# Skateboard (fast forward motion along a coastal road, a distant scene): a 100 m cap and 5 cm voxels
python -m vipe.metrictrail.pointcloud vipe_results/Skateboard --name Skateboard \
    --video assets/examples/Skateboard.mp4 --max-depth 100 --voxel-size 0.05
# DynamicRobot (a walking robot, 24 fps): every 12th frame, every pixel
python -m vipe.metrictrail.pointcloud vipe_results/DynamicRobot --name DynamicRobot \
    --video assets/examples/DynamicRobot.mp4 --frame-stride 12 --pixel-stride 1
```

**Pixel stride.** `--pixel-stride 2` (the default) back-projects every second pixel. After
the 2 cm voxel average that loses little: at 280x504 one pixel spans about 1 cm at 2 m, and
the point budget (`--max-input-points`) is shared by all fused frames anyway, so on a long
video every pixel would mostly be subsampled away again, at four times the work of the
consistency filter. Use `--pixel-stride 1` when only a few frames are fused, as with a large
`--frame-stride`: DynamicRobot fuses 11 frames, and every pixel keeps its cloud dense.

## Output

By default the cloud goes to `<output_dir>/pointcloud/<name>.ply`, or to `--output`. The
format is binary little-endian PLY with one `vertex` element of `float x, y, z` and
`uchar red, green, blue` (15 bytes per point). The command prints one JSON line with the
settings, the filter preset (`filter`), the output frame (`z_up` or `vipe_world`), the
colour rule (`rgb`) and the frame, voxel and rejected-point counts.

From Python, `fuse_point_cloud(depth, intrinsics, c2w, rgb, FusionSettings(...))` returns
(`fusion_settings("strict", max_depth_m=100)` builds the settings of a preset with overrides)
the points (z-up; `world_frame=True` for ViPE's world frame), colours and stats, and
`write_binary_point_cloud(path, points, colors)` writes them.
`load_vipe_artifacts(output_dir, name)` reads a run and `open_rgb_source(...)` gives its
tracked frames.
