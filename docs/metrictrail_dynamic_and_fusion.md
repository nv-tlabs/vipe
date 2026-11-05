# MetricTrail: ViPE dynamic masks and VDA depth fusion

MetricTrail (`pipeline=metrictrail`) can use two optional ViPE features. Both
work as they do in ViPE's default pipeline and keep ViPE's field names. The only
difference is the SLAM backend. Both are **off by default**, and
with both off nothing of them runs: the code paths are the ones without these
features, with the same random draws.
MetricTrail's own optional inputs and outputs (known K, loop closure, stereo) are
in [optional inputs and outputs](metrictrail_optional.md).

## What they do

- **Dynamic masks** segment objects of the prompted categories (ViPE's defaults:
  people, animals, vehicles, ...) and keep DPVO's patches off them (except on fallback
  frames, below), so moving objects no longer pull the camera poses. On their own they
  change the **tracking only**; the saved depth and the point cloud still contain the
  objects (with depth fusion on, the masks also steer its alignment). Typically worth
  turning on when moving objects cover a large part of the frames.
- **Depth fusion** replaces the per-frame depth with ViPE's fused depth: Video
  Depth Anything aligned to the tracked trajectory, smooth from frame to frame. It
  changes the **depth only** (and so the point cloud), not the poses, and does not
  depend on the masks. Whether it gives a better point cloud depends on the video:
  smoother depth between neighbouring frames does not guarantee better agreement
  between distant views, so compare with the default. It runs ViPE's depth models over
  the whole video after tracking.

The two are independent and both are **off by default**. When you turn depth fusion on,
pick the setting of the matching ViPE preset:

| Scene | As ViPE's preset | Masks | Depth fusion |
|---|---|---|---|
| With moving objects | `default` | on | `adaptive_unidepth-l_svda` (Video Depth Anything Small) |
| Static | `static_vda` | off | `adaptive_unidepth-l_vda` (Video Depth Anything Large) |

Neither feature removes moving objects from the exported point cloud.

| Feature | Switch (as in ViPE) | Default |
|---|---|---|
| Dynamic-object masking | `pipeline.init.instance` (non-null turns it on) | `null` (off) |
| Depth fusion (VDA) | `pipeline.post.depth_align_model` | `null` (no depth fusion) |

```bash
# Dynamic masks with ViPE's defaults (kf_gap_sec 2.0, ViPE's phrase list, add_sky true)
python run.py pipeline=metrictrail streams.base_path=VIDEO.mp4 'pipeline.init.instance={}'

# ...or with your own fields (the same fields as ViPE's init.instance)
python run.py pipeline=metrictrail streams.base_path=VIDEO.mp4 \
  'pipeline.init.instance={kf_gap_sec:2.0,phrases:[person,car],add_sky:true}'

# Static scene: depth fusion with Video Depth Anything Large (as ViPE's static_vda preset)
python run.py pipeline=metrictrail streams.base_path=VIDEO.mp4 \
  pipeline.post.depth_align_model=adaptive_unidepth-l_vda

# Scene with moving objects: masks + depth fusion with Video Depth Anything Small
# (as ViPE's default preset)
python run.py pipeline=metrictrail streams.base_path=VIDEO.mp4 'pipeline.init.instance={}' \
  pipeline.post.depth_align_model=adaptive_unidepth-l_svda
```

## Dynamic masks (`init.instance`)

ViPE's `TrackAnythingProcessor` runs in the init stage, after the shared K
(SharedK, or the known K of `pipeline.optional.intrinsics`) and before SLAM. It uses ViPE's own arguments and gives `frame.mask = erode((instance == 0)
| sky)`, where True means a static pixel. It runs once per video. The same masks
are replayed for an FP32 recovery pass (non-finite poses), for the focal-refinement
re-track and for the outputs.

| ViPE | MetricTrail |
|---|---|
| The mask is resized bilinearly to the 1/8 feature grid, and a cell is valid where the value is `> 0.9`. | The same rule, applied on the patchifier's 1/4 grid. The mask first goes through MetricTrail's reflect-padding transport. |
| Flow weights of invalid source pixels are set to 0, so dynamic pixels add nothing to BA. | DPVO's BA only sees patches, so no patch centroid is placed on an invalid cell. DPVO's uniform random draw is rejection-sampled from extra candidates. This removes the dynamic pixels from BA without wasting the patch budget. |
| The metric depth prior. | `metric_mask` is ANDed with the static mask. |
| Dynamic pixels are left out of the exported `SLAMMap`. | Patch points on invalid cells are left out of the exported `SLAMMap`. |
| Masks are written to `mask/<n>.zip` and `mask/<n>.txt` (phrases). | The same ViPE mask artifacts. The saved metric depth (`depth/<n>.zip`, default) keeps the depth of dynamic pixels; only DPVO's prior leaves them out. |

Fallback: a frame is tracked unmasked, with unrestricted centroids and its
depth mask left as it is, in two cases:

- it has fewer valid interior cells than `PATCHES_PER_FRAME`;
- its static cells carry too little metric depth. DPVO's metric API rejects a
  frame in which no sampled patch has a metric prior, and that aborts the run.
  A static region that is only sky, for example, has no metric depth.

## Depth fusion (`post.depth_align_model`)

ViPE's own `AdaptiveDepthProcessor` runs on the DPVO output, after the final pose
and K are assigned, exactly as in ViPE's `_add_post_processors`. It uses:

- the DPVO trajectory (camera-to-world, one pose per frame);
- the input-resolution K;
- a dense keyframe map as the `SLAMMap`, in world coordinates and indexed by frame
  ordinal (below).

**The dense keyframe map (`vipe/metrictrail/fused_map.py`).** ViPE's processor prompts
its depth model (Prior-Depth-Anything) with the SLAM map projected into every frame,
and needs at least a few known points per frame. ViPE's own SLAM exports the
BA-refined dense depth of its keyframes (1/8 resolution) as that map. DPVO refines
only its sparse patches (96 per keyframe); with them alone a frame can get too few
projected points, above all once the dynamic masks remove a moving object's patches,
and the run then fails ("There are not enough known points"). So with depth fusion on,
MetricTrail fuses the two depths of every keyframe DPVO keeps:

- the depth provider's dense metric depth, sampled every 8 input pixels, with the dynamic 
  pixels (masks) and the pixels without depth left out;
- scaled to the BA-refined patch depths of that keyframe: the median ratio of patch
  depth to provider depth over the keyframe's patches (at least 8 of them; otherwise the
  median over all keyframes), so the dense points agree with the optimised geometry;
- plus DPVO's patch points themselves, unchanged.

This runs only when depth fusion is on; without it the map is DPVO's patch points
(`output.save_slam_map`).

With masks on, the masks steer the alignment, as in ViPE. The aligned depth is
written with ViPE's `io.save_depth_artifacts` to `depth/<n>.zip`, before the pose
and the run record. That file then is the run's depth artifact, as in ViPE's
default pipeline: the tracker's per-frame metric depth
(`output.save_metric_depth`) is not written in such a run, and the run record says
so (`metric_depth: {mode: async, saved: false, reason: depth_align_model}`).

Depth fusion is a ViPE post-processing output. It is separate from the per-frame
metric depth that DPVO consumes (see [the pipeline page](metrictrail.md)).
