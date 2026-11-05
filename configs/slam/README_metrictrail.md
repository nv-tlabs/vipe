# MetricTrail benchmark settings

`pipeline=metrictrail` tracks RGB frames with DINO-fused DPVO and per-frame MoGe-2 (default)
metric depth. The DPVO backend is one of the slam presets in this directory, selected
with `slam@pipeline.slam=<preset>`. The table lists each benchmark's settings and the
commands below apply them.

| benchmark | slam preset | input raster | focal refinement | loop closure |
| --- | --- | --- | --- | --- |
| tum | `metrictrail_gba_refine` | prepared 388x518, kept | off | off |
| wild_slam | `metrictrail_gba` | native 720x1280, resized to 288x512 at read time | off | off |
| 7scenes | `metrictrail_gba` | raw 480x640, stretched to 434x574 at read time (no crop); every 5th frame tracked | off | off |
| kitti | `metrictrail_local` | prepared 280x504, kept | off | off |
| kitti_calibrated | `metrictrail_local` | prepared 280x504, kept | off | off |
| oxford | `metrictrail_local` | prepared 378x518 (stride-1 frames), kept | off | off |
| vbr | `metrictrail_local` | prepared 280x504, kept | on | off |
| lamaria | `metrictrail_local` | raw cam0 frames, resized to 392x504 at read time | off | off |

- **Input.** ViPE's frame-directory stream (`streams=frame_dir_stream`) reads one directory
  of frames per sequence. The directory name is the sequence name. It names the outputs
  and seeds DPVO, so link the frames as `data/<Benchmark>/rgb/<seq>`. The directory must
  hold only that sequence's RGB frames.
- **Frames.** Every frame is read, because the pipeline sets `streams.frame_end=-1`.
  7-Scenes uses `pipeline.init.tracking_stride=5`: the intrinsics are
  calibrated on every frame of the clip, and DPVO tracks frames 0, 5, 10, and so on. The other
  benchmarks track every frame. `source_frame_ids` in `pose/<seq>.npz` holds the input
  frame of each tracked frame, and so does `tracking_stride` in the run record.
- **Raster.** "kept" is `pipeline.init.input_raster.policy=unchanged`. "resized at read
  time" is `policy=cover_crop` with the target `height`/`width`: a cover-fit and an
  integer centre crop. "stretched" is `policy=resize`: a plain resize to `height`/`width`,
  no crop.
  - Every read-time resize uses the default resampler, bicubic with antialiasing
    (`input_raster.resampler=bicubic_antialias`; see the pipeline page). The validation numbers
    in this repository's history were produced with ViPE's bilinear resize (`VideoFrame.resize`,
    no antialiasing), like the linear resizes of the original records; add
    `pipeline.init.input_raster.resampler=bilinear` to these commands to reproduce them
    exactly. Results with the default are expected to be within run-to-run noise.
  - 720x1280 and 288x512 have the same aspect, so Wild-SLAM is not cropped. 7-Scenes is
    stretched from 480x640 to 434x574. SharedK calibrates on the same resized frames that DPVO tracks, so the K fits the tracked frames.
  - `cover_crop` and `resize` keep one frame in k above `input_raster.max_frames` (1500).
    Wild-SLAM (at most 1268 frames) and the 7-Scenes clips (at most 1000) are below that.
    LaMaria raises it.
- **Seed:** DPVO's per-sequence seed is `pipeline.slam.seed` (1234) plus the directory name.
- **Depth precision:** all 8 commands track with MoGe-2 in FP16 (`pipeline.slam.depth_fp16: true`,
  the default of every preset); SharedK calibrates with MoGe-2 in FP32 (`pipeline.init.calibration.depth_use_fp16: false`).
- **Non-finite poses:** every preset tracks a sequence once, at the sequence seed, in the
  preset's mixed precision. Only if a pose comes back non-finite does a freshly loaded
  network track it again in full FP32 (DPVO mixed precision, DINOv3 autocast and TF32
  off). This is `pipeline.slam.nonfinite_recovery`, on by default, so no command sets it.
- **Runtime:** the run record's `runtime` block holds the tracking stage's wall clock
  (calibration and model loading excluded, no warm-up), `tracking_fps`, and
  `torch.cuda.max_memory_allocated` / `max_memory_reserved` in GiB with the peak reset
  after the models are loaded, plus the attention backend's DINOv2 ran with (`xformers` or `torch_sdpa`).
- **Focal refinement:** `pipeline.focal_refine.enabled` (RoMa). Its defaults are the
  settings used here. GT-free routing on the first tracking pass decides per sequence
  whether it runs. RoMa runs in its own interpreter (`pipeline.focal_refine.roma_python`
  or `$VIPE_METRICTRAIL_ROMA_PYTHON`).
- **Loop closure:** optional PyPose CPU postprocessing consumes externally generated
  candidate JSON and writes `pose_lc/<seq>.npz`. Install `pypose==0.6.8` in the
  ViPE environment; SALAD candidate generation uses a separate environment. See the
  [installation and file workflow](../../docs/metrictrail_optional.md#loop-closure).
- **Outputs.** Each benchmark writes to its own `pipeline.output.path`,
  `vipe_results/<benchmark>/`. Each sequence gets `pose/<seq>.npz`, `intrinsics/<seq>.npz`,
  `depth/<seq>.zip` and the run record `vipe/<seq>_metrictrail.json`. With loop closure on,
  it also gets `pose_lc/<seq>.npz`.

## Commands

tum

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_gba_refine \
  streams=frame_dir_stream streams.base_path=data/TUM/rgb/<seq> \
  pipeline.init.input_raster.policy=unchanged \
  pipeline.init.calibration.allow_unstable=true \
  pipeline.output.path=vipe_results/tum
```

wild_slam

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_gba \
  streams=frame_dir_stream streams.base_path=data/Wild-SLAM/rgb/<seq> \
  pipeline.init.input_raster.policy=cover_crop \
  pipeline.init.input_raster.height=288 pipeline.init.input_raster.width=512 \
  pipeline.init.calibration.allow_unstable=true \
  pipeline.output.path=vipe_results/wild_slam
```

7scenes

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_gba \
  streams=frame_dir_stream streams.base_path=data/7-Scenes/rgb/<scene>__seq-NN \
  pipeline.init.input_raster.policy=resize \
  pipeline.init.input_raster.height=434 pipeline.init.input_raster.width=574 \
  pipeline.init.tracking_stride=5 \
  pipeline.init.calibration.allow_unstable=true \
  pipeline.output.path=vipe_results/sevenscenes
```

kitti

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_local \
  streams=frame_dir_stream streams.base_path=data/KITTI/rgb/<seq> \
  pipeline.init.input_raster.policy=unchanged \
  pipeline.init.calibration.allow_unstable=true \
  pipeline.output.path=vipe_results/kitti

# Generate this sequence's SALAD JSON externally, then:
python -m vipe.metrictrail.optional.loop_closure vipe_results/kitti \
  --name <seq> --preset kitti --candidates /path/to/<seq>-loops.json
```

kitti_calibrated

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_local \
  streams=frame_dir_stream streams.base_path=data/KITTI/rgb/<seq> \
  pipeline.init.input_raster.policy=unchanged \
  'pipeline.optional.intrinsics={path: data/KITTI/prepared_280x504/<seq>.npz}' \
  pipeline.output.path=vipe_results/kitti_calibrated
```

oxford

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_local \
  streams=frame_dir_stream streams.base_path=data/Oxford-Spires/rgb/<seq> \
  pipeline.init.input_raster.policy=unchanged \
  pipeline.init.calibration.allow_unstable=true \
  pipeline.output.path=vipe_results/oxford
```

vbr

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_local \
  streams=frame_dir_stream streams.base_path=data/VBR/rgb/<seq> \
  pipeline.init.input_raster.policy=unchanged \
  pipeline.init.calibration.allow_unstable=true \
  pipeline.focal_refine.enabled=true \
  pipeline.output.path=vipe_results/vbr

# Generate this sequence's SALAD JSON externally, then:
python -m vipe.metrictrail.optional.loop_closure vipe_results/vbr \
  --name <seq> --preset vbr --candidates /path/to/<seq>-loops.json
```

lamaria

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_local \
  streams=frame_dir_stream streams.base_path=data/LaMaria/rgb/<seq> \
  pipeline.init.input_raster.policy=cover_crop \
  pipeline.init.input_raster.height=392 pipeline.init.input_raster.width=504 \
  pipeline.init.input_raster.max_frames=100000 \
  pipeline.init.calibration.allow_unstable=true \
  pipeline.slam.seed=null \
  pipeline.output.path=vipe_results/lamaria
```

## KITTI stereo

KITTI stereo (known K and the stereo baseline) runs with the separate stereo runner
`python -m vipe.metrictrail.stereo`, not with a `pipeline=metrictrail` command. Its loop closure runs afterwards, with the loop-closure
CLI on the runner's output directory. Build the stereo kernels once with
`python -m vipe.metrictrail.build --stereo`.

- **Frames.** The right frames must be prepared on the same 280x504 raster as the left ones. Prepare them as the transform that produced the
  prepared left frames: read the raw KITTI `image_3/NNNNNN.png` as RGB, cover-fit it with
  `cv2.INTER_AREA` (scale `max(504 / W, 280 / H)`, size rounded to integers), then take the
  integer centre crop (offsets `(resized - target) // 2`). The left frames are the `images`
  of `prepared_280x504/<seq>.npz`, as for the mono commands. Both directories use the same
  file names (`000000.png`, ...): the runner pairs frame directories by frame number.
- **K.** `--intrinsics` is fx, fy, cx, cy of the 3x3 `K` stored in
  `prepared_280x504/<seq>.npz`.

Prepare the frames of one sequence: save the script below as, for example,
`prepare_kitti_stereo.py` and run `python prepare_kitti_stereo.py <seq>` (default `00`) from
the directory that holds `data/`:

```python
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

seq = sys.argv[1] if len(sys.argv) > 1 else "00"
height, width = 280, 504
left_dir, right_dir = Path(f"data/KITTI/rgb/{seq}"), Path(f"data/KITTI/rgb_right/{seq}")
left_dir.mkdir(parents=True, exist_ok=True)
right_dir.mkdir(parents=True, exist_ok=True)
with np.load(f"data/KITTI/prepared_280x504/{seq}.npz") as npz:
    images = npz["images"]  # (N, 3, 280, 504) uint8 RGB
for index, chw in enumerate(images):
    name = f"{index:06d}.png"
    cv2.imwrite(str(left_dir / name), np.ascontiguousarray(chw.transpose(1, 2, 0)[:, :, ::-1]))
    with Image.open(f"data/KITTI/dataset/sequences/{seq}/image_3/{name}") as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    scale = max(width / rgb.shape[1], height / rgb.shape[0])
    size = (int(round(rgb.shape[1] * scale)), int(round(rgb.shape[0] * scale)))
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    crop = resized[top : top + height, left : left + width]
    cv2.imwrite(str(right_dir / name), np.ascontiguousarray(crop[:, :, ::-1]))
```

Track the sequence, then apply optional loop closure with candidate files for that trajectory:

```bash
K=$(python -c "import numpy as np; K = np.load('data/KITTI/prepared_280x504/<seq>.npz')['K']; print(*K[[0, 1, 0, 1], [0, 1, 2, 2]].tolist())")
python -m vipe.metrictrail.stereo --left data/KITTI/rgb/<seq> --right data/KITTI/rgb_right/<seq> \
  --intrinsics $K --baseline <baseline> --name <seq> --output vipe_results/kitti_stereo
python -m vipe.metrictrail.optional.loop_closure vipe_results/kitti_stereo --name <seq> --preset kitti \
  --candidates /path/to/<seq>-loops.json
```

`pose/<seq>.npz` keeps the stereo trajectory and `pose_lc/<seq>.npz` holds the loop-closed
one.
