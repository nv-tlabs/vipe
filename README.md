# ViPE: Video Pose Engine for Geometric 3D Perception

<p align="center">
  <img src="assets/teaser.gif" alt="teaser"/>
</p>

**TL;DR: ViPE is a useful open-source spatial AI tool for annotating camera poses and dense depth maps from raw videos!**

ViPE estimates camera intrinsics, camera motion, and dense near-metric depth maps from unconstrained raw videos, including pinhole, wide-angle, and 360-degree panorama footage.

<p align="center">
  <a href="https://research.nvidia.com/labs/toronto-ai/vipe"><img src="https://img.shields.io/badge/Project%20Page-76B900?logo=nvidia&logoColor=white" alt="Project Page"/></a>
  <a href="https://arxiv.org/abs/2508.10934"><img src="https://img.shields.io/badge/arXiv-2508.10934-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"/></a>
  <a href="https://pypi.org/project/nvidia-vipe/"><img src="https://img.shields.io/pypi/v/nvidia-vipe?logo=pypi&logoColor=white&label=PyPI" alt="PyPI"/></a>
  <a href="https://nv-tlabs.github.io/vipe/"><img src="https://img.shields.io/badge/Documentation-blue?logo=readthedocs&logoColor=white" alt="Documentation"/></a>
  <a href="https://nv-tlabs.github.io/vipe/dataset/"><img src="https://img.shields.io/badge/Datasets-4B8BBE?logo=databricks&logoColor=white" alt="Datasets"/></a>
  <a href="https://research.nvidia.com/labs/sil/projects/metrictrail/"><img src="https://img.shields.io/badge/MetricTrail%20Project%20Page-76B900?logo=nvidia&logoColor=white" alt="MetricTrail Project Page"/></a>
</p>

## News

- **2026/11**: 🚀🚀🚀 Released ViPE-Long: Added long-sequence pipeline (`metrictrail`) for streaming long videos reconstruction, with metric depth and local optimization as key features. Uses the MoGe v2 (`moge2-l`) keyframe-depth prior by default.
- **2026/09**: Added long-sequence SLAM (`pose_only_long`) for arbitrarily long videos, with GPU/CPU memory bounded by a sliding keyframe window instead of growing with video length. Uses the MoGe v2 (`moge2-l`) keyframe-depth prior by default.
- **2026/06**: 🚀🚀🚀 Released ViPE 1.2.0: **2.7x speed-up with no loss of accuracy**, enabled by CUDA fused kernels, model and pipeline caching, prefetching, and other optimizations.
- **2026/05**: Merged Panorama estimation pipeline & bump release version to 1.0.0.
- **2026/01**: Integration with [Depth-Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for depth estimation (use `dav3` pipeline).
- **2025/10**: Add support to run on wide-angle videos.
- **2025/09**: Add support to run [Lyra](https://github.com/nv-tlabs/lyra) pipeline.
- **2025/08**: Initial release of ViPE.

## Installation

```bash
# From PyPI
pip install nvidia-vipe

# From source (conda for CUDA/native deps, uv for the Python env)
conda env create -f envs/cu128.yml
conda activate cu128
uv sync
uv run vipe infer YOUR_VIDEO.mp4
```

See [docs/installation.md](docs/installation.md) for details (dev/docs dependency groups, etc).

## Long-Sequence Videos

For videos too long for the default pipeline's fixed-size keyframe buffer, use the `pose_only_long` pipeline: it retires old keyframes to a compact trajectory ledger as it streams, so GPU and CPU memory stay bounded (roughly constant, not growing with video length) regardless of how many frames the video has. It uses the MoGe v2 (`moge2-l`) keyframe-depth prior by default.

```bash
# Single video
uv run vipe infer YOUR_LONG_VIDEO.mp4 -p pose_only_long -o vipe_results/

# A folder of videos: every .mp4 in the folder is processed in turn, through a single
# loaded model instance, writing each video's trajectory to vipe_results/pose/<name>.npz
uv run vipe infer YOUR_VIDEO_FOLDER/ -p pose_only_long -o vipe_results/
```

## MetricTrail

MetricTrail is a **streaming metric geometry system** that combines pretrained spatial priors with bounded local geometric optimization to recover accurate camera trajectories and dense 3D reconstruction from long videos. To use metrictrail, build MetricTrail's DPVO kernels after the installation of ViPE.
### Build

```bash
uv pip install yacs numba   # DPVO runtime requirements (not ViPE dependencies)
uv pip install xformers==0.0.33.post1  # MoGe-2's DINOv2 attention
python -m vipe.metrictrail.build       # cuda_corr, cuda_ba, lietorch_backends
python -m vipe.metrictrail.build --stereo   # also the optional stereo DPVO kernels
python -m vipe.metrictrail.build --dry-run  # print the plan only
```
Point `CUDA_HOME` at the conda environment's toolkit (`export CUDA_HOME=$CONDA_PREFIX` after
`conda activate cu128`) when the shell presets another CUDA, so all extensions use nvcc 12.8.

The sources travel with ViPE's sdist (`MANIFEST.in`) and are package data of
`vipe.metrictrail`, so the build also works from an installed (non-editable) ViPE. 

### Weights

MetricTrail weights are obtained separately, with a Hugging Face release planned under the
[NVIDIA License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf).
Set `pipeline.slam.weights` or `VIPE_METRICTRAIL_DPVO_CHECKPOINT` to the DINO-DPVO checkpoint.

| weight | setting | default |
| --- | --- | --- |
| MoGe-2 ViT-L | ViPE's own | https://github.com/microsoft/moge |
| DINO-DPVO | `pipeline.slam.weights` | https://huggingface.co/nvidia/MetricTrail |
| DINOv3 ViT-S/16| `pipeline.slam.dinov3_weights` | https://github.com/facebookresearch/dinov3|
| DINO SALAD| External loop-candidate generation | https://github.com/serizba/salad/|

For SALAD candidates, follow the [separate installation and export instructions](docs/metrictrail_optional.md#install-salad-separately).

### Quick start
**The demo video** (`assets/examples/metrictrail_demo.mp4`): the in-the-wild defaults, i.e. a
280x504 cover-crop at the video's own frame rate, RGB-only SharedK calibration, local tracking.

```bash
python run.py pipeline=metrictrail streams.base_path=assets/examples/metrictrail_demo.mp4 \
    pipeline.output.path=vipe_results/metrictrail_demo
# -> vipe_results/metrictrail_demo/{pose,intrinsics,depth}/metrictrail_demo.*  and the run record
# vipe_results/metrictrail_demo/vipe/metrictrail_demo_metrictrail.json 

# point cloud export
python -m vipe.metrictrail.pointcloud vipe_results/metrictrail_demo --name metrictrail_demo \
    --video assets/examples/metrictrail_demo.mp4 --filter strict
# -> vipe_results/metrictrail_demo/pointcloud/metrictrail_demo.ply
```

**KITTI odometry**, one sequence, uncalibrated, with loop closure (frames prepared at 280x504;
data layout and every benchmark in `configs/slam/README_metrictrail.md`):

```bash
python run.py pipeline=metrictrail slam@pipeline.slam=metrictrail_local \
    streams=frame_dir_stream streams.base_path=data/KITTI/rgb/02 \
    pipeline.init.input_raster.policy=unchanged \
    pipeline.init.calibration.allow_unstable=true \
    pipeline.output.path=vipe_results/kitti

# After exporting candidate JSON for the saved trajectory:
python -m vipe.metrictrail.optional.loop_closure vipe_results/kitti --name 02 --preset kitti \
    --candidates /path/to/02-loops.json
# vipe_results/kitti/pose/02.npz (before loop closure), vipe_results/kitti/pose_lc/02.npz (after)
```

Scoring is a separate package on the `metrictrail_eval` branch (its README lists every dataset and
how to plot the aligned trajectories, `python -m metrictrail_eval.plot`):

```bash
git worktree add ../vipe-metrictrail-eval metrictrail_eval
cd ../vipe-metrictrail-eval/metrictrail_eval
python -m metrictrail_eval.score --dataset kitti --pred-format vipe --pred <repo>/vipe_results/kitti \
    --gt-root <KITTI GT, e.g. data/KITTI/prepared_280x504> --cases 02 --allow-partial --stage LC  # --stage local: before LC
```

Refer to [MetricTrail](./docs/metrictrail.md) for detailed explanations and further usages.

## License

This project will download and install additional third-party **models and softwares**. Note that these models or softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use. NVIDIA-authored source is released under the [Apache License 2.0](LICENSE). Incorporated third-party code retains its applicable terms, including Unik3D (BY-NC-SA 4.0), DPVO (MIT), and LieTorch (BSD-3-Clause). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) and [NOTICE](NOTICE).
