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
</p>

## News

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

## License

This project will download and install additional third-party **models and softwares**. Note that these models or softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use. This source code, **except for the Unik3D part (which is under the BY-NC-SA 4.0 license)** , is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
