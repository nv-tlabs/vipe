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

- **2026/05**: Merged Panorama estimation pipeline & bump release version to 1.0.0.
- **2026/01**: Integration with [Depth-Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for depth estimation (use `dav3` pipeline).
- **2025/10**: Add support to run on wide-angle videos.
- **2025/09**: Add support to run [Lyra](https://github.com/nv-tlabs/lyra) pipeline.
- **2025/08**: Initial release of ViPE.

## License

This project will download and install additional third-party **models and softwares**. Note that these models or softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use. This source code, **except for the Unik3D part (which is under the BY-NC-SA 4.0 license)** , is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
