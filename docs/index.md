# ViPE: Video Pose Engine for Geometric 3D Perception

<p align="center">
  <img src="https://raw.githubusercontent.com/nv-tlabs/vipe/main/assets/teaser.gif" alt="ViPE teaser">
</p>

**TL;DR:** ViPE is an open-source spatial AI tool for annotating camera poses and dense depth maps from raw videos.

**Contributors:** NVIDIA Spatial Intelligence Lab, Dynamic Vision Lab, NVIDIA Isaac, and NVIDIA Research.

ViPE estimates camera intrinsics, camera motion, and dense near-metric depth maps from unconstrained raw videos. It is designed for varied real-world footage, including dynamic selfie videos, cinematic shots, dashcams, wide-angle videos, and 360-degree panoramas.

ViPE was used to annotate a large-scale video collection containing around 100K real-world internet videos, 1M high-quality AI-generated videos, and 2K panoramic videos, totaling approximately 96M frames.

## Links

- [Technical whitepaper](https://research.nvidia.com/labs/toronto-ai/vipe/assets/paper.pdf)
- [Project page](https://research.nvidia.com/labs/toronto-ai/vipe)
- [Dataset documentation](dataset.md)
- [Configuration reference](reference/configuration.md)

## What ViPE Produces

- Camera poses
- Camera intrinsics
- Dense depth maps
- Optional instance masks
- Optional visualization videos
- Optional reusable artifacts for downstream tools
