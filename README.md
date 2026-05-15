# ViPE: Video Pose Engine for Geometric 3D Perception

<p align="center">
  <img src="assets/teaser.gif" alt="teaser"/>
</p>

**TL;DR: ViPE is a useful open-source spatial AI tool for annotating camera poses and dense depth maps from raw videos!**

ViPE estimates camera intrinsics, camera motion, and dense near-metric depth maps from unconstrained raw videos, including pinhole, wide-angle, and 360-degree panorama footage.

## Links

- 📖 [Documentation](https://nv-tlabs.github.io/vipe/)
- 🌐 [Project page](https://research.nvidia.com/labs/toronto-ai/vipe)
- 📄 [Technical whitepaper](https://research.nvidia.com/labs/toronto-ai/vipe/assets/paper.pdf)
- 📊 [Datasets](https://nv-tlabs.github.io/vipe/dataset/)

## Installation

```bash
pip install nvidia-vipe
```

This installs the `vipe` Python package and the `vipe` CLI. ViPE releases are published as source distributions, so pip builds the native CUDA extensions during installation. The environment needs a CUDA-enabled PyTorch build and an available CUDA toolkit with `nvcc`.

## License

This project will download and install additional third-party **models and softwares**. Note that these models or softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use. This source code, **except for the Unik3D part (which is under the BY-NC-SA 4.0 license)** , is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
