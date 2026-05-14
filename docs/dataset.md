# Dataset

<p align="center">
  <img src="https://raw.githubusercontent.com/nv-tlabs/vipe/main/assets/dataset.gif" alt="ViPE dataset">
</p>

Together with ViPE, we release a large-scale dataset containing around 1M high-quality videos with accurate camera poses and dense depth maps.

| Dataset Name | # Videos | # Frames | Hugging Face Link | License | Prefix |
| --- | ---: | ---: | --- | --- | --- |
| Dynpose-100K++ | 99,501 | 15.8M | [Link](https://huggingface.co/datasets/nvidia/vipe-dynpose-100kpp) | CC-BY-NC 4.0 | `dpsp` |
| Wild-SDG-1M | 966,448 | 78.2M | [Link](https://huggingface.co/datasets/nvidia/vipe-wild-sdg-1m) | CC-BY-NC 4.0 | `wsdg` |
| Web360 | 2,114 | 212K | [Link](https://huggingface.co/datasets/nvidia/vipe-web360) | CC-BY 4.0 | `w360` |

Download datasets with:

```bash
# Replace YOUR_PREFIX with a prefix from the table.
# More specific prefixes, such as wsdg-003e2c86, download a specific shard.
uv run python scripts/download_dataset.py --prefix YOUR_PREFIX --output_base YOUR_OUTPUT_DIR --rgb --depth
```

!!! note
    The depth component is very large and can take a long time to download. For the RGB component of Dynpose-100K++, ViPE retrieves frames from YouTube. Install `yt_dlp` and `ffmpeg-python` to use that path. See the original [Dynpose-100K dataset](https://huggingface.co/datasets/nvidia/dynpose-100k) for alternative retrieval approaches.

Visualize downloaded dataset artifacts with:

```bash
uv run vipe visualize YOUR_OUTPUT_DIR
```
