# ViPE Installation for Modal

This document describes how to install and use ViPE as a Modal app.

## Package Changes

The following changes have been made to make ViPE installable as a proper Python package:

1. **Moved configs to package**: `configs/` directory is now `vipe/configs/` (part of the package)
2. **Updated config path resolution**: `get_config_path()` now returns the package location
3. **Added package data configuration**: `pyproject.toml` includes config files in package data
4. **Created MANIFEST.in**: Explicit file inclusion rules for distributions
5. **Updated run.py**: Uses `get_config_path()` for portable config loading

## Installation in Modal

### Method 1: From Git Repository

```python
import modal

# Create Modal image with ViPE installed
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04")
    .apt_install("git")  # Needed for git+https pip install
    .pip_install(
        "torch==2.7.0", 
        "torchvision==0.22.0",
        extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    .pip_install("git+https://github.com/your-org/vipe.git")
)

app = modal.App("vipe-app", image=image)
```

### Method 2: From Local Directory

```python
import modal

# Mount local ViPE directory and install
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04")
    .pip_install(
        "torch==2.7.0", 
        "torchvision==0.22.0",
        extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    .pip_install("/path/to/vipe", force_build=True)  # Will build CUDA extensions
)

app = modal.App("vipe-app", image=image)
```

### Method 3: Using requirements.txt

First install the dependencies, then ViPE:

```python
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04")
    .pip_install_from_requirements("envs/requirements.txt")
    .pip_install("/path/to/vipe", force_build=True)
)

app = modal.App("vipe-app", image=image)
```

## Usage in Modal Functions

### Example 1: CLI Usage

```python
import modal

app = modal.App("vipe-cli")

@app.function(
    image=image,
    gpu="A100",  # Or any GPU with CUDA support
    timeout=3600,
)
def process_video(video_url: str):
    import subprocess
    
    # Download video
    subprocess.run(["wget", video_url, "-O", "/tmp/video.mp4"])
    
    # Run ViPE
    result = subprocess.run(
        ["vipe", "infer", "/tmp/video.mp4", "--output", "/tmp/results"],
        capture_output=True,
        text=True,
    )
    
    return result.stdout
```

### Example 2: Python API Usage

```python
import modal
from pathlib import Path

app = modal.App("vipe-api")

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
)
def process_video_python(video_path: str):
    import hydra
    from vipe import get_config_path, make_pipeline
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.streams.base import ProcessedVideoStream
    
    # Initialize Hydra with ViPE configs
    with hydra.initialize_config_dir(
        config_dir=str(get_config_path()), 
        version_base=None
    ):
        args = hydra.compose(
            "default",
            overrides=[
                "pipeline=default",
                "pipeline.output.path=/tmp/results",
                "pipeline.output.save_artifacts=true",
            ]
        )
    
    # Create pipeline and process video
    pipeline = make_pipeline(args.pipeline)
    video_stream = ProcessedVideoStream(
        RawMp4Stream(Path(video_path)), []
    ).cache(desc="Reading video")
    
    pipeline.run(video_stream)
    
    return "/tmp/results"
```

### Example 3: Batch Processing

```python
import modal

app = modal.App("vipe-batch")

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
)
def process_single_video(video_path: str, output_dir: str):
    """Process a single video."""
    import subprocess
    subprocess.run([
        "vipe", "infer", video_path,
        "--output", output_dir,
        "--pipeline", "default",
    ])
    return output_dir

@app.local_entrypoint()
def main(video_list: str):
    """Process multiple videos in parallel."""
    with open(video_list) as f:
        videos = [line.strip() for line in f]
    
    # Process videos in parallel
    results = list(process_single_video.map(
        [(video, f"/tmp/results/{i}") 
         for i, video in enumerate(videos)]
    ))
    
    print(f"Processed {len(results)} videos")
```

## Building CUDA Extensions

The CUDA extensions will be built automatically when the package is installed in Modal's GPU containers. The build process requires:

- CUDA toolkit (provided by the base image)
- PyTorch (installed before ViPE)
- C++ compiler and build tools (included in nvidia/cuda devel images)

The build happens during `pip install` and uses:
- `setuptools` with CUDA extension support
- Ninja build system (faster builds)
- Eigen 3.4.0 (downloaded automatically if not present)

## Troubleshooting

### CUDA Extension Build Fails

If CUDA extension compilation fails:

1. Ensure you're using a CUDA-enabled base image (e.g., `nvidia/cuda:*-devel-*`)
2. Verify PyTorch is installed with CUDA support before installing ViPE
3. Check that the GPU type in Modal supports the CUDA version

### Config Files Not Found

If you get errors about missing config files:

1. Verify `vipe/configs/` directory exists in the installation
2. Check that `get_config_path()` returns the correct path
3. Ensure package was installed with `--no-build-isolation` if using editable mode

### Import Errors

If you get import errors:

1. Make sure all dependencies from `envs/requirements.txt` are installed
2. Verify PyTorch is installed with the correct CUDA version
3. Check that the CUDA extensions built successfully

## Testing Installation

You can verify the installation works by running:

```python
@app.function(image=image, gpu="A100")
def test_installation():
    import vipe
    from vipe import get_config_path
    import os
    
    config_path = get_config_path()
    assert os.path.exists(config_path), "Config path doesn't exist"
    assert os.path.exists(config_path / "default.yaml"), "Config files missing"
    
    print(f"ViPE version: {vipe.__version__}")
    print(f"Config path: {config_path}")
    print("✓ Installation verified!")
```

## Next Steps

Once ViPE is packaged and working in Modal, you can:

1. Create Modal apps in `/home/shivin/ml-monorepo/backend/modal/apps/`
2. Use ViPE for video processing pipelines
3. Expose ViPE functionality via Modal endpoints
4. Scale processing across multiple GPUs


