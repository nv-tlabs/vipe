# Quick Start: ViPE in Modal

## TL;DR

ViPE is now packaged and ready for Modal. Copy-paste this to get started:

```python
import modal

# Create image with ViPE installed
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04")
    .pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0", 
        extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    .pip_install("/home/shivin/ml-testing/vipe")  # Local path
)

app = modal.App("vipe-processor", image=image)

# Process a video
@app.function(gpu="A100", timeout=3600)
def process_video(video_path: str):
    import subprocess
    result = subprocess.run(
        ["vipe", "infer", video_path, "--output", "/tmp/results"],
        capture_output=True, text=True
    )
    return result.stdout

# Test it
@app.local_entrypoint()
def main():
    result = process_video.remote("/path/to/video.mp4")
    print(result)
```

## What Changed?

✅ Configs are now inside the package (`vipe/configs/`)  
✅ Package properly configured with `pyproject.toml`  
✅ CUDA extensions build automatically  
✅ Works in Modal GPU containers  

## Run It

```bash
# Test locally (with dependencies)
cd /home/shivin/ml-testing/vipe
pip install --no-build-isolation -e .
vipe infer assets/examples/dog-example.mp4

# Deploy to Modal
cd /home/shivin/ml-monorepo/backend/modal/apps/
# Create your app file, then:
modal deploy your_app.py
```

## Need More?

- **Full Modal Examples**: See `MODAL_INSTALLATION.md`
- **What Changed**: See `PACKAGING_CHANGES.md`
- **Complete Summary**: See `PACKAGING_COMPLETE.md`

## Next: Create Modal App

```bash
cd /home/shivin/ml-monorepo/backend/modal/apps/
mkdir vipe-app && cd vipe-app
# Create your Modal app using the template above
```


