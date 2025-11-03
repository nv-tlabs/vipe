# ViPE Manifest Format Specification

## Overview
ViPE outputs a single zipped archive containing separate data files organized by type, with a `manifest.json` file describing the structure and metadata.

## Output Structure

```
{artifact_name}.zip
├── manifest.json              # Metadata and file structure
├── rgb/
│   └── rgb.mp4                # H264 encoded RGB video
├── depth/
│   └── depth.zip              # Zipped fp16 binary depth maps
├── pose/
│   └── pose.npz               # Camera poses (cam2world + world2cam)
├── intrinsics/
│   └── intrinsics.npz         # Per-frame camera intrinsics
├── mask/                      # Optional: instance segmentation masks
│   ├── frame_00000.png
│   └── ...
└── phrase/                    # Optional: phrase annotations
    └── phrases.json
```

## manifest.json Schema

```json
{
  "format_version": "1.0",
  "data": {
    "rgb": {
      "format": "mp4",
      "file": "rgb/rgb.mp4",
      "frame_count": 122,
      "resolution": [1280, 704]
    },
    "depth": {
      "format": "fp16_binary_zipped",
      "file": "depth/depth.zip",
      "frame_count": 122,
      "dtype": "float16",
      "units": "meters",
      "description": "Zipped fp16 binary files with shape.txt"
    },
    "poses": {
      "file": "pose/pose.npz",
      "dtype": "float32",
      "shape": [T, 4, 4],
      "description": "cam2world transforms (OpenCV convention)"
    },
    "poses_inv": {
      "file": "pose/pose.npz",
      "key": "poses_inv",
      "dtype": "float32",
      "shape": [T, 4, 4],
      "description": "world2cam transforms (computed as inverse)"
    },
    "intrinsics": {
      "file": "intrinsics/intrinsics.npz",
      "dtype": "float64",
      "shape": [T, 4],
      "description": "camera intrinsics per frame [fx, fy, cx, cy]"
    }
  },
  "metadata": {
    "depth_range": [min, max],
    "total_frames": T,
    "resolution": [W, H],
    "base_fps": fps,
    "fov_x": fov_x_degrees,
    "fov_y": fov_y_degrees,
    "aspect_ratio": ratio
  }
}
```

## Data File Formats

### RGB Video (`rgb/rgb.mp4`)
- **Format**: H.264 encoded MP4
- **Color**: RGB (8-bit per channel)
- **Encoding**: Standard MP4 container
- **Access**: Use OpenCV or any video reader

### Depth Maps (`depth/depth.zip`)
- **Format**: Zipped collection of fp16 binary files
- **Contents**:
  - `shape.txt`: Dimensions as "H,W"
  - `00000.bin`, `00001.bin`, ...: Per-frame depth maps
- **Data Type**: float16 (half-precision)
- **Units**: Meters (metric depth)
- **Reading**: Use `vipe.utils.io.read_depth_artifacts()`

### Camera Poses (`pose/pose.npz`)
NumPy archive containing:
- **`data`**: cam2world transforms [T, 4, 4] (float32)
  - OpenCV convention
  - Row-major 4x4 transformation matrices
- **`poses_inv`**: world2cam transforms [T, 4, 4] (float32)
  - Computed as matrix inverse of `data`
  - Pre-computed for convenience
- **`inds`**: Frame indices [T] (int)

### Camera Intrinsics (`intrinsics/intrinsics.npz`)
NumPy archive containing:
- **`data`**: Intrinsics [T, 4] (float64)
  - Format: [fx, fy, cx, cy] per frame
- **`inds`**: Frame indices [T] (int)

### Instance Masks (Optional, `mask/`)
- **Format**: PNG images per frame
- **Naming**: `frame_{idx:05d}.png`
- **Content**: Instance segmentation masks

### Phrases (Optional, `phrase/phrases.json`)
- **Format**: JSON
- **Content**: Phrase annotations for segmented instances

## Usage Examples

### Python

```python
import json
import zipfile
import numpy as np
import cv2
from pathlib import Path
from vipe.utils.io import read_depth_artifacts

# Extract the zip
with zipfile.ZipFile('dog-example.zip', 'r') as z:
    z.extractall('output/')

# Read manifest
with open('output/manifest.json') as f:
    manifest = json.load(f)

print(f"Format version: {manifest['format_version']}")
print(f"Total frames: {manifest['metadata']['total_frames']}")
print(f"Resolution: {manifest['metadata']['resolution']}")
print(f"Depth range: {manifest['metadata']['depth_range']}")

# Load RGB video
rgb_path = manifest['data']['rgb']['file']
video = cv2.VideoCapture(f'output/{rgb_path}')
ret, frame = video.read()

# Load depth maps
depth_path = manifest['data']['depth']['file']
depths = list(read_depth_artifacts(Path(f'output/{depth_path}')))
frame_idx, depth_map = depths[0]

# Load camera poses
pose_path = manifest['data']['poses']['file']
poses_data = np.load(f'output/{pose_path}')
cam2world = poses_data['data']          # [T, 4, 4]
world2cam = poses_data['poses_inv']     # [T, 4, 4]
frame_indices = poses_data['inds']      # [T]

# Load intrinsics
intrinsics_path = manifest['data']['intrinsics']['file']
intrinsics_data = np.load(f'output/{intrinsics_path}')
intrinsics = intrinsics_data['data']    # [T, 4] - [fx, fy, cx, cy]
```

## Benefits

1. **Separate Files**: Each component can be loaded independently
2. **Standard Formats**: MP4, NPZ, PNG - widely supported
3. **Metadata Rich**: manifest.json contains all necessary information
4. **Both Pose Types**: cam2world and world2cam included
5. **Easy Distribution**: Single zip file
6. **Readable**: JSON manifest is human-readable

## File Size Example

For a 122-frame video at 1280x704:
- **RGB**: ~15 MB (H.264 compressed)
- **Depth**: ~20 MB (fp16 zipped)
- **Poses**: ~32 KB (both cam2world and world2cam)
- **Intrinsics**: ~4 KB
- **Manifest**: ~1 KB
- **Total**: ~35 MB (zipped)

## Version History

- **1.0**: Initial manifest format with separate files and pose inverses
