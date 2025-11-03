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
│   ├── poses.bin              # cam2world transforms (float32)
│   ├── poses_inv.bin          # world2cam transforms (float32)
│   └── poses_meta.json        # Shape and metadata
├── intrinsics/
│   ├── intrinsics.bin         # Camera intrinsics (float32)
│   └── intrinsics_meta.json   # Shape and metadata
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
      "file": "pose/poses.bin",
      "meta_file": "pose/poses_meta.json",
      "dtype": "float32",
      "shape": [T, 4, 4],
      "byte_order": "little",
      "description": "cam2world transforms (OpenCV convention) - raw binary"
    },
    "poses_inv": {
      "file": "pose/poses_inv.bin",
      "meta_file": "pose/poses_meta.json",
      "dtype": "float32",
      "shape": [T, 4, 4],
      "byte_order": "little",
      "description": "world2cam transforms (computed as inverse) - raw binary"
    },
    "intrinsics": {
      "file": "intrinsics/intrinsics.bin",
      "meta_file": "intrinsics/intrinsics_meta.json",
      "dtype": "float32",
      "shape": [T, 4],
      "byte_order": "little",
      "description": "camera intrinsics per frame [fx, fy, cx, cy] - raw binary"
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

### Camera Poses (`pose/poses.bin`, `pose/poses_inv.bin`)
Raw binary files (little-endian float32):
- **`poses.bin`**: cam2world transforms [T, 4, 4]
  - OpenCV convention
  - Row-major 4x4 transformation matrices
  - Size: T * 4 * 4 * 4 bytes
- **`poses_inv.bin`**: world2cam transforms [T, 4, 4]
  - Computed as matrix inverse of poses
  - Same format as poses.bin
- **`poses_meta.json`**: Metadata
  ```json
  {
    "shape": [T, 4, 4],
    "dtype": "float32",
    "frame_indices": [0, 1, 2, ...]
  }
  ```

### Camera Intrinsics (`intrinsics/intrinsics.bin`)
Raw binary file (little-endian float32):
- **`intrinsics.bin`**: Camera intrinsics [T, 4]
  - Format: [fx, fy, cx, cy] per frame
  - Size: T * 4 * 4 bytes
- **`intrinsics_meta.json`**: Metadata
  ```json
  {
    "shape": [T, 4],
    "dtype": "float32",
    "format": "[fx, fy, cx, cy] per frame",
    "frame_indices": [0, 1, 2, ...]
  }
  ```

### Instance Masks (Optional, `mask/`)
- **Format**: PNG images per frame
- **Naming**: `frame_{idx:05d}.png`
- **Content**: Instance segmentation masks

### Phrases (Optional, `phrase/phrases.json`)
- **Format**: JSON
- **Content**: Phrase annotations for segmented instances

## Usage Examples

### JavaScript

```javascript
// Load and parse manifest
const manifestResponse = await fetch('manifest.json');
const manifest = await manifestResponse.json();

console.log(`Format: ${manifest.format_version}`);
console.log(`Frames: ${manifest.metadata.total_frames}`);
console.log(`Resolution: ${manifest.metadata.resolution}`);

// Load camera poses (cam2world)
const posesResponse = await fetch(manifest.data.poses.file);
const posesBuffer = await posesResponse.arrayBuffer();
const posesArray = new Float32Array(posesBuffer);

// Reshape to [T, 4, 4]
const [T, rows, cols] = manifest.data.poses.shape;
const poses = [];
for (let i = 0; i < T; i++) {
  const matrix = [];
  for (let r = 0; r < 4; r++) {
    const row = [];
    for (let c = 0; c < 4; c++) {
      row.push(posesArray[i * 16 + r * 4 + c]);
    }
    matrix.push(row);
  }
  poses.push(matrix);
}

// Load inverse poses (world2cam)
const posesInvResponse = await fetch(manifest.data.poses_inv.file);
const posesInvBuffer = await posesInvResponse.arrayBuffer();
const posesInvArray = new Float32Array(posesInvBuffer);

// Load intrinsics
const intrinsicsResponse = await fetch(manifest.data.intrinsics.file);
const intrinsicsBuffer = await intrinsicsResponse.arrayBuffer();
const intrinsicsArray = new Float32Array(intrinsicsBuffer);

// Reshape to [T, 4] - [fx, fy, cx, cy] per frame
const intrinsics = [];
for (let i = 0; i < T; i++) {
  intrinsics.push({
    fx: intrinsicsArray[i * 4 + 0],
    fy: intrinsicsArray[i * 4 + 1],
    cx: intrinsicsArray[i * 4 + 2],
    cy: intrinsicsArray[i * 4 + 3]
  });
}

// Load RGB video
const videoElement = document.createElement('video');
videoElement.src = manifest.data.rgb.file;
```

### Python

```python
import json
import numpy as np
import cv2
from pathlib import Path

# Read manifest
with open('manifest.json') as f:
    manifest = json.load(f)

print(f"Format version: {manifest['format_version']}")
print(f"Total frames: {manifest['metadata']['total_frames']}")
print(f"Resolution: {manifest['metadata']['resolution']}")

# Load RGB video
rgb_path = manifest['data']['rgb']['file']
video = cv2.VideoCapture(rgb_path)

# Load camera poses (cam2world)
pose_path = manifest['data']['poses']['file']
poses_meta_path = manifest['data']['poses']['meta_file']

with open(poses_meta_path) as f:
    poses_meta = json.load(f)

poses = np.fromfile(pose_path, dtype=np.float32).reshape(poses_meta['shape'])

# Load inverse poses (world2cam)
poses_inv_path = manifest['data']['poses_inv']['file']
poses_inv = np.fromfile(poses_inv_path, dtype=np.float32).reshape(poses_meta['shape'])

# Load intrinsics
intrinsics_path = manifest['data']['intrinsics']['file']
intrinsics_meta_path = manifest['data']['intrinsics']['meta_file']

with open(intrinsics_meta_path) as f:
    intrinsics_meta = json.load(f)

intrinsics = np.fromfile(intrinsics_path, dtype=np.float32).reshape(intrinsics_meta['shape'])
# intrinsics[i] = [fx, fy, cx, cy] for frame i
```

## Benefits

1. **JavaScript Compatible**: Raw binary files work directly with TypedArrays
2. **Separate Files**: Each component can be loaded independently
3. **Standard Formats**: MP4, raw binary, PNG - universal support
4. **Metadata Rich**: manifest.json + individual meta.json files
5. **Both Pose Types**: cam2world and world2cam included
6. **Easy Distribution**: Single zip file
7. **No Dependencies**: No need for NumPy or special decoders in JS

## File Size Example

For a 122-frame video at 1280x704:
- **RGB**: ~15 MB (H.264 compressed)
- **Depth**: ~20 MB (fp16 zipped)
- **Poses**: ~32 KB (both cam2world and world2cam)
- **Intrinsics**: ~4 KB
- **Manifest**: ~1 KB
- **Total**: ~35 MB (zipped)

## Version History

- **1.0**: Initial manifest format with binary files (.bin) for JavaScript compatibility
  - Raw binary float32 for poses and intrinsics
  - Separate metadata JSON files
  - Little-endian byte order
