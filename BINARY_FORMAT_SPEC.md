# 4D Time-Series Point Cloud Binary Format Specification

## Overview
Single compressed binary file format for 4D time-series point cloud visualization, containing RGB images, depth maps, camera poses, and metadata.

## File Structure

```
[4 bytes] Header length (uint32, little-endian)
[N bytes] JSON header (UTF-8 encoded)
[M bytes] Compressed data blob (per-array compression)
```

## JSON Header Schema

```json
{
  "rgb": {
    "offset": 0,
    "dtype": "uint8",
    "shape": [T, H, W, 3],
    "compression": "webp|jpeg|png",
    "quality": 95,
    "frame_offsets": [...]
  },
  "depth": {
    "offset": N,
    "dtype": "float16",
    "shape": [T, H, W],
    "length": L
  },
  "intrinsics": {
    "offset": N,
    "dtype": "float64",
    "shape": [T, 3, 3],
    "length": L
  },
  "poses": {
    "offset": N,
    "dtype": "float32",
    "shape": [T, 4, 4],
    "encoding": "delta",
    "length": L
  },
  "poses_inv": {
    "offset": N,
    "dtype": "float32",
    "shape": [T, 4, 4],
    "encoding": "delta",
    "length": L
  },
  "meta": {
    "depth_range": [min, max],
    "total_frames": T,
    "resolution": [W, H],
    "base_fps": fps,
    "fov": fov_y,
    "fov_x": fov_x,
    "original_aspect_ratio": ratio,
    "fixed_aspect_ratio": ratio
  }
}
```

## Data Compression Strategy

### RGB Frames
- **Format**: WebP (default), JPEG, or PNG
- **Quality**: 95 (configurable)
- **Storage**: Per-frame compressed with size prefix
- **Structure**: `[4 bytes size][compressed frame data]`
- **Fallback**: WebP → JPEG if WebP encoding fails

### Depth Maps
- **Format**: Raw fp16 (float16) binary
- **Compression**: Zstandard (level 19, multi-threaded) or gzip (fallback)
- **Precision**: Half-precision floating point
- **Units**: Meters (metric depth)

### Intrinsics
- **Format**: Raw fp64 (float64) 3x3 matrices
- **Compression**: Zstandard or gzip
- **Storage**: Per-frame camera intrinsics

### Camera Poses
- **Format**: 4x4 transformation matrices (float32)
- **Encoding**: Delta-encoded (first-order differences)
- **Compression**: Zstandard or gzip
- **Types**: 
  - `poses`: Camera-to-world transforms
  - `poses_inv`: World-to-camera transforms (pre-computed)

## Compression Results

For a 122-frame video (1280x704 resolution):
- **Uncompressed**: 524.2 MB
- **Compressed**: 56.6 MB
- **Compression ratio**: 89.2% reduction

### Per-component breakdown:
- RGB: ~31.2 MB (WebP @ 95 quality)
- Depth: ~28.2 MB (fp16 + Zstandard)
- Intrinsics: ~55 bytes (constant intrinsics)
- Poses: ~5 KB (delta-encoded)
- Poses inverse: ~5 KB (delta-encoded)

## Implementation

### Saving
```python
from vipe.utils.io import save_binary_artifacts, ArtifactPath

out_path = ArtifactPath(base_path, artifact_name)
save_binary_artifacts(
    out_path, 
    cached_final_stream,
    rgb_format="webp",  # "webp", "jpeg", or "png"
    rgb_quality=95
)
```

### Reading
```python
import struct
import json
import numpy as np
import zstandard as zstd

with open(binary_path, 'rb') as f:
    # Read header
    header_length = struct.unpack('<I', f.read(4))[0]
    header_json = f.read(header_length)
    header = json.loads(header_json.decode('utf-8'))
    
    # Read compressed data blob
    data_blob = f.read()
    
    # Extract depth (example)
    depth_info = header['depth']
    depth_compressed = data_blob[depth_info['offset']:depth_info['offset'] + depth_info['length']]
    depth_bytes = zstd.ZstdDecompressor().decompress(depth_compressed)
    depth_array = np.frombuffer(depth_bytes, dtype=np.float16).reshape(tuple(depth_info['shape']))
```

## Validation

Run the test script to validate format integrity:

```bash
python test_binary_format.py path/to/file.bin
```

This validates:
- Header structure and JSON parsing
- Array shapes and data types
- Depth value ranges
- Camera intrinsics
- Pose matrix inversions
- Compression ratios
- File size

## Design Decisions

1. **fp16 for depth**: Balances precision and size. Range [0.01, 65504] covers typical scene depths.
2. **WebP default**: Best compression ratio for RGB with minimal quality loss.
3. **Delta encoding for poses**: Camera motion is smooth; delta encoding compresses well.
4. **Pre-computed inverse poses**: Avoids runtime matrix inversion overhead.
5. **Per-array compression**: Different data types benefit from different compression strategies.
6. **Zstandard preferred**: Better compression ratio and speed than gzip.

## Future Extensions

1. **Segmentation masks**: Add `masks` section for instance segmentation
2. **Sparse point cloud**: Add `points` section for 3D point cloud data
3. **Camera z-axis**: Reserved section for camera depth/near-far planes
4. **Multiple resolutions**: LOD support for progressive loading
