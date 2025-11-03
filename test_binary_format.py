#!/usr/bin/env python3
"""Test script to verify binary format export."""

import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


def read_binary_format(binary_path: Path):
    """Read and validate the binary format file."""
    print(f"Reading binary file: {binary_path}")
    
    with binary_path.open('rb') as f:
        # Read header length
        header_length_bytes = f.read(4)
        header_length = struct.unpack('<I', header_length_bytes)[0]
        print(f"Header length: {header_length} bytes")
        
        # Read JSON header
        header_json = f.read(header_length)
        header = json.loads(header_json.decode('utf-8'))
        print(f"\nJSON Header:")
        print(json.dumps(header, indent=2))
        
        # Read compressed data blob (per-array compressed)
        data_blob = f.read()
        print(f"\nCompressed data blob size: {len(data_blob)} bytes")
        
        # Validate data arrays
        print("\n=== Validating Arrays ===")
        
        # RGB (per-frame compressed)
        rgb_info = header['rgb']
        print(f"RGB: compression={rgb_info.get('compression', 'unknown')}, "
              f"frames={rgb_info['shape'][0]}, resolution={rgb_info['shape'][2]}x{rgb_info['shape'][1]}")
        print(f"  Stored as {len(rgb_info.get('frame_offsets', []))} compressed frames")
        
        # Depth (compressed fp16)
        depth_info = header['depth']
        depth_offset = depth_info['offset']
        depth_length = depth_info['length']
        depth_shape = tuple(depth_info['shape'])
        depth_compressed = data_blob[depth_offset:depth_offset + depth_length]
        
        # Decompress depth
        if HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            depth_bytes = dctx.decompress(depth_compressed)
        else:
            depth_bytes = gzip.decompress(depth_compressed)
        
        depth_array = np.frombuffer(depth_bytes, dtype=np.float16).reshape(depth_shape)
        depth_valid = depth_array[~np.isnan(depth_array)]
        print(f"Depth: shape={depth_array.shape}, dtype={depth_array.dtype}, "
              f"range=[{depth_valid.min():.2f}, {depth_valid.max():.2f}]m")
        
        # Intrinsics (compressed)
        intr_info = header['intrinsics']
        intr_compressed = data_blob[intr_info['offset']:intr_info['offset'] + intr_info['length']]
        if HAS_ZSTD:
            intr_bytes = zstd.ZstdDecompressor().decompress(intr_compressed)
        else:
            intr_bytes = gzip.decompress(intr_compressed)
        intr_array = np.frombuffer(intr_bytes, dtype=np.float64).reshape(tuple(intr_info['shape']))
        print(f"Intrinsics: shape={intr_array.shape}, dtype={intr_array.dtype}")
        print(f"  fx={intr_array[0,0,0]:.1f}, fy={intr_array[0,1,1]:.1f}")
        
        # Poses (delta-encoded + compressed)
        poses_info = header['poses']
        poses_compressed = data_blob[poses_info['offset']:poses_info['offset'] + poses_info['length']]
        if HAS_ZSTD:
            poses_bytes = zstd.ZstdDecompressor().decompress(poses_compressed)
        else:
            poses_bytes = gzip.decompress(poses_compressed)
        poses_delta = np.frombuffer(poses_bytes, dtype=np.float32).reshape(tuple(poses_info['shape']))
        poses_array = np.cumsum(poses_delta, axis=0)
        print(f"Poses (cam2world): shape={poses_array.shape}, encoding={poses_info.get('encoding', 'raw')}")
        
        # Poses inverse (delta-encoded + compressed)
        poses_inv_info = header['poses_inv']
        poses_inv_compressed = data_blob[poses_inv_info['offset']:poses_inv_info['offset'] + poses_inv_info['length']]
        if HAS_ZSTD:
            poses_inv_bytes = zstd.ZstdDecompressor().decompress(poses_inv_compressed)
        else:
            poses_inv_bytes = gzip.decompress(poses_inv_compressed)
        poses_inv_delta = np.frombuffer(poses_inv_bytes, dtype=np.float32).reshape(tuple(poses_inv_info['shape']))
        poses_inv_array = np.cumsum(poses_inv_delta, axis=0)
        print(f"Poses inverse (world2cam): shape={poses_inv_array.shape}, encoding={poses_inv_info.get('encoding', 'raw')}")
        
        # Verify poses are inverses (check middle frame for better test)
        test_idx = len(poses_array) // 2
        identity_check = np.matmul(poses_array[test_idx], poses_inv_array[test_idx])
        identity_error = np.abs(identity_check - np.eye(4)).max()
        print(f"  Pose inverse check (frame {test_idx}): max error = {identity_error:.8f} {'✓' if identity_error < 1e-5 else '✗'}")
        
        # Metadata
        print(f"\n=== Metadata ===")
        meta = header['meta']
        for key, value in meta.items():
            print(f"{key}: {value}")
        
        # Compression summary
        print(f"\n=== Compression Summary ===")
        total_compressed = len(data_blob)
        T, H, W = depth_shape
        
        # Calculate uncompressed sizes
        rgb_uncompressed = T * H * W * 3  # uint8
        depth_uncompressed = T * H * W * 2  # float16
        intr_uncompressed = T * 3 * 3 * 8  # float64
        poses_uncompressed = T * 4 * 4 * 4  # float32
        poses_inv_uncompressed = T * 4 * 4 * 4  # float32
        total_uncompressed = rgb_uncompressed + depth_uncompressed + intr_uncompressed + poses_uncompressed + poses_inv_uncompressed
        
        print(f"Total uncompressed: {total_uncompressed / (1024**2):.1f} MB")
        print(f"Total compressed: {total_compressed / (1024**2):.1f} MB")
        print(f"Compression ratio: {(1 - total_compressed / total_uncompressed) * 100:.1f}% reduction")
        print(f"File size on disk: {binary_path.stat().st_size / (1024**2):.1f} MB")
        
        print("\n✅ Binary format validation successful!")
        return True


def main():
    if len(sys.argv) > 1:
        binary_path = Path(sys.argv[1])
    else:
        # Find the first binary file in vipe output
        vipe_results = Path("/home/shivin/ml-testing/vipe/vipe_results")
        binary_files = list(vipe_results.rglob("*.bin"))
        if not binary_files:
            print("❌ No binary files found. Please run the pipeline first.")
            print(f"   Searched in: {vipe_results}")
            return 1
        binary_path = binary_files[0]
    
    if not binary_path.exists():
        print(f"❌ File not found: {binary_path}")
        return 1
    
    try:
        read_binary_format(binary_path)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

