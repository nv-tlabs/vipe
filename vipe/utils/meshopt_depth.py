"""
Depth Map Compression Using meshoptimizer

Efficient, real-time-friendly compression for Float16 depth maps using
meshoptimizer's vertex buffer codec.

This module provides a Python integration layer using ctypes + NumPy.
"""

import ctypes
import os
from typing import Tuple

import numpy as np

# Try to find the meshoptimizer library
# Priority: 1. Environment variable, 2. Local lib, 3. System installation
LIB_PATH = os.getenv("MESHOPT_LIB_PATH")
if not LIB_PATH:
    # Try local lib directory first
    local_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "lib", "libmeshoptimizer.so")
    if os.path.exists(local_lib):
        LIB_PATH = local_lib
    else:
        # Fall back to system installation
        LIB_PATH = "/usr/local/lib/libmeshoptimizer.so"

# Load the library (will raise OSError if not found)
try:
    _meshopt = ctypes.CDLL(LIB_PATH)
    
    # Configure function signatures (note: actual functions have "Buffer" suffix)
    _meshopt.meshopt_encodeVertexBufferBound.argtypes = (ctypes.c_size_t, ctypes.c_size_t)
    _meshopt.meshopt_encodeVertexBufferBound.restype = ctypes.c_size_t
    
    _meshopt.meshopt_encodeVertexBuffer.argtypes = (
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    )
    _meshopt.meshopt_encodeVertexBuffer.restype = ctypes.c_size_t
    
    _meshopt.meshopt_decodeVertexBuffer.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_size_t,
    )
    _meshopt.meshopt_decodeVertexBuffer.restype = ctypes.c_int
    
    MESHOPT_AVAILABLE = True
except (OSError, AttributeError) as e:
    MESHOPT_AVAILABLE = False
    _MESHOPT_ERROR = str(e)


def encode_depth_map(depth: np.ndarray) -> Tuple[bytes, int, int]:
    """
    Encode a Float16 depth map using meshoptimizer's vertex buffer codec.
    
    Args:
        depth: 2D numpy array of float16 depth values (H, W)
        
    Returns:
        Tuple of (compressed_bytes, height, width)
        
    Raises:
        ValueError: If depth is not float16 or not 2D
        RuntimeError: If meshoptimizer is not available
        RuntimeError: If encoding fails
    """
    if not MESHOPT_AVAILABLE:
        raise RuntimeError(f"meshoptimizer not available: {_MESHOPT_ERROR}")
    
    if depth.dtype != np.float16:
        raise ValueError("depth must be float16")
    if depth.ndim != 2:
        raise ValueError("depth must be 2D (H, W)")
    
    h, w = depth.shape
    
    # meshoptimizer requires vertex_size % 4 == 0
    # So we treat pairs of float16 values as 4-byte vertices
    # If we have odd total pixels, pad with one extra float16
    total_pixels = h * w
    if total_pixels % 2 != 0:
        # Pad to even number
        depth_padded = np.zeros(total_pixels + 1, dtype=np.float16)
        depth_padded[:total_pixels] = depth.flatten()
        count = (total_pixels + 1) // 2
        is_padded = True
    else:
        depth_padded = depth.flatten()
        count = total_pixels // 2
        is_padded = False
    
    vertex_size = 4  # Two float16 values = 4 bytes
    
    # Ensure contiguous array
    depth_c = np.ascontiguousarray(depth_padded)
    depth_bytes = depth_c.view(np.uint8)
    src_ptr = depth_bytes.ctypes.data_as(ctypes.c_void_p)
    
    # Allocate output buffer (get upper bound on compressed size)
    bound = _meshopt.meshopt_encodeVertexBufferBound(count, vertex_size)
    dst = (ctypes.c_ubyte * bound)()
    
    # Encode
    encoded_size = _meshopt.meshopt_encodeVertexBuffer(dst, bound, src_ptr, count, vertex_size)
    
    if encoded_size == 0:
        raise RuntimeError("meshopt_encodeVertex failed (returned 0)")
    
    return bytes(bytearray(dst[:encoded_size])), h, w


def decode_depth_map(compressed: bytes, height: int, width: int) -> np.ndarray:
    """
    Decode a meshoptimizer-compressed depth map back to Float16.
    
    Args:
        compressed: Compressed bytes from encode_depth_map()
        height: Original height
        width: Original width
        
    Returns:
        2D numpy array of float16 depth values (H, W)
        
    Raises:
        RuntimeError: If meshoptimizer is not available
        RuntimeError: If decoding fails
    """
    if not MESHOPT_AVAILABLE:
        raise RuntimeError(f"meshoptimizer not available: {_MESHOPT_ERROR}")
    
    total_pixels = height * width
    
    # meshoptimizer uses 4-byte vertices (pairs of float16)
    # Need to account for padding if odd number of pixels
    if total_pixels % 2 != 0:
        padded_pixels = total_pixels + 1
        count = padded_pixels // 2
        is_padded = True
    else:
        padded_pixels = total_pixels
        count = total_pixels // 2
        is_padded = False
    
    vertex_size = 4  # Two float16 values = 4 bytes
    
    # Allocate output buffer (for potentially padded data)
    depth_bytes = np.empty(padded_pixels * 2, dtype=np.uint8)  # 2 bytes per float16
    dst_ptr = depth_bytes.ctypes.data_as(ctypes.c_void_p)
    
    # Prepare source buffer
    src_arr = (ctypes.c_ubyte * len(compressed)).from_buffer_copy(compressed)
    src_ptr = ctypes.cast(src_arr, ctypes.POINTER(ctypes.c_ubyte))
    
    # Decode
    res = _meshopt.meshopt_decodeVertexBuffer(dst_ptr, count, vertex_size, src_ptr, len(compressed))
    
    if res != 0:
        raise RuntimeError(f"meshopt_decodeVertexBuffer failed with code {res}")
    
    # Convert back to float16 and trim padding if necessary
    depth_flat = depth_bytes.view(np.float16)
    if is_padded:
        depth_flat = depth_flat[:total_pixels]
    
    return depth_flat.reshape((height, width))


def is_available() -> bool:
    """Check if meshoptimizer library is available."""
    return MESHOPT_AVAILABLE


def get_library_path() -> str:
    """Get the path to the loaded meshoptimizer library."""
    return LIB_PATH if MESHOPT_AVAILABLE else None

