#!/usr/bin/env python3
"""
Simple test script for hybrid metric depth model.
Tests scale computation on synthetic data.
"""

import torch
from vipe.priors.depth.scale_optimizer import ScaleOptimizer
from vipe.priors.depth import make_depth_model

print("=" * 60)
print("Testing Hybrid Metric Depth Implementation")
print("=" * 60)

# Test 1: ScaleOptimizer
print("\n1. Testing ScaleOptimizer...")
optimizer = ScaleOptimizer()

# Create synthetic data
H, W = 128, 128
gc_depth = torch.ones(H, W) * 2.0  # Relative depth
moge_depth = torch.ones(H, W) * 5.0  # Metric depth

# Compute scale (should be ~2.5)
scale = optimizer.compute_scale_at_frame(moge_depth, gc_depth)
print(f"   ✓ Scale computation: {scale:.3f} (expected: 2.500)")
assert abs(scale - 2.5) < 0.1, f"Scale computation failed: {scale}"

# Test 2: Scale interpolation
print("\n2. Testing scale interpolation...")
keyframe_scales = {0: 1.0, 10: 2.0, 20: 3.0}
n_frames = 30
scales = optimizer.interpolate_scales(keyframe_scales, n_frames)
print(f"   ✓ Interpolated {len(scales)} scales")
print(f"   Scale at frame 0: {scales[0]:.3f} (expected: 1.000)")
print(f"   Scale at frame 5: {scales[5]:.3f} (expected: ~1.500)")
print(f"   Scale at frame 10: {scales[10]:.3f} (expected: 2.000)")

# Test 3: Model factory
print("\n3. Testing model factory...")
try:
    model = make_depth_model("hybrid_metric")
    print(f"   ✓ HybridMetricDepthModel created successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Keyframe interval: {model.keyframe_interval}")
except Exception as e:
    print(f"   ✗ Failed to create model: {e}")
    raise

# Test 4: Keyframe selection
print("\n4. Testing keyframe selection...")
n_frames = 47
keyframes = model.select_keyframes(n_frames)
print(f"   ✓ Selected {len(keyframes)} keyframes from {n_frames} frames")
print(f"   Keyframes: {keyframes}")
assert 0 in keyframes, "First frame should be a keyframe"
assert (n_frames - 1) in keyframes, "Last frame should be a keyframe"

# Test 5: Model info
print("\n5. Model information...")
info = model.get_model_info()
for key, value in info.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)

