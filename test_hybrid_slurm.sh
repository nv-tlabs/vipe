#!/bin/bash
#SBATCH --job-name=vipe-hybrid-test
#SBATCH --output=test_hybrid_%j.out
#SBATCH --error=test_hybrid_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vipe-test

# Set working directory
cd /home/shivin/ml-testing/vipe

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/shivin/ml-testing/vipe:$PYTHONPATH

echo "=========================================="
echo "ViPE Hybrid Metric Depth Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Test 1: Simple unit tests
echo "=== Running Unit Tests ==="
python -c "
import torch
from vipe.priors.depth.scale_optimizer import ScaleOptimizer
from vipe.priors.depth import make_depth_model

print('Testing ScaleOptimizer...')
optimizer = ScaleOptimizer()
gc_depth = torch.ones(128, 128) * 2.0
moge_depth = torch.ones(128, 128) * 5.0
scale = optimizer.compute_scale_at_frame(moge_depth, gc_depth)
print(f'✓ Scale computation: {scale:.3f}')

print('\\nTesting HybridMetricDepthModel...')
model = make_depth_model('hybrid_metric')
print(f'✓ Model created: {type(model).__name__}')
print(f'✓ Keyframe interval: {model.keyframe_interval}')

keyframes = model.select_keyframes(50)
print(f'✓ Keyframes for 50 frames: {keyframes}')
print('\\nAll unit tests passed!')
"

echo ""
echo "=== Testing on Example Video ==="

# Create output directory
OUTPUT_DIR="vipe_results_hybrid_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Test with one of the example videos
INPUT_VIDEO="assets/examples/cosmos-example.mp4"

echo "Input video: $INPUT_VIDEO"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video not found: $INPUT_VIDEO"
    exit 1
fi

# Get video info
echo "Video information:"
ffmpeg -i "$INPUT_VIDEO" 2>&1 | grep -E 'Duration|Stream.*Video'
echo ""

# Create a Python test script
cat > test_hybrid_video.py << 'EOF'
import sys
import torch
import cv2
from pathlib import Path
from vipe.priors.depth import make_depth_model
from vipe.priors.depth.base import DepthEstimationInput
from vipe.utils.cameras import CameraType

print("Initializing hybrid metric depth model...")
model = make_depth_model("hybrid_metric")
model = model.cuda()

# Load video
input_video = sys.argv[1]
output_dir = sys.argv[2]

print(f"Loading video: {input_video}")
cap = cv2.VideoCapture(input_video)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video properties:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Frames: {frame_count}")
print(f"  Duration: {frame_count/fps:.2f}s")
print()

# Limit to first N frames for testing
max_frames = min(30, frame_count)
print(f"Processing first {max_frames} frames...")
print()

# Read frames
frames = []
rgb_frames = []
for i in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        break
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    rgb_tensor = torch.from_numpy(rgb_frame).float() / 255.0
    rgb_frames.append(rgb_tensor)
    frames.append(frame)
    
cap.release()

print(f"Loaded {len(frames)} frames")
print()

# Create synthetic GeometryCrafter depths for testing
# In real usage, these would come from GeometryCrafter model
print("Creating synthetic GeometryCrafter depths (relative)...")
gc_depths = torch.ones(len(rgb_frames), height, width) * 2.0  # Relative depth = 2.0

# Create DepthEstimationInput objects
print("Preparing depth estimation inputs...")
depth_inputs = []
intrinsics = torch.eye(3) * 500  # Synthetic intrinsics
intrinsics[2, 2] = 1.0

for rgb in rgb_frames:
    depth_input = DepthEstimationInput(
        rgb=rgb.cuda(),
        intrinsics=intrinsics.cuda(),
        camera_type=CameraType.PINHOLE
    )
    depth_inputs.append(depth_input)

# Run hybrid depth estimation
print("Running hybrid metric depth estimation...")
print("  (This will run MoGe on keyframes only)")
print()

try:
    results = model.estimate_video(depth_inputs, gc_depths.cuda())
    
    print(f"✓ Successfully processed {len(results)} frames")
    print()
    
    # Analyze results
    print("Results analysis:")
    metric_depths = [r.metric_depth.cpu() for r in results]
    
    for i, depth in enumerate(metric_depths[:5]):  # Show first 5
        print(f"  Frame {i}: depth range [{depth.min():.3f}, {depth.max():.3f}], mean={depth.mean():.3f}")
    
    if len(metric_depths) > 5:
        print(f"  ... ({len(metric_depths) - 5} more frames)")
    
    print()
    print("✓ Hybrid metric depth test SUCCESSFUL!")
    
    # Save a sample depth map
    sample_depth = metric_depths[0].numpy()
    import numpy as np
    depth_normalized = (sample_depth - sample_depth.min()) / (sample_depth.max() - sample_depth.min())
    depth_vis = (depth_normalized * 255).astype('uint8')
    cv2.imwrite(f"{output_dir}/sample_depth_frame0.png", depth_vis)
    print(f"Saved sample depth map to: {output_dir}/sample_depth_frame0.png")
    
except Exception as e:
    print(f"✗ Error during hybrid depth estimation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("Test completed successfully!")
print("=" * 60)
EOF

# Run the test
python test_hybrid_video.py "$INPUT_VIDEO" "$OUTPUT_DIR"

# Cleanup
rm test_hybrid_video.py

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

