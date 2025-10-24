#!/bin/bash
#SBATCH --job-name=vipe_hybrid_e2e
#SBATCH --output=vipe_hybrid_e2e_%j.out
#SBATCH --error=vipe_hybrid_e2e_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --partition=batch

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 40G"
echo "Date: $(date)"
echo "========================================"
echo ""

# Load the conda environment
source /home/shivin/miniconda3/etc/profile.d/conda.sh
conda activate vipe-test

# Navigate to the ViPE directory
cd /home/shivin/ml-testing/vipe

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv
echo ""

# Set PyTorch memory management for better fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable JIT compilation - use pre-built extensions
export VIPE_EXT_JIT=0

echo "========================================"
echo "Running hybrid metric depth E2E test..."
echo "========================================"
echo ""

# Run the end-to-end test
python test_hybrid_e2e.py 2>&1

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job finished with exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "========================================"

exit $EXIT_CODE

