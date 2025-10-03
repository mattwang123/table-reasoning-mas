#!/bin/bash

echo "🚀 Integrated REINFORCE Training Launch"

# Setup environment
source ~/.bashrc
conda activate rlhf

export WANDB_API_KEY="2c72972c9432bc92518ad59f67f278bbf47c0e5a"
export CUDA_HOME=$CONDA_PREFIX
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Create output directory
mkdir -p experiments/reinforce_integrated

echo "🔧 Environment Check:"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Working directory: $(pwd)"

# Check if we have multiple GPUs
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "🚀 Launching Multi-GPU training with $GPU_COUNT GPUs"
    accelerate launch --num_processes $GPU_COUNT --mixed_precision bf16 reinforce_trainer.py
else
    echo "🚀 Launching Single-GPU training"
    python reinforce_trainer.py
fi