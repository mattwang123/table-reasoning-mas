#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "🚀 Launching verifier training on multiple GPUs..."
echo "GPUs: $CUDA_VISIBLE_DEVICES"

accelerate launch \
    --num_processes=3 \
    --main_process_port 0 \
    training/train_verifier.py

echo "✅ Training completed!"