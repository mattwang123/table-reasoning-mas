#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "🚀 Launching verifier training on multiple GPUs..."
echo "GPUs: $CUDA_VISIBLE_DEVICES"

accelerate launch \
    --num_processes=2 \
    --main_process_port=29500 \
    training/train_verifier.py

echo "✅ Training completed!"