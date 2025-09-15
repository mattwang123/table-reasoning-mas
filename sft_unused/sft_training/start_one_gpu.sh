#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "🚀 Launching verifier training on 1 GPU..."
echo "GPU: $CUDA_VISIBLE_DEVICES"

python training/train_verifier.py

echo "✅ Training completed!"