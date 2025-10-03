#!/bin/bash

set -e  # Exit on error

echo "🔥 Complete Data Pipeline"
echo "========================"

echo "🚀 Step 1: Generating sample indices..."
python scripts/generate_indices.py

echo "🚀 Step 2: Collecting reasoning outputs..."
python scripts/collect_data.py --agent_type reasoning

echo "🚀 Step 3: Collecting coder outputs..."
python scripts/collect_data.py --agent_type coder

echo "🚀 Step 4: Combining and analyzing..."
python scripts/combine_and_analyze.py --auto_find

echo "🚀 Step 5: Formatting training data..."
python scripts/format_training_data.py --auto_find

echo "✅ Complete pipeline finished!"
echo ""
echo "📁 Output files:"
echo "   Sample indices: data/sample_indices_*.json"
echo "   Expert outputs: data/collected/"
echo "   Combined database: data/combined/combined_analyzed_database_*.jsonl"
echo "   Training data: data/training/"
echo ""
echo "🎯 Ready for training!"

accelerate launch --config_file configs/deepspeed_zero2.yaml scripts/train_verifier.py --config configs/training.yaml