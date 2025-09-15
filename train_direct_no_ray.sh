#!/bin/bash
set -x

echo "🚀 Direct Training (No Ray)"

# Activate environment
source ~/.bashrc
conda activate rlhf
export CUDA_HOME=$CONDA_PREFIX
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

# Configuration
EXPERIMENT_NAME="direct-no-ray-$(date +%Y%m%d_%H%M%S)"
WORKING_DIR="$HOME/.local/reasoning/table-reasoning-mas/rl_training"
TRAIN_DATA="$HOME/.local/reasoning/table-reasoning-mas/data/reinforce_train_20250913_224659.jsonl"
SAVE_PATH="$WORKING_DIR/experiments/$EXPERIMENT_NAME"

mkdir -p "$SAVE_PATH"
cd "$WORKING_DIR"

echo "🎯 Starting direct training (no Ray)..."

# Use the direct training script instead of Ray
python3 -m openrlhf.cli.train_ppo \
   --pretrain "Qwen/Qwen-7B-Chat" \
   --reward_pretrain "$WORKING_DIR/rl_src/reward_func.py" \
   --save_path "$SAVE_PATH" \
   --micro_train_batch_size 1 \
   --train_batch_size 2 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 2 \
   --n_samples_per_prompt 1 \
   --max_samples 10 \
   --max_epochs 1 \
   --generate_max_len 64 \
   --prompt_max_len 256 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 3e-7 \
   --init_kl_coef 0.01 \
   --advantage_estimator reinforce_baseline \
   --prompt_data "$TRAIN_DATA" \
   --input_key "prompt" \
   --normalize_reward \
   --gradient_checkpointing \
   --logging_steps 1

