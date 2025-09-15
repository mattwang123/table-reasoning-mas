#!/bin/bash
set -x

echo "🚀 REINFORCE Table Verifier Training (CUDA Fixed)"
echo "=============================================="

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="table-verifier-reinforce-${TIMESTAMP}"
WANDB_TOKEN="2c72972c9432bc92518ad59f67f278bbf47c0e5a"

# Get absolute paths
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
RL_TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$RL_TRAINING_DIR")"
WORKING_DIR="$RL_TRAINING_DIR"

# Data files
TRAIN_DATA_PATH="$(realpath "${PROJECT_ROOT}/data/reinforce_train_20250913_224659.jsonl")"
REWARD_FUNC_PATH="$(realpath "${RL_TRAINING_DIR}/rl_src/reward_func.py")"

# Create experiment directories
EXPERIMENT_DIR="${RL_TRAINING_DIR}/experiments/${EXPERIMENT_NAME}"
CHECKPOINT_PATH="${EXPERIMENT_DIR}/checkpoints"
SAVE_PATH="${EXPERIMENT_DIR}/save"
mkdir -p "${CHECKPOINT_PATH}" "${SAVE_PATH}"

echo "🔧 Configuration:"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  Train data: ${TRAIN_DATA_PATH}"
echo "  Reward func: ${REWARD_FUNC_PATH}"
echo "  Save path: ${SAVE_PATH}"
echo ""

# Check files exist
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "❌ Training data not found: $TRAIN_DATA_PATH"
    exit 1
fi

if [ ! -f "$REWARD_FUNC_PATH" ]; then
    echo "❌ Reward function not found: $REWARD_FUNC_PATH"
    exit 1
fi

# Set CUDA environment variables
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

# Environment setup
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Disable Flash Attention in favor of xformers
export VLLM_USE_FLASH_ATTN=0
export VLLM_ATTENTION_BACKEND="XFORMERS"
export TRANSFORMERS_USE_FLASH_ATTENTION_2=false

echo "🔧 CUDA Environment:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

cd "$RL_TRAINING_DIR"

# Setup Ray (local)
if ray status >/dev/null 2>&1; then
    echo "🛑 Stopping existing Ray cluster"
    ray stop
    sleep 2
fi

echo "🚀 Starting Ray cluster"
ray start --head --node-ip-address 0.0.0.0 --port 6379 --dashboard-port 8265 --num-gpus 2
sleep 5

RAY_ADDRESS="http://127.0.0.1:8265"

echo "🎯 Starting REINFORCE training..."

ray job submit --address="${RAY_ADDRESS}" \
   --runtime-env-json="{
     \"working_dir\": \"${WORKING_DIR}\",
     \"excludes\": [\"../data/*\", \"../shared/*\", \"../data_collection/*\", \"../sft_training/*\", \"../organized_project/*\", \"../.git/*\", \"../wandb/*\", \"experiments/*\", \"__pycache__/*\"],
     \"env_vars\": {
       \"WANDB_API_KEY\": \"${WANDB_TOKEN}\",
       \"CUDA_HOME\": \"${CUDA_HOME}\",
       \"CUDA_ROOT\": \"${CUDA_HOME}\",
       \"PATH\": \"${CUDA_HOME}/bin:${PATH}\",
       \"LD_LIBRARY_PATH\": \"${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}\",
       \"LIBRARY_PATH\": \"${CUDA_HOME}/lib64:${LIBRARY_PATH}\",
       \"C_INCLUDE_PATH\": \"${CUDA_HOME}/include:${C_INCLUDE_PATH}\",
       \"CPLUS_INCLUDE_PATH\": \"${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}\",
       \"CC\": \"$(which x86_64-conda-linux-gnu-gcc)\",
       \"CXX\": \"$(which x86_64-conda-linux-gnu-g++)\",
       \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
       \"PYTHONPATH\": \"${WORKING_DIR}:${WORKING_DIR}/rl_src:${PYTHONPATH}\",
       \"VLLM_USE_FLASH_ATTN\": \"0\",
       \"VLLM_ATTENTION_BACKEND\": \"XFORMERS\",
       \"TRANSFORMERS_USE_FLASH_ATTENTION_2\": \"false\"
     }
   }" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --colocate_actor_ref \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --enforce_eager \
   --pretrain "Qwen/Qwen-7B-Chat" \
   --remote_rm_url "${REWARD_FUNC_PATH}" \
   --save_path "${SAVE_PATH}" \
   --ckpt_path "${CHECKPOINT_PATH}" \
   --micro_train_batch_size 4 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 4 \
   --max_samples 10000 \
   --max_epochs 2 \
   --generate_max_len 256 \
   --prompt_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 3e-7 \
   --init_kl_coef 0.01 \
   --advantage_estimator reinforce_baseline \
   --prompt_data "${TRAIN_DATA_PATH}" \
   --input_key "prompt" \
   --normalize_reward \
   --gradient_checkpointing \
   --wandb_project "table-reasoning-reinforce" \
   --wandb_run_name "${EXPERIMENT_NAME}" \
   --use_wandb "${WANDB_TOKEN}" \
   --packing_samples \
   --kl_estimator k2 \
   --use_kl_loss

TRAINING_EXIT_CODE=$?

echo ""
echo "🧹 Cleaning up Ray..."
ray stop

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "🎉 Training completed successfully!"
    echo "📁 Checkpoints: ${CHECKPOINT_PATH}/"
    echo "📁 Final model: ${SAVE_PATH}/"
    echo "📊 WandB: https://wandb.ai/hwang302-johns-hopkins-university/table-reasoning-reinforce"
else
    echo "❌ Training failed with exit code: ${TRAINING_EXIT_CODE}"
fi