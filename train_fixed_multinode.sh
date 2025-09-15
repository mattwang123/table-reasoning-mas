#!/bin/bash
set -x

echo "🚀 Fixed Multi-Node Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"

# Activate conda environment first
source ~/.bashrc
conda activate openrl_working

# Check if environment is working
python -c "import torch; print(f'PyTorch available: {torch.cuda.is_available()}')" || {
    echo "❌ Environment not working, trying different activation"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate openrl_working
}

# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export VLLM_USE_FLASH_ATTN=0
export VLLM_ATTENTION_BACKEND="XFORMERS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Environment check:"
echo "CUDA_HOME: $CUDA_HOME"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Test imports
python -c "
try:
    import openrlhf
    print('✅ OpenRLHF available')
except ImportError as e:
    print(f'❌ OpenRLHF not available: {e}')
    exit(1)
"

# Get node info
HEAD_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n1)
HEAD_NODE_IP=$(host $HEAD_NODE | awk '{print $4}' | head -1)

echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"

# Configuration
EXPERIMENT_NAME="reinforce-fixed-$(date +%Y%m%d_%H%M%S)"
WORKING_DIR="$HOME/.local/reasoning/table-reasoning-mas/rl_training"
TRAIN_DATA="$HOME/.local/reasoning/table-reasoning-mas/data/reinforce_train_20250913_224659.jsonl"
SAVE_PATH="$WORKING_DIR/experiments/$EXPERIMENT_NAME"

mkdir -p "$SAVE_PATH"

# Start Ray cluster manually (without srun since it's failing)
echo "🔧 Starting Ray head node locally..."
ray stop 2>/dev/null || true
sleep 2

# Start head node
ray start --head \
    --node-ip-address=$HEAD_NODE_IP \
    --port=6379 \
    --dashboard-port=8265 \
    --num-gpus=$(nvidia-smi -L | wc -l) \
    --dashboard-host=0.0.0.0

sleep 10

echo "📊 Ray cluster status:"
ray status

echo "🎯 Starting training with Ray job submission..."
cd "$WORKING_DIR"

# Submit training job to Ray
ray job submit --address="http://$HEAD_NODE_IP:8265" \
   --runtime-env-json="{
     \"working_dir\": \"$WORKING_DIR\",
     \"excludes\": [\"../data/*\", \"experiments/*\"],
     \"env_vars\": {
       \"CUDA_HOME\": \"$CUDA_HOME\",
       \"DS_BUILD_OPS\": \"0\",
       \"DS_SKIP_CUDA_CHECK\": \"1\",
       \"VLLM_USE_FLASH_ATTN\": \"0\",
       \"VLLM_ATTENTION_BACKEND\": \"XFORMERS\",
       \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
       \"PYTHONPATH\": \"$WORKING_DIR:$WORKING_DIR/rl_src:\$PYTHONPATH\"
     }
   }" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --colocate_actor_ref \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --enforce_eager \
   --pretrain "Qwen/Qwen-7B-Chat" \
   --remote_rm_url "$WORKING_DIR/rl_src/reward_func.py" \
   --save_path "$SAVE_PATH" \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 64 \
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
   --prompt_data "$TRAIN_DATA" \
   --input_key "prompt" \
   --normalize_reward \
   --gradient_checkpointing \
   --wandb_project "table-reasoning-reinforce-fixed" \
   --wandb_run_name "$EXPERIMENT_NAME"

echo "🧹 Cleaning up..."
ray stop

