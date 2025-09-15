#!/bin/bash
set -x

echo "🚀 Simplified Multi-Node Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"

# Get node info
HEAD_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n1)
ALL_NODES=$(scontrol show hostname $SLURM_NODELIST)

echo "Head node: $HEAD_NODE"
echo "All nodes: $ALL_NODES"

# Try to get head node IP without srun
HEAD_NODE_IP=$(host $HEAD_NODE | awk '{print $4}' | head -1)
if [ -z "$HEAD_NODE_IP" ]; then
    HEAD_NODE_IP=$HEAD_NODE  # Use hostname if IP resolution fails
fi

echo "Head node IP: $HEAD_NODE_IP"

# Configuration
EXPERIMENT_NAME="reinforce-simple-$(date +%Y%m%d_%H%M%S)"
WORKING_DIR="$HOME/.local/reasoning/table-reasoning-mas/rl_training"
TRAIN_DATA="$HOME/.local/reasoning/table-reasoning-mas/data/reinforce_train_20250913_224659.jsonl"

# Create a simple Ray startup script
cat > /tmp/start_ray_head.sh << 'RAYHEAD'
#!/bin/bash
source ~/.bashrc
conda activate rlhf
export CUDA_HOME=$CONDA_PREFIX
ray stop 2>/dev/null || true
sleep 2
ray start --head --node-ip-address=HEAD_NODE_IP --port=6379 --dashboard-port=8265 --num-gpus=4
RAYHEAD

sed -i "s/HEAD_NODE_IP/$HEAD_NODE_IP/g" /tmp/start_ray_head.sh
chmod +x /tmp/start_ray_head.sh

cat > /tmp/start_ray_worker.sh << 'RAYWORKER'  
#!/bin/bash
source ~/.bashrc
conda activate rlhf
export CUDA_HOME=$CONDA_PREFIX
ray start --address=HEAD_NODE_IP:6379 --num-gpus=4
RAYWORKER

sed -i "s/HEAD_NODE_IP/$HEAD_NODE_IP/g" /tmp/start_ray_worker.sh
chmod +x /tmp/start_ray_worker.sh

# Start head node with timeout
echo "🔧 Starting Ray head node..."
timeout 60 srun --nodes=1 --ntasks=1 -w $HEAD_NODE /tmp/start_ray_head.sh &
HEAD_PID=$!

sleep 20

# Check if head node started
if kill -0 $HEAD_PID 2>/dev/null; then
    echo "✅ Head node starting..."
else
    echo "❌ Head node failed to start"
    exit 1
fi

# Start worker nodes
echo "🔗 Starting worker nodes..."
WORKER_NODES=$(echo "$ALL_NODES" | grep -v $HEAD_NODE)
for node in $WORKER_NODES; do
    echo "Starting worker on $node"
    timeout 60 srun --nodes=1 --ntasks=1 -w $node /tmp/start_ray_worker.sh &
    sleep 5
done

sleep 30

echo "📊 Checking cluster status..."
timeout 30 srun --nodes=1 --ntasks=1 -w $HEAD_NODE bash -c "
    source ~/.bashrc
    conda activate rlhf
    ray status
" || echo "⚠️ Could not get Ray status, but continuing..."

echo "🎯 Starting training..."
cd $WORKING_DIR

# Set environment
export CUDA_HOME=$CONDA_PREFIX
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export VLLM_USE_FLASH_ATTN=0

# Submit training job directly (not through srun)
python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 2 \
    --ref_num_gpus_per_node 4 \
    --actor_num_nodes 2 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --enforce_eager \
    --pretrain "Qwen/Qwen-7B-Chat" \
    --remote_rm_url "$WORKING_DIR/rl_src/reward_func.py" \
    --save_path "$WORKING_DIR/experiments/$EXPERIMENT_NAME" \
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
    --wandb_project "table-reasoning-reinforce-simple" \
    --wandb_run_name "$EXPERIMENT_NAME"

echo "🧹 Cleaning up..."
# Kill all background processes
jobs -p | xargs -r kill 2>/dev/null || true

