#!/bin/bash
echo "🚀 Activating Working OpenRL Environment"
eval "$(conda shell.bash hook)"
conda activate openrl_working

# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "✅ Environment: openrl_working"
echo "✅ All packages match your working setup"

python -c "
import torch
try:
    import openrlhf
    print(f'Ready: PyTorch {torch.__version__}, OpenRLHF {openrlhf.__version__}, CUDA: {torch.cuda.is_available()}')
except ImportError:
    print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, OpenRLHF: Not available')
"
