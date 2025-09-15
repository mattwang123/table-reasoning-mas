#!/bin/bash
echo "🚀 Activating OpenRL Environment"
eval "$(conda shell.bash hook)"
conda activate openrl

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "✅ Environment: openrl (Python 3.10)"
echo "✅ CUDA_HOME: $CUDA_HOME"

python -c "
import torch
try:
    import openrlhf
    print(f'PyTorch: {torch.__version__}, OpenRLHF: {openrlhf.__version__}, CUDA: {torch.cuda.is_available()}')
except ImportError:
    print(f'PyTorch: {torch.__version__}, OpenRLHF: Not available, CUDA: {torch.cuda.is_available()}')
"
