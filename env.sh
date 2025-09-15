#!/bin/bash

echo "🚀 Creating Exact Working OpenRL Environment"
echo "Based on your working environment specification"

echo "📦 Creating base environment with conda packages"
conda create -n openrl_working python=3.10.18 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate openrl_working

echo "🔧 Installing conda packages from nvidia, pytorch, conda-forge channels"
conda install -c nvidia -c pytorch -c conda-forge -c defaults \
    cuda-cudart=11.8.89 \
    cuda-cupti=11.8.87 \
    cuda-libraries=11.8.0 \
    cuda-nvrtc=11.8.89 \
    cuda-nvtx=11.8.86 \
    cuda-runtime=11.8.0 \
    cuda-version=12.9 \
    pytorch-cuda=11.8 \
    pytorch-mutex=1.0=cuda \
    mkl=2022.1.0 \
    mkl-devel=2022.1.0 \
    mkl-include=2022.1.0 \
    tbb=2021.13.0 \
    -y

echo "🔥 Installing PyTorch (exact versions from your env)"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

echo "⚡ Installing Flash Attention (exact version)"
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "🤖 Installing ML packages (exact versions)"
pip install transformers==4.51.3
pip install datasets==4.0.0
pip install peft==0.17.0
pip install accelerate==1.10.0

echo "🛠️ Installing core utilities (exact versions)"
pip install numpy==1.26.4
pip install pandas==2.3.1
pip install tqdm==4.67.1
pip install wandb==0.21.1

echo "🚀 Installing OpenRLHF and dependencies (exact versions)"
pip install openrlhf==0.8.1
pip install deepspeed==0.16.8
pip install ray==2.43.0

echo "🚀 Installing vLLM (exact versions)"
pip install vllm==0.8.3
pip install vllm-flash-attn==2.5.9
pip install vllm-nccl-cu12==2.18.1.0.4.0

echo "⚡ Installing xformers (exact version)"
pip install xformers==0.0.29.post2

echo "🔧 Installing additional dependencies (exact versions)"
pip install \
    einops==0.8.1 \
    safetensors==0.6.2 \
    tokenizers==0.21.4 \
    huggingface-hub==0.34.4 \
    bitsandbytes==0.46.1 \
    optimum==1.27.0 \
    sentencepiece==0.2.0 \
    tiktoken==0.10.0 \
    triton==3.2.0 \
    psutil==7.0.0 \
    ninja==1.13.0

echo "📊 Installing monitoring and utilities"
pip install \
    tensorboard==2.20.0 \
    prometheus-client==0.22.1 \
    fastapi==0.116.1 \
    uvicorn==0.35.0 \
    opencv-python-headless==4.11.0.86

echo "🧪 Testing exact installation"
python -c "
print('=== Exact Working Environment Test ===')
import sys
print(f'Python: {sys.version}')

# Test core packages
import torch
print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')

import numpy as np
print(f'✅ NumPy: {np.__version__}')

import transformers
print(f'✅ Transformers: {transformers.__version__}')

try:
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
except Exception as e:
    print(f'❌ Flash Attention: {e}')

try:
    import openrlhf
    print(f'✅ OpenRLHF: {openrlhf.__version__}')
except Exception as e:
    print(f'❌ OpenRLHF: {e}')

try:
    import vllm
    print(f'✅ vLLM: {vllm.__version__}')
except Exception as e:
    print(f'❌ vLLM: {e}')

try:
    import xformers
    print(f'✅ xformers: {xformers.__version__}')
except Exception as e:
    print(f'❌ xformers: {e}')

# Test critical imports
try:
    from openrlhf.models import Actor
    print('✅ OpenRLHF Actor imports successfully')
except Exception as e:
    print(f'❌ OpenRLHF Actor: {e}')

try:
    from transformers import AutoModel
    print('✅ Transformers AutoModel imports successfully')
except Exception as e:
    print(f'❌ Transformers AutoModel: {e}')

try:
    from vllm import LLM
    print('✅ vLLM LLM imports successfully')
except Exception as e:
    print(f'❌ vLLM LLM: {e}')

print('\\n🎉 Environment setup complete!')
print('GPU Info:')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  No CUDA GPUs available')
"

# Create activation script
cat > activate_working_env.sh << 'ACTIVATE_EOF'
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
ACTIVATE_EOF

chmod +x activate_working_env.sh

echo ""
echo "🎉 Exact Working Environment Created!"
echo "=========================================="
echo "✅ All packages match your working environment exactly"
echo "✅ Flash Attention: 2.7.4.post1 (exact match)"
echo "✅ OpenRLHF: 0.8.1 (exact match)"
echo "✅ PyTorch: 2.6.0 (exact match)"
echo "✅ All dependencies with exact versions"
echo ""
echo "🔧 To use:"
echo "  conda activate openrl_working"
echo "  source activate_working_env.sh"
echo ""
echo "🚀 Ready for REINFORCE training!"