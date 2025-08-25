import torch
import json
import re
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional, Tuple
from vllm import LLM, SamplingParams

def load_models_vllm(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load models using VLLM - stable version"""
    
    models = {}
    model_configs = {
        "coder": config["models"]["coder"],
        "reasoning": config["models"]["reasoning"]
    }
    
    print(f"Loading models with VLLM")
    
    for model_name, model_path in model_configs.items():
        print(f"Loading {model_name}...")
        
        models[f"llm_{model_name}"] = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,  # Conservative for stability
            trust_remote_code=True,
            max_model_len=4096,
            disable_log_stats=True,
            max_num_seqs=8,  # Conservative batch size
        )
        
        print(f"  {model_name} loaded successfully")
    
    return models

def query_llm_vllm(llm: LLM, prompt: str, max_tokens: int = 1000) -> str:
    """Query VLLM model"""
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=max_tokens,
        stop=None,
    )
    
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

def normalize_prediction(response: str) -> int:
    """Extract prediction from model response"""
    response_lower = response.lower().strip()
    
    if any(phrase in response_lower for phrase in ['true', 'yes', 'correct', 'valid']):
        if any(phrase in response_lower for phrase in ['false', 'no', 'incorrect', 'invalid']):
            return 2
        return 1
    elif any(phrase in response_lower for phrase in ['false', 'no', 'incorrect', 'invalid']):
        return 0
    else:
        return 2

def extract_code(response: str) -> str:
    """Extract Python code from model response"""
    code_patterns = [
        r'```python\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'```python(.*?)```',
        r'```(.*?)```'
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    lines = response.split('\n')
    code_lines = []
    for line in lines:
        line = line.strip()
        if (line.startswith('import ') or line.startswith('from ') or
            line.startswith('df ') or line.startswith('result ') or
            line.startswith('print(') or 'pandas' in line or 'pd.' in line):
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return "# No valid code found"

def execute_code(code: str) -> Tuple[str, Optional[str]]:
    """Execute Python code and return output and error"""
    if not code or code.startswith("# No valid code") or code.startswith("# Generation error"):
        return "", "No executable code provided"
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return result.stdout.strip(), None
        else:
            return "", result.stderr.strip()
            
    except subprocess.TimeoutExpired:
        return "", "Code execution timeout"
    except Exception as e:
        return "", str(e)

def extract_code_prediction(output: str) -> int:
    """Extract prediction from code execution output"""
    if not output:
        return 2
    
    output_lower = output.lower().strip()
    
    if 'true' in output_lower:
        return 1
    elif 'false' in output_lower:
        return 0
    else:
        return 2

def load_prompt(file_path: str) -> str:
    """Load prompt from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file {file_path} not found")
        return ""

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {memory_gb:.2f} GB")
            print(f"  Allocated Memory: {allocated_gb:.2f} GB")
    else:
        print("CUDA not available")