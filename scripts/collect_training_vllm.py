#!/usr/bin/env python3

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from utils_vllm import print_gpu_info
from vllm_data_collector import VllmDataCollector, combine_separate_results

def main():
    print("VLLM Data Collection with Pre-Generated Indices")
    print_gpu_info()
    
    # STEP 1: Generate indices first (run generate_sample_indices.py)
    # STEP 2: Use the generated indices file here
    
    config = {
        "models": {
            "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "coder": "Qwen/Qwen2.5-Coder-14B-Instruct",
        },
        
        # CRITICAL: Path to pre-generated indices file
        "indices_file": "data/sample_indices_10000_42_20250821_210255.json",  # UPDATE THIS PATH
        
        # Agent selection
        "agent_type": "reasoning",    # "reasoning", "coder", "both"
        
        "batch_size": 16,
        "output_dir": "data/deterministic_training",
        "enable_debug": True,
        "debug_rounds": 3,
        "max_tokens": 3000,
    }
    
    # Verify indices file exists
    if not os.path.exists(config["indices_file"]):
        print(f"ERROR: Indices file not found: {config['indices_file']}")
        print("Please run: python scripts/generate_sample_indices.py")
        return
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize collector
    collector = VllmDataCollector(config)
    
    # Collect data (no dataset_path needed - using indices file)
    results, output_file = collector.collect_data(config["output_dir"])
    
    print(f"Completed!")
    print(f"Agent type: {config['agent_type']}")
    print(f"Indices file: {config['indices_file']}")
    print(f"Results: {len(results)} samples")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()