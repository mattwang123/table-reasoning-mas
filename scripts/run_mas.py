#!/usr/bin/env python3

import os
import sys
import yaml
from tqdm import tqdm
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_models, clear_gpu_memory, print_gpu_info, append_jsonl, log_to_file, analyze_agreement, print_analysis
from agents import MultiAgentSystem

# Your existing load_tabfact_dataset function
def load_tabfact_dataset(dataset_path, raw2clean_path=None, tag="test", first_n=-1):
    dataset = []
    
    if first_n != -1:
        all_lines = []
        for line in open(dataset_path):
            all_lines.append(line)
            if len(all_lines) >= first_n: 
                break
    else:
        all_lines = open(dataset_path).readlines()
    
    for i, line in enumerate(all_lines):
        info = json.loads(line)
        info["id"] = f"{tag}-{i}"
        info["chain"] = []
        info["cleaned_statement"] = info["statement"]
        dataset.append(info)
    
    return dataset

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Settings from your original script
    dataset_path = "data/tabfact_clean/train.jsonl"
    output_file = "mas_results.jsonl"
    log_file = "mas_results.txt"
    first_n = 20
    enable_auto_debug = True
    use_planner = False
    
    print_gpu_info()
    
    # Load models
    models = load_models(config)
    
    # Initialize MAS system
    multi_agent_system = MultiAgentSystem(
        models["pipe_coder"], 
        models["pipe_instruct"], 
        models["pipe_reasoning"], 
        enable_auto_debug, 
        use_planner
    )
    
    # Load dataset
    dataset = load_tabfact_dataset(dataset_path, first_n=first_n)
    final_results = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Processing with Multi-Agent System")):
        table_text = json.dumps(sample["table_text"])
        statement = sample["statement"]
        label = sample["label"]
        
        print(f"\n===== Sample {i + 1} =====")
        print(f"Statement: {statement}")
        
        result = multi_agent_system.process_sample(table_text, statement, str(label), log_file)
        result["sample_id"] = i
        
        print(f"Reasoning: {result['reasoning_prediction']} | Code: {result['code_prediction']} | Verifier: {result['verifier_prediction']}")
        print(f"Methods Agree: {result['methods_agree']} | Final: {result['prediction']} | Match: {result['match']}")
        
        final_results.append(result)
        append_jsonl(result, output_file)
        
        if (i + 1) % 5 == 0:
            clear_gpu_memory()
    
    # Final analysis
    final_stats = analyze_agreement(final_results)
    print_analysis(final_stats, log_file)

if __name__ == "__main__":
    main()