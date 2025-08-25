#!/usr/bin/env python3

import os
import sys
import json
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from vllm_data_collector import generate_sample_indices

def main():
    print("Generate Sample Indices for Deterministic Sampling")
    
    # Configuration
    dataset_path = "data/tabfact_clean/train.jsonl"
    target_samples = 10000  # Adjust as needed
    seed = 42             # Change this to get different sample sets
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    indices_file = f"data/sample_indices_{target_samples}_{seed}_{timestamp}.json"
    
    # Create output directory
    os.makedirs(os.path.dirname(indices_file), exist_ok=True)
    
    print(f"Generating indices for:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Target samples: {target_samples}")
    print(f"  Seed: {seed}")
    print(f"  Output: {indices_file}")
    
    # Generate indices
    indices = generate_sample_indices(
        dataset_path=dataset_path,
        target_samples=target_samples,
        seed=seed,
        save_file=indices_file
    )
    
    print(f"\nGeneration completed!")
    print(f"Generated {len(indices)} sample indices")
    print(f"Indices file: {indices_file}")
    print(f"\nUse this file path in your data collection config:")
    print(f'  "indices_file": "{indices_file}"')

if __name__ == "__main__":
    main()