#!/usr/bin/env python3

import os
import sys
import json
import argparse
import yaml
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from src.data_collection.collectors import generate_sample_indices

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate sample indices for deterministic sampling")
    
    parser.add_argument("--config", help="Config file to load defaults from")
    parser.add_argument("--dataset_path", default="data/tabfact_clean/train.jsonl", 
                       help="Path to dataset")
    parser.add_argument("--target_samples", type=int, default=10000, 
                       help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--output_file", help="Output file path")
    
    args = parser.parse_args()
    
    print("📋 Generate Sample Indices for Deterministic Sampling")
    
    # Load config if provided
    if args.config and os.path.exists(args.config):
        try:
            config = load_config(args.config)
            # Use config defaults if not overridden
            dataset_path = args.dataset_path
            target_samples = args.target_samples or config.get("data", {}).get("target_samples", 10000)
            seed = args.seed or config.get("data", {}).get("seed", 42)
            print(f"✅ Using config from: {args.config}")
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}, using defaults")
            dataset_path = args.dataset_path
            target_samples = args.target_samples
            seed = args.seed
    else:
        dataset_path = args.dataset_path
        target_samples = args.target_samples
        seed = args.seed
    
    # Generate output file name
    if args.output_file:
        indices_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        indices_file = f"data/sample_indices_{target_samples}_{seed}_{timestamp}.json"
    
    # Create output directory
    os.makedirs(os.path.dirname(indices_file), exist_ok=True)
    
    print(f"📋 Configuration:")
    print(f"   Dataset: {dataset_path}")
    print(f"   Target samples: {target_samples}")
    print(f"   Seed: {seed}")
    print(f"   Output: {indices_file}")
    print()
    
    # Generate indices
    indices = generate_sample_indices(
        dataset_path=dataset_path,
        target_samples=target_samples,
        seed=seed,
        save_file=indices_file
    )
    
    print(f"\n✅ Generation completed!")
    print(f"   Generated {len(indices)} sample indices") 
    print(f"   Indices file: {indices_file}")
    print(f"\n💡 Use this file path in your data collection config:")
    print(f'   "indices_file": "{indices_file}"')

if __name__ == "__main__":
    main()