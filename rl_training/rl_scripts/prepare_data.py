#!/usr/bin/env python3
"""
Prepare REINFORCE dataset from VLLM collected agent data
Works with reanalysis output from your data collection pipeline
"""

import argparse
import os
import sys
import random
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rl_src'))

from data_utils import (
    load_collected_agent_data, 
    create_reinforce_dataset, 
    save_dataset,
    find_collected_data_files
)

def find_reanalysis_files(base_dir: str = "../..") -> dict:
    """Find reanalysis output files from your data collection"""
    print(f"🔍 Searching for reanalysis files in {base_dir}")
    
    files_found = {}
    
    # Look for reanalysis output files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Look for various possible reanalysis output names
            if file.endswith(".jsonl"):
                if "reanalysis" in file.lower():
                    files_found["reanalysis"] = file_path
                    print(f"📁 Found reanalysis file: {file_path}")
                elif "combined_results" in file:
                    files_found["combined"] = file_path
                    print(f"📁 Found combined results: {file_path}")
                elif "deterministic_training" in root and "combined" in file:
                    files_found["combined_alt"] = file_path
                    print(f"📁 Found combined file: {file_path}")
                # Also look for individual agent files as backup
                elif "reasoning_results" in file:
                    files_found["reasoning"] = file_path
                elif "coder_results" in file:
                    files_found["coder"] = file_path
    
    return files_found

def main():
    parser = argparse.ArgumentParser(description="Prepare REINFORCE dataset from collected agent data")
    parser.add_argument("--input_file", type=str, help="Path to reanalysis/combined results file")
    parser.add_argument("--auto_find", action="store_true", help="Automatically find reanalysis files")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples to use")
    parser.add_argument("--test_split", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("🚀 REINFORCE Dataset Preparation from Reanalysis Data")
    print("=" * 60)
    
    # Find input file
    input_file = None
    
    if args.input_file:
        input_file = args.input_file
        print(f"📁 Using specified file: {input_file}")
    elif args.auto_find:
        print("🔍 Auto-finding reanalysis files...")
        found_files = find_reanalysis_files()
        
        # Priority order: reanalysis > combined > combined_alt
        if "reanalysis" in found_files:
            input_file = found_files["reanalysis"]
            print(f"✅ Using reanalysis file: {input_file}")
        elif "combined" in found_files:
            input_file = found_files["combined"]
            print(f"✅ Using combined results: {input_file}")
        elif "combined_alt" in found_files:
            input_file = found_files["combined_alt"]
            print(f"✅ Using combined file: {input_file}")
        else:
            print("❌ No reanalysis or combined results found!")
            print("Available files:")
            for key, path in found_files.items():
                print(f"  {key}: {path}")
            print("\nOptions:")
            print("1. Run your data collection script with reanalysis_mode=True")
            print("2. Specify a file manually with --input_file")
            if "reasoning" in found_files and "coder" in found_files:
                print("3. Use our combiner script to merge reasoning and coder files")
            return
    else:
        print("❌ No input file specified!")
        print("Use --input_file or --auto_find")
        print("\nExamples:")
        print("  python rl_scripts/prepare_data.py --auto_find")
        print("  python rl_scripts/prepare_data.py --input_file /path/to/reanalysis_*.jsonl")
        return
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📊 Input file: {input_file}")
    print(f"📊 Max samples: {args.max_samples}")
    print(f"📊 Output dir: {args.output_dir}")
    print()
    
    # Load collected agent data
    samples = load_collected_agent_data(input_file, args.max_samples)
    
    if len(samples) == 0:
        print("❌ No valid samples loaded!")
        print("Possible issues:")
        print("1. Reanalysis hasn't been run yet")
        print("2. Data format doesn't match expected structure")
        print("3. All samples failed processing")
        return
    
    print(f"📈 Loaded {len(samples)} samples from collected agent data")
    
    # Create REINFORCE dataset
    reinforce_samples = create_reinforce_dataset(samples)
    
    if len(reinforce_samples) == 0:
        print("❌ No REINFORCE samples created!")
        return
    
    # Split data
    print(f"✂️ Splitting data...")
    random.seed(args.seed)
    random.shuffle(reinforce_samples)
    
    split_idx = int(len(reinforce_samples) * (1 - args.test_split))
    train_samples = reinforce_samples[:split_idx]
    eval_samples = reinforce_samples[split_idx:]
    
    print(f"Split: {len(train_samples)} train, {len(eval_samples)} eval")
    
    # Save datasets with timestamped names only
    print(f"💾 Saving datasets...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    train_file = os.path.join(args.output_dir, f"reinforce_train_{timestamp}.jsonl")
    eval_file = os.path.join(args.output_dir, f"reinforce_eval_{timestamp}.jsonl")
    
    save_dataset(train_samples, train_file)
    save_dataset(eval_samples, eval_file)
    
    print(f"\n🎉 REINFORCE Dataset Ready!")
    print("=" * 60)
    print(f"✅ Train: {len(train_samples)} samples")
    print(f"✅ Eval: {len(eval_samples)} samples")
    print(f"✅ Train file: {train_file}")
    print(f"✅ Eval file: {eval_file}")
    
    # Show sample
    print(f"\n📋 Sample REINFORCE data:")
    sample = train_samples[0]
    print(f"  Prompt length: {len(sample['prompt'])} chars")
    print(f"  Ground truth: {sample['ground_truth']}")
    print(f"  Prompt preview: {sample['prompt'][:200]}...")
    
    # Return file paths for potential use in training scripts
    return train_file, eval_file

if __name__ == "__main__":
    main()