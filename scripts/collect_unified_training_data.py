#!/usr/bin/env python3

import os
import sys
import json
import asyncio
from datetime import datetime

# Add src to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from utils import load_models, clear_gpu_memory, print_gpu_info
from data_collection import (
    smart_sample_selection, 
    EnhancedDataCollector,
    create_unified_training_data,
    save_training_data_with_splits,
    analyze_training_data
)

async def main():
    print("=== Enhanced Unified Verifier Training Data Collection ===")
    print_gpu_info()
    
    # Optimized configuration
    config = {
        "models": {
            "coder": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "instruct": "Qwen/Qwen2.5-7B-Instruct",
            "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        },
        "target_samples": 10000,  # Increased for better coverage
        "batch_size": 16,        # Process 32 samples concurrently
        "sampling_strategy": "smart",
        "output_dir": "data/unified_training",
        "debug_rounds": 3,
        "max_tokens": 3072,
        "task_types": ["verify"],  # Start with verify only
    }
    
    # Setup
    dataset_path = "data/tabfact_clean/train.jsonl"
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Load models
    print("Loading models...")
    models = load_models(config)
    
    # Smart sampling (now includes ALL labels: 0, 1, 2)
    print(f"Performing smart sampling for {config['target_samples']} samples...")
    samples = smart_sample_selection(dataset_path, config["target_samples"])
    
    # Initialize enhanced data collector
    print("Initializing enhanced data collector...")
    collector = EnhancedDataCollector(models, config)
    
    # Process samples efficiently
    print(f"Processing {len(samples)} samples with batch size {config['batch_size']}...")
    all_results, raw_output_file = await collector.process_samples_efficiently(samples, config["output_dir"])
    
    # Create unified training data (now includes Unknown cases)
    print("Creating unified training data...")
    training_data = create_unified_training_data(all_results, config["task_types"])
    
    # Save training data with splits
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_file, val_file, full_file = save_training_data_with_splits(training_data, config["output_dir"], timestamp)
    
    # Analyze data
    analysis = analyze_training_data(training_data)
    
    # Save analysis
    analysis_file = f"{config['output_dir']}/training_analysis_{timestamp}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print results
    print(f"\n=== Enhanced Data Collection Complete ===")
    print(f"Raw results: {len(all_results)} samples")
    print(f"Training data: {len(training_data)} examples")
    print(f"Success rate: {len(training_data)/len(all_results)*100:.1f}%")
    
    print(f"\nTask Type Distribution:")
    for task_type, count in analysis["task_type_distribution"].items():
        print(f"  {task_type}: {count}")
    
    print(f"\nTarget Distribution (now includes Unknown):")
    for target, count in analysis["target_distribution"].items():
        print(f"  {target}: {count}")
    
    print(f"\nAgreement Types:")
    for agreement_type, count in analysis["agreement_analysis"].items():
        print(f"  {agreement_type}: {count}")
    
    print(f"\nLearning Scenarios:")
    for scenario, count in analysis["learning_scenarios"].items():
        print(f"  {scenario}: {count}")
    
    print(f"\nFiles Created:")
    print(f"  Raw data: {raw_output_file}")
    print(f"  Training data: {train_file}")
    print(f"  Validation data: {val_file}")
    print(f"  Full dataset: {full_file}")
    print(f"  Analysis: {analysis_file}")
    
    print(f"\nTraining Quality Checks:")
    readiness = analysis["training_quality"]
    for check, status in readiness.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {check}: {status}")
    
    if all(readiness.values()):
        print(f"\nData is ready for unified LoRA training!")
        print(f"Next step: Train unified verifier with {train_file}")
        print(f"The verifier will learn to handle:")
        print(f"  - Clear True/False cases")
        print(f"  - Code error cases (trust reasoning)")
        print(f"  - Ambiguous cases (predict Unknown)")
    else:
        print(f"\nConsider reviewing data quality or adjusting parameters")
    
    # Special note about Unknown cases
    unknown_count = analysis["learning_scenarios"]["unknown_cases"].split()[0]
    print(f"\nNote: {unknown_count} Unknown cases included for robust training!")

if __name__ == "__main__":
    asyncio.run(main())