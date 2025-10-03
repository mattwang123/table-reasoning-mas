#!/usr/bin/env python3

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from src.agents.utils_vllm import print_gpu_info
from src.data_collection.collectors import VllmDataCollector

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Collect expert outputs")
    
    # Required
    parser.add_argument("--agent_type", required=True, choices=["reasoning", "coder"])
    
    # Config files - updated defaults to use separated configs
    parser.add_argument("--config", default="configs/data_collection.yaml", 
                       help="Main data collection config file")
    
    # Common overrides
    parser.add_argument("--indices_file")
    parser.add_argument("--output_dir") 
    parser.add_argument("--batch_size", type=int)
    
    args = parser.parse_args()
    
    print(f"🚀 VLLM Data Collection - {args.agent_type.title()} Agent")
    print_gpu_info()
    
    # Load unified config
    try:
        config_data = load_config(args.config)
        print(f"✅ Loaded config from: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Build collector config with overrides
    config = {
        "models": config_data["models"],
        "indices_file": args.indices_file or config_data["data"]["indices_file"],
        "agent_type": args.agent_type,
        "batch_size": args.batch_size or config_data["collection"]["batch_size"],
        "output_dir": args.output_dir or config_data["collection"]["output_dir"],
        "enable_debug": config_data["collection"]["enable_debug"],
        "debug_rounds": config_data["collection"]["debug_rounds"],
        "max_tokens": config_data["collection"]["max_tokens"],
        "vllm": config_data.get("vllm", {}),
        "generation": config_data.get("generation", {}),
    }
    
    print(f"📋 Config: {config['agent_type']} | Batch: {config['batch_size']} | Output: {config['output_dir']}")
    
    # Verify indices file
    if not os.path.exists(config["indices_file"]):
        print(f"❌ Indices file not found: {config['indices_file']}")
        return
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Run collection
    collector = VllmDataCollector(config)
    results, output_file = collector.collect_data(config["output_dir"])
    
    print(f"✅ Completed: {len(results)} samples → {output_file}")

if __name__ == "__main__":
    main()