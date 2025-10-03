#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from pathlib import Path

# Fix path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from src.data_collection.collectors import combine_and_analyze_results

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def find_latest_file(directory: str, pattern: str) -> str:
    """Find the latest file matching pattern"""
    try:
        files = list(Path(directory).glob(pattern))
        return str(max(files, key=os.path.getctime)) if files else None
    except Exception as e:
        print(f"Error finding files: {e}")
        return None

def main():
    print("🔗 Combine and Analyze Expert Outputs")
    
    parser = argparse.ArgumentParser(description="Combine and analyze expert outputs")
    parser.add_argument("--config", default="configs/data_processing.yaml", 
                       help="Data processing config file")
    parser.add_argument("--reasoning_file", help="Path to reasoning results file")
    parser.add_argument("--coder_file", help="Path to coder results file")
    parser.add_argument("--output_dir", help="Output directory (overrides config)")
    parser.add_argument("--auto_find", action="store_true", help="Auto-find latest files")
    parser.add_argument("--search_dir", help="Directory to search for files (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
        print(f"✅ Loaded config from: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        # Fallback to defaults
        config = {
            "data": {
                "collected_dir": "data/collected",
                "combined_dir": "data/combined"
            }
        }
    
    # Get paths from config with overrides
    search_dir = args.search_dir or config["data"]["collected_dir"]
    output_dir = args.output_dir or config["data"]["combined_dir"]
    
    reasoning_file = args.reasoning_file
    coder_file = args.coder_file
    
    # Auto-find files
    if args.auto_find:
        print(f"🔍 Auto-finding files in: {search_dir}")
        reasoning_file = find_latest_file(search_dir, "reasoning_results_*.jsonl")
        coder_file = find_latest_file(search_dir, "coder_results_*.jsonl")
        
        if reasoning_file:
            print(f"   Found reasoning: {reasoning_file}")
        else:
            print("   ❌ No reasoning results file found")
        
        if coder_file:
            print(f"   Found coder: {coder_file}")
        else:
            print("   ❌ No coder results file found")
    
    # Interactive input if needed
    if not reasoning_file:
        reasoning_file = input("Path to reasoning results file: ").strip()
    if not coder_file:
        coder_file = input("Path to coder results file: ").strip()
    
    # Verify files exist
    if not os.path.exists(reasoning_file):
        print(f"❌ Reasoning file not found: {reasoning_file}")
        return
    if not os.path.exists(coder_file):
        print(f"❌ Coder file not found: {coder_file}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"   Config: {args.config}")
    print(f"   Reasoning file: {reasoning_file}")
    print(f"   Coder file: {coder_file}")
    print(f"   Output directory: {output_dir}")
    print()
    
    # Run combination and analysis
    print("🚀 Starting combination and analysis...")
    try:
        combined_file = combine_and_analyze_results(reasoning_file, coder_file, output_dir)
        print(f"✅ Combined database created: {combined_file}")
    except Exception as e:
        print(f"❌ Error during combination: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()