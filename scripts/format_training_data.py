#!/usr/bin/env python3
import sys
import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_processing.data_formatter import create_training_data_from_database

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def find_latest_database(search_dir: str = "data/combined") -> str:
    """Find the latest combined database file"""
    if not os.path.exists(search_dir):
        return None
    
    files = list(Path(search_dir).glob("combined_analyzed_database_*.jsonl"))
    return str(max(files, key=os.path.getctime)) if files else None

def get_default_output_dir(base_dir: str) -> str:
    """Generate default output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"formatted_{timestamp}")

def main():
    parser = argparse.ArgumentParser(description="Format training data from database")
    
    parser.add_argument("--config", default="configs/data_processing.yaml", 
                       help="Data processing config file")
    parser.add_argument("--database_file", help="Path to combined database file (overrides config)")
    parser.add_argument("--output_dir", help="Output directory (overrides config)")
    parser.add_argument("--prompt_file", help="Prompt template file (overrides config)")
    parser.add_argument("--sample_filter", 
                       choices=["all", "disagreement_only", "hard_only", "both_wrong_only"],
                       help="Sample filter to apply (overrides config)")
    parser.add_argument("--auto_find", action="store_true", help="Auto-find latest database file")
    
    args = parser.parse_args()
    
    print("🎯 Format Training Data")
    
    # Load config
    try:
        config = load_config(args.config)
        print(f"✅ Loaded config from: {args.config}")
        print(f"🔍 Config contents: {config}")  # DEBUG: Print entire config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # DEBUG: Print what we're getting from config
    print(f"🔍 Config data section: {config.get('data', 'NOT FOUND')}")
    print(f"🔍 Config prompts section: {config.get('prompts', 'NOT FOUND')}")
    print(f"🔍 Config processing section: {config.get('processing', 'NOT FOUND')}")
    
    # Get values from config with command line overrides
    database_file = args.database_file or config.get("data", {}).get("combined_database_file")
    output_dir = args.output_dir or get_default_output_dir(config.get("data", {}).get("training_dir", "data/training"))
    prompt_file = args.prompt_file or config.get("prompts", {}).get("verifier_prompt", "prompts/verifier_prompt.txt")
    sample_filter = args.sample_filter or config.get("processing", {}).get("sample_filter", "all")
    
    print(f"🔍 Values after config loading:")
    print(f"   database_file from config: {config.get('data', {}).get('combined_database_file')}")
    print(f"   prompt_file from config: {config.get('prompts', {}).get('verifier_prompt')}")
    print(f"   sample_filter from config: {config.get('processing', {}).get('sample_filter')}")
    
    # Auto-find database file if requested or not found in config
    if args.auto_find or not database_file:
        print("🔍 Auto-finding latest database file...")
        database_file = find_latest_database()
        if database_file:
            print(f"   Found: {database_file}")
        else:
            print("❌ No database file found in data/combined/")
            return
    
    # Interactive input if still needed
    if not database_file:
        database_file = input("Path to combined database file: ").strip()
    
    # Validate files exist
    if not os.path.exists(database_file):
        print(f"❌ Database file not found: {database_file}")
        return
    
    if not os.path.exists(prompt_file):
        print(f"❌ Prompt file not found: {prompt_file}")
        return
    
    print(f"📋 Configuration:")
    print(f"   Config: {args.config}")
    print(f"   Database: {database_file}")
    print(f"   Prompt: {prompt_file}")
    print(f"   Filter: {sample_filter}")
    print(f"   Output: {output_dir}")
    print()
    
    # Create training data
    create_training_data_from_database(
        database_file=database_file,
        output_dir=output_dir,
        prompt_file=prompt_file,
        sample_filter=sample_filter
    )

if __name__ == "__main__":
    main()