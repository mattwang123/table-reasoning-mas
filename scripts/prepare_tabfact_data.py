#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loaders.tabfact_loader import create_clean_format_datasets, verify_format

def main():
    print("=== Data Preparation ===")
    
    # Dataset selection
    dataset = input("Select dataset (tabfact/wikitq/both) [tabfact]: ").strip().lower() or "tabfact"
    
    if dataset in ["tabfact", "both"]:
        print("\n--- Preparing TabFact ---")
        tabfact_path = "Table-Fact-Checking"
        output_dir = "data/tabfact_clean"
        first_n = -1  # Process all
        
        create_clean_format_datasets(tabfact_path, output_dir, first_n)
        verify_format(output_dir)
        print("✅ TabFact preparation completed!")
    
    if dataset in ["wikitq", "both"]:
        print("\n--- WikiTQ (Placeholder) ---")
        print("WikiTQ loader not implemented yet")
    
    print(f"\n✅ Data preparation completed for: {dataset}")

if __name__ == "__main__":
    main()