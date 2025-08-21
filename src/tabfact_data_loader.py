import json
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

def load_csv_with_hash_separator(csv_path: str) -> pd.DataFrame:
    """Load CSV file that uses # as separator"""
    try:
        df = pd.read_csv(csv_path, sep='#', engine='python')
        return df
    except Exception:
        return pd.DataFrame()

def create_clean_format_datasets(tabfact_path: str, output_dir: str, first_n: int = -1):
    """Create datasets using tokenized data with proper CSV parsing"""
    
    print(f"Creating datasets from: {tabfact_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenized data
    tokenized_data = {}
    tokenized_data_path = os.path.join(tabfact_path, "tokenized_data")
    
    for split in ['train', 'val']:
        file_path = os.path.join(tokenized_data_path, f"{split}_examples.json")
        if os.path.exists(file_path):
            print(f"Loading {split}_examples.json...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert dict to list of examples
                examples_list = []
                for table_id, table_data in data.items():
                    if isinstance(table_data, list) and len(table_data) >= 3:
                        statements = table_data[0]
                        labels = table_data[1] 
                        caption = table_data[2]
                        
                        # Create examples for each statement
                        for statement, label in zip(statements, labels):
                            example = {
                                "statement": statement,
                                "label": label,
                                "table_caption": caption,
                                "table_id": table_id
                            }
                            examples_list.append(example)
                
                tokenized_data[split] = examples_list
                print(f"  Loaded {len(examples_list)} examples from {split}")
                
            except Exception as e:
                print(f"  Error loading {split}_examples.json: {e}")
                tokenized_data[split] = []
    
    # CSV directory
    csv_dir = os.path.join(tabfact_path, "data", "all_csv")
    
    # Process each split
    for split_name, examples in tokenized_data.items():
        if not examples:
            continue
        
        print(f"\nProcessing {split_name} split ({len(examples)} examples)...")
        
        # Limit examples if requested
        if first_n != -1:
            examples = examples[:first_n]
            print(f"Limited to first {first_n} examples")
        
        dataset = []
        processed_tables = {}  # Cache loaded tables
        failed_tables = set()
        
        for example in tqdm(examples, desc=f"Processing {split_name}"):
            table_id = example["table_id"]
            
            # Skip if we already know this table failed
            if table_id in failed_tables:
                continue
            
            # Load table (use cache to avoid reloading)
            if table_id not in processed_tables:
                csv_path = os.path.join(csv_dir, table_id)
                if os.path.exists(csv_path):
                    df = load_csv_with_hash_separator(csv_path)
                    if not df.empty:
                        # Convert to clean format (list of lists)
                        table_text = []
                        
                        # Headers - clean and strip
                        headers = [str(col).strip() for col in df.columns.tolist()]
                        table_text.append(headers)
                        
                        # Rows - clean and strip
                        for _, row in df.iterrows():
                            row_data = [str(val).strip() if pd.notna(val) else "" for val in row.tolist()]
                            table_text.append(row_data)
                        
                        processed_tables[table_id] = {
                            "table_text": table_text,
                            "table_caption": example["table_caption"]
                        }
                    else:
                        failed_tables.add(table_id)
                        continue
                else:
                    failed_tables.add(table_id)
                    continue
            
            # Create example using cached table
            if table_id in processed_tables:
                clean_example = {
                    "statement": example["statement"],
                    "label": example["label"],
                    "table_caption": processed_tables[table_id]["table_caption"],
                    "table_text": processed_tables[table_id]["table_text"],
                    "table_id": table_id
                }
                dataset.append(clean_example)
        
        print(f"Created {len(dataset)} examples for {split_name}")
        print(f"Successfully processed {len(processed_tables)} unique tables")
        
        # Save dataset
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved to: {output_file}")

def verify_format(output_dir: str):
    """Verify output format is correct"""
    
    train_file = os.path.join(output_dir, "train.jsonl")
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            example = json.loads(f.readline())
        
        # Check for # symbols in table
        has_hash = any('#' in str(item) for sublist in example['table_text'] for item in sublist)
        
        print(f"\nFormat verification:")
        print(f"  Required fields present: {all(field in example for field in ['statement', 'label', 'table_caption', 'table_text', 'table_id'])}")
        print(f"  Clean format (no # symbols): {not has_hash}")
        print(f"  Sample statement: {example['statement'][:50]}...")

if __name__ == "__main__":
    print("TabFact Data Preparation")
    print("=" * 30)
    
    # Configuration
    tabfact_path = "Table-Fact-Checking"
    output_dir = "data/tabfact_clean"
    first_n = -1  # Process all data
    
    # Create clean datasets
    create_clean_format_datasets(tabfact_path, output_dir, first_n)
    
    # Verify format
    verify_format(output_dir)
    
    print(f"\nDatasets created in: {output_dir}")
    print("Ready for MAS system integration")