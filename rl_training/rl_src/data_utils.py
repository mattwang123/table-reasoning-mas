"""
Data utilities for REINFORCE - works with VLLM collected agent data
"""

import json
import os
import random
from typing import List, Dict

VERIFIER_PROMPT_TEMPLATE = """You are a table reasoning verifier. You have access to a table, a query, and analysis from two agents. Based on all this information, determine the correct response.

Table: {table}
Query: {query}

=== Reasoning Agent Analysis ===
{reasoning_output}
Reasoning Agent Prediction: {reasoning_prediction}

=== Code Agent Analysis ===
Generated Code:
{extracted_code}

Code Execution Output:
{code_output}

Code Execution Error: {code_error}
Code Agent Prediction: {code_prediction}

=== Your Task ===
Based on the analysis above, determine the correct response to the query.

Instructions:
- For statements to verify: Answer "True" or "False" 
- For questions to answer: Provide the direct answer from the table
- You may explain your reasoning first
- Always conclude with: Final Answer: [your answer]"""

def convert_prediction_to_text(prediction) -> str:
    """Convert predictions to text format"""
    if isinstance(prediction, (int, float)):
        return "True" if prediction == 1 else ("False" if prediction == 0 else "Unknown")
    else:
        pred_str = str(prediction).strip().lower()
        if pred_str in ["true", "1"]:
            return "True"
        elif pred_str in ["false", "0"]:
            return "False"
        else:
            return pred_str.title()

def format_table(table_data) -> str:
    """Format table data into readable string format"""
    if isinstance(table_data, list) and len(table_data) > 0:
        if isinstance(table_data[0], list):
            headers = table_data[0]
            rows = table_data[1:]
            formatted = f"Headers: {', '.join(str(h) for h in headers)}\n"
            for i, row in enumerate(rows):
                formatted += f"Row {i+1}: {', '.join(str(cell) for cell in row)}\n"
            return formatted.strip()
    return json.dumps(table_data)

def clean_text(text: str) -> str:
    """Clean text for better formatting"""
    if not text:
        return "No content"
    text = ' '.join(text.split())
    if len(text) > 2000:
        text = text[:2000] + "..."
    return text

def inspect_collected_data(file_path: str, num_samples: int = 3):
    """Inspect your collected agent data format"""
    print(f"🔍 Inspecting collected data: {file_path}")
    print("-" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i in range(num_samples):
            line = f.readline().strip()
            if not line:
                break
            try:
                sample = json.loads(line)
                print(f"\n📋 Sample {i+1}:")
                print(f"Keys: {list(sample.keys())}")
                
                # Show key fields for collected data
                key_fields = [
                    'sample_id', 'statement', 'label', 'table_text',
                    'reasoning_output', 'reasoning_prediction', 
                    'code_output', 'code_prediction', 'extracted_code',
                    'processing_failed'
                ]
                
                for key in key_fields:
                    if key in sample:
                        value = sample[key]
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: [NOT FOUND]")
                        
            except json.JSONDecodeError as e:
                print(f"❌ JSON error in sample {i+1}: {e}")
    print("-" * 60)

def load_collected_agent_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load data from your VLLM agent collection (combined results)"""
    print(f"📊 Loading collected agent data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    # First inspect the format
    inspect_collected_data(file_path, 2)
    
    samples = []
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            if max_samples and len(samples) >= max_samples:
                break
                
            try:
                sample = json.loads(line)
                
                # Skip failed samples
                if sample.get("processing_failed", False):
                    skipped += 1
                    continue
                
                # Check required fields for REINFORCE
                required_fields = ["statement", "table_text", "label"]
                if not all(field in sample for field in required_fields):
                    print(f"⚠️  Missing required fields in sample {line_num}: {list(sample.keys())}")
                    skipped += 1
                    continue
                
                # The sample should already be in the right format from your collector
                # Just ensure we have all needed fields
                processed_sample = {
                    "statement": sample["statement"],
                    "table_text": sample["table_text"], 
                    "label": sample["label"],
                    
                    # Agent outputs (should exist from your collection)
                    "reasoning_output": sample.get("reasoning_output", "No reasoning output"),
                    "reasoning_prediction": sample.get("reasoning_prediction", sample.get("label", 0)),
                    "code_prediction": sample.get("code_prediction", sample.get("label", 0)),
                    "code_output": sample.get("code_output", "No code output"),
                    "code_error": sample.get("code_error", "None"),
                    "extracted_code": sample.get("extracted_code", "No code extracted"),
                    
                    "sample_id": sample.get("sample_id", line_num)
                }
                
                samples.append(processed_sample)
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON error on line {line_num}: {e}")
                skipped += 1
                continue
            except Exception as e:
                print(f"❌ Processing error on line {line_num}: {e}")
                skipped += 1
                continue
    
    print(f"✅ Loaded {len(samples)} samples")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} samples due to errors or missing fields")
    
    return samples

def find_collected_data_files(base_dir: str = "../..") -> Dict[str, str]:
    """Find your collected data files"""
    print(f"🔍 Searching for collected data files in {base_dir}")
    
    files_found = {}
    
    # Look for combined results (preferred)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "combined_results" in file and file.endswith(".jsonl"):
                files_found["combined"] = file_path
                print(f"📁 Found combined results: {file_path}")
            elif "reasoning_results" in file and file.endswith(".jsonl"):
                files_found["reasoning"] = file_path
                print(f"📁 Found reasoning results: {file_path}")
            elif "coder_results" in file and file.endswith(".jsonl"):
                files_found["coder"] = file_path
                print(f"📁 Found coder results: {file_path}")
    
    return files_found

def create_reinforce_dataset(samples: List[Dict]) -> List[Dict]:
    """Create REINFORCE dataset from collected agent data"""
    print(f"🎯 Creating REINFORCE dataset from {len(samples)} collected samples...")
    
    reinforce_samples = []
    
    for sample in samples:
        try:
            prompt = VERIFIER_PROMPT_TEMPLATE.format(
                table=format_table(sample["table_text"]),
                query=sample["statement"],
                reasoning_output=clean_text(sample.get("reasoning_output", "")),
                reasoning_prediction=convert_prediction_to_text(sample["reasoning_prediction"]),
                extracted_code=clean_text(sample.get("extracted_code", "")),
                code_output=clean_text(sample.get("code_output", "")),
                code_error=clean_text(sample.get("code_error", "None")),
                code_prediction=convert_prediction_to_text(sample["code_prediction"])
            )
            
            ground_truth = convert_prediction_to_text(sample["label"])
            
            reinforce_samples.append({
                "prompt": prompt,
                "ground_truth": ground_truth
            })
            
        except Exception as e:
            print(f"❌ Error processing sample {sample.get('sample_id', '?')}: {e}")
            continue
    
    print(f"✅ Created {len(reinforce_samples)} REINFORCE samples")
    return reinforce_samples

def save_dataset(samples: List[Dict], output_file: str):
    """Save dataset"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(samples)} samples to {output_file}")