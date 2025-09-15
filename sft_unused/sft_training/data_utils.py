import json
import random
import os
from typing import List, Dict, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer

# Updated prompt template to match your inference format
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
    """Convert 0/1/2 predictions to True/False/Unknown text"""
    if prediction == 1:
        return "True"
    elif prediction == 0:
        return "False"
    elif prediction == 2:
        return "Unknown"
    else:
        # Handle string inputs
        pred_str = str(prediction).lower().strip()
        if pred_str in ["true", "1"]:
            return "True"
        elif pred_str in ["false", "0"]:
            return "False"
        else:
            return "Unknown"

def format_table(table_data) -> str:
    """Format table data into readable string format"""
    if isinstance(table_data, list) and len(table_data) > 0:
        if isinstance(table_data[0], list):
            # List of lists format
            headers = table_data[0]
            rows = table_data[1:]
            
            formatted = f"Headers: {', '.join(str(h) for h in headers)}\n"
            for i, row in enumerate(rows):
                formatted += f"Row {i+1}: {', '.join(str(cell) for cell in row)}\n"
            return formatted.strip()
    
    # Fallback to JSON string
    return json.dumps(table_data)

def clean_text(text: str) -> str:
    """Clean text for better formatting"""
    if not text:
        return "No content"
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long (keep reasonable length)
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    return text

def create_training_sample(sample: Dict) -> Dict:
    """Convert collected sample to verifier training format with minimal target"""
    
    try:
        # Format table
        table_str = format_table(sample["table_text"])
        
        # Get query (statement in your case)
        query = sample["statement"]
        
        # Convert all predictions to text format
        reasoning_pred_text = convert_prediction_to_text(sample["reasoning_prediction"])
        code_pred_text = convert_prediction_to_text(sample["code_prediction"])
        label_text = convert_prediction_to_text(sample["label"])
        
        # Get reasoning output (clean it up)
        reasoning_output = clean_text(sample.get("reasoning_output", "No reasoning output"))
        
        # Get code information
        extracted_code = sample.get("extracted_code", "No code generated")
        code_output = sample.get("code_output", "No output")
        code_error = sample.get("code_error", "None")
        
        # Handle None values and clean up
        if code_error == "None" or code_error is None:
            code_error = "None"
        if not code_output or code_output.strip() == "":
            code_output = "No output"
        
        # Clean code fields
        extracted_code = clean_text(extracted_code)
        code_output = clean_text(code_output)
        code_error = clean_text(code_error)
        
        # Create full prompt
        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            table=table_str,
            query=query,
            reasoning_output=reasoning_output,
            reasoning_prediction=reasoning_pred_text,
            extracted_code=extracted_code,
            code_output=code_output,
            code_error=code_error,
            code_prediction=code_pred_text
        )
        
        # UPDATED: Minimal target format - just what your normalization function needs
        target = f"Final Answer: {label_text}"
        
        return {
            "prompt": prompt,
            "target": target,  # Minimal: "Final Answer: True" or "Final Answer: False"
            "query": query,
            "sample_id": sample.get("sample_id", -1),
            "label_numeric": sample["label"],  # Keep for analysis
            "reasoning_pred_numeric": sample["reasoning_prediction"],
            "code_pred_numeric": sample["code_prediction"],
        }
        
    except Exception as e:
        print(f"Error processing sample {sample.get('sample_id', 'unknown')}: {e}")
        return None

def load_and_process_data(file_path: str) -> List[Dict]:
    """Load and process your collected JSONL data file"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    samples = []
    skipped = 0
    
    print(f"Loading data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    raw_sample = json.loads(line)
                    
                    # Skip if missing required fields
                    required_fields = ["statement", "table_text", "label", "reasoning_output", 
                                     "reasoning_prediction", "code_prediction"]
                    if not all(field in raw_sample for field in required_fields):
                        skipped += 1
                        continue
                    
                    processed_sample = create_training_sample(raw_sample)
                    if processed_sample is not None:
                        samples.append(processed_sample)
                    else:
                        skipped += 1
                    
                except json.JSONDecodeError as e:
                    print(f"JSON error on line {line_num}: {e}")
                    skipped += 1
                    continue
                except Exception as e:
                    print(f"Processing error on line {line_num}: {e}")
                    skipped += 1
                    continue
    
    print(f"Successfully loaded {len(samples)} samples")
    if skipped > 0:
        print(f"Skipped {skipped} samples due to errors or missing fields")
    
    # Show statistics
    if samples:
        labels = [s["label_numeric"] for s in samples]
        label_counts = {0: labels.count(0), 1: labels.count(1)}
        print(f"Label distribution: False={label_counts[0]}, True={label_counts[1]}")
        
        # Show example
        print("\n=== EXAMPLE TRAINING SAMPLE ===")
        example = samples[0]
        print(f"Query: {example['query'][:100]}...")
        print(f"Target: {example['target']}")  # Will show "Final Answer: True/False"
        print(f"Prompt length: {len(example['prompt'])} characters")
        print("=" * 50)
    
    return samples

def create_train_eval_split(samples: List[Dict], test_split: float = 0.15, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and eval sets with stratification"""
    
    # Stratify by label to ensure balanced split
    true_samples = [s for s in samples if s["label_numeric"] == 1]
    false_samples = [s for s in samples if s["label_numeric"] == 0]
    
    random.seed(seed)
    random.shuffle(true_samples)
    random.shuffle(false_samples)
    
    # Split each class
    true_split = int(len(true_samples) * (1 - test_split))
    false_split = int(len(false_samples) * (1 - test_split))
    
    train_samples = true_samples[:true_split] + false_samples[:false_split]
    eval_samples = true_samples[true_split:] + false_samples[false_split:]
    
    # Shuffle final splits
    random.shuffle(train_samples)
    random.shuffle(eval_samples)
    
    print(f"Stratified data split:")
    print(f"  Train samples: {len(train_samples)} (True: {sum(1 for s in train_samples if s['label_numeric'] == 1)}, False: {sum(1 for s in train_samples if s['label_numeric'] == 0)})")
    print(f"  Eval samples: {len(eval_samples)} (True: {sum(1 for s in eval_samples if s['label_numeric'] == 1)}, False: {sum(1 for s in eval_samples if s['label_numeric'] == 0)})")
    
    return train_samples, eval_samples

def tokenize_function(examples: Dict, tokenizer: PreTrainedTokenizer, max_length: int = 3072) -> Dict:
    """Tokenize samples for training"""
    
    # Combine prompt and target for training
    full_texts = []
    for prompt, target in zip(examples["prompt"], examples["target"]):
        full_text = prompt + target  # e.g., "[long prompt]Final Answer: True"
        full_texts.append(full_text)
    
    # Tokenize with proper settings
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,  # Let the trainer handle tensor conversion
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def prepare_datasets(config, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset]:
    """Prepare train and eval datasets"""
    
    print("=" * 50)
    print("PREPARING DATASETS")
    print("=" * 50)
    
    # Load your collected data
    all_samples = load_and_process_data(config.train_file)
    
    if len(all_samples) == 0:
        raise ValueError("No valid samples found in training data")
    
    # Split into train/eval
    train_samples, eval_samples = create_train_eval_split(
        all_samples, config.test_split
    )
    
    # Convert to datasets
    train_dataset = Dataset.from_list(train_samples)
    eval_dataset = Dataset.from_list(eval_samples)
    
    print("\nTokenizing datasets...")
    
    # Tokenization function
    def tokenize_batch(examples):
        return tokenize_function(examples, tokenizer, config.max_seq_length)
    
    # Remove ALL non-tensor columns before tokenization
    columns_to_remove = ["prompt", "target", "query", "sample_id", "label_numeric", 
                        "reasoning_pred_numeric", "code_pred_numeric"]
    
    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing train data"
    )

    if len(eval_dataset) > 0:
        eval_dataset = eval_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, config.max_length),
            batched=True,
            remove_columns=columns_to_remove,
            desc="Tokenizing eval data"
        )
    else:
        # Create an empty dataset with the right structure
        eval_dataset = train_dataset.select([])  # Empty dataset with same structure
        print("Warning: eval_dataset is empty, using empty dataset with train structure")
    
    print(f"\nDataset preparation complete:")
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Eval dataset: {len(eval_dataset)} samples")
    print(f"  Max sequence length: {config.max_seq_length}")
    
    # Sample verification
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Input length: {len(sample['input_ids'])}")
        print(f"  Labels length: {len(sample['labels'])}")
        print(f"  Input type: {type(sample['input_ids'])}")
    
    return train_dataset, eval_dataset