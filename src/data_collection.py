import json
import random
import asyncio
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm
import torch
import concurrent.futures
import multiprocessing as mp

def smart_sample_selection(dataset_path: str, target_samples: int = 1000) -> List[Dict]:
    """Improved smart sampling for better training diversity"""
    
    print("Loading dataset...")
    with open(dataset_path, 'r') as f:
        all_data = [json.loads(line) for line in f]
    
    print(f"Total available examples: {len(all_data)}")
    
    # Keep ALL labels (0, 1, 2) for comprehensive training
    print(f"Label distribution:")
    label_counts = {}
    for ex in tqdm(all_data, desc="Analyzing labels"):
        label = ex['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in label_counts.items():
        label_name = {0: "False", 1: "True", 2: "Unknown"}[label]
        print(f"  {label_name} ({label}): {count}")
    
    # Define complexity metrics
    def calculate_complexity_score(example):
        table = example['table_text']
        statement = example['statement']
        
        # Table complexity
        table_rows = len(table) if table else 0
        table_cols = len(table[0]) if table and len(table) > 0 else 0
        table_size = table_rows * table_cols
        
        # Statement complexity
        statement_words = len(statement.split())
        
        # Numerical content (harder for reasoning)
        has_numbers = any(char.isdigit() for char in statement)
        
        # Comparison words (require more reasoning)
        comparison_words = ['more', 'less', 'greater', 'smaller', 'higher', 'lower', 'most', 'least', 'average', 'total']
        has_comparison = any(word in statement.lower() for word in comparison_words)
        
        # Temporal content (dates, years)
        temporal_words = ['year', 'date', 'month', 'day', 'before', 'after', 'during']
        has_temporal = any(word in statement.lower() for word in temporal_words)
        
        # Negation (harder to process)
        negation_words = ['not', 'no', 'never', 'none', 'neither']
        has_negation = any(word in statement.lower() for word in negation_words)
        
        # Calculate composite complexity score
        complexity = 0.0
        complexity += min(table_size / 50.0, 1.0) * 0.3  # Table size factor
        complexity += min(statement_words / 20.0, 1.0) * 0.2  # Statement length factor
        complexity += 0.15 if has_numbers else 0.0  # Numerical reasoning
        complexity += 0.15 if has_comparison else 0.0  # Comparison reasoning
        complexity += 0.1 if has_temporal else 0.0  # Temporal reasoning
        complexity += 0.1 if has_negation else 0.0  # Negation handling
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    # Add complexity scores
    print("Calculating complexity scores...")
    for example in tqdm(all_data, desc="Computing complexity"):
        example['complexity_score'] = calculate_complexity_score(example)
    
    # Stratified sampling by complexity for each label
    def stratified_sample_by_complexity(data, n_samples, label_name):
        if len(data) <= n_samples:
            return data
        
        # Sort by complexity
        sorted_data = sorted(data, key=lambda x: x['complexity_score'])
        
        # Create complexity bins
        n_bins = 5
        bin_size = len(sorted_data) // n_bins
        samples_per_bin = n_samples // n_bins
        
        selected = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(sorted_data)
            bin_data = sorted_data[start_idx:end_idx]
            
            # Sample from this complexity bin
            bin_samples = min(samples_per_bin, len(bin_data))
            if bin_samples > 0:
                selected.extend(random.sample(bin_data, bin_samples))
        
        # Fill remaining samples randomly
        remaining = n_samples - len(selected)
        if remaining > 0:
            available = [ex for ex in sorted_data if ex not in selected]
            if available:
                selected.extend(random.sample(available, min(remaining, len(available))))
        
        return selected[:n_samples]
    
    # Sample proportionally from each label
    print("Performing stratified sampling...")
    total_examples = len(all_data)
    selected_samples = []
    
    for label, count in label_counts.items():
        label_name = {0: "False", 1: "True", 2: "Unknown"}[label]
        
        # Calculate proportional sample size
        proportion = count / total_examples
        label_samples = int(target_samples * proportion)
        
        print(f"Sampling {label_samples} examples from {label_name} ({count} available)")
        
        # Get data for this label
        label_data = [ex for ex in all_data if ex['label'] == label]
        
        # Sample with complexity stratification
        label_selected = stratified_sample_by_complexity(label_data, label_samples, label_name)
        selected_samples.extend(label_selected)
    
    # Shuffle final selection
    print("Shuffling final selection...")
    random.shuffle(selected_samples)
    
    # Print sampling statistics
    final_label_counts = {}
    complexity_stats = []
    for ex in selected_samples:
        label = ex['label']
        final_label_counts[label] = final_label_counts.get(label, 0) + 1
        complexity_stats.append(ex['complexity_score'])
    
    print(f"Selected {len(selected_samples)} examples:")
    for label, count in final_label_counts.items():
        label_name = {0: "False", 1: "True", 2: "Unknown"}[label]
        print(f"  {label_name}: {count}")
    print(f"  Complexity range: {min(complexity_stats):.2f} - {max(complexity_stats):.2f}")
    print(f"  Average complexity: {sum(complexity_stats)/len(complexity_stats):.2f}")
    
    return selected_samples

class EnhancedDataCollector:
    """Enhanced data collector with efficient batch processing and tqdm progress bars"""
    
    def __init__(self, models: Dict[str, Any], config: Dict[str, Any]):
        self.models = models
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.max_workers = min(mp.cpu_count(), 8)
        
        # Initialize agents
        from agents import ReasoningAgent, CoderAgent
        self.reasoning_agent = ReasoningAgent(models["pipe_reasoning"], config)
        self.coder_agent = CoderAgent(models["pipe_coder"], enable_auto_debug=True, config=config)
    
    async def process_samples_efficiently(self, samples: List[Dict], output_dir: str) -> Tuple[List[Dict], str]:
        """Process samples efficiently with batching and concurrency"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/unified_training_data_{timestamp}.jsonl"
        
        all_results = []
        
        # Process in batches for better memory management
        total_batches = (len(samples) - 1) // self.batch_size + 1
        
        # Main progress bar for batches
        batch_pbar = tqdm(total=total_batches, desc="Processing batches", unit="batch")
        
        # Overall progress bar for samples
        sample_pbar = tqdm(total=len(samples), desc="Overall progress", unit="sample")
        
        for batch_start in range(0, len(samples), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            
            # Update batch progress bar description
            batch_pbar.set_description(f"Batch {batch_num}/{total_batches} ({len(batch_samples)} samples)")
            
            # Process batch with concurrency
            batch_results = await self._process_batch_concurrent(batch_samples, batch_start, sample_pbar)
            
            # Save batch results immediately
            for result in batch_results:
                all_results.append(result)
                self._append_jsonl(result, output_file)
            
            # Memory cleanup after each batch
            from utils import clear_gpu_memory
            clear_gpu_memory()
            
            # Update batch progress
            batch_pbar.update(1)
            
            # Update batch progress bar with success rate
            successful = len([r for r in all_results if not r.get("processing_failed", False)])
            success_rate = successful / len(all_results) * 100 if all_results else 0
            batch_pbar.set_postfix({"Success": f"{success_rate:.1f}%", "Total": len(all_results)})
        
        # Close progress bars
        batch_pbar.close()
        sample_pbar.close()
        
        return all_results, output_file
    
    async def _process_batch_concurrent(self, batch_samples: List[Dict], start_idx: int, sample_pbar: tqdm) -> List[Dict]:
        """Process batch with async concurrency"""
        
        # Create tasks for concurrent processing
        tasks = []
        for i, sample in enumerate(batch_samples):
            task = asyncio.create_task(
                self._process_single_sample_async(sample, start_idx + i, sample_pbar)
            )
            tasks.append(task)
        
        # Wait for all tasks in batch to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                tqdm.write(f"Error in batch processing sample {start_idx + i}: {result}")
                error_result = {
                    "sample_id": start_idx + i,
                    "processing_failed": True,
                    "error": str(result),
                    "statement": batch_samples[i].get("statement", ""),
                    "label": batch_samples[i].get("label", -1)
                }
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_sample_async(self, sample: Dict, sample_id: int, sample_pbar: tqdm) -> Dict:
        """Process single sample asynchronously"""
        
        start_time = time.time()
        
        # Extract sample data
        statement = sample["statement"]
        table_text = json.dumps(sample["table_text"])
        label = sample["label"]
        
        # Process ALL labels now (including Unknown=2)
        # No more skipping unknown labels
        
        try:
            # Process with reasoning agent
            reasoning_output, reasoning_prediction = self.reasoning_agent.analyze(table_text, statement)
            
            # Process with coder agent
            raw_code, code, code_output, code_error, debug_attempts, code_prediction = self.coder_agent.generate_code(table_text, statement)
            
            # Calculate derived features
            methods_agree = (reasoning_prediction == code_prediction)
            reasoning_correct = (reasoning_prediction == label)
            code_correct = (code_prediction == label)
            
            # Determine agreement type
            if reasoning_correct and code_correct:
                agreement_type = "both_correct"
            elif not reasoning_correct and not code_correct:
                agreement_type = "both_wrong"
            elif reasoning_correct and not code_correct:
                agreement_type = "reasoning_correct"
            elif code_correct and not reasoning_correct:
                agreement_type = "code_correct"
            else:
                agreement_type = "mixed"  # For cases involving Unknown predictions
            
            # Build result with ALL core features
            result = {
                # Basic info
                "sample_id": sample_id,
                "statement": statement,
                "table_text": sample["table_text"],
                "table_caption": sample.get("table_caption", ""),
                "table_id": sample.get("table_id", ""),
                "label": label,  # Keep ALL labels (0/1/2)
                
                # Core features for unified verifier
                "reasoning_output": reasoning_output,
                "reasoning_prediction": reasoning_prediction,  # 0/1/2
                "code_output": code_output or "",
                "code_prediction": code_prediction,  # 0/1/2
                "code_error": code_error or "None",
                "debug_attempts": len(debug_attempts),
                "methods_agree": methods_agree,
                
                # Additional context
                "extracted_code": code,
                "raw_code_output": raw_code,
                
                # Metadata
                "metadata": {
                    "reasoning_correct": reasoning_correct,
                    "code_correct": code_correct,
                    "agreement_type": agreement_type,
                    "processing_time": time.time() - start_time,
                    "complexity_score": sample.get('complexity_score', 0.5),
                    "useful_for_training": True,  # ALL examples are useful now
                }
            }
            
            # Update sample progress bar
            sample_pbar.update(1)
            
            # Occasionally update sample progress bar with details
            if sample_id % 10 == 0:
                sample_pbar.set_postfix({
                    "Current": f"#{sample_id}",
                    "Agree": "Yes" if methods_agree else "No",
                    "Time": f"{time.time() - start_time:.1f}s"
                })
            
            return result
            
        except Exception as e:
            # Update progress bar even on error
            sample_pbar.update(1)
            
            # Return error result instead of raising
            return {
                "sample_id": sample_id,
                "processing_failed": True,
                "error": str(e),
                "statement": statement,
                "label": label
            }
    
    def _append_jsonl(self, result: Dict, file_path: str):
        """Append result to JSONL file"""
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

# ... rest of the functions remain the same as before ...
def create_unified_input_format(result: Dict, task_type: str = "verify") -> str:
    """Create unified input format for verify and answer tasks"""
    
    # Format agent predictions clearly
    pred_map = {0: "False", 1: "True", 2: "Unknown"}
    reasoning_pred_text = pred_map[result['reasoning_prediction']]
    code_pred_text = pred_map[result['code_prediction']]
    
    # Clean outputs
    reasoning_output = result['reasoning_output'].strip()
    code_output = result['code_output'].strip() if result['code_output'] else "No output"
    code_error = result['code_error'] if result['code_error'] != "None" else "No error"
    
    # Task-specific instructions
    if task_type == "verify":
        task_instruction = f"Verify if the following statement is true or false: {result['statement']}"
    elif task_type == "answer":
        task_instruction = f"Answer the following question: {result['statement']}"
    else:
        task_instruction = f"Analyze the following: {result['statement']}"
    
    # Unified format
    input_text = f"""{task_instruction}

Table Data:
{json.dumps(result['table_text'], indent=2)}

Analysis from Reasoning Agent:
{reasoning_output}
Reasoning Agent's Conclusion: {reasoning_pred_text}

Analysis from Code Agent:
{code_output}
Code Agent's Conclusion: {code_pred_text}
Code Execution Status: {code_error}
Number of Debug Attempts: {result['debug_attempts']}

Do the agents agree? {"Yes" if result['methods_agree'] else "No"}

Your answer:"""

    return input_text

def create_training_example(result: Dict, task_type: str) -> Dict:
    """Create a single training example for specified task type"""
    
    # Generate unified input
    input_text = create_unified_input_format(result, task_type)
    
    # Generate appropriate target based on task type
    if task_type == "verify":
        # Handle ALL cases, including Unknown (2)
        target_map = {0: "False", 1: "True", 2: "Unknown"}
        target_text = target_map[result['label']]
        
    elif task_type == "answer":
        # For open-ended questions - NOT IMPLEMENTED YET
        raise NotImplementedError("Open-ended question answering not yet implemented. Need different dataset with numerical/text answers.")
        
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    return {
        # Core training data
        "input_text": input_text,
        "target_text": target_text,
        "task_type": task_type,
        
        # All core features preserved (exact match to requirements)
        "sample_id": result['sample_id'],
        "statement": result['statement'],
        "table_text": result['table_text'],
        "reasoning_output": result['reasoning_output'],
        "reasoning_prediction": result['reasoning_prediction'],
        "code_output": result['code_output'],
        "code_prediction": result['code_prediction'],
        "code_error": result['code_error'],
        "debug_attempts": result['debug_attempts'],
        "methods_agree": result['methods_agree'],
        
        # Additional preserved fields
        "table_caption": result.get('table_caption', ''),
        "table_id": result.get('table_id', ''),
        "ground_truth_label": result['label'],
        "extracted_code": result.get('extracted_code', ''),
        "raw_code_output": result.get('raw_code_output', ''),
        
        # Metadata
        "metadata": result['metadata']
    }

def create_unified_training_data(results: List[Dict], task_types: List[str] = ["verify"]) -> List[Dict]:
    """Create unified training data supporting multiple question types"""
    
    training_data = []
    
    # Progress bar for creating training examples
    for result in tqdm(results, desc="Creating training examples"):
        # Skip failed processing
        if result.get("processing_failed", False):
            continue
        
        # DON'T skip unknown labels anymore - include ALL labels (0/1/2)
        # This allows verifier to learn when to be uncertain
        
        # Create examples for each requested task type
        for task_type in task_types:
            try:
                example = create_training_example(result, task_type)
                training_data.append(example)
            except NotImplementedError as e:
                tqdm.write(f"Skipping task_type '{task_type}': {e}")
                continue
    
    return training_data

def save_training_data_with_splits(training_data: List[Dict], output_dir: str, timestamp: str):
    """Save training data with proper train/validation splits"""
    
    print("Shuffling and splitting data...")
    # Shuffle data
    random.shuffle(training_data)
    
    # Create splits (80% train, 20% validation)
    split_idx = int(len(training_data) * 0.9)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Save training split
    train_file = f"{output_dir}/train_data_{timestamp}.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in tqdm(train_data, desc="Saving training data"):
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    # Save validation split
    val_file = f"{output_dir}/val_data_{timestamp}.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in tqdm(val_data, desc="Saving validation data"):
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    # Save complete dataset
    full_file = f"{output_dir}/full_data_{timestamp}.jsonl"
    with open(full_file, 'w', encoding='utf-8') as f:
        for example in tqdm(training_data, desc="Saving full dataset"):
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Data splits saved:")
    print(f"  Training: {len(train_data)} examples -> {train_file}")
    print(f"  Validation: {len(val_data)} examples -> {val_file}")
    print(f"  Full dataset: {len(training_data)} examples -> {full_file}")
    
    return train_file, val_file, full_file

def analyze_training_data(training_data: List[Dict]) -> Dict:
    """Comprehensive analysis of training data quality"""
    
    total = len(training_data)
    if total == 0:
        return {"error": "No training data"}
    
    print("Analyzing training data...")
    
    # Task type distribution
    task_type_counts = {}
    for example in training_data:
        task_type = example["task_type"]
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    
    # Target distribution (now includes Unknown)
    target_counts = {}
    for example in training_data:
        target = example["target_text"]
        target_counts[target] = target_counts.get(target, 0) + 1
    
    # Agreement analysis
    agreement_types = {}
    for example in training_data:
        agreement_type = example["metadata"]["agreement_type"]
        agreement_types[agreement_type] = agreement_types.get(agreement_type, 0) + 1
    
    # Complexity analysis
    complexities = [ex["metadata"]["complexity_score"] for ex in training_data]
    
    # Error analysis
    error_stats = {
        "has_code_error": sum(1 for ex in training_data if ex["code_error"] != "None"),
        "methods_agree": sum(1 for ex in training_data if ex["methods_agree"]),
        "debug_attempts_avg": sum(ex["debug_attempts"] for ex in training_data) / total,
        "zero_debug_attempts": sum(1 for ex in training_data if ex["debug_attempts"] == 0),
        "unknown_predictions": sum(1 for ex in training_data if ex["target_text"] == "Unknown"),
    }
    
    # Learning scenario distribution
    learning_scenarios = {
        "easy_agree_correct": sum(1 for ex in training_data 
                                 if ex["methods_agree"] and ex["metadata"]["agreement_type"] == "both_correct"),
        "hard_disagree": sum(1 for ex in training_data 
                           if not ex["methods_agree"]),
        "code_errors": sum(1 for ex in training_data 
                          if ex["code_error"] != "None"),
        "both_wrong": sum(1 for ex in training_data 
                         if ex["metadata"]["agreement_type"] == "both_wrong"),
        "unknown_cases": sum(1 for ex in training_data 
                           if ex["target_text"] == "Unknown"),
    }
    
    analysis = {
        "dataset_summary": {
            "total_examples": total,
            "complexity_range": f"{min(complexities):.2f} - {max(complexities):.2f}",
            "average_complexity": f"{sum(complexities)/len(complexities):.2f}",
        },
        "task_type_distribution": {
            f"{k}": f"{v} ({v/total*100:.1f}%)" for k, v in task_type_counts.items()
        },
        "target_distribution": {
            f"{k}": f"{v} ({v/total*100:.1f}%)" for k, v in target_counts.items()
        },
        "agreement_analysis": {
            f"{k}": f"{v} ({v/total*100:.1f}%)" for k, v in agreement_types.items()
        },
        "error_statistics": {
            "code_errors": f"{error_stats['has_code_error']} ({error_stats['has_code_error']/total*100:.1f}%)",
            "methods_agree": f"{error_stats['methods_agree']} ({error_stats['methods_agree']/total*100:.1f}%)",
            "avg_debug_attempts": f"{error_stats['debug_attempts_avg']:.2f}",
            "clean_code_execution": f"{error_stats['zero_debug_attempts']} ({error_stats['zero_debug_attempts']/total*100:.1f}%)",
            "unknown_predictions": f"{error_stats['unknown_predictions']} ({error_stats['unknown_predictions']/total*100:.1f}%)",
        },
        "learning_scenarios": {
            f"{k}": f"{v} ({v/total*100:.1f}%)" for k, v in learning_scenarios.items()
        },
        "training_quality": {
            "has_all_targets": len(set(target_counts.keys())) >= 2,  # At least True/False
            "sufficient_disagreements": learning_scenarios["hard_disagree"] >= total * 0.15,
            "diverse_complexity": max(complexities) - min(complexities) > 0.3,
            "sufficient_data": total >= 500,
            "good_error_coverage": error_stats['has_code_error'] >= total * 0.1,
            "has_unknown_cases": error_stats['unknown_predictions'] > 0,
        }
    }
    
    return analysis