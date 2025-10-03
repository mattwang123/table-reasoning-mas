import json
import random
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import torch
import os

from ..agents.utils_vllm import (
    load_single_model_vllm, unload_model, print_gpu_info
)
from ..agents.agents_vllm import ReasoningAgent, CoderAgent

def setup_logging(output_dir: str, timestamp: str, agent_type: str) -> logging.Logger:
    """Setup logging"""
    log_file = f"{output_dir}/{agent_type}_collection_{timestamp}.log"
    
    logger = logging.getLogger(f'{agent_type.title()}Collection')
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def generate_sample_indices(dataset_path: str, target_samples: int = 1000, 
                           seed: int = 42, save_file: str = None) -> List[int]:
    """Generate and save sample indices for deterministic sampling"""
    
    print(f"Generating sample indices...")
    print(f"Dataset: {dataset_path}")
    print(f"Target samples: {target_samples}")
    print(f"Seed: {seed}")
    
    with open(dataset_path, 'r') as f:
        total_samples = sum(1 for _ in f)
    
    print(f"Total samples in dataset: {total_samples}")
    
    # Generate deterministic indices
    random.seed(seed)
    if total_samples <= target_samples:
        indices = list(range(total_samples))
        print(f"Using all {total_samples} samples (target was {target_samples})")
    else:
        indices = random.sample(range(total_samples), target_samples)
        print(f"Randomly selected {target_samples} samples from {total_samples}")
    
    # Shuffle for good measure
    random.shuffle(indices)
    
    # Create indices data
    indices_data = {
        "dataset_path": dataset_path,
        "target_samples": target_samples,
        "actual_samples": len(indices),
        "seed": seed,
        "total_samples": total_samples,
        "selected_indices": indices,
        "generation_timestamp": datetime.now().isoformat(),
        "first_few_indices": indices[:5],
    }
    
    if save_file:
        with open(save_file, 'w') as f:
            json.dump(indices_data, f, indent=2)
        print(f"Sample indices saved to: {save_file}")
        print(f"First 5 indices: {indices[:5]}")
    
    return indices

def load_samples_by_indices(dataset_path: str, indices: List[int], logger: logging.Logger = None) -> List[Dict]:
    """Load specific samples by indices"""
    
    if logger:
        logger.info(f"Loading {len(indices)} samples by indices from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        all_data = [json.loads(line) for line in f]
    
    selected_samples = []
    for i, idx in enumerate(indices):
        if idx < len(all_data):
            sample = all_data[idx]
            # Add original index for tracking
            sample['original_index'] = idx
            sample['selection_order'] = i
            selected_samples.append(sample)
        else:
            if logger:
                logger.warning(f"Index {idx} is out of range (dataset has {len(all_data)} samples)")
    
    if logger:
        logger.info(f"Successfully loaded {len(selected_samples)} samples")
        # Log first few for verification
        if selected_samples:
            first_queries = [s.get("statement", s.get("question", ""))[:50] + "..." for s in selected_samples[:3]]
            logger.info(f"First 3 samples for verification: {first_queries}")
    
    return selected_samples

def load_samples_from_indices_file(indices_file: str, logger: logging.Logger = None) -> List[Dict]:
    """Load samples using saved indices file"""
    
    if not os.path.exists(indices_file):
        raise FileNotFoundError(f"Indices file not found: {indices_file}")
    
    if logger:
        logger.info(f"Loading samples from indices file: {indices_file}")
    
    with open(indices_file, 'r') as f:
        indices_data = json.load(f)
    
    dataset_path = indices_data["dataset_path"]
    indices = indices_data["selected_indices"]
    
    if logger:
        logger.info(f"Indices file info:")
        logger.info(f"  Dataset: {dataset_path}")
        logger.info(f"  Target samples: {indices_data['target_samples']}")
        logger.info(f"  Actual samples: {indices_data['actual_samples']}")
        logger.info(f"  Seed: {indices_data['seed']}")
        logger.info(f"  Generated: {indices_data['generation_timestamp']}")
        logger.info(f"  First few indices: {indices_data['first_few_indices']}")
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    samples = load_samples_by_indices(dataset_path, indices, logger)
    
    return samples

class VllmDataCollector:
    """Data collector - ONLY collects, NO combination or analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get("batch_size", 16)
        self.agent_type = config.get("agent_type", "reasoning")
        self.indices_file = config.get("indices_file", None)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logging(config["output_dir"], timestamp, self.agent_type)
        self.timestamp = timestamp
        
        self.logger.info(f"Initializing VLLM Data Collector - Agent Type: {self.agent_type}")
        self.logger.info(f"Indices file: {self.indices_file}")
        self.logger.info(f"Config: {config}")
    
    def collect_data(self, output_dir: str) -> Tuple[List[Dict], str]:
        """Main collection method - ONLY for single agent"""
        
        if not self.indices_file:
            raise ValueError("indices_file must be specified in config")
        
        self.logger.info("Step 1: Loading samples from indices file")
        samples = load_samples_from_indices_file(self.indices_file, self.logger)
        
        if self.agent_type == "reasoning":
            return self._run_reasoning_only(samples, output_dir)
        elif self.agent_type == "coder":
            return self._run_coder_only(samples, output_dir)
        else:
            raise ValueError(f"Unknown agent_type: {self.agent_type}. Use 'reasoning' or 'coder' only")
    
    def _get_query_from_sample(self, sample: Dict) -> str:
        """Extract query from sample - works for both TabFact and WikiTQ"""
        return sample.get("statement", sample.get("question", ""))
    
    def _run_reasoning_only(self, samples: List[Dict], output_dir: str) -> Tuple[List[Dict], str]:
        """Run only reasoning agent - PURE COLLECTION"""
        
        self.logger.info("Step 2: Processing with reasoning agent only")
        
        # Output file
        output_file = f"{output_dir}/reasoning_results_{self.timestamp}.jsonl"
        
        # Load model and create agent
        model_path = self.config["models"]["reasoning"]
        self.logger.info(f"Loading reasoning model: {model_path}")
        llm = load_single_model_vllm(model_path, "reasoning")
        reasoning_agent = ReasoningAgent(llm, self.config)
        
        results = []
        
        # Process in batches
        total_batches = (len(samples) - 1) // self.batch_size + 1
        self.logger.info(f"Processing {len(samples)} samples in {total_batches} batches")
        
        for batch_start in range(0, len(samples), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            for i, sample in enumerate(tqdm(batch_samples, desc=f"Reasoning Batch {batch_num}")):
                sample_id = batch_start + i
                
                try:
                    start_time = time.time()
                    query = self._get_query_from_sample(sample)
                    table_text = json.dumps(sample["table_text"])
                    
                    # Use reasoning agent
                    response, prediction = reasoning_agent.analyze(table_text, query)
                    
                    result = {
                        "sample_id": sample_id,
                        "original_index": sample.get("original_index", -1),
                        "selection_order": sample.get("selection_order", -1),
                        "agent_type": "reasoning",
                        
                        # Original sample data
                        "query": query,
                        "table_text": sample["table_text"],
                        "label": sample["label"],
                        "table_id": sample.get("table_id", ""),
                        "table_caption": sample.get("table_caption", ""),
                        
                        # Reasoning outputs
                        "reasoning_output": response,
                        "reasoning_prediction": prediction,
                        "processing_time": time.time() - start_time,
                        
                        # Minimal metadata
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "dataset_type": sample.get("dataset_type", "unknown")
                        }
                    }
                    
                    results.append(result)
                    
                    # Save incrementally
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write('\n')
                    
                except Exception as e:
                    self.logger.error(f"Error processing reasoning sample {sample_id}: {e}")
                    error_result = {
                        "sample_id": sample_id,
                        "original_index": sample.get("original_index", -1),
                        "processing_failed": True,
                        "error": str(e),
                        "agent_type": "reasoning"
                    }
                    results.append(error_result)
            
            # Simple progress update
            successful = len([r for r in results if not r.get("processing_failed", False)])
            success_rate = successful / len(results) * 100 if results else 0
            self.logger.info(f"Batch {batch_num} completed. Success rate: {success_rate:.1f}%")
        
        # Unload model
        self.logger.info("Unloading reasoning model")
        unload_model(llm)
        
        # Simple summary
        successful = len([r for r in results if not r.get("processing_failed", False)])
        self.logger.info(f"Reasoning processing completed: {successful}/{len(results)} successful")
        
        return results, output_file
    
    def _run_coder_only(self, samples: List[Dict], output_dir: str) -> Tuple[List[Dict], str]:
        """Run only coder agent - PURE COLLECTION"""
        
        self.logger.info("Step 2: Processing with coder agent only")
        
        # Output file
        output_file = f"{output_dir}/coder_results_{self.timestamp}.jsonl"
        
        # Load model and create agent
        model_path = self.config["models"]["coder"]
        self.logger.info(f"Loading coder model: {model_path}")
        llm = load_single_model_vllm(model_path, "coder")
        coder_agent = CoderAgent(
            llm, 
            enable_auto_debug=self.config.get("enable_debug", True), 
            config=self.config
        )
        
        self.logger.info(f"CoderAgent initialized: enable_auto_debug={coder_agent.enable_auto_debug}, debug_rounds={coder_agent.debug_rounds}")
        
        results = []
        
        # Process in batches
        total_batches = (len(samples) - 1) // self.batch_size + 1
        self.logger.info(f"Processing {len(samples)} samples in {total_batches} batches")
        
        for batch_start in range(0, len(samples), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            for i, sample in enumerate(tqdm(batch_samples, desc=f"Coder Batch {batch_num}")):
                sample_id = batch_start + i
                
                try:
                    start_time = time.time()
                    query = self._get_query_from_sample(sample)
                    table_text = json.dumps(sample["table_text"])
                    
                    # Use coder agent
                    raw_code, code, code_output, code_error, debug_attempts, code_prediction = coder_agent.generate_code(table_text, query)
                    
                    result = {
                        "sample_id": sample_id,
                        "original_index": sample.get("original_index", -1),
                        "selection_order": sample.get("selection_order", -1),
                        "agent_type": "coder",
                        
                        # Original sample data
                        "query": query,
                        "table_text": sample["table_text"],
                        "label": sample["label"],
                        "table_id": sample.get("table_id", ""),
                        "table_caption": sample.get("table_caption", ""),
                        
                        # Coder outputs
                        "raw_code_output": raw_code,
                        "extracted_code": code,
                        "code_output": code_output or "",
                        "code_error": code_error or "None",
                        "debug_attempts": len(debug_attempts) - 1,
                        "debug_attempts_detail": debug_attempts,
                        "code_prediction": code_prediction,
                        "processing_time": time.time() - start_time,
                        
                        # Minimal metadata
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "dataset_type": sample.get("dataset_type", "unknown")
                        }
                    }
                    
                    results.append(result)
                    
                    # Save incrementally
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write('\n')
                    
                except Exception as e:
                    self.logger.error(f"Error processing coder sample {sample_id}: {e}")
                    error_result = {
                        "sample_id": sample_id,
                        "original_index": sample.get("original_index", -1),
                        "processing_failed": True,
                        "error": str(e),
                        "agent_type": "coder"
                    }
                    results.append(error_result)
            
            # Simple progress update
            successful = len([r for r in results if not r.get("processing_failed", False)])
            success_rate = successful / len(results) * 100 if results else 0
            self.logger.info(f"Batch {batch_num} completed. Success rate: {success_rate:.1f}%")
        
        # Unload model
        self.logger.info("Unloading coder model")
        unload_model(llm)
        
        # Simple summary
        successful = len([r for r in results if not r.get("processing_failed", False)])
        self.logger.info(f"Coder processing completed: {successful}/{len(results)} successful")
        
        return results, output_file

# SEPARATE FUNCTION for combining with full analysis
def combine_and_analyze_results(reasoning_file: str, coder_file: str, output_dir: str) -> str:
    """Combine reasoning and coder results with FULL ANALYSIS - creates DATABASE"""
    
    from ..agents.utils_vllm import normalize_prediction, extract_code_prediction
    
    print("=" * 60)
    print("COMBINING AND ANALYZING EXPERT OUTPUTS")
    print("=" * 60)
    
    print("Loading reasoning results...")
    reasoning_results = []
    with open(reasoning_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                if not result.get("processing_failed", False):
                    # RE-ANALYZE reasoning prediction with current utils
                    original_output = result["reasoning_output"]
                    new_prediction = normalize_prediction(original_output)
                    result["reasoning_prediction"] = new_prediction
                    reasoning_results.append(result)
    
    print("Loading coder results...")
    coder_results = []
    with open(coder_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                if not result.get("processing_failed", False):
                    # RE-ANALYZE code prediction with current utils
                    code_output = result["code_output"]
                    new_prediction = extract_code_prediction(code_output)
                    result["code_prediction"] = new_prediction
                    coder_results.append(result)
    
    print(f"Loaded {len(reasoning_results)} reasoning results")
    print(f"Loaded {len(coder_results)} coder results")
    
    # Create lookup dictionaries
    reasoning_lookup = {r["sample_id"]: r for r in reasoning_results}
    coder_lookup = {r["sample_id"]: r for r in coder_results}
    
    print("Combining with full analysis...")
    combined_results = []
    
    for sample_id in reasoning_lookup.keys():
        if sample_id in coder_lookup:
            reasoning_result = reasoning_lookup[sample_id]
            coder_result = coder_lookup[sample_id]
            
            # FULL ANALYSIS - all the information we need
            reasoning_pred = reasoning_result["reasoning_prediction"]
            code_pred = coder_result["code_prediction"]
            label = reasoning_result["label"]
            
            # Agreement analysis
            methods_agree = (reasoning_pred == code_pred)
            reasoning_correct = (reasoning_pred == label)
            code_correct = (code_pred == label)
            
            # Sample type categorization
            if reasoning_correct and code_correct:
                agreement_type = "both_correct"
                sample_type = "both_correct"
                difficulty_score = 1.0  # Easy
            elif not reasoning_correct and not code_correct:
                agreement_type = "both_wrong"
                sample_type = "both_wrong"
                difficulty_score = 3.0  # Hard
            elif reasoning_correct and not code_correct:
                agreement_type = "reasoning_correct"
                sample_type = "reasoning_correct"
                difficulty_score = 2.0  # Medium
            else:
                agreement_type = "code_correct"
                sample_type = "code_correct"
                difficulty_score = 2.0  # Medium
            
            # Additional analysis
            has_code_error = (coder_result["code_error"] != "None" and coder_result["code_error"] != "")
            code_complexity = len(coder_result["extracted_code"].split('\n')) if coder_result["extracted_code"] else 0
            reasoning_length = len(reasoning_result["reasoning_output"].split()) if reasoning_result["reasoning_output"] else 0
            
            # Build COMPLETE combined result - DATABASE ENTRY
            combined_result = {
                "sample_id": sample_id,
                "original_index": reasoning_result.get("original_index", -1),
                "selection_order": reasoning_result.get("selection_order", -1),
                
                # Original data
                "query": reasoning_result["query"],
                "table_text": reasoning_result["table_text"],
                "label": label,
                "table_id": reasoning_result.get("table_id", ""),
                "table_caption": reasoning_result.get("table_caption", ""),
                
                # COMPLETE reasoning results
                "reasoning_output": reasoning_result["reasoning_output"],
                "reasoning_prediction": reasoning_pred,
                "reasoning_processing_time": reasoning_result["processing_time"],
                
                # COMPLETE coder results
                "raw_code_output": coder_result["raw_code_output"],
                "extracted_code": coder_result["extracted_code"],
                "code_output": coder_result["code_output"],
                "code_error": coder_result["code_error"],
                "debug_attempts": coder_result["debug_attempts"],
                "debug_attempts_detail": coder_result["debug_attempts_detail"],
                "code_prediction": code_pred,
                "code_processing_time": coder_result["processing_time"],
                
                # COMPLETE analysis - all the metadata we need
                "methods_agree": methods_agree,
                "reasoning_correct": reasoning_correct,
                "code_correct": code_correct,
                "agreement_type": agreement_type,
                "sample_type": sample_type,
                "difficulty_score": difficulty_score,
                
                # Additional analysis for curriculum learning
                "has_code_error": has_code_error,
                "code_complexity": code_complexity,
                "reasoning_length": reasoning_length,
                
                # Metadata
                "total_processing_time": reasoning_result["processing_time"] + coder_result["processing_time"],
                "dataset_type": reasoning_result.get("metadata", {}).get("dataset_type", "unknown"),
                "combination_timestamp": datetime.now().isoformat()
            }
            
            combined_results.append(combined_result)
    
    # Save COMPLETE DATABASE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/combined_analyzed_database_{timestamp}.jsonl"
    
    print(f"Saving {len(combined_results)} combined results with full analysis...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in combined_results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    # COMPREHENSIVE ANALYSIS
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    total = len(combined_results)
    reasoning_acc = sum(1 for r in combined_results if r["reasoning_correct"]) / total
    code_acc = sum(1 for r in combined_results if r["code_correct"]) / total
    agreement_rate = sum(1 for r in combined_results if r["methods_agree"]) / total
    
    # Sample type distribution
    sample_types = {}
    difficulty_dist = {}
    for result in combined_results:
        sample_type = result["sample_type"]
        sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
        
        difficulty = result["difficulty_score"]
        difficulty_dist[difficulty] = difficulty_dist.get(difficulty, 0) + 1
    
    print(f"Total samples: {total}")
    print(f"Reasoning accuracy: {reasoning_acc:.3f}")
    print(f"Code accuracy: {code_acc:.3f}")
    print(f"Agreement rate: {agreement_rate:.3f}")
    print(f"Sample type distribution: {sample_types}")
    print(f"Difficulty distribution: {difficulty_dist}")
    
    # Save analysis summary
    analysis_summary = {
        "total_samples": total,
        "reasoning_accuracy": reasoning_acc,
        "code_accuracy": code_acc,
        "agreement_rate": agreement_rate,
        "sample_type_distribution": sample_types,
        "difficulty_distribution": difficulty_dist,
        "analysis_timestamp": datetime.now().isoformat(),
        "reasoning_file": reasoning_file,
        "coder_file": coder_file,
        "combined_file": output_file
    }
    
    analysis_file = f"{output_dir}/analysis_summary_{timestamp}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"\nDatabase saved to: {output_file}")
    print(f"Analysis saved to: {analysis_file}")
    print("=" * 60)
    
    return output_file