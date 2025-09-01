import json
import random
import time
import logging
import pickle
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import torch
import os

from utils_vllm import (
    load_single_model_vllm, unload_model, print_gpu_info
)

from agents_vllm import ReasoningAgent, CoderAgent

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
        "first_few_indices": indices[:5],  # For verification
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
            first_statements = [s["statement"][:50] + "..." for s in selected_samples[:3]]
            logger.info(f"First 3 samples for verification: {first_statements}")
    
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
    """Flexible data collector with deterministic sampling from indices file"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get("batch_size", 16)
        self.agent_type = config.get("agent_type", "reasoning")
        self.indices_file = config.get("indices_file", None)  # Path to indices file
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logging(config["output_dir"], timestamp, self.agent_type)
        self.timestamp = timestamp
        
        self.logger.info(f"Initializing VLLM Data Collector - Agent Type: {self.agent_type}")
        self.logger.info(f"Indices file: {self.indices_file}")
        self.logger.info(f"Config: {config}")
    
    def collect_data(self, output_dir: str) -> Tuple[List[Dict], str]:
        """Main collection method using pre-generated indices"""
        
        if not self.indices_file:
            raise ValueError("indices_file must be specified in config")
        
        self.logger.info("Step 1: Loading samples from indices file")
        samples = load_samples_from_indices_file(self.indices_file, self.logger)
        
        if self.agent_type == "reasoning":
            return self._run_reasoning_only(samples, output_dir)
        elif self.agent_type == "coder":
            return self._run_coder_only(samples, output_dir)
        elif self.agent_type == "both":
            return self._run_both_agents(samples, output_dir)
        else:
            raise ValueError(f"Unknown agent_type: {self.agent_type}. Use 'reasoning', 'coder', or 'both'")
    
    def _run_reasoning_only(self, samples: List[Dict], output_dir: str) -> Tuple[List[Dict], str]:
        """Run only reasoning agent"""
        
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
                    statement = sample["statement"]
                    table_text = json.dumps(sample["table_text"])
                    
                    # Use reasoning agent
                    response, prediction = reasoning_agent.analyze(table_text, statement)
                    
                    result = {
                        "sample_id": sample_id,
                        "original_index": sample.get("original_index", -1),  # Track original dataset index
                        "selection_order": sample.get("selection_order", -1),  # Track selection order
                        "agent_type": "reasoning",
                        "statement": statement,
                        "table_text": sample["table_text"],
                        "label": sample["label"],
                        "reasoning_output": response,
                        "reasoning_prediction": prediction,
                        "processing_time": time.time() - start_time,
                        "metadata": {
                            "reasoning_correct": (prediction == sample["label"]),
                            "timestamp": datetime.now().isoformat()
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
            
            # Progress update
            successful = len([r for r in results if not r.get("processing_failed", False)])
            success_rate = successful / len(results) * 100 if results else 0
            self.logger.info(f"Batch {batch_num} completed. Success rate: {success_rate:.1f}%")
        
        # Unload model
        self.logger.info("Unloading reasoning model")
        unload_model(llm)
        
        # Analysis
        self._analyze_reasoning_results(results)
        
        self.logger.info(f"Reasoning processing completed: {len(results)} results saved to {output_file}")
        return results, output_file
    
    def _run_coder_only(self, samples: List[Dict], output_dir: str) -> Tuple[List[Dict], str]:
        """Run only coder agent"""
        
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
                    statement = sample["statement"]
                    table_text = json.dumps(sample["table_text"])
                    
                    # Use coder agent
                    raw_code, code, code_output, code_error, debug_attempts, code_prediction = coder_agent.generate_code(table_text, statement)
                    
                    result = {
                        "sample_id": sample_id,
                        "original_index": sample.get("original_index", -1),
                        "selection_order": sample.get("selection_order", -1),
                        "agent_type": "coder",
                        "statement": statement,
                        "table_text": sample["table_text"],
                        "label": sample["label"],
                        "raw_code_output": raw_code,
                        "extracted_code": code,
                        "code_output": code_output or "",
                        "code_error": code_error or "None",
                        "debug_attempts": len(debug_attempts) - 1,
                        "debug_attempts_detail": debug_attempts,
                        "code_prediction": code_prediction,
                        "processing_time": time.time() - start_time,
                        "metadata": {
                            "code_correct": (code_prediction == sample["label"]),
                            "timestamp": datetime.now().isoformat()
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
            
            # Progress update
            successful = len([r for r in results if not r.get("processing_failed", False)])
            success_rate = successful / len(results) * 100 if results else 0
            self.logger.info(f"Batch {batch_num} completed. Success rate: {success_rate:.1f}%")
        
        # Unload model
        self.logger.info("Unloading coder model")
        unload_model(llm)
        
        # Analysis
        self._analyze_coder_results(results)
        
        self.logger.info(f"Coder processing completed: {len(results)} results saved to {output_file}")
        return results, output_file
    
    def _run_both_agents(self, samples: List[Dict], output_dir: str) -> Tuple[List[Dict], str]:
        """Run both agents sequentially (for single machine)"""
        
        self.logger.info("Step 2: Processing with both agents sequentially")
        
        # Run reasoning first
        reasoning_config = self.config.copy()
        reasoning_config["agent_type"] = "reasoning"
        reasoning_collector = VllmDataCollector(reasoning_config)
        reasoning_results, reasoning_file = reasoning_collector._run_reasoning_only(samples, output_dir)
        
        # Run coder second
        coder_config = self.config.copy()
        coder_config["agent_type"] = "coder"
        coder_collector = VllmDataCollector(coder_config)
        coder_results, coder_file = coder_collector._run_coder_only(samples, output_dir)
        
        # Combine results
        self.logger.info("Step 3: Combining results and calculating agreement")
        final_results = self._combine_agent_results(reasoning_results, coder_results, samples, output_dir)
        
        return final_results, f"{output_dir}/combined_results_{self.timestamp}.jsonl"
    
    def _combine_agent_results(self, reasoning_results: List[Dict], coder_results: List[Dict], 
                              samples: List[Dict], output_dir: str) -> List[Dict]:
        """Combine reasoning and coder results"""
        
        output_file = f"{output_dir}/combined_results_{self.timestamp}.jsonl"
        
        # Create lookup dictionaries
        reasoning_lookup = {r["sample_id"]: r for r in reasoning_results if not r.get("processing_failed", False)}
        coder_lookup = {r["sample_id"]: r for r in coder_results if not r.get("processing_failed", False)}
        
        final_results = []
        
        for i, sample in enumerate(samples):
            sample_id = i
            
            # Get results for this sample
            reasoning_result = reasoning_lookup.get(sample_id)
            coder_result = coder_lookup.get(sample_id)
            
            if not reasoning_result or not coder_result:
                # Skip if either failed
                continue
            
            # Calculate agreement metrics
            reasoning_pred = reasoning_result["reasoning_prediction"]
            code_pred = coder_result["code_prediction"]
            label = sample["label"]
            
            methods_agree = (reasoning_pred == code_pred)
            reasoning_correct = (reasoning_pred == label)
            code_correct = (code_pred == label)
            
            if reasoning_correct and code_correct:
                agreement_type = "both_correct"
            elif not reasoning_correct and not code_correct:
                agreement_type = "both_wrong"
            elif reasoning_correct and not code_correct:
                agreement_type = "reasoning_correct"
            else:
                agreement_type = "code_correct"
            
            # Build combined result
            combined_result = {
                "sample_id": sample_id,
                "statement": sample["statement"],
                "table_text": sample["table_text"],
                "label": label,
                
                # Reasoning results
                "reasoning_output": reasoning_result["reasoning_output"],
                "reasoning_prediction": reasoning_pred,
                
                # Coder results  
                "code_output": coder_result["code_output"],
                "code_prediction": code_pred,
                "code_error": coder_result["code_error"],
                "debug_attempts": coder_result["debug_attempts"],
                "extracted_code": coder_result["extracted_code"],
                "raw_code_output": coder_result["raw_code_output"],
                
                # Agreement metrics
                "methods_agree": methods_agree,
                
                # Metadata
                "metadata": {
                    "reasoning_correct": reasoning_correct,
                    "code_correct": code_correct,
                    "agreement_type": agreement_type,
                    "reasoning_time": reasoning_result["processing_time"],
                    "coder_time": coder_result["processing_time"],
                    "total_time": reasoning_result["processing_time"] + coder_result["processing_time"]
                }
            }
            
            final_results.append(combined_result)
            
            # Save incrementally
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(combined_result, f, ensure_ascii=False)
                f.write('\n')
        
        # Analysis
        self._analyze_combined_results(final_results)
        
        return final_results
    
    def _analyze_reasoning_results(self, results: List[Dict]):
        """Analyze reasoning-only results"""
        successful_results = [r for r in results if not r.get("processing_failed", False)]
        
        if not successful_results:
            self.logger.info("No successful reasoning results to analyze")
            return
        
        n = len(successful_results)
        correct = sum(1 for r in successful_results if r["metadata"]["reasoning_correct"])
        accuracy = correct / n
        
        self.logger.info("REASONING RESULTS SUMMARY")
        self.logger.info(f"Total Samples: {len(results)}")
        self.logger.info(f"Successful Samples: {n}")
        self.logger.info(f"Reasoning Accuracy: {accuracy:.3f} ({correct}/{n})")
        
        # Save analysis
        analysis = {
            "agent_type": "reasoning",
            "total_samples": len(results),
            "successful_samples": n,
            "reasoning_accuracy": accuracy,
            "reasoning_correct": correct
        }
        
        analysis_file = f"{self.config['output_dir']}/reasoning_analysis_{self.timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _analyze_coder_results(self, results: List[Dict]):
        """Analyze coder-only results"""
        successful_results = [r for r in results if not r.get("processing_failed", False)]
        
        if not successful_results:
            self.logger.info("No successful coder results to analyze")
            return
        
        n = len(successful_results)
        correct = sum(1 for r in successful_results if r["metadata"]["code_correct"])
        accuracy = correct / n
        
        debug_counts = [r["debug_attempts"] for r in successful_results]
        avg_debug = sum(debug_counts) / len(debug_counts) if debug_counts else 0
        
        self.logger.info("CODER RESULTS SUMMARY")
        self.logger.info(f"Total Samples: {len(results)}")
        self.logger.info(f"Successful Samples: {n}")
        self.logger.info(f"Code Accuracy: {accuracy:.3f} ({correct}/{n})")
        self.logger.info(f"Average Debug Rounds: {avg_debug:.2f}")
        
        # Save analysis
        analysis = {
            "agent_type": "coder",
            "total_samples": len(results),
            "successful_samples": n,
            "code_accuracy": accuracy,
            "code_correct": correct,
            "avg_debug_rounds": avg_debug,
            "max_debug_rounds": max(debug_counts) if debug_counts else 0
        }
        
        analysis_file = f"{self.config['output_dir']}/coder_analysis_{self.timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _analyze_combined_results(self, results: List[Dict]):
        """Analyze combined results using exact same method as original"""
        
        if not results:
            self.logger.info("No combined results to analyze")
            return
        
        # Use the exact same analysis function from original
        from utils_vllm import analyze_agreement, print_analysis, log_to_file
        
        stats = analyze_agreement(results)
        
        # Log using exact same format as original
        log_file = f"{self.config['output_dir']}/combined_analysis_{self.timestamp}.log"
        print_analysis(stats, log_file)
        
        # Also log to main logger with enhanced info
        self.logger.info("COMBINED RESULTS SUMMARY")
        self.logger.info(f"Total Samples: {stats['total_samples']}")
        self.logger.info(f"Final Accuracy: {stats['final_accuracy']:.3f}")
        self.logger.info(f"Reasoning Accuracy: {stats['reasoning_accuracy']:.3f}")
        self.logger.info(f"Code Accuracy: {stats['code_accuracy']:.3f}")
        self.logger.info(f"Verifier Accuracy: {stats['verifier_accuracy']:.3f}")
        self.logger.info(f"Agreement Rate: {stats['agreement_rate']:.3f}")
        self.logger.info(f"Both Correct: {stats['both_correct']}")
        self.logger.info(f"Both Wrong: {stats['both_incorrect']}")
        self.logger.info(f"Reasoning Correct, Code Wrong: {stats['disagree_reasoning_correct']}")
        self.logger.info(f"Code Correct, Reasoning Wrong: {stats['disagree_code_correct']}")
        
        # Save analysis as JSON
        analysis_file = f"{self.config['output_dir']}/combined_analysis_{self.timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(stats, f, indent=2)

def combine_separate_results(reasoning_file: str, coder_file: str, output_dir: str, 
                           dataset_path: str, target_samples: int = None) -> str:
    """Utility function to combine results from separate runs with reanalysis"""
    
    from utils_vllm import normalize_prediction, extract_code_prediction, analyze_agreement, print_analysis
    
    print("Loading reasoning results...")
    reasoning_results = []
    with open(reasoning_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                if not result.get("processing_failed", False):
                    # RE-ANALYZE reasoning prediction with new utils
                    original_output = result["reasoning_output"]
                    new_prediction = normalize_prediction(original_output)
                    result["reasoning_prediction"] = new_prediction  # Update with reanalyzed prediction
                    reasoning_results.append(result)
    
    print("Loading coder results...")
    coder_results = []
    with open(coder_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                if not result.get("processing_failed", False):
                    # RE-ANALYZE code prediction with new utils
                    code_output = result["code_output"]
                    new_prediction = extract_code_prediction(code_output)
                    result["code_prediction"] = new_prediction  # Update with reanalyzed prediction
                    coder_results.append(result)
    
    print("Loading original samples...")
    with open(dataset_path, 'r') as f:
        all_samples = [json.loads(line) for line in f]
    
    if target_samples:
        all_samples = all_samples[:target_samples]
    
    # Create lookup dictionaries
    reasoning_lookup = {r["sample_id"]: r for r in reasoning_results}
    coder_lookup = {r["sample_id"]: r for r in coder_results}
    
    print("Combining results with reanalyzed predictions...")
    final_results = []
    
    for sample_id in reasoning_lookup.keys():
        if sample_id in coder_lookup:
            reasoning_result = reasoning_lookup[sample_id]
            coder_result = coder_lookup[sample_id]
            
            # Calculate agreement metrics with reanalyzed predictions
            reasoning_pred = reasoning_result["reasoning_prediction"]
            code_pred = coder_result["code_prediction"]
            label = reasoning_result["label"]
            
            methods_agree = (reasoning_pred == code_pred)
            reasoning_correct = (reasoning_pred == label)
            code_correct = (code_pred == label)
            
            if reasoning_correct and code_correct:
                agreement_type = "both_correct"
            elif not reasoning_correct and not code_correct:
                agreement_type = "both_wrong"
            elif reasoning_correct and not code_correct:
                agreement_type = "reasoning_correct"
            else:
                agreement_type = "code_correct"
            
            # Build combined result
            combined_result = {
                "sample_id": sample_id,
                "statement": reasoning_result["statement"],
                "table_text": reasoning_result["table_text"],
                "label": label,
                
                # Reanalyzed results
                "reasoning_output": reasoning_result["reasoning_output"],
                "reasoning_prediction": reasoning_pred,
                "code_output": coder_result["code_output"],
                "code_prediction": code_pred,
                "code_error": coder_result["code_error"],
                "debug_attempts": coder_result.get("debug_attempts", 0),
                "extracted_code": coder_result.get("extracted_code", ""),
                "raw_code_output": coder_result.get("raw_code_output", ""),
                
                # Agreement metrics
                "methods_agree": methods_agree,
                
                # Metadata
                "metadata": {
                    "reasoning_correct": reasoning_correct,
                    "code_correct": code_correct,
                    "agreement_type": agreement_type,
                    "reasoning_time": reasoning_result.get("processing_time", 0),
                    "coder_time": coder_result.get("processing_time", 0),
                    "total_time": reasoning_result.get("processing_time", 0) + coder_result.get("processing_time", 0)
                }
            }
            
            final_results.append(combined_result)
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/reanalyzed_combined_{timestamp}.jsonl"
    
    print(f"Saving {len(final_results)} combined results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in final_results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    # Run analysis with reanalyzed predictions
    print("Running analysis with reanalyzed predictions...")
    stats = analyze_agreement(final_results)
    
    # Print and save analysis
    log_file = f"{output_dir}/reanalysis_{timestamp}.log"
    print_analysis(stats, log_file)
    
    # Save as JSON
    json_file = f"{output_dir}/reanalysis_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Analysis saved to:")
    print(f"  Log: {log_file}")
    print(f"  JSON: {json_file}")
    
    return output_file