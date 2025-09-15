#!/usr/bin/env python3
"""
Evaluate REINFORCE trained table verifier model.
Complete version with comprehensive analysis.
"""

import argparse
import os
import sys
import random
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (
    load_collected_data,
    VERIFIER_PROMPT_TEMPLATE,
    format_table,
    clean_text,
    convert_prediction_to_text
)
from reward_utils import normalize_prediction, compute_binary_reward
from evaluation_utils import (
    analyze_reinforce_results,
    print_reinforce_analysis,
    save_evaluation_results,
    compare_with_baseline,
    print_comparison_analysis
)

def load_reinforce_model(model_path: str, base_model: str = "Qwen/Qwen-7B-Chat"):
    """Load REINFORCE trained model"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print(f"Loading REINFORCE model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    
    print("✅ REINFORCE model loaded successfully")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate response using REINFORCE model"""
    import torch
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Lower for evaluation
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs.size(1):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()

def evaluate_reinforce_model(model_path: str, 
                           test_data_file: str, 
                           num_samples: int = 200,
                           base_model: str = "Qwen/Qwen-7B-Chat",
                           output_dir: str = "./outputs",
                           seed: int = 42) -> dict:
    """Evaluate REINFORCE model on test data"""
    
    print("🎯 REINFORCE Model Evaluation")
    print("=" * 50)
    print(f"Model path: {model_path}")
    print(f"Base model: {base_model}")
    print(f"Test data: {test_data_file}")
    print(f"Samples: {num_samples}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Load model
    print("🤖 Loading REINFORCE model...")
    model, tokenizer = load_reinforce_model(model_path, base_model)
    
    # Load test data
    print("📊 Loading test data...")
    samples = load_collected_data(test_data_file)
    
    if len(samples) > num_samples:
        random.seed(seed)
        samples = random.sample(samples, num_samples)
        print(f"Randomly selected {num_samples} samples from {len(samples)} total")
    
    print(f"Evaluating on {len(samples)} samples...")
    
    # Evaluate each sample
    results = []
    
    for i, sample in enumerate(samples):
        print(f"Evaluating sample {i+1}/{len(samples)}", end='\r')
        
        try:
            # Create prompt using existing template
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
            
            # Generate response with REINFORCE model
            response = generate_response(model, tokenizer, prompt)
            
            # Compute reward and extract prediction
            ground_truth = convert_prediction_to_text(sample["label"])
            reward = compute_binary_reward(response, ground_truth)
            pred_answer = normalize_prediction(response)
            
            # Create result record
            result = {
                "sample_id": i,
                "query": sample["statement"],
                "ground_truth": ground_truth,
                "predicted": pred_answer,
                "response": response,
                "reward": reward,
                "correct": reward == 1.0,
                
                # Baseline agent predictions for comparison
                "reasoning_pred": convert_prediction_to_text(sample["reasoning_prediction"]),
                "code_pred": convert_prediction_to_text(sample["code_prediction"]),
                
                # Metadata
                "metadata": {
                    "original_sample_id": sample.get("sample_id", -1),
                    "reasoning_correct": sample["reasoning_prediction"] == sample["label"],
                    "code_correct": sample["code_prediction"] == sample["label"]
                }
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError evaluating sample {i}: {e}")
            continue
    
    print(f"\n✅ Evaluation completed on {len(results)} samples")
    
    # Run comprehensive analysis
    print("\n📊 Running analysis...")
    stats = analyze_reinforce_results(results)
    
    # Print detailed analysis
    print_reinforce_analysis(stats)
    
    # Compare with baseline agents
    print("\n📈 Comparing with baseline agents...")
    comparison = compare_with_baseline(results)
    print_comparison_analysis(comparison)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(output_dir, f"reinforce_evaluation_{timestamp}.jsonl")
    save_evaluation_results(results, results_file)
    
    # Save analysis
    analysis_file = os.path.join(output_dir, f"reinforce_analysis_{timestamp}.json")
    analysis_data = {
        "stats": stats,
        "comparison": comparison,
        "config": {
            "model_path": model_path,
            "test_data": test_data_file,
            "num_samples": len(results),
            "timestamp": timestamp
        }
    }
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"\n📁 Results saved:")
    print(f"  Detailed results: {results_file}")
    print(f"  Analysis: {analysis_file}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate REINFORCE trained table verifier")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to REINFORCE trained model (LoRA adapter)")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data (collected agent results)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-7B-Chat",
                       help="Base model name")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        return
    
    if not os.path.exists(args.test_data):
        print(f"❌ Test data not found: {args.test_data}")
        return
    
    # Run evaluation
    stats = evaluate_reinforce_model(
        model_path=args.model_path,
        test_data_file=args.test_data,
        num_samples=args.num_samples,
        base_model=args.base_model,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Final summary
    print("\n🏆 FINAL REINFORCE EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Training Method: PPO (REINFORCE with baseline)")
    print(f"Reward Function: Binary final answer correctness")
    print(f"Model Accuracy: {stats['accuracy']:.3f}")
    print(f"Average Reward: {stats['avg_reward']:.3f}")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Correct Predictions: {stats['correct']}")
    print("=" * 50)

if __name__ == "__main__":
    main()