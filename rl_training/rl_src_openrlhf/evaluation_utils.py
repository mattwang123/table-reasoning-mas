"""
Evaluation utilities for REINFORCE table verifier.
"""

import json
from typing import List, Dict, Any
from collections import Counter
import re

def normalize_answer_for_comparison(answer: Any) -> str:
    """Normalize answers for fair comparison"""
    if answer is None:
        return "unknown"
    
    answer_str = str(answer).strip().lower()
    
    # Handle common variations
    if answer_str in ["true", "yes", "1"]:
        return "true"
    elif answer_str in ["false", "no", "0"]:
        return "false"
    elif answer_str in ["unknown", "uncertain", "unclear", "none", "not found"]:
        return "unknown"
    else:
        # For open-ended answers, return as-is but normalized
        return answer_str

def analyze_reinforce_results(results: List[Dict]) -> Dict:
    """
    Analyze REINFORCE evaluation results
    """
    stats = {
        "total_samples": len(results),
        "correct": 0,
        "incorrect": 0,
        "accuracy": 0.0,
        "avg_reward": 0.0,
        "answer_distribution": Counter(),
        "prediction_distribution": Counter(),
        "error_analysis": Counter()
    }
    
    total_reward = 0.0
    
    for result in results:
        # Basic metrics
        is_correct = result.get("correct", False)
        reward = result.get("reward", 0.0)
        
        if is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
        
        total_reward += reward
        
        # Answer distributions
        ground_truth = normalize_answer_for_comparison(result.get("ground_truth", ""))
        predicted = normalize_answer_for_comparison(result.get("predicted", ""))
        
        stats["answer_distribution"][ground_truth] += 1
        stats["prediction_distribution"][predicted] += 1
        
        # Error analysis
        if not is_correct:
            error_key = f"{ground_truth} -> {predicted}"
            stats["error_analysis"][error_key] += 1
    
    # Calculate rates
    if stats["total_samples"] > 0:
        stats["accuracy"] = stats["correct"] / stats["total_samples"]
        stats["avg_reward"] = total_reward / stats["total_samples"]
    
    return stats

def print_reinforce_analysis(stats: Dict, log_file: str = None):
    """Print detailed REINFORCE analysis"""
    
    analysis_text = "\n" + "="*50 + "\n"
    analysis_text += "REINFORCE EVALUATION ANALYSIS\n"
    analysis_text += "="*50 + "\n"
    analysis_text += f"Total Samples: {stats['total_samples']}\n"
    analysis_text += f"Accuracy: {stats['accuracy']:.3f}\n"
    analysis_text += f"Average Reward: {stats['avg_reward']:.3f}\n"
    analysis_text += f"Correct: {stats['correct']}\n"
    analysis_text += f"Incorrect: {stats['incorrect']}\n"
    
    # Answer distribution
    if stats.get("answer_distribution"):
        analysis_text += "\nGROUND TRUTH DISTRIBUTION:\n"
        for answer, count in stats["answer_distribution"].most_common():
            percentage = count / stats["total_samples"] * 100
            analysis_text += f"  '{answer}': {count} ({percentage:.1f}%)\n"
    
    # Prediction distribution
    if stats.get("prediction_distribution"):
        analysis_text += "\nPREDICTION DISTRIBUTION:\n"
        for answer, count in stats["prediction_distribution"].most_common():
            percentage = count / stats["total_samples"] * 100
            analysis_text += f"  '{answer}': {count} ({percentage:.1f}%)\n"
    
    # Error analysis
    if stats.get("error_analysis"):
        analysis_text += "\nERROR ANALYSIS (Top 10):\n"
        for error, count in stats["error_analysis"].most_common(10):
            percentage = count / stats["total_samples"] * 100
            analysis_text += f"  {error}: {count} ({percentage:.1f}%)\n"
    
    analysis_text += "="*50
    
    print(analysis_text)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(analysis_text + '\n')

def save_evaluation_results(results: List[Dict], output_file: str):
    """Save evaluation results to JSONL file"""
    
    print(f"Saving {len(results)} evaluation results to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Results saved to {output_file}")

def compare_with_baseline(reinforce_results: List[Dict]) -> Dict:
    """
    Compare REINFORCE model with baseline agent performance
    """
    
    comparison = {
        "reinforce_accuracy": 0.0,
        "reasoning_accuracy": 0.0,
        "code_accuracy": 0.0,
        "improvement_vs_reasoning": 0.0,
        "improvement_vs_code": 0.0,
        "improvement_vs_best": 0.0
    }
    
    if not reinforce_results:
        return comparison
    
    # Calculate accuracies
    reinforce_correct = sum(1 for r in reinforce_results if r.get("correct", False))
    reasoning_correct = 0
    code_correct = 0
    
    for result in reinforce_results:
        ground_truth = normalize_answer_for_comparison(result.get("ground_truth", ""))
        reasoning_pred = normalize_answer_for_comparison(result.get("reasoning_pred", ""))
        code_pred = normalize_answer_for_comparison(result.get("code_pred", ""))
        
        if reasoning_pred == ground_truth:
            reasoning_correct += 1
        if code_pred == ground_truth:
            code_correct += 1
    
    total_samples = len(reinforce_results)
    
    comparison["reinforce_accuracy"] = reinforce_correct / total_samples
    comparison["reasoning_accuracy"] = reasoning_correct / total_samples
    comparison["code_accuracy"] = code_correct / total_samples
    
    comparison["improvement_vs_reasoning"] = comparison["reinforce_accuracy"] - comparison["reasoning_accuracy"]
    comparison["improvement_vs_code"] = comparison["reinforce_accuracy"] - comparison["code_accuracy"]
    comparison["improvement_vs_best"] = comparison["reinforce_accuracy"] - max(comparison["reasoning_accuracy"], comparison["code_accuracy"])
    
    return comparison

def print_comparison_analysis(comparison: Dict):
    """Print comparison with baseline agents"""
    
    print("\n" + "="*50)
    print("REINFORCE vs BASELINE COMPARISON")
    print("="*50)
    print(f"REINFORCE Accuracy: {comparison['reinforce_accuracy']:.3f}")
    print(f"Reasoning Accuracy: {comparison['reasoning_accuracy']:.3f}")
    print(f"Code Accuracy: {comparison['code_accuracy']:.3f}")
    print()
    print("IMPROVEMENTS:")
    print(f"  vs Reasoning: {comparison['improvement_vs_reasoning']:+.3f}")
    print(f"  vs Code: {comparison['improvement_vs_code']:+.3f}")
    print(f"  vs Best Individual: {comparison['improvement_vs_best']:+.3f}")
    print("="*50)