import re
import logging

logger = logging.getLogger(__name__)

def extract_final_answer(text: str) -> str:
    """Extract the final answer from model response"""
    if not text:
        return ""
    
    # Look for "Final Answer:" pattern
    patterns = [
        r'Final Answer:\s*(.+?)(?:\n|$)',
        r'final answer:\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    if not answer:
        return ""
    
    normalized = str(answer).strip()
    
    # Handle boolean cases (case insensitive)
    answer_lower = normalized.lower()
    if answer_lower in ['true', 'yes', '1']:
        return "True"
    elif answer_lower in ['false', 'no', '0']:
        return "False"
    
    # For other answers, return as-is but cleaned
    return normalized

def table_reasoning_reward_func(completions, target, **kwargs):
    """
    REQUIRED: Reward function for table reasoning - this is the core of GRPO!
    
    This function evaluates how good each completion is by comparing
    the model's predicted answer with the ground truth.
    """
    rewards = []
    
    for completion, ground_truth in zip(completions, target):
        # Extract predicted answer from completion
        pred_answer = extract_final_answer(completion)
        pred_normalized = normalize_answer(pred_answer)
        
        # Handle ground truth format
        truth_answer = ground_truth
        if isinstance(truth_answer, str) and truth_answer.startswith("Final Answer:"):
            truth_answer = truth_answer.replace("Final Answer:", "").strip()
        truth_normalized = normalize_answer(truth_answer)
        
        # Binary reward: 1.0 for correct, 0.0 for incorrect
        reward = 1.0 if pred_normalized == truth_normalized else 0.0
        rewards.append(reward)
    
    return rewards
