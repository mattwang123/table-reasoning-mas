import torch
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_final_answer(text: str) -> str:
    """Extract the final answer from model response"""
    if not text:
        return ""
    
    text = text.strip()
    
    # Look for "Final Answer:" pattern (case insensitive)
    patterns = [
        r'Final Answer:\s*(.+?)(?:\n|$)',
        r'final answer:\s*(.+?)(?:\n|$)',
        r'FINAL ANSWER:\s*(.+?)(?:\n|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'[.!?]*$', '', answer).strip()
            return answer
    
    return ""

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    if not answer:
        return ""
    
    normalized = str(answer).strip()
    
    # Handle boolean cases
    answer_lower = normalized.lower()
    if answer_lower in ['true', 'yes', '1']:
        return "True"
    elif answer_lower in ['false', 'no', '0']:
        return "False"
    
    # For open-ended answers, clean up
    normalized = ' '.join(normalized.split())
    return normalized

# TRL-compatible reward function
def table_reasoning_reward_func(completions, prompts=None, target=None, **kwargs):
    """
    TRL-compatible reward function for table reasoning
    
    Args:
        completions: List of generated completions
        prompts: List of prompts (not used but passed by TRL)
        target: List of target answers from dataset
        **kwargs: Additional arguments from dataset
    
    Returns:
        List[float]: Rewards for each completion
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        # Extract predicted answer
        pred_answer = extract_final_answer(completion)
        pred_normalized = normalize_answer(pred_answer)
        
        # Get ground truth - TRL passes dataset columns as kwargs
        if target and i < len(target):
            truth_answer = target[i]
            if isinstance(truth_answer, str) and truth_answer.startswith("Final Answer:"):
                truth_answer = truth_answer.replace("Final Answer:", "").strip()
            truth_normalized = normalize_answer(truth_answer)
        else:
            # Fallback: try to get from kwargs
            truth_normalized = ""
        
        # Binary reward
        reward = 1.0 if pred_normalized == truth_normalized else 0.0
        rewards.append(reward)
    
    return rewards

# For backward compatibility
class SimpleRewardFunction:
    """Wrapper class for backward compatibility"""
    
    def __init__(self):
        self.total_samples = 0
        self.correct_count = 0
        self.recent_rewards = []
    
    def __call__(self, completions, **kwargs):
        """Call the reward function"""
        rewards = table_reasoning_reward_func(completions, **kwargs)
        
        # Update statistics
        self.total_samples += len(rewards)
        self.correct_count += sum(rewards)
        self.recent_rewards.extend(rewards)
        
        # Keep only recent rewards
        if len(self.recent_rewards) > 100:
            self.recent_rewards = self.recent_rewards[-100:]
        
        return rewards
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        overall_acc = self.correct_count / self.total_samples if self.total_samples > 0 else 0.0
        recent_acc = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0
        
        return {
            'reward/overall_accuracy': overall_acc,
            'reward/recent_accuracy': recent_acc,
            'reward/total_samples': float(self.total_samples)
        }