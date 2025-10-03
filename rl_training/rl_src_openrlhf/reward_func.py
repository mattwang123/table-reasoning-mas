"""
Rule-based binary reward function for table reasoning REINFORCE training.
Compatible with OpenRLHF format.
"""

import torch
import re
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_prediction(pred: str) -> str:
    """Extract final answer from model response"""
    text = pred.lower().strip()

    # Layer 1: Extract "Final Answer: X" pattern
    final_answer_match = re.search(r'final\s+answer\s*:?\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).strip()
        # Fix the regex pattern - escape the dash properly
        answer = re.sub(r'[^\w\s.\-]', '', answer).strip()
        
        if answer.lower() in ["true", "yes"]:
            return "True"
        elif answer.lower() in ["false", "no"]:
            return "False"
        elif answer.lower() in ["unknown", "uncertain"]:
            return "Unknown"
        else:
            return answer.title()

    # Layer 2: Look for patterns in the last part
    last_part = " ".join(text.split()[-20:])
    
    if "false" in last_part and "true" not in last_part:
        return "False"
    if "true" in last_part and "false" not in last_part:
        return "True"
    if "unknown" in last_part:
        return "Unknown"
    
    # Layer 3: Try to extract any meaningful answer
    answer_patterns = [
        r'(?:the )?answer is (.+?)(?:\.|$)',
        r'(?:it is|this is) (.+?)(?:\.|$)',
        r'(?:result|conclusion):\s*(.+?)(?:\.|$)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Fix the regex pattern here too
            answer = re.sub(r'[^\w\s.\-]', '', answer).strip()
            if answer:
                return answer.title()
    
    return "Unknown"

def extract_ground_truth_from_prompt(prompt: str) -> str:
    """
    Extract ground truth from the REINFORCE prompt.
    Our prompts should contain the ground truth somewhere.
    """
    try:
        # Look for patterns in our table reasoning prompts
        # The ground truth should be embedded in the prompt structure
        
        # Pattern 1: Look for explicit ground truth markers
        gt_patterns = [
            r'Ground Truth:\s*(.+?)(?:\n|$)',
            r'Correct Answer:\s*(.+?)(?:\n|$)',
            r'Expected:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in gt_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Pattern 2: Extract from our specific table reasoning format
        # This might need adjustment based on your actual prompt format
        if "Final Answer:" in prompt:
            # Find the last occurrence which might be the expected answer
            matches = list(re.finditer(r'Final Answer:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE))
            if matches:
                return matches[-1].group(1).strip()
        
        # Fallback: return empty string
        return ""
        
    except Exception as e:
        logger.warning(f"Error extracting ground truth from prompt: {e}")
        return ""

def extract_response_from_query(query: str) -> str:
    """Extract the model's response from the full conversation"""
    try:
        # OpenRLHF format - look for assistant response
        if "assistant<|end_header_id|>" in query:
            response = query.split("assistant<|end_header_id|>", 1)[1].strip()
            return response
        elif "assistant\n\n" in query:
            response = query.split("assistant\n\n", 1)[1].strip()
            return response
        else:
            # Fallback: assume the query IS the response
            return query.strip()
            
    except Exception as e:
        logger.warning(f"Error extracting response from query: {e}")
        return query.strip()

class TableReasoningRewardCalculator:
    """Binary reward calculator for table reasoning"""
    
    def __init__(self):
        self.total_samples = 0
        self.correct_count = 0
    
    def calculate_reward(self, generated_response: str, ground_truth: str) -> float:
        """Calculate binary reward: 1.0 if correct, 0.0 if incorrect"""
        self.total_samples += 1
        
        # Extract predicted answer
        pred_answer = normalize_prediction(generated_response)
        
        # Normalize both for comparison
        pred_normalized = pred_answer.lower().strip()
        truth_normalized = ground_truth.lower().strip()
        
        # Binary reward
        reward = 1.0 if pred_normalized == truth_normalized else 0.0
        
        if reward == 1.0:
            self.correct_count += 1
        
        return reward

# Global calculator
calculator = TableReasoningRewardCalculator()

def reward_func(queries, prompts, labels, **kwargs):
    """
    Binary reward function for table reasoning REINFORCE training.
    
    Args:
        queries: Full conversations (prompt + model response)
        prompts: Input prompts (contain ground truth)
        labels: Not used in our case
        **kwargs: Additional parameters
    
    Returns:
        torch.Tensor: Binary rewards (1.0 for correct, 0.0 for incorrect)
    """
    global calculator
    
    try:
        rewards = []
        
        # Debug logging for first few samples
        if calculator.total_samples < 3:
            logger.info(f"🎯 Processing batch of {len(queries)} table reasoning samples")
        
        for i, query in enumerate(queries):
            # Extract model's response from the full conversation
            generated_response = extract_response_from_query(str(query))
            
            # Extract ground truth from the prompt
            prompt_text = str(prompts[i]) if i < len(prompts) else ""
            ground_truth = extract_ground_truth_from_prompt(prompt_text)
            
            # Debug logging
            if calculator.total_samples < 3:
                logger.info(f"🔍 Sample {i}:")
                logger.info(f"   Generated response: {generated_response[:100]}...")
                logger.info(f"   Ground truth: {ground_truth}")
            
            # Calculate binary reward
            if ground_truth and generated_response:
                reward = calculator.calculate_reward(generated_response, ground_truth)
            else:
                # If we can't extract ground truth or response, give 0 reward
                reward = 0.0
                if calculator.total_samples < 3:
                    logger.warning(f"Missing ground truth or response for sample {i}")
            
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Periodic logging
        if calculator.total_samples % 50 == 0:
            accuracy = calculator.correct_count / calculator.total_samples if calculator.total_samples > 0 else 0.0
            avg_reward = rewards_tensor.mean().item()
            logger.info(f"📊 Batch reward: {avg_reward:.3f}, Overall accuracy: {accuracy:.3f}, Total samples: {calculator.total_samples}")
        
        return rewards_tensor
        
    except Exception as e:
        logger.error(f"Error in reward function: {e}")
        # Return zero rewards as fallback
        return torch.zeros(len(queries), dtype=torch.float32)