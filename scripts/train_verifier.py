#!/usr/bin/env python3
import os
import sys
import yaml
import json
import argparse
import re
import logging
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
import wandb
from datetime import datetime

# --- Start of Reward Functions ---
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
    Reward function for table reasoning - follows TRL format exactly
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
        
        # Binary reward: 1.0 if exact match, 0.0 otherwise
        reward = 1.0 if pred_normalized == truth_normalized else 0.0
        rewards.append(reward)
    
    return rewards
# --- End of Reward Functions ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DebugCallback(TrainerCallback):
    """Debug callback to understand step counting"""
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            logger.debug(f"Step begin - Global step: {state.global_step}, Epoch: {state.epoch}")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            logger.debug(f"Step end - Global step: {state.global_step}")

class AccuracyTrackingCallback(TrainerCallback):
    """Callback to track accuracy and other important metrics"""
    def __init__(self):
        self.best_eval_accuracy = 0.0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not logs:
            return
        
        metrics_to_log = {}
        
        if 'loss' in logs:
            metrics_to_log['train/loss'] = logs['loss']
        if 'learning_rate' in logs:
            metrics_to_log['train/learning_rate'] = logs['learning_rate']
        if 'reward' in logs:
            metrics_to_log['train/reward'] = logs['reward']
            metrics_to_log['train/accuracy'] = logs['reward']
        if 'reward_std' in logs:
            metrics_to_log['train/reward_std'] = logs['reward_std']
        if 'completions/mean_length' in logs:
            metrics_to_log['train/completion_length'] = logs['completions/mean_length']
        if 'entropy' in logs:
            metrics_to_log['train/entropy'] = logs['entropy']
        if 'clip_ratio/region_mean' in logs:
            metrics_to_log['train/clip_ratio'] = logs['clip_ratio/region_mean']
        
        if wandb.run and metrics_to_log:
            wandb.log(metrics_to_log, step=state.global_step)
        
        if state.global_step % args.logging_steps == 0:
            accuracy = logs.get('reward', 0.0)
            loss = logs.get('loss', 0.0)
            logger.info(f"Step {state.global_step}: Loss={loss:.4f}, Accuracy={accuracy:.3f}")
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if not logs:
            return
        
        eval_metrics = {}
        for key, value in logs.items():
            if key.startswith('eval_'):
                clean_key = key.replace('eval_', '')
                eval_metrics[f'eval/{clean_key}'] = value
        
        if 'eval_reward' in logs:
            eval_reward = logs['eval_reward']
            eval_metrics['eval/accuracy'] = eval_reward
            
            if eval_reward > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_reward
                eval_metrics['eval/best_accuracy'] = self.best_eval_accuracy
            
            logger.info(f"🔍 Validation at step {state.global_step}:")
            logger.info(f"   Accuracy: {eval_reward:.3f} (Best: {self.best_eval_accuracy:.3f})")
            
        if wandb.run and eval_metrics:
            wandb.log(eval_metrics, step=state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"\n🏆 Training Summary:")
        logger.info(f"   Total steps: {state.global_step}")
        logger.info(f"   Best validation accuracy: {self.best_eval_accuracy:.3f}")
        
        if wandb.run:
            wandb.log({
                'final/best_eval_accuracy': self.best_eval_accuracy,
                'final/total_steps': state.global_step,
            })

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def setup_wandb(config, unique_run_name):
    """Setup wandb from config with a unique run name"""
    wandb_config = config.get('wandb', {})
    if not wandb_config.get('enabled', True):
        logger.info("📊 Wandb disabled in config")
        return None
    
    try:
        wandb_init_config = {
            'project': wandb_config.get('project'),
            'entity': wandb_config.get('entity'),
            'name': unique_run_name,
            'tags': wandb_config.get('tags', []),
            'notes': wandb_config.get('notes', ''),
            'config': {
                **config,
                'total_parameters': '7B (with LoRA)',
                'framework': 'TRL GRPO',
                'task': 'table_reasoning_verification'
            },
            'save_code': wandb_config.get('save_code', True),
        }
        
        run = wandb.init(**wandb_init_config)
        logger.info(f"📊 Wandb initialized: {wandb.run.url}")
        return run
    except Exception as e:
        logger.error(f"❌ Wandb initialization failed: {e}")
        return None

def load_datasets(config):
    """Load datasets"""
    train_data = []
    with open(config['data']['train_file'], 'r') as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))
    
    subset_size = config['training'].get('subset_size')
    if subset_size and subset_size < len(train_data):
        train_data = train_data[:subset_size]
        logger.info(f"Using training subset: {len(train_data)} samples")
    
    eval_data = []
    if os.path.exists(config['data']['eval_file']):
        with open(config['data']['eval_file'], 'r') as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))

    eval_subset_size = config['training'].get('eval_subset_size')
    if eval_subset_size and eval_subset_size < len(eval_data):
        eval_data = eval_data[:eval_subset_size]
        logger.info(f"Using evaluation subset: {len(eval_data)} samples")
    
    logger.info(f"📊 Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None
    
    return train_dataset, eval_dataset

def main():
    parser = argparse.ArgumentParser(description="Train table reasoning verifier")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb even if enabled in config")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 TRL GRPO Training - Table Reasoning Verifier")
    logger.info("=" * 60)
    
    try:
        config = load_config(args.config)
        logger.info(f"✅ Loaded config from: {args.config}")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return
    
    if args.no_wandb:
        config['wandb']['enabled'] = False
    
    if not os.path.exists(config['data']['train_file']):
        logger.error(f"❌ Training file not found: {config['data']['train_file']}")
        return
    
    eval_exists = os.path.exists(config['data']['eval_file'])
    if not eval_exists:
        logger.warning(f"⚠️  Evaluation file not found: {config['data']['eval_file']}. Evaluation will be skipped.")
    
    # Generate unique run name with timestamp
    run_name_from_config = config['training']['run_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_name = f"{run_name_from_config}_{timestamp}"
    
    wandb_run = setup_wandb(config, unique_run_name)
    train_dataset, eval_dataset = load_datasets(config)
    
    callbacks = [AccuracyTrackingCallback()]
    if args.debug:
        callbacks.append(DebugCallback())
    
    training_args = GRPOConfig(
        output_dir=config['training']['output_dir'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        eval_accumulation_steps=config['training']['eval_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        num_train_epochs=config['training']['num_epochs'],
        warmup_steps=config['training']['warmup_steps'],
        
        # GRPO specific parameters
        num_generations=config['training']['num_generations'],
        temperature=config['training']['temperature'],
        top_p=config['training']['top_p'],
        beta=config['training']['beta'],
        max_prompt_length=config['training']['max_prompt_length'],
        max_completion_length=config['training']['max_completion_length'],
        scale_rewards=config['training']['scale_rewards'],
        mask_truncated_completions=config['training']['mask_truncated_completions'],
        shuffle_dataset=config['training']['shuffle_dataset'],
        log_completions=True,
        
        # Logging & Evaluation
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'] if eval_exists else None,
        save_steps=config['training']['save_steps'],
        eval_strategy="steps" if eval_exists else "no",
        save_strategy="steps",
        
        # Wandb reporting
        report_to=["wandb"] if wandb_run else [],
        run_name=unique_run_name,
        
        # Best model tracking
        load_best_model_at_end=True if eval_exists else False,
        metric_for_best_model="eval_reward" if eval_exists else None,
        greater_is_better=True,
        
        # Performance & Memory Optimization
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        
        # vLLM Integration for accelerated generation
        use_vllm=config['training'].get('use_vllm', False),
    )
    
    trainer = GRPOTrainer(
        model=config['model']['name'],
        reward_funcs=table_reasoning_reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['target_modules']
        ),
        callbacks=callbacks,
    )
    
    # Print training info for verification
    logger.info(f"📋 Training Configuration:")
    logger.info(f"   Model: {config['model']['name']}")
    logger.info(f"   Dataset: {len(train_dataset)} train, {len(eval_dataset) if eval_exists else 0} eval")
    logger.info(f"   Per-device train batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"   Per-device eval batch size: {training_args.per_device_eval_batch_size}")
    logger.info(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"   Eval accumulation: {training_args.eval_accumulation_steps}")
    logger.info(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"   Generations per prompt: {training_args.num_generations}")
    logger.info(f"   Learning rate: {training_args.learning_rate}")
    logger.info(f"   Total epochs: {training_args.num_train_epochs}")
    logger.info(f"   Wandb: {'✅ Enabled' if wandb_run else '❌ Disabled'}")
    logger.info(f"   Using vLLM: {'✅ Enabled' if training_args.use_vllm else '❌ Disabled'}")
    logger.info("")
    
    logger.info("🎯 Starting GRPO training...")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ CUDA OOM Error: {e}")
        logger.error("Try reducing batch_size, gradient_accumulation_steps, or num_generations in the config")
        raise
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    logger.info("💾 Saving model...")
    trainer.save_model()
    
    logger.info(f"\n✅ Training completed!")
    if wandb.run:
        logger.info(f"📊 Results: {wandb.run.url}")
        wandb.finish()

if __name__ == "__main__":
    main()