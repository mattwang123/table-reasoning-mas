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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.training.reward_functions import table_reasoning_reward_func

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_main_process():
    """Check if this is the main process"""
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except:
        return True

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def setup_wandb(config, unique_run_name):
    """Setup wandb - only on main process, let TRL handle the actual logging"""
    if not is_main_process():
        return None
        
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

# --- FIXED: Single Logging Callback ---
class AccuracyTrackingCallback(TrainerCallback):
    """
    Callback to track accuracy - CONSOLE ONLY, let TRL handle wandb
    This preserves all your functionality but eliminates double logging
    """
    def __init__(self):
        self.best_eval_accuracy = 0.0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Console logging only - NO wandb.log() calls"""
        if not logs or not is_main_process():
            return
        
        # Detailed console logging every logging_steps
        if state.global_step % args.logging_steps == 0:
            accuracy = logs.get('reward', 0.0)
            loss = logs.get('loss', 0.0)
            lr = logs.get('learning_rate', 0.0)
            
            # Main metrics
            logger.info(f"Step {state.global_step}: Loss={loss:.4f}, Accuracy={accuracy:.3f}, LR={lr:.2e}")
            
            # Additional useful metrics (console only)
            if 'reward_std' in logs:
                logger.info(f"   Reward std: {logs['reward_std']:.3f}")
            if 'completions/mean_length' in logs:
                logger.info(f"   Avg completion length: {logs['completions/mean_length']:.1f}")
            if 'entropy' in logs:
                logger.info(f"   Entropy: {logs['entropy']:.3f}")
            if 'clip_ratio/region_mean' in logs:
                logger.info(f"   Clip ratio: {logs['clip_ratio/region_mean']:.3f}")
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Evaluation logging - CONSOLE ONLY, NO wandb.log() calls"""
        if not logs or not is_main_process():
            return
        
        if 'eval_reward' in logs:
            eval_reward = logs['eval_reward']
            eval_loss = logs.get('eval_loss', 0.0)
            
            # Track best accuracy
            if eval_reward > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_reward
                logger.info(f"🎉 New best validation accuracy: {eval_reward:.3f}")
            
            logger.info(f"🔍 Validation at step {state.global_step}:")
            logger.info(f"   Accuracy: {eval_reward:.3f} (Best: {self.best_eval_accuracy:.3f})")
            logger.info(f"   Loss: {eval_loss:.4f}")
            
            # Show trend if we have previous evaluations
            trend_info = ""
            if hasattr(self, 'prev_eval_reward'):
                diff = eval_reward - self.prev_eval_reward
                trend = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                trend_info = f" {trend} ({diff:+.3f})"
            
            if trend_info:
                logger.info(f"   Trend: {trend_info}")
            
            self.prev_eval_reward = eval_reward
    
    def on_train_end(self, args, state, control, **kwargs):
        """Final training summary - console only"""
        if not is_main_process():
            return
            
        logger.info(f"\n🏆 Training Summary:")
        logger.info(f"   Total steps: {state.global_step}")
        logger.info(f"   Total epochs: {state.epoch:.2f}")
        logger.info(f"   Best validation accuracy: {self.best_eval_accuracy:.3f}")

class DebugCallback(TrainerCallback):
    """Debug callback for development"""
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and is_main_process():
            logger.debug(f"Debug - Step {state.global_step}, Epoch: {state.epoch:.2f}")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and is_main_process():
            logger.debug(f"Debug - Step {state.global_step} completed")

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
    
    # Log process info for debugging
    if is_main_process():
        logger.info("Running on main process")
    else:
        logger.info(f"Running on worker process")
    
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
    
    # Setup wandb (main process only)
    wandb_run = setup_wandb(config, unique_run_name)
    
    # Load datasets
    train_dataset, eval_dataset = load_datasets(config)
    
    # Setup callbacks
    callbacks = [AccuracyTrackingCallback()]  # Console logging only
    if args.debug:
        callbacks.append(DebugCallback())
    
    # Training configuration
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
        evaluation_strategy="steps" if eval_exists else "no",  # Fixed: was eval_strategy
        save_strategy="steps",
        
        # FIXED: Let TRL handle wandb logging (no double logging)
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
    
    # Create trainer with REQUIRED reward function
    trainer = GRPOTrainer(
        model=config['model']['name'],
        reward_funcs=table_reasoning_reward_func,  # REQUIRED: This is the core of GRPO!
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
    
    # Print training info for verification (main process only)
    if is_main_process():
        effective_batch_size = (training_args.per_device_train_batch_size * 
                               training_args.gradient_accumulation_steps * 
                               torch.cuda.device_count())
        
        logger.info(f"📋 Training Configuration:")
        logger.info(f"   Model: {config['model']['name']}")
        logger.info(f"   Dataset: {len(train_dataset)} train, {len(eval_dataset) if eval_exists else 0} eval")
        logger.info(f"   Per-device train batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"   Per-device eval batch size: {training_args.per_device_eval_batch_size}")
        logger.info(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
        logger.info(f"   Eval accumulation: {training_args.eval_accumulation_steps}")
        logger.info(f"   Effective batch size: {effective_batch_size}")
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
    
    if is_main_process():
        logger.info(f"\n✅ Training completed!")
        if wandb.run:
            logger.info(f"📊 Results: {wandb.run.url}")
            wandb.finish()

if __name__ == "__main__":
    main()