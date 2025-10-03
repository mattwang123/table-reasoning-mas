"""
Integrated REINFORCE Training for Table Reasoning
Uses existing reward_func, data_utils, and model_utils
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
import json
import wandb
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, field
import os
import sys
from datetime import datetime
import random

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'rl_src'))

# Import existing utilities
try:
    from reward_func import get_reward
    print("✅ Using existing reward function")
    REWARD_FUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import reward_func: {e}")
    REWARD_FUNC_AVAILABLE = False

try:
    from data_utils import load_data, preprocess_data
    print("✅ Using existing data utilities")
    DATA_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import data_utils: {e}")
    DATA_UTILS_AVAILABLE = False

try:
    from model_utils import setup_model, setup_tokenizer
    print("✅ Using existing model utilities")
    MODEL_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import model_utils: {e}")
    MODEL_UTILS_AVAILABLE = False

@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE training"""
    # Model config
    model_name: str = "Qwen/Qwen-7B-Chat"
    max_length: int = 1024
    max_new_tokens: int = 256
    
    # Training config
    batch_size: int = 4
    learning_rate: float = 3e-7
    num_epochs: int = 3
    max_samples: int = 10000
    samples_per_prompt: int = 4
    
    # LoRA config
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Optimization config
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Generation config
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True
    
    # Baseline config
    baseline_alpha: float = 0.1
    
    # Logging and saving
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 250
    
    # Paths
    train_data_path: str = "../data/reinforce_train_20250913_224659.jsonl"
    output_dir: str = "./experiments/reinforce_integrated"
    
    # WandB
    wandb_project: str = "table-reasoning-reinforce"
    wandb_run_name: Optional[str] = None
    
    # Reward config
    reward_clip_range: tuple = (-10.0, 10.0)
    normalize_rewards: bool = True
    
    def __post_init__(self):
        if self.wandb_run_name is None:
            self.wandb_run_name = f"reinforce-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class IntegratedTableDataset(Dataset):
    """Dataset that uses existing data_utils if available"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Try to use existing data_utils
        if DATA_UTILS_AVAILABLE:
            try:
                self.data = load_data(data_path)
                if hasattr(self, 'data') and len(self.data) > 0:
                    print(f"✅ Loaded {len(self.data)} examples using data_utils")
                    return
            except Exception as e:
                print(f"⚠️ data_utils failed, falling back to simple loading: {e}")
        
        # Fallback to simple loading
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        print(f"📁 Loaded {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Try to use existing data preprocessing
        if DATA_UTILS_AVAILABLE:
            try:
                processed_item = preprocess_data(item)
                if processed_item:
                    item = processed_item
            except:
                pass  # Fall back to original item
        
        # Extract prompt
        prompt = item.get('prompt', item.get('input', item.get('question', '')))
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'prompt': prompt,
            'original_item': item
        }

class IntegratedRewardFunction:
    """Reward function that uses existing reward_func if available"""
    
    def __init__(self):
        self.reward_func = self._setup_reward_function()
    
    def _setup_reward_function(self):
        """Setup reward function using existing or fallback"""
        if REWARD_FUNC_AVAILABLE:
            try:
                print("✅ Using existing reward function from reward_func.py")
                return get_reward
            except Exception as e:
                print(f"⚠️ Error with existing reward function: {e}")
        
        print("🔄 Using fallback reward function")
        return self._fallback_reward
    
    def _fallback_reward(self, prompt: str, response: str) -> float:
        """Enhanced fallback reward for table reasoning"""
        response_lower = response.lower().strip()
        
        if not response_lower or len(response_lower) < 3:
            return 0.0
        
        score = 0.0
        
        # Check for direct answers
        if any(word in response_lower for word in ['yes', 'no', 'true', 'false']):
            score += 0.4
        
        # Check for table-related reasoning
        table_words = ['table', 'row', 'column', 'cell', 'header', 'data']
        if any(word in response_lower for word in table_words):
            score += 0.3
        
        # Check for reasoning indicators
        reasoning_words = ['because', 'since', 'therefore', 'thus', 'so', 'hence']
        if any(word in response_lower for word in reasoning_words):
            score += 0.2
        
        # Length penalty (not too short, not too long)
        word_count = len(response.split())
        if 5 <= word_count <= 100:
            score += 0.2
        elif word_count > 100:
            score -= 0.1
        
        # Penalty for non-answers
        negative_words = ['sorry', 'cannot', 'unable', 'don\'t know', 'unclear']
        if any(word in response_lower for word in negative_words):
            score -= 0.5
        
        return max(0.0, min(1.0, score))
    
    def __call__(self, prompt: str, response: str) -> float:
        """Compute reward for prompt-response pair"""
        try:
            reward = self.reward_func(prompt, response)
            return float(reward)
        except Exception as e:
            print(f"⚠️ Reward computation error: {e}")
            return 0.0

class REINFORCETrainer:
    """Integrated REINFORCE trainer using Accelerate"""
    
    def __init__(self, config: REINFORCEConfig):
        self.config = config
        
        # Initialize Accelerate
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.wandb_project else None,
            project_dir=config.output_dir
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if self.accelerator.is_main_process:
            self.logger.info(f"🚀 Starting REINFORCE training")
            self.logger.info(f"📋 Config: {config}")
        
        # Initialize components
        self._setup_model()
        self._setup_reward_function()
        self._setup_dataset()
        self._setup_optimizer()
        
        # Initialize baseline
        self.baseline = 0.0
        
        # Initialize WandB
        if self.accelerator.is_main_process and config.wandb_project:
            self._setup_wandb()
    
    def _setup_model(self):
        """Setup model using existing model_utils if available"""
        if self.accelerator.is_main_process:
            self.logger.info(f"🤖 Loading model: {self.config.model_name}")
        
        # Try to use existing model utilities
        if MODEL_UTILS_AVAILABLE:
            try:
                self.tokenizer = setup_tokenizer(self.config.model_name)
                self.model = setup_model(self.config.model_name)
                if self.accelerator.is_main_process:
                    print("✅ Using existing model utilities")
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"⚠️ model_utils failed, using fallback: {e}")
                MODEL_UTILS_AVAILABLE = False
        
        if not MODEL_UTILS_AVAILABLE:
            # Fallback model setup
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None
            )
        
        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if self.accelerator.is_main_process:
            self.model.print_trainable_parameters()
        
        # Reference model for potential KL penalty
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None
        )
        self.ref_model.eval()
    
    def _setup_reward_function(self):
        """Setup integrated reward function"""
        self.reward_function = IntegratedRewardFunction()
    
    def _setup_dataset(self):
        """Setup dataset using integrated data loading"""
        self.dataset = IntegratedTableDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_length
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=2,
            pin_memory=True
        )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.config.weight_decay
        )
        
        num_update_steps_per_epoch = len(self.dataloader) // self.config.gradient_accumulation_steps
        total_steps = num_update_steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare with Accelerate
        self.model, self.ref_model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.ref_model, self.optimizer, self.dataloader, self.scheduler
        )
        
        if self.accelerator.is_main_process:
            self.logger.info(f"⚙️ Optimizer: {total_steps} steps, {warmup_steps} warmup")
    
    def _setup_wandb(self):
        """Setup WandB tracking"""
        self.accelerator.init_trackers(
            project_name=self.config.wandb_project,
            config=self.config.__dict__,
            init_kwargs={"wandb": {"name": self.config.wandb_run_name}}
        )
    
    def _collate_fn(self, batch):
        """Collate function with padding"""
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            pad_len = max_len - len(ids)
            padded_ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            padded_mask = torch.cat([mask, torch.zeros(pad_len)])
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'prompts': [item['prompt'] for item in batch],
            'original_items': [item['original_item'] for item in batch]
        }
    
    def generate_responses(self, input_ids, attention_mask, num_samples=1):
        """Generate responses"""
        self.model.eval()
        batch_size = input_ids.size(0)
        all_responses = []
        all_response_ids = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
                
                generated_ids = outputs.sequences
                new_tokens = generated_ids[:, input_ids.size(1):]
                
                responses = []
                for i in range(batch_size):
                    response = self.tokenizer.decode(new_tokens[i], skip_special_tokens=True)
                    responses.append(response.strip())
                
                all_responses.append(responses)
                all_response_ids.append(new_tokens)
        
        return all_responses, all_response_ids
    
    def compute_log_probs(self, input_ids, response_ids):
        """Compute log probabilities for responses"""
        self.model.eval()
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits
        
        response_start = input_ids.size(1)
        response_logits = logits[:, response_start-1:-1]
        response_log_probs = F.log_softmax(response_logits, dim=-1)
        
        log_probs = torch.gather(response_log_probs, -1, response_ids.unsqueeze(-1)).squeeze(-1)
        
        sequence_log_probs = []
        for i in range(response_ids.size(0)):
            response_length = (response_ids[i] != self.tokenizer.pad_token_id).sum().item()
            if response_length > 0:
                seq_log_prob = log_probs[i, :response_length].sum()
            else:
                seq_log_prob = torch.tensor(0.0, device=log_probs.device)
            sequence_log_probs.append(seq_log_prob)
        
        return torch.stack(sequence_log_probs)
    
    def compute_rewards(self, prompts, responses):
        """Compute rewards using integrated reward function"""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            reward = self.reward_function(prompt, response)
            reward = max(self.config.reward_clip_range[0], 
                        min(self.config.reward_clip_range[1], reward))
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.accelerator.device)
        
        if self.config.normalize_rewards and len(rewards) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        return rewards_tensor
    
    def train_step(self, batch):
        """Single REINFORCE training step"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prompts = batch['prompts']
        
        # Generate responses
        all_responses, all_response_ids = self.generate_responses(
            input_ids, attention_mask, num_samples=self.config.samples_per_prompt
        )
        
        total_loss = 0
        total_reward = 0
        num_samples = 0
        
        self.model.train()
        
        for sample_idx in range(self.config.samples_per_prompt):
            responses = all_responses[sample_idx]
            response_ids = all_response_ids[sample_idx]
            
            # Compute log probabilities and rewards
            log_probs = self.compute_log_probs(input_ids, response_ids)
            rewards = self.compute_rewards(prompts, responses)
            
            # Update baseline
            current_reward_mean = rewards.mean().item()
            self.baseline = (1 - self.config.baseline_alpha) * self.baseline + \
                           self.config.baseline_alpha * current_reward_mean
            
            # Compute advantages
            advantages = rewards - self.baseline
            
            # REINFORCE loss
            loss = -(log_probs * advantages).mean()
            
            total_loss += loss
            total_reward += current_reward_mean
            num_samples += 1
        
        avg_loss = total_loss / num_samples
        avg_reward = total_reward / num_samples
        
        return avg_loss, avg_reward, self.baseline
    
    def train(self):
        """Main training loop"""
        if self.accelerator.is_main_process:
            self.logger.info("🎯 Starting REINFORCE training")
        
        global_step = 0
        samples_processed = 0
        
        for epoch in range(self.config.num_epochs):
            if self.accelerator.is_main_process:
                self.logger.info(f"📈 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            epoch_reward = 0
            num_batches = 0
            
            progress_bar = tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    loss, reward, baseline = self.train_step(batch)
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    if self.accelerator.sync_gradients:
                        global_step += 1
                
                epoch_loss += loss.item()
                epoch_reward += reward
                num_batches += 1
                samples_processed += len(batch['prompts']) * self.config.samples_per_prompt
                
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'reward': f"{reward:.4f}",
                        'baseline': f"{baseline:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # Logging
                if global_step % self.config.logging_steps == 0 and self.accelerator.is_main_process:
                    self.accelerator.log({
                        'train/loss': loss.item(),
                        'train/reward': reward,
                        'train/baseline': baseline,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/global_step': global_step,
                        'train/samples_processed': samples_processed
                    })
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0 and self.accelerator.is_main_process:
                    self.save_checkpoint(global_step)
                
                if samples_processed >= self.config.max_samples:
                    if self.accelerator.is_main_process:
                        self.logger.info(f"✅ Reached max samples: {self.config.max_samples}")
                    break
            
            # Epoch logging
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                avg_epoch_reward = epoch_reward / num_batches
                
                if self.accelerator.is_main_process:
                    self.logger.info(f"Epoch {epoch + 1} - Loss: {avg_epoch_loss:.4f}, Reward: {avg_epoch_reward:.4f}")
                    
                    self.accelerator.log({
                        'epoch/loss': avg_epoch_loss,
                        'epoch/reward': avg_epoch_reward,
                        'epoch/baseline': self.baseline,
                        'epoch/number': epoch + 1
                    })
            
            if samples_processed >= self.config.max_samples:
                break
        
        if self.accelerator.is_main_process:
            self.save_final_model()
            self.logger.info("🎉 Training completed!")
    
    def save_checkpoint(self, step):
        """Save checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        self.accelerator.save_model(self.model, checkpoint_dir)
        
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(checkpoint_dir)
            self.logger.info(f"💾 Checkpoint saved: {checkpoint_dir}")
    
    def save_final_model(self):
        """Save final model"""
        final_dir = os.path.join(self.config.output_dir, "final_model")
        self.accelerator.save_model(self.model, final_dir)
        self.tokenizer.save_pretrained(final_dir)
        self.logger.info(f"💾 Final model saved: {final_dir}")

def main():
    """Main training function"""
    config = REINFORCEConfig(
        model_name="Qwen/Qwen-7B-Chat",
        train_data_path="../data/reinforce_train_20250913_224659.jsonl",
        output_dir="./experiments/reinforce_integrated",
        batch_size=2,
        learning_rate=3e-7,
        num_epochs=2,
        max_samples=10000,
        samples_per_prompt=4,
        gradient_accumulation_steps=8,
        wandb_project="table-reasoning-reinforce-integrated",
        logging_steps=5,
        save_steps=250
    )
    
    trainer = REINFORCETrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()