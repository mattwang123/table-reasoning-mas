from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 3072
    
    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # Training settings - MEMORY OPTIMIZED
    output_dir: str = "data/trained_models/verifier_v1"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1   # Conservative for memory
    per_device_eval_batch_size: int = 1    # Conservative for memory
    gradient_accumulation_steps: int = 48  # Maintain effective batch size
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Data settings
    train_file: str = "data/deterministic_training/reanalyzed_combined_20250825_212207.jsonl"
    eval_file: Optional[str] = None
    test_split: float = 0.15
    
    # Logging and evaluation - OPTIMIZED
    logging_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 500
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    
    # Wandb settings
    wandb_project: str = "verifier-training"
    wandb_name: Optional[str] = None
    report_to: str = "wandb"
    
    # System settings
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    
    # Training stability
    max_grad_norm: float = 1.0
    save_total_limit: int = 3
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]
        
        if self.wandb_name is None:
            from datetime import datetime
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            self.wandb_name = f"verifier-{num_gpus}gpu-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.wandb_project = f"verifier-training-{num_gpus}gpu"
        
        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)

# Global config instance
config = TrainingConfig()