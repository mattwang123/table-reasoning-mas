from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 3072
    
    # LoRA settings
    lora_r: int = 16                        # REDUCED: 32 → 16
    lora_alpha: int = 32                    # REDUCED: 64 → 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # Training settings - MEMORY OPTIMIZED FOR 3 GPUs
    output_dir: str = "data/trained_models/verifier_v1"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1    # REDUCED: 2 → 1
    per_device_eval_batch_size: int = 1     # REDUCED: 2 → 1
    gradient_accumulation_steps: int = 48   # INCREASED: 24 → 48 (maintain effective batch)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine_with_restarts"
    
    # Data settings
    train_file: str = "data/deterministic_training/reanalyzed_combined_20250825_212207.jsonl"
    eval_file: Optional[str] = None
    test_split: float = 0.15
    
    # Logging and evaluation - REDUCED FREQUENCY
    logging_steps: int = 5                  # INCREASED: 1 → 5
    eval_steps: int = 100                   # INCREASED: 50 → 100
    save_steps: int = 500                   # Keep same
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    
    # Wandb settings
    wandb_project: str = "verifier-training-3gpu"
    wandb_name: Optional[str] = None
    report_to: str = "wandb"
    
    # System settings - MEMORY OPTIMIZED
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 0         # REDUCED: 2 → 0
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    
    # Training stability
    max_grad_norm: float = 1.0
    save_total_limit: int = 2               # REDUCED: 5 → 2
    
    # Memory optimization
    gradient_checkpointing: bool = True     # ADDED
    dataloader_pin_memory: bool = False     # ADDED
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj"          # REDUCED: Only q_proj, v_proj
            ]
        
        if self.wandb_name is None:
            from datetime import datetime
            self.wandb_name = f"verifier-3gpu-mem-opt-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate effective batch size
        self.effective_batch_size = self.per_device_train_batch_size * 3 * self.gradient_accumulation_steps
        print(f"✓ Memory Optimized 3 GPU Configuration:")
        print(f"  - Per device batch size: {self.per_device_train_batch_size}")
        print(f"  - Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  - Effective batch size: {self.effective_batch_size}")
        print(f"  - Max sequence length: {self.max_seq_length}")
        print(f"  - LoRA rank: {self.lora_r}")
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)

# Global config instance
config = TrainingConfig()