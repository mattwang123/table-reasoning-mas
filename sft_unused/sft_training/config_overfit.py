from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048              # REDUCED for overfitting test
    
    # LoRA settings - OPTIMIZED FOR OVERFITTING
    lora_r: int = 32                        # INCREASED: Better capacity for overfitting
    lora_alpha: int = 64                    # INCREASED: Better capacity for overfitting
    lora_dropout: float = 0.0               # DISABLED: No dropout for overfitting
    target_modules: Optional[List[str]] = None
    
    # OVERFITTING TEST SETTINGS
    overfit_test: bool = True               # Enable overfitting mode
    overfit_samples: int = 10               # Number of samples to overfit
    
    # Training settings - OPTIMIZED FOR OVERFITTING
    output_dir: str = "data/trained_models/verifier_overfit_test"
    num_train_epochs: int = 20              # MORE EPOCHS: For complete overfitting
    per_device_train_batch_size: int = 1    # Keep small for memory
    per_device_eval_batch_size: int = 1     # Keep small for memory
    gradient_accumulation_steps: int = 1    # REDUCED: Faster updates for overfitting
    learning_rate: float = 1e-3             # HIGHER: Faster learning for overfitting
    weight_decay: float = 0.0               # DISABLED: No regularization for overfitting
    warmup_ratio: float = 0.0               # DISABLED: No warmup for small dataset
    lr_scheduler_type: str = "constant"     # CONSTANT: No decay for overfitting
    
    # Data settings
    train_file: str = "data/deterministic_training/reanalyzed_combined_20250825_212207.jsonl"
    eval_file: Optional[str] = None
    test_split: float = 0.0                 # NO SPLIT: Use same data for train/eval
    
    # Logging and evaluation - FREQUENT FOR OVERFITTING
    logging_steps: int = 1                  # Log every step
    eval_steps: int = 5                     # Evaluate very frequently
    save_steps: int = 20                    # Save frequently
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = False    # Don't need best model for overfitting test
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    
    # Wandb settings - OVERFITTING PROJECT
    wandb_project: str = "verifier-overfit-test"
    wandb_name: Optional[str] = None
    report_to: str = "wandb"
    
    # System settings - MEMORY OPTIMIZED
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    
    # Training stability - RELAXED FOR OVERFITTING
    max_grad_norm: float = 10.0             # HIGHER: Allow larger gradients
    save_total_limit: int = 3               # Keep few checkpoints
    
    # Memory optimization
    gradient_checkpointing: bool = False    # DISABLED: Faster for small dataset
    dataloader_pin_memory: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            if self.overfit_test:
                # MORE MODULES for better overfitting capacity
                self.target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            else:
                self.target_modules = ["q_proj", "v_proj"]
        
        if self.wandb_name is None:
            from datetime import datetime
            if self.overfit_test:
                self.wandb_name = f"overfit-{self.overfit_samples}samples-{datetime.now().strftime('%m%d_%H%M')}"
            else:
                self.wandb_name = f"verifier-3gpu-mem-opt-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate effective batch size
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.effective_batch_size = self.per_device_train_batch_size * num_gpus * self.gradient_accumulation_steps
        
        if self.overfit_test:
            print(f"🧪 OVERFITTING TEST CONFIGURATION:")
            print(f"  - Samples to overfit: {self.overfit_samples}")
            print(f"  - Epochs: {self.num_train_epochs}")
            print(f"  - Learning rate: {self.learning_rate}")
            print(f"  - LoRA rank: {self.lora_r}")
            print(f"  - LoRA alpha: {self.lora_alpha}")
            print(f"  - No dropout, no weight decay, no warmup")
            print(f"  - Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"  - Effective batch size: {self.effective_batch_size}")
            print(f"  - Target modules: {len(self.target_modules)} modules")
            print(f"  - Expected: 100% accuracy, loss < 0.1")
        else:
            print(f"✓ Memory Optimized 3 GPU Configuration:")
            print(f"  - Per device batch size: {self.per_device_train_batch_size}")
            print(f"  - Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"  - Effective batch size: {self.effective_batch_size}")
            print(f"  - Max sequence length: {self.max_seq_length}")
            print(f"  - LoRA rank: {self.lora_r}")
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def enable_normal_training(self):
        """Switch back to normal training configuration"""
        self.overfit_test = False
        self.num_train_epochs = 3
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.lr_scheduler_type = "cosine_with_restarts"
        self.gradient_accumulation_steps = 48
        self.test_split = 0.15
        self.lora_dropout = 0.1
        self.gradient_checkpointing = True
        self.max_grad_norm = 1.0
        self.output_dir = "data/trained_models/verifier_v1"
        self.wandb_project = "verifier-training-3gpu"
        self.target_modules = ["q_proj", "v_proj"]
        print("🔄 Switched to normal training configuration")

# Global config instance
config = TrainingConfig()