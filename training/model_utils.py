import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total")

def load_model_and_tokenizer(config):
    """Load model and tokenizer with LoRA configuration - Fixed for multi-GPU training"""
    
    print("=" * 50)
    print("LOADING MODEL AND TOKENIZER")
    print("=" * 50)
    
    print(f"Model: {config.model_name}")
    print(f"Precision: FP16 (no quantization)")
    print(f"Max sequence length: {config.max_seq_length}")
    
    # Print initial GPU memory
    print("\nInitial GPU memory:")
    print_gpu_memory()
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Added pad token (using eos_token)")
    
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
    
    # Load model - FIXED: Remove device_map for distributed training
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        # device_map="auto",  # REMOVE THIS LINE for multi-GPU training
        trust_remote_code=True,
        torch_dtype=torch.float16,  # FP16 precision
        # No quantization config
    )
    
    print(f"Base model loaded")
    
    # Print memory after base model loading
    print("\nGPU memory after base model loading:")
    print_gpu_memory()
    
    # LoRA configuration
    print(f"\nApplying LoRA configuration:")
    print(f"  Rank (r): {config.lora_r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Dropout: {config.lora_dropout}")
    print(f"  Target modules: {config.target_modules}")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    print("\nLoRA applied successfully!")
    model.print_trainable_parameters()
    
    # Print final memory usage
    print("\nFinal GPU memory after LoRA:")
    print_gpu_memory()
    
    return model, tokenizer

def prepare_model_for_training(model):
    """Prepare model for training"""
    
    print("\nPreparing model for training...")
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  ✓ Gradient checkpointing enabled")
    
    # Enable input require grads for LoRA
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
        print("  ✓ Input gradients enabled")
    
    # Set model to training mode
    model.train()
    print("  ✓ Model set to training mode")
    
    return model

def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    """Save trained model and tokenizer"""
    
    print(f"\nSaving model to {output_dir}")
    
    # Save LoRA weights
    model.save_pretrained(output_dir)
    print("  ✓ LoRA weights saved")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print("  ✓ Tokenizer saved")
    
    # Save training info
    import json
    from datetime import datetime
    
    training_info = {
        "model_name": model.config.name_or_path if hasattr(model.config, 'name_or_path') else "unknown",
        "training_completed": datetime.now().isoformat(),
        "lora_config": {
            "r": model.peft_config['default'].r,
            "lora_alpha": model.peft_config['default'].lora_alpha,
            "target_modules": model.peft_config['default'].target_modules,
            "lora_dropout": model.peft_config['default'].lora_dropout,
        }
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    print("  ✓ Training info saved")
    
    print("Model saving completed!")