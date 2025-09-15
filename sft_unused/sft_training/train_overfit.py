#!/usr/bin/env python3

import os
import sys
import torch
from config_overfit import config
from data_utils import prepare_datasets
from model_utils import load_model_and_tokenizer, prepare_model_for_training, save_model_and_tokenizer
from trainer_utils import setup_trainer, initialize_wandb

def main():
    # OVERFITTING TEST CONFIGURATION
    OVERFIT_TEST = True  # Set to False for normal training
    OVERFIT_SAMPLES = 10  # Number of samples to overfit on
    
    print("🚀 VERIFIER TRAINING STARTED")
    if OVERFIT_TEST:
        print("🧪 OVERFITTING TEST MODE")
        print(f"   Using only {OVERFIT_SAMPLES} samples for overfitting")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Training data: {config.train_file}")
    print(f"Output dir: {config.output_dir}")
    print(f"Multi-GPU: {torch.cuda.device_count()} GPUs available")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps * torch.cuda.device_count()}")
    
    # Modify config for overfitting test
    if OVERFIT_TEST:
        # Override config for overfitting
        config.num_train_epochs = 10  # More epochs for overfitting
        config.learning_rate = 5e-4   # Higher learning rate
        config.eval_steps = 5         # Evaluate more frequently
        config.logging_steps = 1      # Log every step
        config.save_steps = 50        # Save more frequently
        config.warmup_ratio = 0.0     # No warmup for small dataset
        
        print(f"🔧 Overfitting config:")
        print(f"   Epochs: {config.num_train_epochs}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Eval every: {config.eval_steps} steps")
    
    # Initialize wandb
    if config.report_to == "wandb":
        if OVERFIT_TEST:
            config.wandb_name = f"overfit-test-{OVERFIT_SAMPLES}samples"
            config.wandb_project = "verifier-overfit-test"
        initialize_wandb(config)
        print("✓ Wandb initialized")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    model = prepare_model_for_training(model)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)
    
    # OVERFIT: Use only small subset
    if OVERFIT_TEST:
        print(f"\n🧪 Creating overfitting subset...")
        print(f"   Original train size: {len(train_dataset)}")
        print(f"   Original eval size: {len(eval_dataset)}")
        
        # Use same small subset for both train and eval (to check overfitting)
        overfit_indices = list(range(min(OVERFIT_SAMPLES, len(train_dataset))))
        
        # Create subset datasets
        train_subset = torch.utils.data.Subset(train_dataset, overfit_indices)
        eval_subset = torch.utils.data.Subset(train_dataset, overfit_indices)  # Same data for eval!
        
        print(f"   Overfit train size: {len(train_subset)}")
        print(f"   Overfit eval size: {len(eval_subset)}")
        print(f"   ✅ Using SAME samples for train and eval (should overfit to 100%)")
        
        train_dataset = train_subset
        eval_dataset = eval_subset
    
    # Setup trainer
    print("\n🏋️ Setting up trainer...")
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, config)
    
    # Print training info
    print(f"\nTraining configuration:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size per GPU: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps * torch.cuda.device_count()}")
    
    if OVERFIT_TEST:
        print(f"\n🎯 OVERFITTING EXPECTATIONS:")
        print(f"   ✅ Training accuracy should reach 100%")
        print(f"   ✅ Eval accuracy should reach 100% (same data)")
        print(f"   ✅ Loss should decrease to near 0")
        print(f"   ✅ Model should memorize all {OVERFIT_SAMPLES} samples")
    
    # Start training
    print("\n🎯 Starting training...")
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    save_model_and_tokenizer(trainer.model, tokenizer, config.output_dir)
    
    # Final evaluation
    print("\n📈 Final evaluation...")
    eval_results = trainer.evaluate()
    print("Final results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    
    # Overfitting test results
    if OVERFIT_TEST:
        print(f"\n🧪 OVERFITTING TEST RESULTS:")
        final_accuracy = eval_results.get('eval_accuracy', 0.0)
        final_loss = eval_results.get('eval_loss', float('inf'))
        
        if final_accuracy >= 0.95:
            print(f"   ✅ SUCCESS: Accuracy = {final_accuracy:.3f} (≥95%)")
        else:
            print(f"   ❌ FAILED: Accuracy = {final_accuracy:.3f} (<95%)")
            
        if final_loss <= 0.1:
            print(f"   ✅ SUCCESS: Loss = {final_loss:.4f} (≤0.1)")
        else:
            print(f"   ❌ FAILED: Loss = {final_loss:.4f} (>0.1)")
            
        print(f"\n🔍 If overfitting failed, check:")
        print(f"   - Learning rate (try higher: 1e-3)")
        print(f"   - More epochs (try 20-50)")
        print(f"   - Gradient accumulation (try lower)")
        print(f"   - Model capacity (LoRA rank)")
    
    print("\n✅ TRAINING COMPLETED!")
    print(f"Model saved to: {config.output_dir}")

if __name__ == "__main__":
    main()