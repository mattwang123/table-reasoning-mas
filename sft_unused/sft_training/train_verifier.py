#!/usr/bin/env python3

import os
import sys
import torch
from config import config
from data_utils import prepare_datasets
from model_utils import load_model_and_tokenizer, prepare_model_for_training, save_model_and_tokenizer
from trainer_utils import setup_trainer, initialize_wandb

def main():
    print("🚀 VERIFIER TRAINING STARTED")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Training data: {config.train_file}")
    print(f"Output dir: {config.output_dir}")
    print(f"Multi-GPU: {torch.cuda.device_count()} GPUs available")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps * torch.cuda.device_count()}")
    
    # Initialize wandb
    if config.report_to == "wandb":
        initialize_wandb(config)
        print("✓ Wandb initialized")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    model = prepare_model_for_training(model)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)
    
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
    
    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model saved to: {config.output_dir}")

if __name__ == "__main__":
    main()