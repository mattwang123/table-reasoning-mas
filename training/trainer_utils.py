import numpy as np
import torch
import gc
import re
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import wandb
import random
from typing import List, Dict

def normalize_prediction(pred: str) -> int:
    """Your normalization function - exact copy"""
    text = pred.lower()

    # Layer 1: strict pattern match
    match = re.search(r'final answer\s*:?\s*(true|false|unknown)', text)
    if match:
        return {"true": 1, "false": 0, "unknown": 2}[match.group(1)]

    # Layer 2: fuzzy match on trailing words
    last_part = " ".join(text.split()[-30:])
    last_part = re.sub(r'[^\w\s]', '', last_part)

    if "false" in last_part: return 0
    if "true" in last_part: return 1
    if "unknown" in last_part: return 2

    # Layer 3: rule-based weak signals
    if re.search(r'(correct answer|conclusion|determination)\s*(is|:)?\s*false', text):
        return 0
    if re.search(r'(correct answer|conclusion|determination)\s*(is|:)?\s*true', text):
        return 1

    # Layer 4: frequency voting
    score = {"true": text.count("true"), "false": text.count("false"), "unknown": text.count("unknown")}
    if score["false"] > max(score["true"], score["unknown"]):
        return 0
    if score["true"] > max(score["false"], score["unknown"]):
        return 1
    if score["unknown"] > 0:
        return 2

    # Completely uncertain
    return 2

def prediction_to_text(pred_num: int) -> str:
    """Convert numeric prediction to text"""
    return {0: "False", 1: "True", 2: "Unknown"}.get(pred_num, "Unknown")

def compute_metrics(eval_pred, tokenizer):
    """Compute evaluation metrics using your normalization function"""
    predictions, labels = eval_pred
    
    # Handle predictions properly - take argmax if logits
    if len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1)
    
    # Limit samples to prevent memory issues
    max_samples = 50
    if len(predictions) > max_samples:
        indices = np.random.choice(len(predictions), max_samples, replace=False)
        predictions = predictions[indices]
        labels = labels[indices]
    
    try:
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Use your normalization function for predictions
        pred_normalized = []
        label_normalized = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            # Normalize prediction using your function
            pred_num = normalize_prediction(pred)
            pred_normalized.append(pred_num)
            
            # Extract true label
            if "Final Answer:" in label:
                true_answer = label.split("Final Answer:")[-1].strip()
            else:
                true_answer = label.strip()
            
            # Convert true answer to number
            true_answer_lower = true_answer.lower().strip()
            if true_answer_lower == "true":
                label_num = 1
            elif true_answer_lower == "false":
                label_num = 0
            else:
                label_num = 2
            
            label_normalized.append(label_num)
        
        # Calculate accuracy
        exact_matches = [pred == label for pred, label in zip(pred_normalized, label_normalized)]
        accuracy = np.mean(exact_matches) if exact_matches else 0.0
        
        # Calculate True/False accuracy (exclude Unknown)
        tf_matches = []
        for pred, label in zip(pred_normalized, label_normalized):
            if label in [0, 1]:  # Only True/False, not Unknown
                tf_matches.append(pred == label)
        
        tf_accuracy = np.mean(tf_matches) if tf_matches else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "true_false_accuracy": tf_accuracy,
            "num_tf_samples": len(tf_matches),
            "num_total_samples": len(pred_normalized),
            "pred_false_count": sum(1 for p in pred_normalized if p == 0),
            "pred_true_count": sum(1 for p in pred_normalized if p == 1),
            "pred_unknown_count": sum(1 for p in pred_normalized if p == 2),
        }
        
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        metrics = {
            "accuracy": 0.0,
            "true_false_accuracy": 0.0,
            "num_tf_samples": 0,
            "num_total_samples": 0,
            "pred_false_count": 0,
            "pred_true_count": 0,
            "pred_unknown_count": 0,
        }
    
    # Clear memory after metrics computation
    torch.cuda.empty_cache()
    gc.collect()
    
    return metrics

class OptimizedTrainer(Trainer):
    """Trainer with memory management and frequent detailed logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_samples_every = 2  # CHANGED: Log every 2 steps instead of 5
        self.last_train_batch = None
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with memory clearing and batch storage"""
        # Store batch for detailed logging
        self.last_train_batch = {
            'input_ids': inputs['input_ids'].clone().detach(),
            'labels': inputs['labels'].clone().detach()
        }
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Clear memory every 3 steps
        if self.state.global_step % 3 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        return loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluation with memory management"""
        torch.cuda.empty_cache()
        gc.collect()
        
        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return result
    
    def log(self, logs, start_time=None):
        """Enhanced logging with frequent detailed samples"""
        # Call parent log method
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # Log detailed samples every 2 steps (only from main process)
        if (self.state.global_step > 0 and 
            self.state.global_step % self.log_samples_every == 0 and
            (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
            
            self._log_detailed_samples()
        
        # Log memory usage every 20 steps
        if self.state.global_step % 20 == 0:
            self._log_gpu_usage()
    
    def _log_detailed_samples(self):
        """Log detailed sample predictions with extensive console output"""
        try:
            print(f"\n{'='*60}")
            print(f"🔍 DETAILED LOGGING AT STEP {self.state.global_step}")
            print(f"{'='*60}")
            
            self.model.eval()
            all_tables = {}
            
            # 1. Process training samples
            print("📚 Processing training batch...")
            if self.last_train_batch is not None:
                train_table_data = self._process_training_batch_for_logging()
                print(f"   → Got {len(train_table_data)} training samples")
                
                if train_table_data:
                    all_tables["training_samples"] = wandb.Table(
                        columns=[
                            "Sample_ID", "Dataset", "Query", "Table_Preview", "Reasoning_Analysis", 
                            "Code_Analysis", "True_Answer", "Raw_Output", "Normalized_Pred", 
                            "Correct", "Loss"
                        ],
                        data=train_table_data
                    )
                    print("   ✅ Created training samples table")
                    
                    # Print training sample details to console
                    for i, row in enumerate(train_table_data):
                        print(f"\n   📝 TRAINING SAMPLE {i+1}:")
                        print(f"      Query: {row[2][:100]}...")
                        print(f"      True Answer: {row[6]}")
                        print(f"      Raw Output: {row[7]}")
                        print(f"      Normalized: {row[8]}")
                        print(f"      Correct: {row[9]} | Loss: {row[10]}")
            else:
                print("   ❌ No training batch available")
            
            # 2. Process evaluation samples
            print("\n🎯 Processing evaluation samples...")
            if hasattr(self, 'eval_dataset') and len(self.eval_dataset) > 0:
                eval_table_data = self._process_eval_samples_for_logging()
                print(f"   → Got {len(eval_table_data)} eval samples")
                
                if eval_table_data:
                    all_tables["evaluation_samples"] = wandb.Table(
                        columns=[
                            "Sample_ID", "Dataset", "Query", "Table_Preview", "Reasoning_Analysis", 
                            "Code_Analysis", "True_Answer", "Raw_Output", "Normalized_Pred", 
                            "Correct", "Loss"
                        ],
                        data=eval_table_data
                    )
                    print("   ✅ Created evaluation samples table")
                    
                    # Print evaluation sample details to console
                    for i, row in enumerate(eval_table_data):
                        print(f"\n   📊 EVALUATION SAMPLE {i+1}:")
                        print(f"      Query: {row[2][:100]}...")
                        print(f"      True Answer: {row[6]}")
                        print(f"      Raw Output: {row[7]}")
                        print(f"      Normalized: {row[8]}")
                        print(f"      Correct: {row[9]} | Loss: {row[10]}")
            else:
                print("   ❌ No evaluation dataset available")
            
            # 3. Log to Wandb
            if wandb.run is not None and all_tables:
                print(f"\n📤 Logging to Wandb...")
                wandb.log(all_tables, step=self.state.global_step)
                total_samples = sum(len(table.data) for table in all_tables.values())
                print(f"   ✅ Successfully logged {total_samples} samples to Wandb")
                
                for table_name, table in all_tables.items():
                    print(f"      - {table_name}: {len(table.data)} rows")
            else:
                print(f"\n❌ Wandb logging failed:")
                print(f"   - Wandb run available: {wandb.run is not None}")
                print(f"   - Tables created: {len(all_tables)}")
            
            print(f"{'='*60}\n")
            self.model.train()
            
        except Exception as e:
            print(f"❌ ERROR in detailed logging: {e}")
            import traceback
            traceback.print_exc()
            self.model.train()
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    
    def _process_training_batch_for_logging(self):
        """Process training batch with better error handling"""
        try:
            if self.last_train_batch is None:
                return []
            
            input_ids = self.last_train_batch['input_ids']
            labels = self.last_train_batch['labels']
            
            if input_ids.numel() == 0 or labels.numel() == 0:
                print("      ⚠️ Empty batch detected")
                return []
            
            # Take first 2 samples from batch
            num_samples = min(2, len(input_ids))
            table_data = []
            
            for i in range(num_samples):
                try:
                    sample_input = input_ids[i:i+1]
                    sample_labels = labels[i:i+1]
                    
                    sample_info = self._process_single_sample_for_logging(
                        sample_input, sample_labels, f"train_step_{self.state.global_step}_sample_{i}", "training"
                    )
                    if sample_info:
                        table_data.append(sample_info)
                        
                except Exception as e:
                    print(f"      ❌ Error processing training sample {i}: {e}")
                    continue
            
            return table_data
            
        except Exception as e:
            print(f"      ❌ Error in training batch processing: {e}")
            return []
    
    def _process_eval_samples_for_logging(self):
        """Process evaluation samples"""
        try:
            num_samples = min(2, len(self.eval_dataset))
            sample_indices = random.sample(range(len(self.eval_dataset)), num_samples)
            
            table_data = []
            
            for idx in sample_indices:
                try:
                    sample = self.eval_dataset[idx]
                    input_ids = torch.tensor(sample['input_ids']).unsqueeze(0)
                    labels = torch.tensor(sample['labels']).unsqueeze(0)
                    
                    sample_info = self._process_single_sample_for_logging(
                        input_ids, labels, f"eval_step_{self.state.global_step}_sample_{idx}", "evaluation"
                    )
                    if sample_info:
                        table_data.append(sample_info)
                        
                except Exception as e:
                    print(f"      ❌ Error processing eval sample {idx}: {e}")
                    continue
            
            return table_data
            
        except Exception as e:
            print(f"      ❌ Error in eval sample processing: {e}")
            return []
    
    def _process_single_sample_for_logging(self, input_ids, labels, sample_id, dataset_type):
        """Process single sample with comprehensive error handling"""
        try:
            # Move to device
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Ensure proper dimensions
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
            # Decode input
            try:
                full_input = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"         ❌ Input decode error for {sample_id}: {e}")
                return None
            
            # Extract components
            components = self._extract_input_components(full_input)
            
            # Generate model output
            raw_response = "[Generation failed]"
            try:
                with torch.no_grad():
                    # Create attention mask
                    attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                    
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=15,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    raw_response = full_output[len(full_input):].strip()
                    
                    if not raw_response:
                        raw_response = "[Empty generation]"
                        
            except Exception as e:
                print(f"         ❌ Generation error for {sample_id}: {e}")
                raw_response = f"[Gen Error: {str(e)[:30]}]"
            
            # Normalize prediction
            try:
                normalized_pred_num = normalize_prediction(raw_response)
                normalized_pred_text = prediction_to_text(normalized_pred_num)
            except Exception as e:
                print(f"         ❌ Normalization error for {sample_id}: {e}")
                normalized_pred_num = 2
                normalized_pred_text = "Unknown"
            
            # Get true answer
            try:
                label_text = self.tokenizer.decode(labels[0], skip_special_tokens=True)
                if "Final Answer:" in label_text:
                    true_answer = label_text.split("Final Answer:")[-1].strip()
                else:
                    true_answer = label_text.strip()
            except Exception as e:
                print(f"         ❌ Label decode error for {sample_id}: {e}")
                true_answer = "[Label error]"
            
            # Convert true answer to number
            true_answer_lower = true_answer.lower().strip()
            if true_answer_lower == "true":
                true_num = 1
            elif true_answer_lower == "false":
                true_num = 0
            else:
                true_num = 2
            
            # Calculate loss
            try:
                with torch.no_grad():
                    loss_outputs = self.model(input_ids=input_ids, labels=labels)
                    sample_loss = loss_outputs.loss.item()
            except Exception as e:
                print(f"         ❌ Loss calculation error for {sample_id}: {e}")
                sample_loss = 0.0
            
            # Check correctness
            is_correct = normalized_pred_num == true_num
            
            # Cleanup
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            
            return [
                sample_id,
                dataset_type,
                components['query'][:100] + "..." if len(components['query']) > 100 else components['query'],
                components['table_preview'][:150] + "..." if len(components['table_preview']) > 150 else components['table_preview'],
                components['reasoning'][:150] + "..." if len(components['reasoning']) > 150 else components['reasoning'],
                components['code'][:150] + "..." if len(components['code']) > 150 else components['code'],
                true_answer,
                raw_response,
                f"{normalized_pred_text} ({normalized_pred_num})",
                "✅" if is_correct else "❌",
                f"{sample_loss:.4f}"
            ]
            
        except Exception as e:
            print(f"         ❌ Sample processing error for {sample_id}: {e}")
            return None
    
    def _extract_input_components(self, full_text):
        """Extract components from input text"""
        components = {
            'query': 'Not found',
            'table_preview': 'Not found',
            'reasoning': 'Not found',
            'code': 'Not found'
        }
        
        try:
            # Extract query
            if "Query:" in full_text:
                query_part = full_text.split("Query:")[1]
                if "===" in query_part:
                    components['query'] = query_part.split("===")[0].strip()
                else:
                    components['query'] = query_part[:200].strip()
            
            # Extract table preview
            if "Table:" in full_text:
                table_part = full_text.split("Table:")[1]
                if "Query:" in table_part:
                    table_text = table_part.split("Query:")[0].strip()
                    lines = table_text.split('\n')[:2]  # Just 2 lines for preview
                    components['table_preview'] = '\n'.join(lines)
                else:
                    components['table_preview'] = table_part[:200]
            
            # Extract reasoning
            if "=== Reasoning Agent Analysis ===" in full_text:
                reasoning_part = full_text.split("=== Reasoning Agent Analysis ===")[1]
                if "=== Code Agent Analysis ===" in reasoning_part:
                    components['reasoning'] = reasoning_part.split("=== Code Agent Analysis ===")[0].strip()
                else:
                    components['reasoning'] = reasoning_part[:200].strip()
            
            # Extract code
            if "Generated Code:" in full_text:
                code_part = full_text.split("Generated Code:")[1]
                if "=== Your Task ===" in code_part:
                    components['code'] = code_part.split("=== Your Task ===")[0].strip()
                else:
                    components['code'] = code_part[:200].strip()
                    
        except Exception as e:
            print(f"Error extracting components: {e}")
        
        return components
    
    def _log_gpu_usage(self):
        """Log GPU memory usage"""
        try:
            if (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0):
                return
                
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
                
                print(f"💾 Step {self.state.global_step}: GPU - {gpu_allocated:.2f}GB allocated, {gpu_cached:.2f}GB cached")
                
                if wandb.run is not None:
                    wandb.log({
                        "gpu_allocated_gb": gpu_allocated,
                        "gpu_cached_gb": gpu_cached,
                    }, step=self.state.global_step)
        except Exception as e:
            print(f"Error logging GPU usage: {e}")
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Save with memory cleanup"""
        torch.cuda.empty_cache()
        gc.collect()
        super().save_model(output_dir, _internal_call)
        torch.cuda.empty_cache()

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config):
    """Setup trainer with frequent detailed logging"""
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        
        logging_steps=1,  # CHANGED: Log every step
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        prediction_loss_only=False,
        
        report_to=["wandb"] if config.report_to == "wandb" else [],
        run_name=config.wandb_name,
        logging_first_step=True,
        
        ddp_find_unused_parameters=False,
        max_grad_norm=config.max_grad_norm,
        save_total_limit=config.save_total_limit,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = OptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )
    
    # Store datasets for logging
    trainer.eval_dataset = eval_dataset
    trainer.train_dataset = train_dataset
    
    print("✓ Trainer setup complete with frequent detailed logging (every 2 steps)")
    return trainer

def initialize_wandb(config):
    """Initialize Wandb logging"""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
        
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_name,
        config={
            "model_name": config.model_name,
            "max_seq_length": config.max_seq_length,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
            "num_epochs": config.num_train_epochs,
            "logging_frequency": "every_2_steps",
        }
    )
    print("✓ Wandb initialized with frequent logging")