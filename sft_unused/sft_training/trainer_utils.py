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

def normalize_prediction(pred: str) -> str:
    """Extract final answer as lowercase string - ONLY used for evaluation"""
    text = pred.lower()

    # Handle multiple "Final Answer:" occurrences - take first one
    final_answers = re.findall(r'final answer\s*:?\s*(true|false|unknown)', text)
    if final_answers:
        return final_answers[0]

    # Layer 2: fuzzy match on trailing words
    last_part = " ".join(text.split()[-30:])
    last_part = re.sub(r'[^\w\s]', '', last_part)

    if "false" in last_part: return "false"
    if "true" in last_part: return "true"
    if "unknown" in last_part: return "unknown"

    # Layer 3: rule-based weak signals
    if re.search(r'(correct answer|conclusion|determination)\s*(is|:)?\s*false', text):
        return "false"
    if re.search(r'(correct answer|conclusion|determination)\s*(is|:)?\s*true', text):
        return "true"

    # Layer 4: frequency voting
    score = {"true": text.count("true"), "false": text.count("false"), "unknown": text.count("unknown")}
    if score["false"] > max(score["true"], score["unknown"]):
        return "false"
    if score["true"] > max(score["false"], score["unknown"]):
        return "true"
    if score["unknown"] > 0:
        return "unknown"

    return "unknown"

def extract_true_answer_from_label(label_text: str) -> str:
    """Extract the true answer from training label - NO normalization"""
    if "Final Answer:" in label_text:
        return label_text.split("Final Answer:")[-1].strip().lower()
    else:
        return label_text.strip().lower()

def safe_decode(tokenizer, token_ids, skip_special_tokens=True):
    """Safely decode tokens, handling out-of-range values"""
    try:
        if hasattr(token_ids, 'tolist'):
            token_list = token_ids.tolist()
        else:
            token_list = token_ids
        
        vocab_size = tokenizer.vocab_size
        valid_tokens = []
        
        for token_id in token_list:
            if isinstance(token_id, (int, np.integer)) and 0 <= token_id < vocab_size:
                valid_tokens.append(int(token_id))
            elif token_id == -100:  # Ignore label
                continue
            else:
                valid_tokens.append(tokenizer.pad_token_id)
        
        return tokenizer.decode(valid_tokens, skip_special_tokens=skip_special_tokens)
        
    except Exception as e:
        return "[Decode Error]"

def compute_metrics(eval_pred, tokenizer):
    """Compute evaluation metrics - normalize predictions, extract from labels"""
    predictions, labels = eval_pred
    
    if len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1)
    
    max_samples = 100
    if len(predictions) > max_samples:
        indices = np.random.choice(len(predictions), max_samples, replace=False)
        predictions = predictions[indices]
        labels = labels[indices]
    
    try:
        pred_answers = []
        true_answers = []
        
        for pred, label in zip(predictions, labels):
            # Normalize prediction using function
            pred_text = safe_decode(tokenizer, pred)
            pred_answer = normalize_prediction(pred_text)
            pred_answers.append(pred_answer)
            
            # Extract true answer from label (no normalization)
            label_text = safe_decode(tokenizer, label)
            true_answer = extract_true_answer_from_label(label_text)
            true_answers.append(true_answer)
        
        # String-based accuracy calculation
        exact_matches = [pred == true for pred, true in zip(pred_answers, true_answers)]
        accuracy = np.mean(exact_matches) if exact_matches else 0.0
        
        # True/False accuracy (exclude unknown)
        tf_matches = []
        for pred, true in zip(pred_answers, true_answers):
            if true in ["true", "false"]:
                tf_matches.append(pred == true)
        
        tf_accuracy = np.mean(tf_matches) if tf_matches else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "true_false_accuracy": tf_accuracy,
            "num_tf_samples": len(tf_matches),
            "num_total_samples": len(pred_answers),
            "pred_false_count": pred_answers.count("false"),
            "pred_true_count": pred_answers.count("true"),
            "pred_unknown_count": pred_answers.count("unknown"),
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
    
    torch.cuda.empty_cache()
    gc.collect()
    return metrics

class OptimizedTrainer(Trainer):
    """FIXED: Proper accuracy computation and cumulative Wandb logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_train_batch = None
        self.current_train_accuracy = 0.0  # FIXED: Store current accuracy
        self.all_detailed_samples = []     # FIXED: Accumulate all samples
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """FIXED: Compute accuracy every step"""
        # Store batch every 10 steps for detailed logging
        if self.state.global_step % 10 == 0:
            try:
                self.last_train_batch = {
                    'input_ids': inputs['input_ids'][:1].detach().cpu(),
                    'labels': inputs['labels'][:1].detach().cpu()
                }
            except Exception:
                self.last_train_batch = None
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # FIXED: Compute training accuracy EVERY step
        self.current_train_accuracy = self._compute_training_accuracy(model, inputs)
        
        # Clear memory every 10 steps
        if self.state.global_step % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        return loss
    
    def _compute_training_accuracy(self, model, inputs):
        """FIXED: Lightweight accuracy computation every step"""
        try:
            model.eval()
            input_ids = inputs['input_ids'][:1]  # Only 1 sample for speed
            labels = inputs['labels'][:1]
            
            with torch.no_grad():
                try:
                    sample_input = input_ids[0:1]
                    sample_labels = labels[0:1]
                    
                    # Extract true answer
                    label_text = safe_decode(self.processing_class, sample_labels[0])
                    true_answer = extract_true_answer_from_label(label_text)
                    
                    # Generate with minimal parameters for speed
                    full_input = safe_decode(self.processing_class, sample_input[0])
                    attention_mask = (sample_input != self.processing_class.pad_token_id).long()
                    
                    outputs = model.generate(
                        sample_input,
                        attention_mask=attention_mask,
                        max_new_tokens=15,          # Minimal for speed
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.2,
                        early_stopping=True,
                        pad_token_id=self.processing_class.pad_token_id,
                        eos_token_id=self.processing_class.eos_token_id,
                    )
                    
                    full_output = safe_decode(self.processing_class, outputs[0])
                    raw_response = full_output[len(full_input):].strip()
                    
                    # Stop at first "Final Answer:" to avoid repetition
                    if "Final Answer:" in raw_response:
                        parts = raw_response.split("Final Answer:")
                        if len(parts) > 1:
                            raw_response = f"Final Answer: {parts[1].split('.')[0].strip()}"
                    
                    pred_answer = normalize_prediction(raw_response)
                    
                    # Clear immediately
                    del outputs
                    torch.cuda.empty_cache()
                    
                    return 1.0 if pred_answer == true_answer else 0.0
                        
                except Exception:
                    return 0.0
            
        except Exception:
            return 0.0
        finally:
            model.train()
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluation with memory management"""
        torch.cuda.empty_cache()
        gc.collect()
        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        torch.cuda.empty_cache()
        gc.collect()
        return result
    
    def log(self, logs, start_time=None):
        """FIXED: Always include current training accuracy"""
        # FIXED: Always add current training accuracy
        logs["train_accuracy"] = self.current_train_accuracy
        
        # Call parent log
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # Detailed logging every 20 steps
        if (self.state.global_step > 0 and 
            self.state.global_step % 20 == 0 and
            (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
            self._log_detailed_samples()
    
    def _log_detailed_samples(self):
        """FIXED: Accumulate samples instead of overwriting"""
        try:
            if self.last_train_batch is None:
                return
                
            print(f"\n📝 DETAILED LOGGING AT STEP {self.state.global_step}")
            
            self.model.eval()
            torch.cuda.empty_cache()
            
            # Process training sample
            input_ids = self.last_train_batch['input_ids']
            labels = self.last_train_batch['labels']
            
            try:
                sample_input = input_ids[0:1]
                sample_labels = labels[0:1]
                
                device = next(self.model.parameters()).device
                sample_input = sample_input.to(device)
                sample_labels = sample_labels.to(device)
                
                # Extract information
                full_input = safe_decode(self.processing_class, sample_input[0])
                query, reasoning_pred, code_pred = self._extract_predictions(full_input)
                
                label_text = safe_decode(self.processing_class, sample_labels[0])
                true_answer_extracted = extract_true_answer_from_label(label_text)
                
                # Generate prediction
                with torch.no_grad():
                    attention_mask = (sample_input != self.processing_class.pad_token_id).long()
                    
                    outputs = self.model.generate(
                        sample_input,
                        attention_mask=attention_mask,
                        max_new_tokens=30,
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.2,
                        early_stopping=True,
                        pad_token_id=self.processing_class.pad_token_id,
                        eos_token_id=self.processing_class.eos_token_id,
                    )
                    
                    full_output = safe_decode(self.processing_class, outputs[0])
                    raw_response = full_output[len(full_input):].strip()
                    
                    # Stop at first "Final Answer:" to avoid repetition
                    if "Final Answer:" in raw_response:
                        parts = raw_response.split("Final Answer:")
                        if len(parts) > 1:
                            raw_response = f"Final Answer: {parts[1].split('.')[0].strip()}"
                    
                    pred_answer_normalized = normalize_prediction(raw_response)
                    is_correct = pred_answer_normalized == true_answer_extracted
                    
                    # Calculate loss
                    loss_outputs = self.model(input_ids=sample_input, labels=sample_labels)
                    sample_loss = loss_outputs.loss.item()
                    
                    print(f"   Sample: Query: {query[:100]}...")
                    print(f"   True: {true_answer_extracted} | Pred: {pred_answer_normalized}")
                    print(f"   Correct: {'✅' if is_correct else '❌'} | Loss: {sample_loss:.4f}")
                    
                    # FIXED: Add to accumulated samples list
                    new_sample = {
                        "step": self.state.global_step,
                        "sample_id": f"train_step_{self.state.global_step}",
                        "dataset": "training",
                        "query": query[:200],
                        "reasoning_pred": reasoning_pred,
                        "code_pred": code_pred,
                        "training_target": label_text[:100] + "..." if len(label_text) > 100 else label_text,
                        "true_answer": true_answer_extracted,
                        "model_output": raw_response,
                        "final_answer": pred_answer_normalized,
                        "correct": "✅" if is_correct else "❌",
                        "loss": f"{sample_loss:.4f}"
                    }
                    
                    # FIXED: Accumulate samples (keep last 50 to avoid memory issues)
                    self.all_detailed_samples.append(new_sample)
                    if len(self.all_detailed_samples) > 50:
                        self.all_detailed_samples.pop(0)  # Remove oldest
                    
                    # Clear immediately
                    del outputs
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   Error processing sample: {e}")
            
            # FIXED: Log accumulated samples to Wandb
            if wandb.run is not None and self.all_detailed_samples:
                table = wandb.Table(
                    columns=[
                        "Step", "Sample_ID", "Dataset", "Query", "Reasoning_Pred", 
                        "Code_Pred", "Training_Target", "True_Answer", "Model_Output", 
                        "Final_Answer", "Correct", "Loss"
                    ],
                    data=[[r["step"], r["sample_id"], r["dataset"], r["query"], r["reasoning_pred"],
                          r["code_pred"], r["training_target"], r["true_answer"], r["model_output"], 
                          r["final_answer"], r["correct"], r["loss"]] for r in self.all_detailed_samples]
                )
                
                # FIXED: Use different table name each time to avoid overwriting
                wandb.log({f"detailed_samples_step_{self.state.global_step}": table}, step=self.state.global_step)
                print(f"📤 Logged {len(self.all_detailed_samples)} accumulated samples to Wandb")
            
            self.model.train()
            
        except Exception as e:
            print(f"❌ Error in detailed logging: {e}")
            self.model.train()
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with detailed logging"""
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Detailed evaluation logging
        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            self._log_evaluation_samples()
        
        return result
    
    def _log_evaluation_samples(self):
        """Log evaluation samples"""
        try:
            if not hasattr(self, 'eval_dataset') or len(self.eval_dataset) == 0:
                return
                
            print(f"\n🎯 EVALUATION DETAILED LOGGING AT STEP {self.state.global_step}")
            
            self.model.eval()
            test_indices = random.sample(range(len(self.eval_dataset)), min(3, len(self.eval_dataset)))
            all_results = []
            
            for test_idx, idx in enumerate(test_indices):
                sample = self.eval_dataset[idx]
                
                input_ids = torch.tensor(sample['input_ids'], dtype=torch.long).unsqueeze(0)
                labels = torch.tensor(sample['labels'], dtype=torch.long).unsqueeze(0)
                
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                # Extract information
                full_input = safe_decode(self.processing_class, input_ids[0])
                query, reasoning_pred, code_pred = self._extract_predictions(full_input)
                
                label_text = safe_decode(self.processing_class, labels[0])
                true_answer_extracted = extract_true_answer_from_label(label_text)
                
                # Generate prediction
                try:
                    with torch.no_grad():
                        attention_mask = (input_ids != self.processing_class.pad_token_id).long()
                        
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=30,
                            temperature=0.1,
                            do_sample=True,
                            repetition_penalty=1.2,
                            early_stopping=True,
                            pad_token_id=self.processing_class.pad_token_id,
                            eos_token_id=self.processing_class.eos_token_id,
                        )
                        
                        full_output = safe_decode(self.processing_class, outputs[0])
                        raw_response = full_output[len(full_input):].strip()
                        
                        # Stop at first "Final Answer:"
                        if "Final Answer:" in raw_response:
                            parts = raw_response.split("Final Answer:")
                            if len(parts) > 1:
                                raw_response = f"Final Answer: {parts[1].split('.')[0].strip()}"
                        
                        pred_answer_normalized = normalize_prediction(raw_response)
                        is_correct = pred_answer_normalized == true_answer_extracted
                        
                        # Calculate loss
                        loss_outputs = self.model(input_ids=input_ids, labels=labels)
                        sample_loss = loss_outputs.loss.item()
                        
                        print(f"   Eval {test_idx+1}: Query: {query[:100]}...")
                        print(f"   True: {true_answer_extracted} | Pred: {pred_answer_normalized}")
                        print(f"   Correct: {'✅' if is_correct else '❌'} | Loss: {sample_loss:.4f}")
                        
                        all_results.append({
                            "step": self.state.global_step,
                            "sample_id": f"eval_{idx}",
                            "dataset": "evaluation",
                            "query": query[:200],
                            "reasoning_pred": reasoning_pred,
                            "code_pred": code_pred,
                            "training_target": label_text[:100] + "..." if len(label_text) > 100 else label_text,
                            "true_answer": true_answer_extracted,
                            "model_output": raw_response,
                            "final_answer": pred_answer_normalized,
                            "correct": "✅" if is_correct else "❌",
                            "loss": f"{sample_loss:.4f}"
                        })
                        
                except Exception as e:
                    print(f"   Error in eval sample {test_idx}: {e}")
            
            # Log to Wandb with unique table name
            if wandb.run is not None and all_results:
                table = wandb.Table(
                    columns=[
                        "Step", "Sample_ID", "Dataset", "Query", "Reasoning_Pred", 
                        "Code_Pred", "Training_Target", "True_Answer", "Model_Output", 
                        "Final_Answer", "Correct", "Loss"
                    ],
                    data=[[r["step"], r["sample_id"], r["dataset"], r["query"], r["reasoning_pred"],
                          r["code_pred"], r["training_target"], r["true_answer"], r["model_output"], 
                          r["final_answer"], r["correct"], r["loss"]] for r in all_results]
                )
                
                wandb.log({f"evaluation_samples_step_{self.state.global_step}": table}, step=self.state.global_step)
                print(f"📤 Logged {len(all_results)} eval samples to Wandb")
            
        except Exception as e:
            print(f"❌ Error in evaluation logging: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    
    def _extract_predictions(self, full_text):
        """Extract query and agent predictions"""
        query = "Not found"
        reasoning_pred = "Not found"
        code_pred = "Not found"
        
        try:
            # Extract query
            if "Query:" in full_text:
                query_part = full_text.split("Query:")[1]
                if "===" in query_part:
                    query = query_part.split("===")[0].strip()
                else:
                    query = query_part[:300].strip()
            
            # Extract reasoning agent prediction
            if "Reasoning Agent Prediction:" in full_text:
                reasoning_part = full_text.split("Reasoning Agent Prediction:")[1]
                if "===" in reasoning_part:
                    reasoning_pred = reasoning_part.split("===")[0].strip()
                else:
                    reasoning_pred = reasoning_part[:50].strip()
            
            # Extract code agent prediction
            if "Code Agent Prediction:" in full_text:
                code_part = full_text.split("Code Agent Prediction:")[1]
                if "===" in code_part:
                    code_pred = code_part.split("===")[0].strip()
                else:
                    code_pred = code_part[:50].strip()
                    
        except Exception:
            pass
        
        return query, reasoning_pred, code_pred
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Save with memory cleanup"""
        torch.cuda.empty_cache()
        gc.collect()
        super().save_model(output_dir, _internal_call)
        torch.cuda.empty_cache()

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config):
    """Setup trainer: full text training, normalize for evaluation"""
    
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
        
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        prediction_loss_only=False,
        
        gradient_checkpointing=True,
        max_grad_norm=config.max_grad_norm,
        save_total_limit=config.save_total_limit,
        
        report_to=["wandb"] if config.report_to == "wandb" else [],
        run_name=config.wandb_name,
        logging_first_step=True,
        
        ddp_find_unused_parameters=False,
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
    
    trainer.eval_dataset = eval_dataset
    
    print("✓ FIXED Trainer setup:")
    print("  - Training accuracy computed every step (not 0 anymore)")
    print("  - Wandb tables accumulate samples (no overwriting)")
    print("  - Repetition handling in generation")
    print("  - Memory optimized for 3 GPUs")
    
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
            "effective_batch_size": config.per_device_train_batch_size * 3 * config.gradient_accumulation_steps,
            "num_epochs": config.num_train_epochs,
            "training_approach": "full_text_training_normalize_eval",
            "fixes": "accuracy_every_step_accumulative_wandb_tables",
        }
    )
    print("✓ Wandb initialized: Fixed accuracy and accumulative tables")