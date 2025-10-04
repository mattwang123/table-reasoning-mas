import json
import os
import random
from typing import List, Dict, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer

class DataFormatter:
    """Clean data formatter with space for advanced features"""
    
    def __init__(self, prompt_file: str = "prompts/verifier_prompt.txt"):
        """
        Initialize formatter
        
        Args:
            prompt_file: Path to prompt template file
        """
        self.prompt_template = self._load_prompt(prompt_file)
        
        # Placeholder for advanced features
        self.use_neurosymbolic = False
        self.use_curriculum = False
        self.curriculum_strategy = None
    
    def _load_prompt(self, prompt_file: str) -> str:
        """Load prompt template from file"""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"❌ Prompt file not found: {prompt_file}")
            print("Please create the prompt file or check the path")
            raise
        except Exception as e:
            print(f"❌ Error loading prompt: {e}")
            raise
    
    def load_database(self, database_file: str) -> List[Dict]:
        """Load combined analyzed database"""
        print(f"📊 Loading database: {database_file}")
        
        if not os.path.exists(database_file):
            raise FileNotFoundError(f"Database file not found: {database_file}")
        
        samples = []
        skipped = 0
        
        with open(database_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    
                    # Check required fields
                    required_fields = ["query", "table_text", "label", "reasoning_output", 
                                     "reasoning_prediction", "code_prediction"]
                    if not all(field in sample for field in required_fields):
                        print(f"⚠️  Missing required fields in sample {line_num}")
                        skipped += 1
                        continue
                    
                    samples.append(sample)
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON error on line {line_num}: {e}")
                    skipped += 1
                    continue
                except Exception as e:
                    print(f"❌ Processing error on line {line_num}: {e}")
                    skipped += 1
                    continue
        
        print(f"✅ Loaded {len(samples)} samples")
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} samples due to errors")
        
        # Show basic stats
        if samples:
            self._show_database_stats(samples)
        
        return samples
    
    def _show_database_stats(self, samples: List[Dict]):
        """Show basic database statistics"""
        total = len(samples)
        
        # Agreement analysis
        agreement_count = sum(1 for s in samples if s.get("methods_agree", False))
        agreement_rate = agreement_count / total if total > 0 else 0
        
        # Sample type distribution
        sample_types = {}
        for sample in samples:
            sample_type = sample.get("sample_type", "unknown")
            sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
        
        print(f"📈 Database Stats:")
        print(f"   Total samples: {total}")
        print(f"   Agreement rate: {agreement_rate:.3f}")
        print(f"   Sample types: {sample_types}")
    
    def create_training_sample(self, sample: Dict) -> Dict:
        """Create training sample from database entry"""
        
        try:
            # Format table
            table_str = self._format_table(sample["table_text"])
            
            # Get query
            query = sample["query"]
            
            # Convert predictions to text format
            reasoning_pred = self._convert_prediction(sample["reasoning_prediction"])
            code_pred = self._convert_prediction(sample["code_prediction"])
            target_answer = self._convert_prediction(sample["label"])
            
            # Clean expert outputs
            reasoning_output = self._clean_text(sample.get("reasoning_output", "No analysis provided"))
            extracted_code = self._clean_text(sample.get("extracted_code", "No code generated"))
            code_output = self._clean_text(sample.get("code_output", "No output"))
            code_error = sample.get("code_error", "None")
            
            # Handle code error formatting
            if not code_error or code_error == "None" or code_error == "":
                code_error = "No errors"
            
            # Create prompt using loaded template
            prompt = self.prompt_template.format(
                table=table_str,
                query=query,
                reasoning_output=reasoning_output,
                reasoning_prediction=reasoning_pred,
                extracted_code=extracted_code,
                code_output=code_output,
                code_error=code_error,
                code_prediction=code_pred
            )
            
            # Target format (NO ground truth in prompt!)
            target = f"Final Answer: {target_answer}"
            
            # Basic training sample
            training_sample = {
                "prompt": prompt,
                "target": target,
                "query": query,
                "sample_id": sample.get("sample_id", -1),
                
                # Keep metadata for potential advanced features
                "sample_type": sample.get("sample_type", "unknown"),
                "difficulty_score": sample.get("difficulty_score", 1.0),
                "experts_agree": sample.get("methods_agree", True),
                "reasoning_correct": sample.get("reasoning_correct", False),
                "code_correct": sample.get("code_correct", False),
            }
            
            # TODO: Add neurosymbolic enhancements here
            if self.use_neurosymbolic:
                training_sample = self._apply_neurosymbolic_enhancements(training_sample, sample)
            
            return training_sample
            
        except Exception as e:
            print(f"❌ Error processing sample {sample.get('sample_id', 'unknown')}: {e}")
            return None
    
    def create_training_dataset(self, samples: List[Dict], 
                              sample_filter: str = "all") -> List[Dict]:
        """Create training dataset with optional filtering"""
        
        print(f"🎯 Creating training dataset (filter: {sample_filter})")
        
        # Apply filtering
        if sample_filter == "disagreement_only":
            filtered_samples = [s for s in samples if not s.get("methods_agree", True)]
            print(f"   Filtered to {len(filtered_samples)} disagreement samples")
        elif sample_filter == "hard_only":
            filtered_samples = [s for s in samples if s.get("difficulty_score", 1.0) >= 2.0]
            print(f"   Filtered to {len(filtered_samples)} hard samples")
        elif sample_filter == "both_wrong_only":
            filtered_samples = [s for s in samples if s.get("sample_type") == "both_wrong"]
            print(f"   Filtered to {len(filtered_samples)} both_wrong samples")
        else:  # "all"
            filtered_samples = samples
            print(f"   Using all {len(filtered_samples)} samples")
        
        # Convert to training format
        training_samples = []
        skipped = 0
        
        for sample in filtered_samples:
            training_sample = self.create_training_sample(sample)
            if training_sample is not None:
                training_samples.append(training_sample)
            else:
                skipped += 1
        
        print(f"✅ Created {len(training_samples)} training samples")
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} samples due to errors")
        
        # Show sample distribution
        if training_samples:
            self._show_training_stats(training_samples)
        
        # TODO: Add curriculum learning here
        if self.use_curriculum:
            training_samples = self._apply_curriculum_learning(training_samples)
        
        return training_samples
    
    def _show_training_stats(self, training_samples: List[Dict]):
        """Show training dataset statistics"""
        sample_types = {}
        for sample in training_samples:
            sample_type = sample["sample_type"]
            sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
        
        print(f"📊 Training Dataset Stats:")
        print(f"   Sample type distribution: {sample_types}")
        
        # Show example
        if training_samples:
            example = training_samples[0]
            print(f"\n📋 Example Training Sample:")
            print(f"   Query: {example['query'][:100]}...")
            print(f"   Target: {example['target']}")
            print(f"   Sample type: {example['sample_type']}")
            print(f"   Experts agree: {example['experts_agree']}")
    
    def create_train_eval_split(self, training_samples: List[Dict], 
                               test_split: float = 0.05, 
                               seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """Split training data into train and eval sets"""
        
        # Stratify by sample type for balanced evaluation
        sample_types = {}
        for sample in training_samples:
            sample_type = sample["sample_type"]
            if sample_type not in sample_types:
                sample_types[sample_type] = []
            sample_types[sample_type].append(sample)
        
        random.seed(seed)
        train_samples = []
        eval_samples = []
        
        # Split each sample type proportionally
        for sample_type, type_samples in sample_types.items():
            random.shuffle(type_samples)
            split_idx = int(len(type_samples) * (1 - test_split))
            train_samples.extend(type_samples[:split_idx])
            eval_samples.extend(type_samples[split_idx:])
        
        # Final shuffle
        random.shuffle(train_samples)
        random.shuffle(eval_samples)
        
        print(f"📚 Data Split:")
        print(f"   Train samples: {len(train_samples)}")
        print(f"   Eval samples: {len(eval_samples)}")
        
        return train_samples, eval_samples
    
    def save_datasets(self, train_samples: List[Dict], eval_samples: List[Dict], 
                     output_dir: str):
        """Save train and eval datasets"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train dataset
        train_file = f"{output_dir}/train_dataset.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        # Save eval dataset
        eval_file = f"{output_dir}/eval_dataset.jsonl"
        with open(eval_file, 'w', encoding='utf-8') as f:
            for sample in eval_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Datasets saved:")
        print(f"   Train: {train_file}")
        print(f"   Eval: {eval_file}")
    
    # Helper functions
    def _format_table(self, table_data) -> str:
        """Format table data into readable string"""
        if isinstance(table_data, list) and len(table_data) > 0:
            if isinstance(table_data[0], list):
                headers = table_data[0]
                rows = table_data[1:]
                
                formatted = f"Headers: {', '.join(str(h) for h in headers)}\n"
                for i, row in enumerate(rows):
                    formatted += f"Row {i+1}: {', '.join(str(cell) for cell in row)}\n"
                return formatted.strip()
        
        return json.dumps(table_data)
    
    def _convert_prediction(self, prediction) -> str:
        """Convert prediction to text format"""
        if prediction == 1:
            return "True"
        elif prediction == 0:
            return "False"
        elif prediction == 2:
            return "Unknown"
        else:
            # Handle string predictions or other formats
            pred_str = str(prediction).lower().strip()
            if pred_str in ["true", "1"]:
                return "True"
            elif pred_str in ["false", "0"]:
                return "False"
            else:
                return str(prediction)
    
    def _clean_text(self, text: str) -> str:
        """Clean and truncate text for better formatting"""
        if not text:
            return "No content"
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    
    # Placeholder methods for advanced features
    def _apply_neurosymbolic_enhancements(self, training_sample: Dict, 
                                        original_sample: Dict) -> Dict:
        """
        TODO: Apply neurosymbolic enhancements
        
        Ideas:
        - Add confidence scores for each expert
        - Enhance prompt with explicit reasoning strategies
        - Add expert reliability indicators
        """
        # Placeholder for future implementation
        return training_sample
    
    def _apply_curriculum_learning(self, training_samples: List[Dict]) -> List[Dict]:
        """
        TODO: Apply curriculum learning
        
        Ideas:
        - Sort by difficulty (easy -> medium -> hard)
        - Create separate datasets by sample type
        - Implement progressive training strategy
        """
        # Placeholder for future implementation
        return training_samples

# Main function for easy usage
def create_training_data_from_database(database_file: str, 
                                     output_dir: str = "data/training",
                                     prompt_file: str = "prompts/verifier_prompt.txt",
                                     sample_filter: str = "all",
                                     test_split: float = 0.05):
    """
    Main function to create training data from database
    
    Args:
        database_file: Path to combined analyzed database
        output_dir: Output directory for training datasets
        prompt_file: Path to prompt template file
        sample_filter: Filter to apply ("all", "disagreement_only", "hard_only", "both_wrong_only")
        test_split: Fraction of data to use for evaluation
    """
    
    print("🎯 Creating Training Data from Database")
    print("=" * 50)
    
    # Initialize formatter
    formatter = DataFormatter(prompt_file)
    
    # Load database
    samples = formatter.load_database(database_file)
    
    if not samples:
        print("❌ No samples loaded!")
        return
    
    # Create training dataset
    training_samples = formatter.create_training_dataset(samples, sample_filter)
    
    if not training_samples:
        print("❌ No training samples created!")
        return
    
    # Split into train/eval
    train_samples, eval_samples = formatter.create_train_eval_split(training_samples, test_split)
    
    # Save datasets
    formatter.save_datasets(train_samples, eval_samples, output_dir)
    
    print(f"\n✅ Training data creation completed!")
    print(f"   Database: {database_file}")
    print(f"   Prompt: {prompt_file}")
    print(f"   Filter: {sample_filter}")
    print(f"   Output: {output_dir}")

if __name__ == "__main__":
    # Example usage
    create_training_data_from_database(
        database_file="data/combined/combined_analyzed_database_20250101.jsonl",
        output_dir="data/training",
        sample_filter="all"
    )