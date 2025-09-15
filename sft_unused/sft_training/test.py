# training/check_lengths.py
import sys
sys.path.append('.')

from training.config import config
from training.data_utils import load_and_process_data
from training.model_utils import load_model_and_tokenizer

# Load data and tokenizer
samples = load_and_process_data(config.train_file)[:100]
_, tokenizer = load_model_and_tokenizer(config)

# Check lengths
lengths = []
for sample in samples:
    full_text = sample["prompt"] + sample["target"]
    tokens = tokenizer(full_text)
    lengths.append(len(tokens["input_ids"]))

print(f"Token lengths for first 100 samples:")
print(f"Min: {min(lengths)}, Max: {max(lengths)}")
print(f"Average: {sum(lengths)/len(lengths):.1f}")
print(f"95th percentile: {sorted(lengths)[int(0.95*len(lengths))]}")