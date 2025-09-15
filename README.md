# REINFORCE Table Verifier (OpenRLHF)

REINFORCE training for table reasoning verification using OpenRLHF PPO with binary reward optimization.

## Overview

This system uses REINFORCE (implemented via PPO) to directly optimize for final answer correctness:

- **Training Method**: PPO (Proximal Policy Optimization) 
- **Reward Function**: Binary (1.0 if correct, 0.0 if incorrect)
- **Objective**: Maximize probability of correct final answers
- **Framework**: OpenRLHF for stable and efficient training

## Key Advantages

- ✅ **Direct Optimization**: Optimizes exactly what we want (final answer correctness)
- ✅ **Simple Reward**: Binary reward is interpretable and robust
- ✅ **Universal**: Works with any answer format (True/False, numbers, text)
- ✅ **Proven Framework**: Uses battle-tested OpenRLHF implementation
- ✅ **Efficient**: Optimized for 2 GPU setup with LoRA
- ✅ **Comprehensive Evaluation**: Detailed analysis and baseline comparison

## Complete Workflow

### Prerequisites
Complete your data collection first:
1. `generate_sample_indices.py` → sample indices
2. `collect_data_vllm.py` → reasoning + code agent outputs → `combined_results_*.jsonl`

### REINFORCE Training Pipeline

#### 1. Setup Environment
```bash
cd reinforce_table_verifier
pip install -r requirements.txt