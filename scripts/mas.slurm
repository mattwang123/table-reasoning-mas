#!/bin/bash -l
#SBATCH --job-name=mas
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=a100
#SBATCH --account=a100acct
#SBATCH --output=mas_noplan%j.log
#SBATCH --mail-user=hwang302@jh.edu
#SBATCH --mail-type=END

module load conda

conda activate llama3

export CUDA_VISIBLE_DEVICES=0,1
# Run your Python script
python mas.py

