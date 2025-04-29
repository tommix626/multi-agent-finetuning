#!/bin/bash
#SBATCH --job-name=lora_ft
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lofa_ft_%j.out

source ~/.bashrc
export PATH=/home/cs601-lwang216/.conda/envs/final_nlp/bin:$PATH
export PYTHONPATH=/home/cs601-lwang216/.conda/envs/final_nlp/lib/python3.10.13/site-packages:$PYTHONPATH

# Load modules
module load conda
module load cuda/12.5.0

# Activate your environment
conda activate multi-agent-finetuning
#final_nlp

# Check if CUDA available
nvidia-smi || echo "nvidia-smi failed"
which python
python --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Launch training
python ../scripts/lora_base.py --device cuda --rank 48