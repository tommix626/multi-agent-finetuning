#!/bin/bash
#SBATCH --job-name=lora_ft
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lofa_ft_test%j.out

# Load modules
module load anaconda
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
python scripts/lora_base.py --config configs/experimentation_lora/lora_base_13.yaml