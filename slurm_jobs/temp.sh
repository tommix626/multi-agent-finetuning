#!/bin/bash
#SBATCH --job-name=expert_trainer_run
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  # assuming dataloader might want more CPUs
#SBATCH --mem=32G          # adjust memory if needed
#SBATCH --time=12:00:00
#SBATCH --output=logs/expert_trainer_%j.out

# Load necessary modules
module load anaconda
module load cuda/11.6.0

# Activate your conda environment
conda activate multi-agent-finetuning

# (Optional) Print environment info for debugging
nvidia-smi
which python
python --version

# Create logs directory if it doesn't exist
mkdir -p logs
export PYTHONPATH=/home/xwang397/scr4_jeisner1/tomwang/multi-agent-finetuning

# Run your training script
python -u /home/xwang397/scr4_jeisner1/tomwang/multi-agent-finetuning/scripts/mixture_training.py --config /home/xwang397/scr4_jeisner1/tomwang/multi-agent-finetuning/configs/mixture_training_4.yml
