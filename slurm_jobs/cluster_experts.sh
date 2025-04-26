#!/bin/bash
#SBATCH --job-name=expert_cluster_training
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/expert_trainer_%j.out
#SBATCH --error=logs/expert_trainer_%j.err
# Load modules
module load anaconda
module load cuda/11.6.0

# Activate your environment
conda activate multi-agent-finetuning

mkdir -p logs

# Sanity check
nvidia-smi
which python
python --version

# Launch training
srun python scripts/train_expert_cluster.py --config configs/cluster_training.yaml
