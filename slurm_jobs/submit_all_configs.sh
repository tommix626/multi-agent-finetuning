#!/bin/bash

# Usage: ./submit_all_configs.sh --config_dir=path/to/configs

# Parse the argument
for arg in "$@"; do
  case $arg in
    --config_dir=*)
      CONFIG_DIR="${arg#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

# Check config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
  echo "Config directory $CONFIG_DIR does not exist!"
  exit 1
fi

# Create logs directory if needed
mkdir -p logs

# Loop over each yaml config file
for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
  CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
  
  sbatch --job-name="trainer_${CONFIG_NAME}" \
         -A jeisner1_gpu \
         --partition=a100 \
         --gres=gpu:1 \
         --nodes=1 \
         --ntasks-per-node=1 \
         --cpus-per-task=8 \
         --mem=32G \
         --time=12:00:00 \
         --output="logs/expert_trainer_${CONFIG_NAME}_%j.out" \
         --wrap="module load anaconda && module load cuda/11.6.0 && conda activate multi-agent-finetuning && nvidia-smi && which python && python --version && mkdir -p logs && python -u cluster_training_with_config.py --config \"$CONFIG_FILE\""
done
