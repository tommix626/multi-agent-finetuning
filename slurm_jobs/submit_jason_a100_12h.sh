#!/bin/bash

# Usage: ./submit_jason.sh your_script.py [args...]
if [ $# -lt 1 ]; then
  echo "Usage: $0 <python_script.py> [args...]"
  exit 1
fi

PYTHON_SCRIPT=$1
shift
PYTHON_ARGS="$@"
JOB_NAME=$(basename "$PYTHON_SCRIPT" .py)

SLURM_SCRIPT=$(mktemp submit_${JOB_NAME}_XXXXXX.sh)

cat <<EOF > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/${JOB_NAME}_%j.out

module load anaconda
module load cuda/11.6.0

conda activate multi-agent-finetuning

nvidia-smi
which python
python --version

mkdir -p logs

python -u ${PYTHON_SCRIPT} ${PYTHON_ARGS}
EOF

echo "[SLURM] Submitting job: ${SLURM_SCRIPT}"
sbatch "$SLURM_SCRIPT"
