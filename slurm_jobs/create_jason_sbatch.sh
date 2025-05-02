#!/bin/bash

# Default SBATCH values
JOB_NAME=""
TIME="12:00:00"
MEM="32G"
SCRIPT_ONLY=false

print_usage() {
  echo "Usage: $0 [-j NAME] [-t TIME] [-m MEM] [-s] <python_script.py> [args...]"
  echo ""
  echo "Options:"
  echo "  -j, --job-name NAME     Set SLURM job name"
  echo "  -t, --time TIME         Set SLURM time limit (default: 12:00:00)"
  echo "  -m, --mem MEM           Set SLURM memory (default: 32G)"
  echo "  -s, --script-only       Only generate SLURM script, do not submit"
  echo ""
  echo "Example:"
  echo "  $0 -j train_exp1 -t 08:00:00 -m 16G -s train.py --config conf.yaml"
}

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    -t|--time)
      TIME="$2"
      shift 2
      ;;
    -m|--mem)
      MEM="$2"
      shift 2
      ;;
    -s|--script-only)
      SCRIPT_ONLY=true
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

# Restore positional args
set -- "${POSITIONAL[@]}"
if [ $# -lt 1 ]; then
  echo "Error: No Python script provided."
  print_usage
  exit 1
fi

PYTHON_SCRIPT=$1
shift
PYTHON_ARGS="$@"

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script '$PYTHON_SCRIPT' not found."
  exit 1
fi

DEFAULT_JOB_NAME=$(basename "$PYTHON_SCRIPT" .py)
JOB_NAME=${JOB_NAME:-$DEFAULT_JOB_NAME}

mkdir -p slurm_scripts logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SLURM_SCRIPT="slurm_scripts/${JOB_NAME}_${TIMESTAMP}.slurm"

cat <<EOF > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
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

echo "[✔] SLURM script generated at: $SLURM_SCRIPT"

if [ "$SCRIPT_ONLY" = false ]; then
  echo "[🚀] Submitting job to SLURM..."
  sbatch "$SLURM_SCRIPT"
else
  echo "[📝] Script-only mode: not submitting to SLURM"
fi
