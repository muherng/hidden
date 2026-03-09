#!/bin/bash
#SBATCH --job-name=tree_scan
#SBATCH --output=logs/%A/%A_%a.out
#SBATCH --error=logs/%A/%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=16G
#SBATCH --array=0-0  # overridden by the launcher below

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${PROJ_ROOT}/scripts/train.yaml"

# ----------------------------
# Self-Submission Launcher
# ----------------------------
if [ -z "$SLURM_JOB_ID" ]; then
  echo "[Launcher] Computing number of combinations from $CONFIG_PATH"
  NUM_JOBS=$(python3 - <<EOF
import yaml
from itertools import product
with open("$CONFIG_PATH") as f:
    sweep_args = yaml.safe_load(f)
keys, values = zip(*sweep_args.items())
combinations = list(product(*[v if isinstance(v, list) else [v] for v in values]))
print(len(combinations))
EOF
  )

  echo "[Launcher] Detected $NUM_JOBS job combinations."
  echo "[Launcher] Submitting array job..."
  sbatch --array=0-$((NUM_JOBS - 1)) "$0"
  exit 0
fi

# ----------------------------
# SLURM JOB SECTION (Compute Node)
# ----------------------------
echo "[SLURM Job] Running job with task ID: $SLURM_ARRAY_TASK_ID"

conda activate base

echo "==== Python path and version ===="
which python
python --version
python -c "import torch; print('PyTorch version:', torch.__version__)"

export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH}"
cd "$PROJ_ROOT"

python -m models.tree_model6 -c scripts/train.yaml
