#!/bin/bash
#SBATCH --job-name=compressed
#SBATCH --output=/data/vision/torralba/selfmanaged/isola/projects/sharut/code/Compressive-Transformer/hidden/logs/%A/%A_%a.out
#SBATCH --error=/data/vision/torralba/selfmanaged/isola/projects/sharut/code/Compressive-Transformer/hidden/logs/%A/%A_%a.err
#SBATCH --account=vision-phillipi
#SBATCH --partition=vision-phillipi
#SBATCH --qos=vision-phillipi-main
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=16G
#SBATCH --nodelist=isola-h200-1
#SBATCH --array=0-0  # dummy; will be overridden by the launcher

# ----------------------------
# Self-Submission Launcher
# ----------------------------
if [ -z "$SLURM_JOB_ID" ]; then
  CONFIG_PATH=scripts/train.yaml

  echo "🚧 [Launcher] Computing number of combinations from $CONFIG_PATH"
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

  echo "🔢 [Launcher] Detected $NUM_JOBS job combinations."
  echo "🚀 [Launcher] Submitting array job..."
  sbatch --array=0-$((NUM_JOBS - 1)) "$0"
  exit 0
fi


# ----------------------------
# SLURM JOB SECTION (Compute Node)
# ----------------------------
echo "🔢 [SLURM Job] Running job with task ID: $SLURM_ARRAY_TASK_ID"

# Set up the environment
source /data/vision/torralba/selfmanaged/isola/u/sharut/.bashrc_hpc
source /data/scratch/sharut/anaconda3/etc/profile.d/conda.sh
conda activate /data/scratch/sharut/anaconda3/envs/multimae

# Debug info
echo "==== Conda Environment List ===="
conda env list

echo "==== Python path and version ===="
which python
python --version

echo "==== Checking if torch is installed ===="
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Set Python path and change directory
export PYTHONPATH=$PYTHONPATH:/data/vision/torralba/selfmanaged/isola/projects/sharut/code/Compressive-Transformer/hidden
cd /data/vision/torralba/selfmanaged/isola/projects/sharut/code/Compressive-Transformer/hidden

# Run your Python script
python tree_model6.py -s -c scripts/train.yaml
