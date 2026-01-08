#!/bin/bash
#
#SBATCH --job-name=your_job_name
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:01:00 # (hh:mm:ss)
#SBATCH --output=/data/lingo/morrisyau/hidden/logs/job_output_%j.log  # CHANGE THIS
#SBATCH --error=/data/lingo/morrisyau/hidden/logs/job_output_%j.err  # CHANGE THIS
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Initialize conda and activate environment
source /data/lingo/morrisyau/miniforge3/etc/profile.d/conda.sh
conda activate base

python -m models.tree_model6 -c scripts/train.yaml