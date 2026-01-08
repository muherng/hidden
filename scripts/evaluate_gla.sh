#!/bin/bash
#SBATCH --job-name=eval-gla
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=01:00:00
#SBATCH --output=/data/lingo/morrisyau/hidden/logs/eval_gla_%j.log
#SBATCH --error=/data/lingo/morrisyau/hidden/logs/eval_gla_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================
# GLA Model Evaluation Script
# ============================================
# Usage: 
#   sbatch scripts/evaluate_gla.sh <model_path>
#
# Example:
#   sbatch scripts/evaluate_gla.sh exp/gla-170M-wikitext103-comparison-258221/hf
# ============================================

# Check if model path is provided
MODEL_PATH=${1:-""}
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Please provide model path as argument"
    echo "Usage: sbatch scripts/evaluate_gla.sh <model_path>"
    exit 1
fi

# Initialize conda
source /data/lingo/morrisyau/miniforge3/etc/profile.d/conda.sh
conda activate fla2

cd /data/lingo/morrisyau/hidden

echo "============================================"
echo "Evaluating GLA Model"
echo "Model: $MODEL_PATH"
echo "============================================"

python scripts/evaluate_gla.py \
    --model_path "$MODEL_PATH" \
    --dataset wikitext-103 \
    --seq_len 512 \
    --batch_size 16

echo "============================================"
echo "Evaluation Complete!"
echo "============================================"

