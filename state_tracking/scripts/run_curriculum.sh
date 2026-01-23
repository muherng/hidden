#!/bin/bash
#SBATCH --job-name=tree_curriculum
#SBATCH --account=lingo
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --time=24:00:00
#SBATCH --output=state_tracking/logs/curriculum_%j.log
#SBATCH --error=state_tracking/logs/curriculum_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ============================================
# Tree Model Curriculum Learning (max_len 2 → 18)
# ============================================
# This script trains the tree model on progressively longer sequences,
# using checkpoints from each length to initialize the next.
# Training stops early if eval_loss falls below the threshold.

# Configuration
export WANDB_PROJECT=spd_icml
# Disable wandb for batch jobs (AFS permissions prevent reading API key on compute nodes)
# Metrics are still logged to output files. Enable wandb for interactive runs.
export WANDB_MODE=disabled
source /data/lingo/morrisyau/miniforge3/etc/profile.d/conda.sh
conda activate base

cd /data/lingo/morrisyau/hidden

# Training parameters
EVAL_LOSS_THRESHOLD=0.001
NUM_STORIES=100000
EPOCHS=10
BATCH_SIZE=32
MIN_LEN=2
MAX_LEN=18

echo "============================================"
echo "Curriculum Learning: max_len ${MIN_LEN} → ${MAX_LEN}"
echo "eval_loss_threshold: ${EVAL_LOSS_THRESHOLD}"
echo "num_stories: ${NUM_STORIES}"
echo "epochs: ${EPOCHS}"
echo "============================================"

# Loop: max_len from MIN_LEN to MAX_LEN
for max_len in $(seq $MIN_LEN $MAX_LEN); do
    echo ""
    echo "=========================================="
    echo "Training max_len = $max_len"
    echo "=========================================="
    
    # Determine from_checkpoint argument
    if [ $max_len -eq $MIN_LEN ]; then
        FROM_CKPT=""
    else
        prev_len=$((max_len - 1))
        FROM_CKPT="--from_checkpoint $prev_len"
    fi
    
    # Only generate dataset if it doesn't exist (never overwrite)
    if [ ! -d "state_tracking/datasets/permutation_5_${max_len}" ]; then
        GENERATE="--generate_dataset"
        echo "Dataset not found, will generate"
    else
        GENERATE=""
        echo "Dataset exists at state_tracking/datasets/permutation_5_${max_len}, skipping generation"
    fi
    
    echo "FROM_CKPT: $FROM_CKPT"
    echo "GENERATE: $GENERATE"
    
    python -m state_tracking.train \
        --model tree \
        --num_items 5 \
        --max_len $max_len \
        --chunk_size 1 \
        --num_stories $NUM_STORIES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --no-early_stopping \
        --eval_loss_threshold $EVAL_LOSS_THRESHOLD \
        --output_dir state_tracking/saved_models \
        --dataset_root state_tracking/datasets \
        $FROM_CKPT \
        $GENERATE
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed at max_len = $max_len"
        exit 1
    fi
    
    echo "Completed max_len = $max_len"
done

echo ""
echo "============================================"
echo "Curriculum training complete!"
echo "Final model saved to: state_tracking/saved_models/tree_1_${MAX_LEN}/"
echo "============================================"
