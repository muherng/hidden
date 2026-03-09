#!/bin/bash
#SBATCH --job-name=gla_curriculum
#SBATCH --account=lingo
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --time=24:00:00
#SBATCH --output=state_tracking/logs/curriculum_gla_%j.log
#SBATCH --error=state_tracking/logs/curriculum_gla_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ============================================
# GLA Curriculum Learning (max_len 2 → 18)
# ============================================
# This script trains GLA on progressively longer sequences,
# using checkpoints from each length to initialize the next.
# Training stops early if eval_loss falls below the threshold.

# Configuration
export WANDB_MODE=disabled
source /data/lingo/morrisyau/miniforge3/etc/profile.d/conda.sh
conda activate fla2

cd "$(cd "$(dirname "$0")/../.." && pwd)"

# Training parameters
EVAL_LOSS_THRESHOLD=0.001
NUM_STORIES=1000000
MAX_EVAL_SAMPLES=1000
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=3e-3  # Higher LR for GLA based on previous experiments
START_LEN=2
MAX_LEN=18

# Use SLURM job ID for unique checkpoint directory
CHECKPOINT_DIR="state_tracking/saved_models/job_${SLURM_JOB_ID}"
mkdir -p "$CHECKPOINT_DIR"

# Function to count datapoints in existing dataset (train + test)
count_datapoints() {
    local dir="$1"
    if [ -d "$dir/train" ]; then
        find "$dir/train" "$dir/test" -name "*.json" 2>/dev/null | wc -l
    else
        echo 0
    fi
}

# Curriculum: 2, 3, 4, ..., 18
CURRICULUM_LENGTHS=$(seq $START_LEN 1 $MAX_LEN)

echo "============================================"
echo "GLA Curriculum Learning: ${START_LEN} → ${MAX_LEN}"
echo "Lengths: ${CURRICULUM_LENGTHS}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Learning rate: ${LEARNING_RATE}"
echo "eval_loss_threshold: ${EVAL_LOSS_THRESHOLD}"
echo "num_stories: ${NUM_STORIES}"
echo "epochs: ${EPOCHS}"
echo "============================================"

# Loop through curriculum lengths
for max_len in $CURRICULUM_LENGTHS; do
    echo ""
    echo "=========================================="
    echo "Training max_len = $max_len"
    echo "=========================================="
    
    # Determine from_checkpoint argument
    if [ $max_len -eq $START_LEN ]; then
        FROM_CKPT=""
    else
        prev_len=$((max_len - 1))
        FROM_CKPT="--from_checkpoint $prev_len"
    fi
    
    # Check if dataset exists with sufficient datapoints
    DATASET_DIR="state_tracking/datasets/permutation_5_${max_len}"
    EXISTING_COUNT=$(count_datapoints "$DATASET_DIR")
    
    if [ $EXISTING_COUNT -lt $NUM_STORIES ]; then
        if [ -d "$DATASET_DIR" ]; then
            echo "Existing dataset has $EXISTING_COUNT < $NUM_STORIES datapoints, regenerating..."
            rm -rf "$DATASET_DIR"
        else
            echo "Dataset not found, will generate $NUM_STORIES datapoints"
        fi
        GENERATE="--generate_dataset"
    else
        echo "Dataset has $EXISTING_COUNT >= $NUM_STORIES datapoints, reusing"
        GENERATE=""
    fi
    
    echo "FROM_CKPT: $FROM_CKPT"
    echo "GENERATE: $GENERATE"
    
    python -m state_tracking.train \
        --model gla \
        --num_items 5 \
        --max_len $max_len \
        --num_stories $NUM_STORIES \
        --max_eval_samples $MAX_EVAL_SAMPLES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --no-early_stopping \
        --eval_loss_threshold $EVAL_LOSS_THRESHOLD \
        --output_dir $CHECKPOINT_DIR \
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
echo "Final model saved to: ${CHECKPOINT_DIR}/gla_len${MAX_LEN}/"
echo "============================================"
