#!/bin/bash
#SBATCH --job-name=state_tracking_gla
#SBATCH --account=lingo
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --time=24:00:00
#SBATCH --output=state_tracking/logs/gla_%j.log
#SBATCH --error=state_tracking/logs/gla_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ============================================
# GLA Training for State Tracking
# ============================================

# Configuration
export WANDB_MODE=disabled
source /data/lingo/morrisyau/miniforge3/etc/profile.d/conda.sh
conda activate fla2

cd /data/lingo/morrisyau/hidden

# Training parameters
MODEL="gla"
NUM_ITEMS=5
MAX_LEN=18
NUM_STORIES=1000000
MAX_EVAL_SAMPLES=1000
EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=3e-3

# Use SLURM job ID for unique checkpoint directory
CHECKPOINT_DIR="state_tracking/saved_models/job_${SLURM_JOB_ID}"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================"
echo "GLA State Tracking Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: ${MODEL}"
echo "Max length: ${MAX_LEN}"
echo "Num stories: ${NUM_STORIES}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "============================================"

python -m state_tracking.train \
    --model $MODEL \
    --num_items $NUM_ITEMS \
    --max_len $MAX_LEN \
    --num_stories $NUM_STORIES \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --no-early_stopping \
    --output_dir $CHECKPOINT_DIR \
    --dataset_root state_tracking/datasets \
    --disable_wandb

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "============================================"
echo "Training complete!"
echo "Model saved to: ${CHECKPOINT_DIR}"
echo "============================================"
