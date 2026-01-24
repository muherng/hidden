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
MAX_LEN=18

# Model architecture
CHUNK_SIZE=2
T1_NUM_LAYERS=2
T2_NUM_LAYERS=2

# Use SLURM job ID for unique checkpoint directory (prevents overwriting)
CHECKPOINT_DIR="state_tracking/saved_models/job_${SLURM_JOB_ID}"
mkdir -p "$CHECKPOINT_DIR"

# Curriculum: train on multiples of chunk_size (chunk_size, 2*chunk_size, ..., MAX_LEN)
# For chunk_size=2, MAX_LEN=18: trains on 2,4,6,8,10,12,14,16,18
# For chunk_size=1, MAX_LEN=18: trains on 1,2,3,...,18
CURRICULUM_LENGTHS=$(seq $CHUNK_SIZE $CHUNK_SIZE $MAX_LEN)
FIRST_LEN=$CHUNK_SIZE

echo "============================================"
echo "Curriculum Learning: ${CHUNK_SIZE}, $((CHUNK_SIZE*2)), ... → ${MAX_LEN}"
echo "Lengths: ${CURRICULUM_LENGTHS}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "chunk_size: ${CHUNK_SIZE}"
echo "T1_num_layers: ${T1_NUM_LAYERS}"
echo "T2_num_layers: ${T2_NUM_LAYERS}"
echo "eval_loss_threshold: ${EVAL_LOSS_THRESHOLD}"
echo "num_stories: ${NUM_STORIES}"
echo "epochs: ${EPOCHS}"
echo "============================================"

# Loop: max_len as multiples of chunk_size
for max_len in $CURRICULUM_LENGTHS; do
    echo ""
    echo "=========================================="
    echo "Training max_len = $max_len"
    echo "=========================================="
    
    # Determine from_checkpoint argument (previous length is max_len - chunk_size)
    if [ $max_len -eq $FIRST_LEN ]; then
        FROM_CKPT=""
    else
        prev_len=$((max_len - CHUNK_SIZE))
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
        --chunk_size $CHUNK_SIZE \
        --T1_num_layers $T1_NUM_LAYERS \
        --T2_num_layers $T2_NUM_LAYERS \
        --num_stories $NUM_STORIES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
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
echo "Final model saved to: ${CHECKPOINT_DIR}/tree_c${CHUNK_SIZE}_T1-${T1_NUM_LAYERS}_T2-${T2_NUM_LAYERS}_len${MAX_LEN}/"
echo "============================================"
