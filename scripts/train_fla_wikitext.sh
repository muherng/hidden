#!/bin/bash
#SBATCH --job-name=gla-wikitext2
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --output=/data/lingo/morrisyau/hidden/logs/fla_%j.log
#SBATCH --error=/data/lingo/morrisyau/hidden/logs/fla_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================
# FLA (Flash Linear Attention) Training Script
# ============================================
# Usage: sbatch scripts/train_fla_wikitext.sh
#
# Configurable via environment variables:
#   MODEL_CONFIG  - path to model config (default: configs/gla_340M.json)
#   STEPS         - training steps (default: 500)
#   BATCH_SIZE    - batch size per device (default: 4)
#   SEQ_LEN       - sequence length (default: 1024)
#   LR            - learning rate (default: 1e-3)
# ============================================

# Initialize conda and activate fla environment
source /data/lingo/morrisyau/miniforge3/etc/profile.d/conda.sh
conda activate fla2

# Change to flame directory
cd /data/lingo/morrisyau/hidden/flame

# Configurable parameters (can override via env vars)
MODEL_CONFIG=${MODEL_CONFIG:-"configs/gla_340M.json"}
STEPS=${STEPS:-500}
BATCH_SIZE=${BATCH_SIZE:-4}
SEQ_LEN=${SEQ_LEN:-1024}
LR=${LR:-1e-3}
WARMUP_STEPS=${WARMUP_STEPS:-100}

# Extract model name from config for experiment folder
MODEL_NAME=$(basename "$MODEL_CONFIG" .json)
EXP_NAME="exp/${MODEL_NAME}-wikitext2-${SLURM_JOB_ID}"

echo "============================================"
echo "Starting FLA Training"
echo "Model Config: $MODEL_CONFIG"
echo "Experiment: $EXP_NAME"
echo "Steps: $STEPS | Batch Size: $BATCH_SIZE | Seq Len: $SEQ_LEN"
echo "Learning Rate: $LR | Warmup: $WARMUP_STEPS"
echo "============================================"

NGPU=1 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder "$EXP_NAME" \
  --model.config "$MODEL_CONFIG" \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.lr "$LR" \
  --lr_scheduler.warmup_steps "$WARMUP_STEPS" \
  --lr_scheduler.decay_type cosine \
  --training.batch_size "$BATCH_SIZE" \
  --training.seq_len "$SEQ_LEN" \
  --training.steps "$STEPS" \
  --training.max_norm 1.0 \
  --training.dataset wikitext \
  --training.dataset_name wikitext-2-raw-v1 \
  --training.dataset_split train \
  --training.num_workers 4 \
  --training.seed 42 \
  --checkpoint.interval 100 \
  --metrics.log_freq 10

echo "============================================"
echo "Training Complete!"
echo "Checkpoint saved to: $EXP_NAME"
echo "============================================"

