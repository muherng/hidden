#!/bin/bash
#SBATCH --job-name=gla-170M-wiki103
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=24:00:00
#SBATCH --output=/data/lingo/morrisyau/hidden/logs/gla_comparison_%j.log
#SBATCH --error=/data/lingo/morrisyau/hidden/logs/gla_comparison_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ============================================
# GLA vs TransformerScanModel Comparison
# ============================================
# This script trains a ~170M parameter GLA model on WikiText-103
# with hyperparameters matched to the TransformerScanModel in train.yaml
#
# TransformerScanModel config (train.yaml):
#   - dataset: wikitext-103
#   - model_size: base (GPT2: 768 hidden, 12 layers, 12 heads)
#   - epochs: 20
#   - batch_size: 64
#   - seq_len: 512
#   - learning_rate: 1e-4
#   - warmup_steps: 1000
#   - lr_scheduler: cosine
#   - weight_decay: 0.01
#
# Calculations:
#   - WikiText-103 train: ~103M tokens
#   - 20 epochs = ~2.06B tokens total
#   - tokens_per_step = batch_size * seq_len = 64 * 512 = 32,768
#   - steps = 2.06B / 32,768 ≈ 62,900 steps
# ============================================

# Initialize conda and activate fla environment
source /data/lingo/morrisyau/miniforge3/etc/profile.d/conda.sh
conda activate fla2

# Change to flame directory
cd /data/lingo/morrisyau/hidden/flame

# ============================================
# Training Parameters (matched to TransformerScanModel)
# ============================================
MODEL_CONFIG="configs/gla_170M.json"
DATASET="wikitext"
DATASET_NAME="wikitext-103-raw-v1"

# Matched to train.yaml
BATCH_SIZE=64
SEQ_LEN=512
LEARNING_RATE=1e-4
WARMUP_STEPS=1000
WEIGHT_DECAY=0.01

# 20 epochs over WikiText-103 (~103M tokens)
# Total tokens: 20 * 103M = 2.06B
# Steps: 2.06B / (64 * 512) ≈ 62,900
TOTAL_STEPS=62900

# Experiment name
EXP_NAME="exp/gla-170M-wikitext103-comparison-${SLURM_JOB_ID:-local}"

echo "============================================"
echo "GLA vs TransformerScanModel Comparison"
echo "============================================"
echo "Model: GLA-170M (configs/gla_170M.json)"
echo "Dataset: WikiText-103"
echo "Batch Size: $BATCH_SIZE"
echo "Seq Length: $SEQ_LEN"
echo "Learning Rate: $LEARNING_RATE"
echo "Warmup Steps: $WARMUP_STEPS"
echo "Total Steps: $TOTAL_STEPS (~20 epochs)"
echo "Output: $EXP_NAME"
echo "============================================"

NGPU=1 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder "$EXP_NAME" \
  --model.config "$MODEL_CONFIG" \
  --model.tokenizer_path gpt2 \
  --optimizer.name AdamW \
  --optimizer.eps 1e-8 \
  --optimizer.lr "$LEARNING_RATE" \
  --lr_scheduler.warmup_steps "$WARMUP_STEPS" \
  --lr_scheduler.lr_min 0.0 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size "$BATCH_SIZE" \
  --training.seq_len "$SEQ_LEN" \
  --training.gradient_accumulation_steps 1 \
  --training.steps "$TOTAL_STEPS" \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset "$DATASET" \
  --training.dataset_name "$DATASET_NAME" \
  --training.dataset_split train \
  --training.num_workers 8 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --checkpoint.interval 5000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 0 \  # 0 = keep ALL checkpoints
  --metrics.log_freq 100 \
  --metrics.enable_wandb

echo "============================================"
echo "Training Complete!"
echo "Checkpoint saved to: $EXP_NAME"
echo "============================================"

