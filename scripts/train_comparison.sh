#!/bin/bash
# ============================================
# Generic Training Script for Comparison Models
# ============================================
# This script provides a simple interface to train any registered model.
# It wraps train_comparison.py with common options.
#
# Usage:
#   ./scripts/train_comparison.sh <model_id> [options]
#
# Examples:
#   # Train GLA-170M (submit to SLURM)
#   ./scripts/train_comparison.sh gla_170M
#
#   # Train Gated DeltaNet locally
#   ./scripts/train_comparison.sh gated_deltanet_340M --local
#
#   # List available models
#   ./scripts/train_comparison.sh --list
#
#   # Dry run (show command without executing)
#   ./scripts/train_comparison.sh mamba_340M --dry_run
# ============================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to show usage
show_usage() {
    echo "Usage: $0 <model_id> [options]"
    echo ""
    echo "Options:"
    echo "  --list          List all available models"
    echo "  --local         Run locally instead of submitting to SLURM"
    echo "  --dry_run       Show commands without executing"
    echo "  --exp_name NAME Custom experiment name"
    echo "  --profile NAME  SLURM profile (default: lingo)"
    echo "  --batch_size N  Override batch size"
    echo "  --seq_len N     Override sequence length"
    echo "  --lr RATE       Override learning rate"
    echo "  --steps N       Override total training steps"
    echo "  --no_wandb      Disable wandb logging"
    echo ""
    echo "Examples:"
    echo "  $0 --list"
    echo "  $0 gla_170M"
    echo "  $0 gated_deltanet_340M --local --dry_run"
    echo "  $0 mamba_340M --batch_size 32 --exp_name my_experiment"
}

# Parse arguments
MODEL_ID=""
EXTRA_ARGS=""
LOCAL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --list)
            python3 "$SCRIPT_DIR/train_comparison.py" --list
            exit 0
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --dry_run)
            EXTRA_ARGS="$EXTRA_ARGS --dry_run"
            shift
            ;;
        --exp_name)
            EXTRA_ARGS="$EXTRA_ARGS --exp_name $2"
            shift 2
            ;;
        --profile)
            EXTRA_ARGS="$EXTRA_ARGS --slurm_profile $2"
            shift 2
            ;;
        --batch_size)
            EXTRA_ARGS="$EXTRA_ARGS --batch_size $2"
            shift 2
            ;;
        --seq_len)
            EXTRA_ARGS="$EXTRA_ARGS --seq_len $2"
            shift 2
            ;;
        --lr)
            EXTRA_ARGS="$EXTRA_ARGS --learning_rate $2"
            shift 2
            ;;
        --steps)
            EXTRA_ARGS="$EXTRA_ARGS --total_steps $2"
            shift 2
            ;;
        --no_wandb)
            EXTRA_ARGS="$EXTRA_ARGS --no_wandb"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$MODEL_ID" ]]; then
                MODEL_ID="$1"
            else
                echo "Error: Multiple model IDs provided"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if model ID is provided
if [[ -z "$MODEL_ID" ]]; then
    echo "Error: Model ID required"
    show_usage
    exit 1
fi

# Build command
if [[ "$LOCAL_MODE" == true ]]; then
    CMD="python3 $SCRIPT_DIR/train_comparison.py --model $MODEL_ID $EXTRA_ARGS"
else
    CMD="python3 $SCRIPT_DIR/train_comparison.py --model $MODEL_ID --submit $EXTRA_ARGS"
fi

echo "============================================"
echo "Training Model: $MODEL_ID"
echo "============================================"
echo "Command: $CMD"
echo "============================================"

# Execute
exec $CMD
