# Tree Model Training for Permutation Tasks

This module trains Tree Models on S5 permutation tasks with curriculum learning.

## Quick Start

### Prerequisites

```bash
conda activate base  # Uses hidden/base.yml environment
cd /data/lingo/morrisyau/hidden  # Run all commands from hidden root
```

### Submit Curriculum Training (Recommended)

Submit a SLURM job that trains on max_len 2 → 18 automatically:

```bash
sbatch state_tracking/scripts/run_curriculum.sh
```

Edit `run_curriculum.sh` to configure:
- `CHUNK_SIZE`: Tokens per chunk (1 or 2)
- `T1_NUM_LAYERS`: Layers in aggregation module (default: 2)
- `T2_NUM_LAYERS`: Layers in prediction module (default: 2)
- `MIN_LEN` / `MAX_LEN`: Curriculum range
- `EVAL_LOSS_THRESHOLD`: Early stopping threshold
- `NUM_STORIES`, `EPOCHS`, `BATCH_SIZE`: Training hyperparameters

---

## Directory Structure

### Datasets
```
state_tracking/datasets/
├── permutation_5_2/     # S5, max_len=2
│   ├── train.pt
│   └── val.pt
├── permutation_5_3/     # S5, max_len=3
├── ...
└── permutation_5_18/    # S5, max_len=18
```

Datasets are generated once and reused. The training script skips generation if the directory exists.

### Checkpoints

Checkpoint directories are named: `tree_c{chunk_size}_T1-{T1_layers}_T2-{T2_layers}_len{max_len}`

**Interactive runs** (default):
```
state_tracking/saved_models/
├── tree_c1_T1-1_T2-1_len2/    # chunk_size=1, T1=1, T2=1, max_len=2
├── tree_c1_T1-1_T2-1_len3/
├── tree_c2_T1-2_T2-2_len2/    # chunk_size=2, T1=2, T2=2, max_len=2
└── ...
```

**SLURM batch jobs** (versioned by job ID):
```
state_tracking/saved_models/
├── job_284522/                 # Job ID prevents overwrites
│   ├── tree_c2_T1-2_T2-2_len2/
│   ├── tree_c2_T1-2_T2-2_len3/
│   └── ...
└── job_284600/
    └── ...
```

Each checkpoint directory contains:
- `model.safetensors` or `pytorch_model.bin`: Model weights
- `config.json`: Model configuration
- `training_args.bin`: Training arguments

### Logs
```
state_tracking/logs/
├── curriculum_284522.log    # stdout
└── curriculum_284522.err    # stderr
```

---

## Manual Training (Interactive)

### Single Length Training

```bash
python -m state_tracking.train \
    --model tree \
    --num_items 5 \
    --max_len 2 \
    --chunk_size 1 \
    --T1_num_layers 1 \
    --T2_num_layers 1 \
    --num_stories 100000 \
    --epochs 10 \
    --batch_size 32 \
    --no-early_stopping \
    --generate_dataset \
    --disable_wandb
```

### Curriculum Learning (Step by Step)

**Step 1: Train on length 2**
```bash
python -m state_tracking.train \
    --model tree --num_items 5 --max_len 2 \
    --chunk_size 1 --T1_num_layers 1 --T2_num_layers 1 \
    --num_stories 100000 --epochs 10 --batch_size 32 \
    --no-early_stopping --generate_dataset --disable_wandb
```

**Step 2: Train on length 3 (from length 2 checkpoint)**
```bash
python -m state_tracking.train \
    --model tree --num_items 5 --max_len 3 \
    --chunk_size 1 --T1_num_layers 1 --T2_num_layers 1 \
    --num_stories 100000 --epochs 10 --batch_size 32 \
    --from_checkpoint 2 \
    --no-early_stopping --generate_dataset --disable_wandb
```

**Continue for each length L from 4 to 18**, using `--from_checkpoint <L-1>`.

**Note:** `--T1_num_layers` and `--T2_num_layers` must match between checkpoints for curriculum learning to work.

---

## Evaluating Length Generalization

After curriculum training, evaluate on sequences longer than training length.

**Important:** Use the same `--chunk_size`, `--T1_num_layers`, `--T2_num_layers` as training.

For `chunk_size=1, T1=1, T2=1`:
```bash
python -m state_tracking.train \
    --model tree --num_items 5 --max_len 18 \
    --chunk_size 1 --T1_num_layers 1 --T2_num_layers 1 \
    --from_checkpoint 18 \
    --eval_lengths \
    --disable_wandb
```

For `chunk_size=2, T1=2, T2=2`:
```bash
python -m state_tracking.train \
    --model tree --num_items 5 --max_len 18 \
    --chunk_size 2 --T1_num_layers 2 --T2_num_layers 2 \
    --from_checkpoint 18 \
    --eval_lengths \
    --disable_wandb
```

This generates plots in the checkpoint directory:
- `length_generalisation_loss_*.png`
- `length_generalisation_error_*.png`

---

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--model tree` | Use Tree Model architecture |
| `--chunk_size` | Tokens per chunk (1 = finest granularity) |
| `--T1_num_layers` | Layers in aggregation module (default: 1) |
| `--T2_num_layers` | Layers in prediction module (default: 1) |
| `--num_items` | Permutation group size (5 for S5) |
| `--max_len` | Sequence length for training |
| `--from_checkpoint <L>` | Initialize from checkpoint trained on max_len=L |
| `--eval_lengths` | Skip training, evaluate on multiple lengths |
| `--eval_loss_threshold` | Stop if eval_loss falls below threshold |
| `--generate_dataset` | Generate dataset (skipped if exists) |
| `--output_dir` | Override default checkpoint directory |
| `--dataset_root` | Override default dataset directory |
| `--disable_wandb` | Disable Weights & Biases logging |
| `--no-early_stopping` | Disable early stopping callback |

---

## Generating Datasets Manually

To pre-generate datasets before training:

```bash
python -m state_tracking.permutation_task \
    --num_items 5 \
    --data_dir state_tracking/datasets/permutation_5_10 \
    --num_stories 100000 \
    --train_ratio 0.9 \
    --story_length 10
```
