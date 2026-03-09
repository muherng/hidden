# State Tracking Experiments

This module runs state tracking experiments on the S5 permutation task with curriculum learning (or single-length training). Supported models: **TransformerScanModel (tree)**, **GPT-2**, **Gated DeltaNet**, and **Gated Linear Attention (GLA)**.

## Prerequisites

- Python 3.10+ and Conda (or Miniforge). Run all commands from the **repository root** (the directory that contains `state_tracking/`).
- A CUDA-capable GPU is required for training.

### Environment setup

Two conda environment files are provided at the repository root:

| File | Env name | Used by |
|------|----------|---------|
| `base.yml` | `base` | TransformerScanModel (tree) |
| `fla.yml` | `fla2` | GPT-2, Gated DeltaNet, GLA |

Create them with:

```bash
conda env create -f base.yml
conda env create -f fla.yml
```

The training scripts source conda and activate the appropriate env automatically. On a different machine you may need to edit the `source` line in the scripts to point to your conda installation, or simply activate the env yourself before running.

## Reproducing Experiments

### TransformerScanModel (tree)

**Recommended:** run the curriculum script (SLURM or locally):

```bash
sbatch state_tracking/scripts/run_curriculum.sh
# Or locally:
bash state_tracking/scripts/run_curriculum.sh
```

**Curriculum sequence:** `chunk_size`, `2*chunk_size`, ..., `MAX_LEN` (e.g. for `CHUNK_SIZE=1`, `MAX_LEN=18`: lengths 1, 2, ..., 18). Edit `run_curriculum.sh` for `CHUNK_SIZE`, `T1_NUM_LAYERS`, `T2_NUM_LAYERS`, `MAX_LEN`, and other hyperparameters.

**Core command -- single length:**

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
    --dataset_root state_tracking/datasets \
    --disable_wandb
```

**Core command -- curriculum step (e.g. length 3 from length 2):**

```bash
python -m state_tracking.train \
    --model tree --num_items 5 --max_len 3 \
    --chunk_size 1 --T1_num_layers 1 --T2_num_layers 1 \
    --num_stories 100000 --epochs 10 --batch_size 32 \
    --from_checkpoint 2 \
    --no-early_stopping --generate_dataset \
    --dataset_root state_tracking/datasets \
    --disable_wandb
```

Continue for each length using `--from_checkpoint <L-1>`. `--T1_num_layers` and `--T2_num_layers` must match across curriculum steps.

---

### GPT-2

**Single-length training** at `max_len=18` (no curriculum):

```bash
sbatch state_tracking/scripts/run_gpt2.sh
# Or locally:
bash state_tracking/scripts/run_gpt2.sh
```

**Core command:**

```bash
python -m state_tracking.train \
    --model gpt2 \
    --num_items 5 \
    --max_len 18 \
    --num_stories 1000000 \
    --max_eval_samples 1000 \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --no-early_stopping \
    --output_dir state_tracking/saved_models \
    --dataset_root state_tracking/datasets \
    --disable_wandb
```

---

### Gated DeltaNet

**Curriculum** from length 2 to 18:

```bash
sbatch state_tracking/scripts/run_curriculum_gated_deltanet.sh
# Or locally:
bash state_tracking/scripts/run_curriculum_gated_deltanet.sh
```

The script loops `max_len` 2, 3, ..., 18 and uses `--from_checkpoint <prev_len>` to chain checkpoints.

**Single-length training** at `max_len=18` (no curriculum):

```bash
sbatch state_tracking/scripts/run_gated_deltanet.sh
# Or locally:
bash state_tracking/scripts/run_gated_deltanet.sh
```

**Core command (one length, e.g. max_len=2):**

```bash
python -m state_tracking.train \
    --model gated_deltanet \
    --num_items 5 \
    --max_len 2 \
    --num_stories 1000000 \
    --max_eval_samples 1000 \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --no-early_stopping \
    --eval_loss_threshold 0.001 \
    --output_dir state_tracking/saved_models \
    --dataset_root state_tracking/datasets \
    --generate_dataset \
    --disable_wandb
```

For the next length use `--max_len <L>` and `--from_checkpoint <L-1>`.

---

### Gated Linear Attention (GLA)

**Curriculum** from length 2 to 18:

```bash
sbatch state_tracking/scripts/run_curriculum_gla.sh
# Or locally:
bash state_tracking/scripts/run_curriculum_gla.sh
```

**Single-length training** at `max_len=18` (no curriculum):

```bash
sbatch state_tracking/scripts/run_gla.sh
# Or locally:
bash state_tracking/scripts/run_gla.sh
```

**Core command (one length, e.g. max_len=2):**

```bash
python -m state_tracking.train \
    --model gla \
    --num_items 5 \
    --max_len 2 \
    --num_stories 1000000 \
    --max_eval_samples 1000 \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 3e-3 \
    --no-early_stopping \
    --eval_loss_threshold 0.001 \
    --output_dir state_tracking/saved_models \
    --dataset_root state_tracking/datasets \
    --generate_dataset \
    --disable_wandb
```

Chain lengths with `--from_checkpoint <L-1>` as in the Gated DeltaNet script.

---

## Summary

| Model                       | Script                             | Mode                         | Conda env |
|-----------------------------|------------------------------------|------------------------------|-----------|
| TransformerScanModel (tree) | `run_curriculum.sh`                | Curriculum (-> MAX_LEN)      | base      |
| GPT-2                       | `run_gpt2.sh`                      | Single length (max_len=18)   | fla2      |
| Gated DeltaNet              | `run_curriculum_gated_deltanet.sh` | Curriculum (2->18)           | fla2      |
| Gated DeltaNet              | `run_gated_deltanet.sh`            | Single length (max_len=18)   | fla2      |
| GLA                         | `run_curriculum_gla.sh`            | Curriculum (2->18)           | fla2      |
| GLA                         | `run_gla.sh`                       | Single length (max_len=18)   | fla2      |

---

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--model` | One of `tree`, `gpt2`, `gated_deltanet`, `gla` |
| `--chunk_size` | Tokens per chunk (tree only; 1 = finest granularity) |
| `--T1_num_layers` | Layers in aggregation module (tree only; default: 1) |
| `--T2_num_layers` | Layers in prediction module (tree only; default: 1) |
| `--num_items` | Permutation group size (5 for S5) |
| `--max_len` | Sequence length for training |
| `--from_checkpoint <L>` | Initialize from checkpoint trained on max_len=L |
| `--num_stories` | Number of training stories (dataset size) |
| `--max_eval_samples` | Max evaluation samples (GPT-2 / Gated DeltaNet / GLA) |
| `--learning_rate` | Learning rate |
| `--eval_loss_threshold` | Stop training if eval_loss drops below this |
| `--eval_lengths` | Skip training, evaluate on multiple lengths (tree) |
| `--generate_dataset` | Generate dataset if missing |
| `--output_dir` | Checkpoint directory |
| `--dataset_root` | Root for datasets (e.g. `state_tracking/datasets`) |
| `--disable_wandb` | Disable Weights & Biases |
| `--no-early_stopping` | Disable early stopping |

---

## Datasets

Datasets live under `state_tracking/datasets/permutation_5_{max_len}/` (e.g. `permutation_5_18`). They are generated automatically when missing, or when `--generate_dataset` is passed. Generated data is reused across runs.

**Pre-generate manually:**

```bash
python -m state_tracking.permutation_task \
    --num_items 5 \
    --data_dir state_tracking/datasets/permutation_5_10 \
    --num_stories 100000 \
    --train_ratio 0.9 \
    --story_length 10
```

---

## Checkpoints and Logs

- **Checkpoints:** written to `state_tracking/saved_models/`. With SLURM, scripts use `state_tracking/saved_models/job_${SLURM_JOB_ID}/` to avoid overwrites. Checkpoint names include model and length (e.g. `tree_c1_T1-1_T2-1_len18`, `gated_deltanet_len18`, `gla_len18`).
- **Logs:** `state_tracking/logs/` is gitignored; nothing there is required to reproduce experiments.

---

## Evaluating Length Generalization (tree)

After curriculum training, evaluate the tree model on longer sequences. Use the same `--chunk_size`, `--T1_num_layers`, `--T2_num_layers` as in training.

Example for `chunk_size=1`, `T1=1`, `T2=1`:

```bash
python -m state_tracking.train \
    --model tree --num_items 5 --max_len 18 \
    --chunk_size 1 --T1_num_layers 1 --T2_num_layers 1 \
    --from_checkpoint 18 \
    --eval_lengths \
    --dataset_root state_tracking/datasets \
    --disable_wandb
```

This produces length-generalization plots in the checkpoint directory (`length_generalisation_loss_*.png`, `length_generalisation_error_*.png`). For GPT-2, Gated DeltaNet, and GLA, evaluation follows the same `python -m state_tracking.train` interface; use the script's checkpoint path and the same `--model` and `--max_len` as in training.

---

## Smoke Tests

A lightweight test suite verifies that the environment files are present and that the training pipeline runs end-to-end.

```bash
# With pytest (from repo root):
pytest state_tracking/tests/test_smoke.py -v

# Or without pytest:
python state_tracking/tests/test_smoke.py
```

The tests check:
1. `base.yml` and `fla.yml` exist and define the expected env names (`base` and `fla2`).
2. A single training step of the tree model completes successfully (requires CUDA; skipped automatically on CPU-only machines).
