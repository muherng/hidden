# TransformerScanModel

A GPT-2-based language model that replaces standard causal self-attention with a
binary-tree parallel prefix scan (Blelloch scan) for sequence aggregation. The
architecture splits processing into three modules:

- **T0** -- token embedding
- **T1** -- aggregation via GPT-2 blocks *without* causal masking, composed through a prefix scan tree
- **T2** -- autoregressive prediction via standard causal GPT-2 blocks

The model is trained on WikiText-103 and compared against GPT-2 Small (125M),
Gated Linear Attention (GLA), and Gated DeltaNet at matched parameter counts.

---

## Repository Structure

```
models/
├── tree_model6.py           # TransformerScanModel (T0/T1/T2) + training loop
├── blelloch_scan.py         # Blelloch parallel prefix scan
├── train_gpt2_small.py      # GPT-2 Small trainer (HuggingFace Trainer)
├── tree.py, tree_ar.py, ... # Earlier model iterations (kept for reference)
scripts/
├── train.sh                 # SLURM launcher for TransformerScanModel
├── train.yaml               # TransformerScanModel hyperparameters
├── train_comparison.sh      # Shell wrapper to train comparison models
├── train_comparison.py      # Comparison training orchestrator
├── evaluate_comparison.py   # Evaluation (perplexity, loss curves, comparison plots)
└── model_registry.yaml      # Central registry of all comparison models + defaults
flame/
├── flame/                   # Flame training infrastructure (fla library)
├── configs/                 # Model architecture configs (JSON)
│   ├── gla_170M.json
│   ├── gated_deltanet_170M.json
│   ├── gpt2_small.json
│   └── ...
├── train.sh                 # Flame SLURM training script
└── exp/                     # Experiment outputs (gitignored)
inference_experiments/
└── inf_plot.py              # Inference speed benchmark (TransformerScanModel vs GPT-2)
state_tracking/              # S5 permutation experiments (see state_tracking/README.md)
```

---

## Environment Setup

Two conda environments are required. The environment files are at the repository root.

| File | Env name | Python | PyTorch | Used by |
|------|----------|--------|---------|---------|
| `base.yml` | `base` | 3.12 | 2.6 | TransformerScanModel |
| `fla.yml` | `fla2` | 3.10 | 2.9 | GPT-2 Small, GLA, Gated DeltaNet (flash-linear-attention + flame) |

```bash
conda env create -f base.yml
conda env create -f fla.yml
```

---

## Training TransformerScanModel

Activate the `base` environment and run from the repository root:

```bash
conda activate base
python -m models.tree_model6 -c scripts/train.yaml
```

The default config (`scripts/train.yaml`) trains on WikiText-103 with these settings:

| Parameter | Value |
|-----------|-------|
| dataset | wikitext-103 |
| model_size | base (GPT-2 base: 768 hidden, 12 heads) |
| train_mode | sequential |
| T1 layers | 1 |
| T2 layers | 12 |
| seq_len | 512 |
| chunk_size | 512 |
| batch_size | 64 |
| epochs | 20 |

To submit via SLURM:

```bash
bash scripts/train.sh
```

The script auto-detects sweep combinations from `train.yaml` and submits a SLURM
array job. Edit the SLURM headers in `scripts/train.sh` for your cluster.

> **Note:** A wandb prompt may appear on first run -- press `3` to skip.
> The initial tokenization of WikiText-103 takes several minutes.

---

## Training Comparison Models

Comparison models (GPT-2 Small, GLA, Gated DeltaNet, etc.) are trained through
a unified pipeline. Activate the `fla2` environment:

```bash
conda activate fla2
```

### Quick start

```bash
# List all registered models
./scripts/train_comparison.sh --list

# Train GPT-2 Small on WikiText-103 (submit to SLURM)
./scripts/train_comparison.sh gpt2_small_wikitext103_drop01

# Train GLA-170M (submit to SLURM)
./scripts/train_comparison.sh gla_170M

# Train locally instead of SLURM
./scripts/train_comparison.sh gla_170M --local

# Dry run (show commands without executing)
./scripts/train_comparison.sh gla_170M --dry_run
```

### Key WikiText-103 model IDs

| Model ID | Description | Key differences from defaults |
|----------|-------------|-------------------------------|
| `gla_170M` | 170M GLA | defaults |
| `gla_170M_baseline` | 170M GLA, GPT-2-matched | expand_k=1, tie_word_embeddings |
| `gated_deltanet_170M` | 170M Gated DeltaNet | defaults |
| `gated_deltanet_170M_baseline` | 170M Gated DeltaNet, GPT-2-matched | expand_v=1, num_heads=12, no short_conv |
| `gpt2_small_wikitext103_drop01` | 125M GPT-2 Small, dropout=0.1 | lr=1e-4, beta2=0.999, 30518 steps |
| `gpt2_small_wikitext103_nodrop` | 125M GPT-2 Small, no dropout | lr=1e-4, beta2=0.999, 30518 steps |

### Default training hyperparameters

Defined in `scripts/model_registry.yaml`:

| Parameter | Value |
|-----------|-------|
| dataset | wikitext-103-raw-v1 |
| batch_size | 64 |
| seq_len | 512 |
| learning_rate | 1e-3 |
| warmup_steps | 1000 |
| weight_decay | 0.01 |
| dropout | 0.1 |
| lr_decay | cosine |
| total_steps | 62,900 (~20 epochs) |
| tokenizer | gpt2 |

### Additional options

```bash
# Custom experiment name
./scripts/train_comparison.sh gla_170M --exp_name my_experiment

# Override hyperparameters
./scripts/train_comparison.sh gla_170M --batch_size 32 --lr 5e-5

# Different SLURM profile (see model_registry.yaml for profiles)
./scripts/train_comparison.sh gla_170M --profile vision

# Disable wandb
./scripts/train_comparison.sh gla_170M --no_wandb
```

---

## Evaluation

Evaluate trained checkpoints with `scripts/evaluate_comparison.py`:

```bash
conda activate fla2

# Evaluate a single experiment
python scripts/evaluate_comparison.py --exp_path flame/exp/<experiment_dir>

# Evaluate specific checkpoint steps
python scripts/evaluate_comparison.py --exp_path flame/exp/... --steps 5000 10000 20000

# Compare multiple experiments
python scripts/evaluate_comparison.py --compare \
    flame/exp/gla_170M-wikitext103-... \
    flame/exp/gated_deltanet_170M-wikitext103-...

# Different evaluation dataset
python scripts/evaluate_comparison.py --exp_path flame/exp/... --dataset wikitext-2
```

The script handles both flame DCP checkpoints and HuggingFace checkpoints,
converting DCP to HuggingFace format when needed. Outputs include validation
loss, perplexity, and optional comparison plots.

---

## Inference Speed Benchmark

`inference_experiments/inf_plot.py` benchmarks autoregressive generation speed,
comparing TransformerScanModel against vanilla GPT-2 with KV caching. Both
models are randomly initialized -- this measures computational scaling, not
generation quality.

The script generates `max_new_tokens` tokens from a WikiText-2 prompt, timing
each token individually. Flash and memory-efficient SDPA are disabled so that
vanilla GPT-2 attention uses the standard O(L^2) kernel, making the asymptotic
difference visible.

```bash
conda activate base
python -m inference_experiments.inf_plot --batch_size 1 --max_new_tokens 5000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 1 | Batch size |
| `--max_new_tokens` | 10000 | Number of tokens to generate |
| `--prompt_len` | 100 | Prompt length (tokens from WikiText-2) |
| `--chunk_size` | 64 | Chunk size for TransformerScanModel |
| `--seed` | 42 | Random seed |
| `--device` | cuda | Device (cuda or cpu) |

Output: `inference_speed.png` -- time per token vs token index for both models.

---

## Adding a New Comparison Model

1. Create a model config JSON in `flame/configs/` (e.g., `my_model_170M.json`).
2. Register the model in `scripts/model_registry.yaml`:

```yaml
models:
  my_model_170M:
    name: "My-Model-170M"
    model_type: my_model
    config: configs/my_model_170M.json
    description: "170M parameter My Model"
    # Optional overrides:
    batch_size: 64
    learning_rate: 1.0e-4
```

3. Train: `./scripts/train_comparison.sh my_model_170M`
4. Evaluate: `python scripts/evaluate_comparison.py --exp_path flame/exp/my_model_170M-...`

---

## State Tracking Experiments

S5 permutation state-tracking experiments (TransformerScanModel, GPT-2, GLA,
Gated DeltaNet with curriculum learning) live in the `state_tracking/` directory.
See [`state_tracking/README.md`](state_tracking/README.md) for setup and
reproduction instructions.
