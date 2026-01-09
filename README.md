# Setup

## Environment

Load the conda environment:

```bash
conda env create -f base_env.yml
conda activate base_env
```

## Training TransformerScanModel

Run the training script:

```bash
python -m models.tree_model6 -c scripts/train.yaml
```

> **Note:** A wandb interface will appear — press `3` to ignore. There is also a long loading time to tokenize the wikitext dataset.

## Inference

Run the inference script:

```bash
python -m inference_experiments.inf_plot --batch_size 32 --max_new_tokens 5000
```

---

# Model Comparison Framework

This framework allows systematic comparison of TransformerScanModel against various baseline models (GLA, Gated DeltaNet, Mamba, etc.) using unified training and evaluation pipelines.

## Quick Start

```bash
# List all available comparison models
./scripts/train_comparison.sh --list

# Train GLA-170M (submit to SLURM)
./scripts/train_comparison.sh gla_170M

# Train Gated DeltaNet (run locally)
./scripts/train_comparison.sh gated_deltanet_340M --local

# Evaluate checkpoints
python scripts/evaluate_comparison.py --exp_path flame/exp/gla_170M-wikitext103-...

# Compare multiple experiments
python scripts/evaluate_comparison.py --compare \
    flame/exp/gla_170M-... \
    flame/exp/gated_deltanet_340M-...
```

## Adding a New Comparison Model

To add a new model (e.g., "my_model"):

### 1. Add the model config JSON (if not already in flame/configs/)

Create `flame/configs/my_model_170M.json` with the appropriate architecture config.

### 2. Register the model in `scripts/model_registry.yaml`

```yaml
models:
  # ... existing models ...
  
  my_model_170M:
    name: "My-Model-170M"
    model_type: my_model          # Must match fla's model type
    config: configs/my_model_170M.json
    description: "170M parameter My Model"
    # Optional overrides (uses defaults if not specified):
    batch_size: 64                # Override if needed
    learning_rate: 1.0e-4
```

### 3. Train the model

```bash
./scripts/train_comparison.sh my_model_170M
```

### 4. Evaluate

```bash
python scripts/evaluate_comparison.py --exp_path flame/exp/my_model_170M-...
```

## Available Models

| Model ID | Name | Description |
|----------|------|-------------|
| `gla_170M` | GLA-170M | 170M Gated Linear Attention |
| `gla_340M` | GLA-340M | 340M Gated Linear Attention |
| `gated_deltanet_340M` | Gated-DeltaNet-340M | 340M Gated DeltaNet |
| `gated_deltanet_1B` | Gated-DeltaNet-1B | 1B Gated DeltaNet |
| `delta_net_340M` | DeltaNet-340M | 340M DeltaNet |
| `mamba_340M` | Mamba-340M | 340M Mamba |
| `mamba2_340M` | Mamba2-340M | 340M Mamba2 |
| `transformer_340M` | Transformer-340M | 340M Transformer baseline |
| `gsa_340M` | GSA-340M | 340M Gated Slot Attention |
| `hgrn2_340M` | HGRN2-340M | 340M HGRN2 |

## File Structure

```
scripts/
├── model_registry.yaml       # Central registry of all comparison models
├── train_comparison.py       # Generic training script
├── train_comparison.sh       # Shell wrapper for easy invocation
├── evaluate_comparison.py    # Generic evaluation script
└── train.yaml                # TransformerScanModel config

flame/
├── configs/                  # Model architecture configs (JSON)
│   ├── gla_170M.json
│   ├── gated_deltanet_340M.json
│   └── ...
└── exp/                      # Experiment outputs
    └── <model>-wikitext103-<timestamp>/
        ├── checkpoint/
        ├── hf-<step>/        # Converted HF checkpoints
        ├── eval_results.json
        └── eval_curve.png
```

## Training Options

```bash
# Custom experiment name
./scripts/train_comparison.sh gla_170M --exp_name my_custom_name

# Override hyperparameters
./scripts/train_comparison.sh gla_170M --batch_size 32 --lr 5e-5

# Dry run (show commands without executing)
./scripts/train_comparison.sh gla_170M --dry_run

# Different SLURM profile
./scripts/train_comparison.sh gla_170M --profile vision

# Disable wandb
./scripts/train_comparison.sh gla_170M --no_wandb
```

## Evaluation Options

```bash
# Evaluate specific checkpoint steps
python scripts/evaluate_comparison.py --exp_path flame/exp/... --steps 5000 10000 20000

# Different dataset
python scripts/evaluate_comparison.py --exp_path flame/exp/... --dataset wikitext-2

# Custom batch size for evaluation
python scripts/evaluate_comparison.py --exp_path flame/exp/... --batch_size 16
```
