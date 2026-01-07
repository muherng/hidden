# Setup

## Environment

Load the conda environment:

```bash
conda env create -f base_env.yml
conda activate base_env
```

## Training

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
