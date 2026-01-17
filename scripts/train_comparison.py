#!/usr/bin/env python3
"""
Generic training script for comparison models.

This script reads from model_registry.yaml and launches training for any registered model.
It works with the flame framework to train various sequence models (GLA, Gated DeltaNet, etc.)

Usage:
    # Train a specific model
    python scripts/train_comparison.py --model gla_170M
    
    # Train with custom experiment name
    python scripts/train_comparison.py --model gated_deltanet_340M --exp_name my_experiment
    
    # List available models
    python scripts/train_comparison.py --list
    
    # Generate SLURM script without running
    python scripts/train_comparison.py --model gla_170M --dry_run
    
    # Submit as SLURM job
    python scripts/train_comparison.py --model gla_170M --submit
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def get_script_dir():
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def get_project_root():
    """Get the project root directory."""
    return get_script_dir().parent


def load_registry():
    """Load the model registry."""
    registry_path = get_script_dir() / "model_registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found at {registry_path}")
    
    with open(registry_path) as f:
        return yaml.safe_load(f)


def get_model_config(registry, model_id):
    """Get configuration for a specific model, merging with defaults."""
    if model_id not in registry["models"]:
        available = list(registry["models"].keys())
        raise ValueError(f"Model '{model_id}' not found. Available models: {available}")
    
    # Start with defaults
    config = dict(registry["defaults"])
    
    # Override with model-specific settings
    model_config = registry["models"][model_id]
    config.update(model_config)
    
    return config


def list_models(registry):
    """Print a table of all available models."""
    print("\n" + "=" * 80)
    print("Available Comparison Models")
    print("=" * 80)
    print(f"{'Model ID':<25} {'Name':<25} {'Description'}")
    print("-" * 80)
    
    for model_id, model_config in registry["models"].items():
        name = model_config.get("name", model_id)
        desc = model_config.get("description", "")[:35]
        print(f"{model_id:<25} {name:<25} {desc}")
    
    print("=" * 80 + "\n")


def generate_train_command(config, exp_name, flame_dir):
    """Generate the training command for flame."""
    cmd_parts = [
        f'NGPU=1 bash train.sh',
        f'--job.config_file flame/models/fla.toml',
        f'--job.dump_folder "{exp_name}"',
        f'--model.config "{config["config"]}"',
        f'--model.tokenizer_path {config["tokenizer"]}',
        f'--optimizer.name AdamW',
        f'--optimizer.eps 1e-15',  # flame default
        f'--optimizer.lr {config["learning_rate"]}',
        f'--optimizer.weight_decay {config["weight_decay"]}',
        f'--lr_scheduler.warmup_steps {config["warmup_steps"]}',
        f'--lr_scheduler.lr_min 0.1',  # flame default (10% of max lr)
        f'--lr_scheduler.decay_type {config["lr_decay_type"]}',
        f'--training.batch_size {config["batch_size"]}',
        f'--training.seq_len {config["seq_len"]}',
        f'--training.gradient_accumulation_steps {config["gradient_accumulation_steps"]}',
        f'--training.steps {config["total_steps"]}',
        f'--training.max_norm {config["max_norm"]}',
        f'--training.dataset {config["dataset"]}',
    ]
    
    # Add optimizer beta parameters if specified (for GPT2 small to match TransformerScanModel or use best practices)
    if config.get("optimizer_beta1") is not None:
        cmd_parts.append(f'--optimizer.beta1 {config["optimizer_beta1"]}')
    if config.get("optimizer_beta2") is not None:
        cmd_parts.append(f'--optimizer.beta2 {config["optimizer_beta2"]}')
    dataset_name_val = config.get("dataset_name")
    if dataset_name_val is not None:
        cmd_parts.append(f'--training.dataset_name {dataset_name_val}')
    
    # Add skip_samples if specified (for reserving validation set)
    skip_samples = config.get("skip_samples", 0)
    if skip_samples > 0:
        cmd_parts.append(f'--training.skip_samples {skip_samples}')
    
    cmd_parts.extend([
        f'--training.dataset_split train',
        f'--training.num_workers {config["num_workers"]}',
        f'--training.prefetch_factor {config["prefetch_factor"]}',
        f'--training.seed {config["seed"]}',
        f'--training.dropout {config.get("dropout", 0.0)}',
        f'--checkpoint.interval {config["checkpoint_interval"]}',
        f'--checkpoint.load_step -1',
        f'--checkpoint.keep_latest_k {config["keep_latest_k"]}',
        f'--metrics.log_freq {config["log_freq"]}',
    ])
    
    if config.get("skip_nan_inf", True):
        cmd_parts.append('--training.skip_nan_inf')
    
    if config.get("enable_wandb", True):
        cmd_parts.append('--metrics.enable_wandb')
    
    return ' \\\n  '.join(cmd_parts)


def generate_slurm_script(config, model_id, exp_name, slurm_profile, flame_dir, partition_override=None):
    """Generate a complete SLURM script."""
    registry = load_registry()
    slurm_config = registry["slurm_profiles"].get(slurm_profile, registry["slurm_profiles"]["lingo"])
    
    # Allow partition override from command line
    partition = partition_override if partition_override else slurm_config["partition"]
    
    train_cmd = generate_train_command(config, exp_name, flame_dir)
    
    script = f'''#!/bin/bash
#SBATCH --job-name={model_id}-wiki103
#SBATCH --account={slurm_config["account"]}
#SBATCH --partition={partition}
#SBATCH --qos={slurm_config["qos"]}
#SBATCH --time={slurm_config["time"]}
#SBATCH --output={flame_dir}/logs/{model_id}_%j.log
#SBATCH --error={flame_dir}/logs/{model_id}_%j.err
#SBATCH --gpus={slurm_config["gpus"]}
#SBATCH --cpus-per-task={slurm_config["cpus_per_task"]}
#SBATCH --mem={slurm_config["mem"]}

# ============================================
# {config.get("name", model_id)} Training
# ============================================
# Model: {config.get("name", model_id)}
# Config: {config["config"]}
# Description: {config.get("description", "")}
# ============================================

# Initialize conda
source {slurm_config["conda_source"]}
conda activate {slurm_config["conda_env"]}

# Change to flame directory
cd {slurm_config["working_dir"]}

# Create logs directory
mkdir -p logs

echo "============================================"
echo "Training: {config.get('name', model_id)}"
echo "============================================"
echo "Model Type: {config['model_type']}"
echo "Config: {config['config']}"
echo "Dataset: {config['dataset_name']}"
echo "Batch Size: {config['batch_size']}"
echo "Seq Length: {config['seq_len']}"
echo "Learning Rate: {config['learning_rate']}"
echo "Warmup Steps: {config['warmup_steps']}"
echo "Total Steps: {config['total_steps']}"
echo "Output: {exp_name}"
echo "============================================"

{train_cmd}

echo "============================================"
echo "Training Complete!"
echo "Checkpoint saved to: {exp_name}"
echo "============================================"
'''
    return script


def run_training(config, model_id, exp_name, flame_dir, dry_run=False):
    """Run training locally (non-SLURM)."""
    train_cmd = generate_train_command(config, exp_name, flame_dir)
    
    print("=" * 60)
    print(f"Training: {config.get('name', model_id)}")
    print("=" * 60)
    print(f"Model Type: {config['model_type']}")
    print(f"Config: {config['config']}")
    print(f"Experiment: {exp_name}")
    print("=" * 60)
    
    if dry_run:
        print("\n[DRY RUN] Would execute:")
        print(train_cmd)
        return
    
    # Change to flame directory and run
    os.chdir(flame_dir)
    subprocess.run(train_cmd, shell=True, check=True)


def submit_slurm_job(config, model_id, exp_name, slurm_profile, flame_dir, dry_run=False, partition_override=None):
    """Submit a SLURM job."""
    script = generate_slurm_script(config, model_id, exp_name, slurm_profile, flame_dir, partition_override)
    
    # Create temp directory for SLURM scripts
    slurm_dir = Path(flame_dir) / "slurm_scripts"
    slurm_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = slurm_dir / f"train_{model_id}_{timestamp}.sh"
    
    with open(script_path, 'w') as f:
        f.write(script)
    
    print(f"Generated SLURM script: {script_path}")
    
    if dry_run:
        print("\n[DRY RUN] SLURM script contents:")
        print("-" * 60)
        print(script)
        return
    
    # Submit the job
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Job submitted: {result.stdout.strip()}")
    else:
        print(f"Failed to submit job: {result.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train comparison models using the flame framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available models
    python scripts/train_comparison.py --list
    
    # Train GLA-170M locally
    python scripts/train_comparison.py --model gla_170M
    
    # Submit Gated DeltaNet as SLURM job
    python scripts/train_comparison.py --model gated_deltanet_340M --submit
    
    # Generate SLURM script without submitting
    python scripts/train_comparison.py --model mamba_340M --submit --dry_run
        """
    )
    
    parser.add_argument("--model", type=str, help="Model ID from registry (e.g., gla_170M)")
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--exp_name", type=str, default=None, help="Custom experiment name")
    parser.add_argument("--submit", action="store_true", help="Submit as SLURM job")
    parser.add_argument("--slurm_profile", type=str, default="lingo", 
                       help="SLURM profile to use (lingo, vision, vision_h100)")
    parser.add_argument("--partition", type=str, default=None,
                       help="Override SLURM partition (e.g., vision-shared-h100)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--flame_dir", type=str, default=None, help="Path to flame directory")
    
    # Override training parameters
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--seq_len", type=int, help="Override sequence length")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--warmup_steps", type=int, help="Override warmup steps")
    parser.add_argument("--total_steps", type=int, help="Override total training steps")
    parser.add_argument("--dropout", type=float, help="Override dropout rate (default: 0.1)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    registry = load_registry()
    
    if args.list:
        list_models(registry)
        return
    
    if not args.model:
        parser.error("--model is required (use --list to see available models)")
    
    # Get model configuration
    config = get_model_config(registry, args.model)
    
    # Apply overrides
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seq_len:
        config["seq_len"] = args.seq_len
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.warmup_steps:
        config["warmup_steps"] = args.warmup_steps
    if args.total_steps:
        config["total_steps"] = args.total_steps
    if args.dropout is not None:
        config["dropout"] = args.dropout
    if args.no_wandb:
        config["enable_wandb"] = False
    
    # Set up paths
    flame_dir = args.flame_dir or str(get_project_root() / "flame")
    
    # Generate experiment name
    if args.exp_name:
        exp_name = f"exp/{args.exp_name}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Get dataset name from config (normalize for filename)
        dataset_name = config.get('dataset', 'unknown')
        # Normalize dataset name: remove special chars, use short form
        if 'wikitext' in dataset_name.lower():
            dataset_short = 'wikitext103'
        elif 'openwebtext' in dataset_name.lower():
            dataset_short = 'openwebtext'
        else:
            dataset_short = dataset_name.replace('-', '_').replace('/', '_')
        
        # Always include dropout in name for clarity
        dropout_val = config.get('dropout', 0.0)
        if dropout_val == 0.0:
            dropout_str = 'nodrop'
        else:
            dropout_str = f'drop{dropout_val}'
        
        # Include other overridden params if specified
        suffix = ""
        if args.learning_rate is not None:
            suffix += f"_lr{config['learning_rate']}"
        
        exp_name = f"exp/{args.model}_{dataset_short}_{dropout_str}{suffix}_{timestamp}"
    
    # Run or submit
    if args.submit:
        submit_slurm_job(
            config, args.model, exp_name, 
            args.slurm_profile, flame_dir, args.dry_run,
            partition_override=args.partition
        )
    else:
        run_training(config, args.model, exp_name, flame_dir, args.dry_run)


if __name__ == "__main__":
    main()
