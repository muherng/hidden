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


def generate_train_command(config, exp_name, flame_dir, use_shell_var=False):
    """Generate the training command for flame.
    
    Args:
        use_shell_var: If True, uses $EXP_NAME shell variable instead of hardcoded path.
                       This is used in SLURM scripts to include job ID dynamically.
    """
    dump_folder = '$EXP_NAME' if use_shell_var else f'"{exp_name}"'
    cmd_parts = [
        f'NGPU=1 bash train.sh',
        f'--job.config_file flame/models/fla.toml',
        f'--job.dump_folder {dump_folder}',
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
    
    # Add enable_qkv_dropout if specified (for Q/K/V projection dropout experiment)
    if config.get("enable_qkv_dropout", False):
        cmd_parts.append('--training.enable_qkv_dropout')
    
    if config.get("skip_nan_inf", True):
        cmd_parts.append('--training.skip_nan_inf')
    
    if config.get("enable_wandb", True):
        cmd_parts.append('--metrics.enable_wandb')
    
    return ' \\\n  '.join(cmd_parts)


def generate_gpt2_small_slurm_script(config, model_id, exp_name, slurm_profile, dry_run=False, partition_override=None, qos_override=None):
    """Generate SLURM script for GPT2 small using HuggingFace Trainer."""
    registry = load_registry()
    slurm_config = registry["slurm_profiles"].get(slurm_profile, registry["slurm_profiles"]["lingo"])
    partition = partition_override if partition_override else slurm_config["partition"]
    qos = qos_override if qos_override else slurm_config["qos"]
    
    project_root = get_project_root()
    flame_dir = str(project_root / "flame")
    script_path = project_root / "models" / "train_gpt2_small.py"
    
    # Build training command
    cmd_parts = [f"python3 {script_path}"]
    
    if 'wikitext' in config.get('dataset', ''):
        cmd_parts.append('--dataset wikitext-103')
    elif 'openwebtext' in config.get('dataset', ''):
        cmd_parts.append('--dataset openwebtext')
        if config.get('skip_samples'):
            cmd_parts.append(f'--skip_samples {config["skip_samples"]}')
    
    cmd_parts.extend([
        f'--dropout {config.get("dropout", 0.0)}',
        f'--learning_rate {config.get("learning_rate", 6e-4)}',
        f'--weight_decay {config.get("weight_decay", 0.1)}',
        f'--warmup_steps {config.get("warmup_steps", 2000)}',
        f'--batch_size {config.get("batch_size", 32)}',
        f'--seq_len {config.get("seq_len", 1024)}',
        f'--gradient_accumulation_steps {config.get("gradient_accumulation_steps", 16)}',
        f'--tokenize_workers {config.get("tokenize_workers", 8)}',
        f'--adam_beta1 {config.get("optimizer_beta1", 0.9)}',
        f'--adam_beta2 {config.get("optimizer_beta2", 0.95)}',
    ])
    
    if config.get('total_steps'):
        cmd_parts.append(f'--total_steps {config["total_steps"]}')
    else:
        cmd_parts.append('--epochs 10')
    
    cmd_parts.extend([
        f'--save_steps {config.get("checkpoint_interval", 5000)}',
        f'--eval_steps {config.get("checkpoint_interval", 5000)}',
        f'--logging_steps {config.get("log_freq", 100)}',
        '--job_id $SLURM_JOB_ID',  # Include SLURM job ID in experiment path
    ])
    
    train_cmd = ' \\\n  '.join(cmd_parts)
    
    script = f'''#!/bin/bash
#SBATCH --job-name={model_id}
#SBATCH --account={slurm_config["account"]}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time={slurm_config["time"]}
#SBATCH --output={flame_dir}/logs/{model_id}_%j.log
#SBATCH --error={flame_dir}/logs/{model_id}_%j.err
#SBATCH --gpus={slurm_config["gpus"]}
#SBATCH --cpus-per-task={slurm_config["cpus_per_task"]}
#SBATCH --mem={slurm_config["mem"]}

# ============================================
# {config.get("name", model_id)} Training (HuggingFace)
# ============================================
# Model: {config.get("name", model_id)}
# Description: {config.get("description", "")}
# Using: HuggingFace Trainer (no Flash Attention required)
# ============================================

# Initialize conda
source {slurm_config["conda_source"]}
conda activate {slurm_config["conda_env"]}

# Ensure CUDA libraries are found (fix for PyTorch CUDA detection)
# PyTorch pip packages store CUDA libs in nvidia subdirectories
NVIDIA_LIB=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/nvjitlink/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Debug: Check GPU visibility
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
nvidia-smi || echo "nvidia-smi failed"
python3 -c "import torch; print(f'PyTorch CUDA check: available={{torch.cuda.is_available()}}, device_count={{torch.cuda.device_count()}}')"

# Change to project root
cd {project_root}

# Create logs directory
mkdir -p {flame_dir}/logs

echo "============================================"
echo "Training: {config.get('name', model_id)}"
echo "============================================"
echo "Using: HuggingFace Trainer (canonical nanoGPT hyperparameters)"
echo "Dataset: {config.get('dataset_name', config.get('dataset', 'unknown'))}"
echo "Batch Size: {config.get('batch_size', 32)} (microbatch)"
echo "Gradient Accumulation: {config.get('gradient_accumulation_steps', 16)}"
echo "Seq Length: {config.get('seq_len', 1024)}"
echo "Effective Tokens/Step: {config.get('batch_size', 32) * config.get('gradient_accumulation_steps', 16) * config.get('seq_len', 1024):,}"
echo "Learning Rate: {config.get('learning_rate', 6e-4)}"
echo "Weight Decay: {config.get('weight_decay', 0.1)}"
echo "Warmup Steps: {config.get('warmup_steps', 2000)}"
echo "Total Steps: {config.get('total_steps', 'N/A (using epochs)')}"
echo "============================================"

{train_cmd}

echo "============================================"
echo "Training Complete!"
echo "============================================"
'''
    return script


def generate_slurm_script(config, model_id, exp_name, slurm_profile, flame_dir, partition_override=None, qos_override=None):
    """Generate a complete SLURM script."""
    registry = load_registry()
    slurm_config = registry["slurm_profiles"].get(slurm_profile, registry["slurm_profiles"]["lingo"])
    
    # Allow partition and qos override from command line
    partition = partition_override if partition_override else slurm_config["partition"]
    qos = qos_override if qos_override else slurm_config["qos"]
    
    # Generate train command using shell variable for exp_name (to include job ID at runtime)
    train_cmd = generate_train_command(config, exp_name, flame_dir, use_shell_var=True)
    
    # Determine dataset short name for job name
    dataset_name = config.get('dataset', 'unknown')
    if 'openwebtext' in dataset_name.lower():
        dataset_short = 'owt'
    elif 'wikitext' in dataset_name.lower():
        dataset_short = 'wiki103'
    else:
        dataset_short = dataset_name[:8]
    
    script = f'''#!/bin/bash
#SBATCH --job-name={model_id}-{dataset_short}
#SBATCH --account={slurm_config["account"]}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
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

# Set wandb API key (read from ~/.netrc - handles indented format)
export WANDB_API_KEY=$(awk '/machine api.wandb.ai/{{getline; getline; print $2}}' ~/.netrc)
export WANDB_PROJECT="spd_icml"
# Use offline mode as fallback if no internet (will sync later)
export WANDB_MODE=${{WANDB_MODE:-offline}}

# Change to flame directory
cd {slurm_config["working_dir"]}

# Create logs directory
mkdir -p logs

# Construct experiment name with SLURM job ID appended
EXP_NAME="{exp_name}_${{SLURM_JOB_ID}}"

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
echo "Output: $EXP_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "============================================"

{train_cmd}

echo "============================================"
echo "Training Complete!"
echo "Checkpoint saved to: $EXP_NAME"
echo "============================================"
'''
    return script


def run_training(config, model_id, exp_name, flame_dir, dry_run=False):
    """Run training locally (non-SLURM)."""
    # Check if this is a GPT2 small model (use HuggingFace instead of flame)
    if model_id.startswith('gpt2_small'):
        run_gpt2_small_training(config, model_id, exp_name, dry_run)
        return
    
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


def run_gpt2_small_training(config, model_id, exp_name, dry_run=False):
    """Run GPT2 small training using HuggingFace Trainer (avoids Flash Attention requirement)."""
    project_root = get_project_root()
    script_path = project_root / "models" / "train_gpt2_small.py"
    
    # Build command
    cmd = ["python3", str(script_path)]
    
    # Dataset
    if 'wikitext' in config.get('dataset', ''):
        cmd.extend(['--dataset', 'wikitext-103'])
    elif 'openwebtext' in config.get('dataset', ''):
        cmd.extend(['--dataset', 'openwebtext'])
        if config.get('skip_samples'):
            cmd.extend(['--skip_samples', str(config['skip_samples'])])
    
    # Hyperparameters (nanoGPT canonical defaults)
    cmd.extend(['--dropout', str(config.get('dropout', 0.0))])
    cmd.extend(['--learning_rate', str(config.get('learning_rate', 6e-4))])
    cmd.extend(['--weight_decay', str(config.get('weight_decay', 0.1))])
    cmd.extend(['--warmup_steps', str(config.get('warmup_steps', 2000))])
    cmd.extend(['--batch_size', str(config.get('batch_size', 32))])
    cmd.extend(['--seq_len', str(config.get('seq_len', 1024))])
    cmd.extend(['--gradient_accumulation_steps', str(config.get('gradient_accumulation_steps', 16))])
    cmd.extend(['--tokenize_workers', str(config.get('tokenize_workers', 8))])
    cmd.extend(['--adam_beta1', str(config.get('optimizer_beta1', 0.9))])
    cmd.extend(['--adam_beta2', str(config.get('optimizer_beta2', 0.95))])
    
    # Training steps
    if config.get('total_steps'):
        cmd.extend(['--total_steps', str(config['total_steps'])])
    else:
        cmd.extend(['--epochs', '10'])
    
    # Checkpointing
    cmd.extend(['--save_steps', str(config.get('checkpoint_interval', 5000))])
    cmd.extend(['--eval_steps', str(config.get('checkpoint_interval', 5000))])
    cmd.extend(['--logging_steps', str(config.get('log_freq', 100))])
    
    print("=" * 60)
    print(f"Training: {config.get('name', model_id)}")
    print("=" * 60)
    print(f"Using: HuggingFace Trainer (no Flash Attention required)")
    print(f"Experiment: {exp_name}")
    print("=" * 60)
    
    if dry_run:
        print("\n[DRY RUN] Would execute:")
        print(' '.join(cmd))
        return
    
    # Run training
    os.chdir(project_root)
    subprocess.run(cmd, check=True)


def submit_slurm_job(config, model_id, exp_name, slurm_profile, flame_dir, dry_run=False, partition_override=None, qos_override=None):
    """Submit a SLURM job."""
    # Check if this is a GPT2 small model (use HuggingFace instead of flame)
    if model_id.startswith('gpt2_small'):
        script = generate_gpt2_small_slurm_script(config, model_id, exp_name, slurm_profile, dry_run, partition_override, qos_override)
    else:
        script = generate_slurm_script(config, model_id, exp_name, slurm_profile, flame_dir, partition_override, qos_override)
    
    # Create temp directory for SLURM scripts
    slurm_dir = Path(flame_dir) / "slurm_scripts"
    slurm_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = slurm_dir / f"train_{model_id}_{timestamp}.sh"
    
    # For GPT2 small, exp_name is generated by the training script, not passed
    if model_id.startswith('gpt2_small'):
        # The training script generates its own exp_name, so we don't need to pass it
        pass
    
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
    parser.add_argument("--qos", type=str, default="lingo-main",
                       help="SLURM QoS (default: lingo-main, alternative: shared-if-available)")
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
        
        # Avoid duplicating dataset name if already in model_id
        if dataset_short.lower() in args.model.lower():
            exp_name = f"exp/{args.model}_{dropout_str}{suffix}_{timestamp}"
        else:
            exp_name = f"exp/{args.model}_{dataset_short}_{dropout_str}{suffix}_{timestamp}"
    
    # Run or submit
    if args.submit:
        submit_slurm_job(
            config, args.model, exp_name, 
            args.slurm_profile, flame_dir, args.dry_run,
            partition_override=args.partition,
            qos_override=args.qos
        )
    else:
        run_training(config, args.model, exp_name, flame_dir, args.dry_run)


if __name__ == "__main__":
    main()
