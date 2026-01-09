#!/usr/bin/env python3
"""
Generic evaluation script for comparison models.

This script evaluates checkpoints from any model trained via train_comparison.py.
It automatically detects the model type and loads appropriate configs.

Usage:
    # Evaluate all checkpoints for an experiment
    python scripts/evaluate_comparison.py --exp_path flame/exp/gla_170M-wikitext103-XXXXX
    
    # Evaluate with explicit model type
    python scripts/evaluate_comparison.py --exp_path flame/exp/my_exp --model gla_170M
    
    # Evaluate specific checkpoint steps
    python scripts/evaluate_comparison.py --exp_path flame/exp/my_exp --steps 5000 10000 15000
    
    # Compare multiple experiments
    python scripts/evaluate_comparison.py --compare \
        flame/exp/gla_170M-... flame/exp/gated_deltanet_340M-...
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

# IMPORTANT: Import fla.models FIRST to register model types with transformers
# This must happen before any transformers imports
# We import specific model modules to ensure their configs get registered
import fla  # noqa: F401

# Import each model module to register their configs with transformers AutoConfig
# This is necessary because just importing fla doesn't auto-register all model types
from fla.models.gla import GLAConfig
from fla.models.delta_net import DeltaNetConfig
from fla.models.gated_deltanet import GatedDeltaNetConfig
from fla.models.mamba import MambaConfig
from fla.models.mamba2 import Mamba2Config
from fla.models.transformer import TransformerConfig
from fla.models.gsa import GSAConfig
from fla.models.hgrn2 import HGRN2Config

# Mapping from model_type string to config class (for direct instantiation fallback)
FLA_CONFIG_MAPPING = {
    "gla": GLAConfig,
    "delta_net": DeltaNetConfig,
    "gated_deltanet": GatedDeltaNetConfig,
    "mamba": MambaConfig,
    "mamba2": Mamba2Config,
    "transformer": TransformerConfig,
    "gsa": GSAConfig,
    "hgrn2": HGRN2Config,
}

import torch
from torch.distributed.checkpoint import load as dcp_load
import matplotlib.pyplot as plt
from tqdm import tqdm
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


def detect_model_from_exp_path(exp_path):
    """Attempt to detect model type from experiment path name."""
    exp_name = Path(exp_path).name
    exp_name_lower = exp_name.lower()
    
    # Try to match against registered model IDs (sorted by length descending for specificity)
    registry = load_registry()
    model_ids = sorted(registry["models"].keys(), key=len, reverse=True)
    
    for model_id in model_ids:
        model_id_lower = model_id.lower()
        # Check various formats: with underscores, with hyphens
        if (model_id_lower in exp_name_lower or 
            model_id_lower.replace("_", "-") in exp_name_lower):
            return model_id
    
    # Fallback: try to match model type (ordered from most specific to least specific)
    # Important: more specific types must come first to avoid partial matches
    model_types = ["gated_deltanet", "delta_net", "mamba2", "mamba", "hgrn2", "transformer", "gsa", "gla"]
    for mt in model_types:
        # Check with underscores, without underscores, and with hyphens
        if (mt in exp_name_lower or 
            mt.replace("_", "") in exp_name_lower or 
            mt.replace("_", "-") in exp_name_lower):
            # Find first matching model ID for this model type
            for model_id, config in registry["models"].items():
                if config.get("model_type") == mt:
                    return model_id
    
    return None


def get_model_config_path(model_id, flame_dir):
    """Get the config path for a model."""
    registry = load_registry()
    if model_id not in registry["models"]:
        raise ValueError(f"Unknown model: {model_id}")
    
    config_rel_path = registry["models"][model_id]["config"]
    return os.path.join(flame_dir, config_rel_path)


def find_checkpoints(exp_path):
    """Find all available checkpoint steps."""
    checkpoint_dir = os.path.join(exp_path, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}")
        return []
    
    steps = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith('step-'):
            try:
                step = int(name.split('-')[1])
                steps.append(step)
            except ValueError:
                continue
    
    return sorted(steps)


def load_fla_config(config_path):
    """Load fla config using direct config class instantiation.
    
    This is more robust than AutoConfig.from_pretrained for local JSON files
    because it directly uses the fla config classes.
    """
    # Read the JSON config to determine model_type
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    model_type = config_dict.get("model_type")
    if model_type is None:
        raise ValueError(f"Config at {config_path} does not have 'model_type' field")
    
    if model_type not in FLA_CONFIG_MAPPING:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported types: {list(FLA_CONFIG_MAPPING.keys())}"
        )
    
    # Get the config class and instantiate from the dict
    config_class = FLA_CONFIG_MAPPING[model_type]
    config = config_class(**config_dict)
    
    return config


def convert_checkpoint(exp_path, step, config_path, tokenizer_path='gpt2'):
    """Convert DCP checkpoint to HuggingFace format using working DCP load method."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    checkpoint_dir = os.path.join(exp_path, f'checkpoint/step-{step}')
    hf_path = os.path.join(exp_path, f'hf-{step}')
    
    # Check if already converted
    if os.path.exists(hf_path) and os.path.exists(os.path.join(hf_path, 'model.safetensors')):
        print(f"  HF checkpoint already exists at {hf_path}")
        return hf_path
    
    print(f"  Converting step {step} to HuggingFace format...")
    
    os.makedirs(hf_path, exist_ok=True)
    
    # Load config using direct fla config class (more robust than AutoConfig)
    config = load_fla_config(config_path)
    config.save_pretrained(hf_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(hf_path)
    
    # Create model and get state dict structure
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Create state dict with the expected structure for DCP
    state_dict = {'model': model.state_dict()}
    
    # Load checkpoint using DCP
    try:
        dcp_load(state_dict, checkpoint_id=checkpoint_dir)
    except Exception as e:
        print(f"  DCP load failed: {e}")
        return None
    
    # Load state dict into model
    model.load_state_dict(state_dict['model'])
    
    # Save in HuggingFace format
    model.save_pretrained(hf_path)
    
    print(f"  Conversion complete!")
    return hf_path


def evaluate_checkpoint(model_path, dataset_name='wikitext-103', seq_len=512, batch_size=8, device='cuda'):
    """Evaluate a single checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load validation data
    if dataset_name == "wikitext-103":
        data = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    else:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    text = " ".join(data["text"])
    tokenizer.model_max_length = int(1e7)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # Create samples
    samples = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        samples.append(token_ids[i:i+seq_len])
    
    # Compute perplexity
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            input_ids = torch.tensor(batch_samples, dtype=torch.long, device=device)
            
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    return avg_loss, perplexity


def evaluate_experiment(exp_path, model_id=None, config_path=None, dataset='wikitext-103', 
                       seq_len=512, batch_size=8, steps=None, output=None):
    """Evaluate all checkpoints for an experiment."""
    
    # Auto-detect model if not provided
    if model_id is None:
        model_id = detect_model_from_exp_path(exp_path)
        if model_id:
            print(f"Auto-detected model: {model_id}")
        else:
            print("Warning: Could not auto-detect model type. Please specify --model")
    
    # Get config path
    if config_path is None and model_id:
        flame_dir = str(get_project_root() / "flame")
        config_path = get_model_config_path(model_id, flame_dir)
    
    if config_path is None:
        raise ValueError("Could not determine model config. Please specify --config or --model")
    
    print("=" * 60)
    print("Checkpoint Evaluation")
    print("=" * 60)
    print(f"Experiment: {exp_path}")
    print(f"Model: {model_id or 'Unknown'}")
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset}")
    print("=" * 60)
    
    # Find checkpoints
    available_steps = find_checkpoints(exp_path)
    if not available_steps:
        print("No checkpoints found!")
        return []
    
    # Filter steps if specified
    if steps:
        eval_steps = [s for s in steps if s in available_steps]
        if len(eval_steps) < len(steps):
            missing = set(steps) - set(eval_steps)
            print(f"Warning: Steps not found: {missing}")
    else:
        eval_steps = available_steps
    
    print(f"Evaluating {len(eval_steps)} checkpoints: {eval_steps}")
    
    results = []
    
    for step in tqdm(eval_steps, desc="Evaluating checkpoints"):
        print(f"\n--- Step {step} ---")
        
        # Convert checkpoint
        hf_path = convert_checkpoint(exp_path, step, config_path)
        
        if hf_path is None:
            print(f"  Skipping step {step} (conversion failed)")
            continue
        
        # Evaluate
        try:
            loss, ppl = evaluate_checkpoint(hf_path, dataset, seq_len, batch_size)
            results.append({
                'step': step,
                'val_loss': loss,
                'val_ppl': ppl
            })
            print(f"  Loss: {loss:.4f}, Perplexity: {ppl:.2f}")
        except Exception as e:
            print(f"  Error evaluating: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Step':>10} | {'Val Loss':>10} | {'Val PPL':>10}")
    print("-" * 36)
    for r in results:
        print(f"{r['step']:>10} | {r['val_loss']:>10.4f} | {r['val_ppl']:>10.2f}")
    print("=" * 60)
    
    # Save results
    output_path = output or os.path.join(exp_path, 'eval_results.json')
    results_data = {
        'model': model_id,
        'config': config_path,
        'dataset': dataset,
        'results': results
    }
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to: {output_path}")
    
    # Plot validation curve
    if results:
        plot_results([{'name': model_id or 'Model', 'results': results}], 
                    os.path.join(exp_path, 'eval_curve.png'))
    
    return results


def plot_results(experiments, output_path):
    """Plot validation curves for one or more experiments."""
    plt.figure(figsize=(12, 5))
    
    colors = plt.cm.tab10.colors
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    for i, exp in enumerate(experiments):
        steps = [r['step'] for r in exp['results']]
        losses = [r['val_loss'] for r in exp['results']]
        color = colors[i % len(colors)]
        plt.plot(steps, losses, '-o', label=exp['name'], color=color)
    
    plt.xlabel('Step')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Perplexity subplot
    plt.subplot(1, 2, 2)
    for i, exp in enumerate(experiments):
        steps = [r['step'] for r in exp['results']]
        ppls = [r['val_ppl'] for r in exp['results']]
        color = colors[i % len(colors)]
        plt.plot(steps, ppls, '-o', label=exp['name'], color=color)
    
    plt.xlabel('Step')
    plt.ylabel('Validation Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()


def compare_experiments(exp_paths, output_dir=None):
    """Compare validation curves across multiple experiments."""
    experiments = []
    
    for exp_path in exp_paths:
        # Try to load existing results
        results_file = os.path.join(exp_path, 'eval_results.json')
        if os.path.exists(results_file):
            with open(results_file) as f:
                data = json.load(f)
            
            name = data.get('model') or Path(exp_path).name
            experiments.append({
                'name': name,
                'path': exp_path,
                'results': data['results']
            })
            print(f"Loaded results for: {name}")
        else:
            print(f"Warning: No eval_results.json found at {exp_path}")
            print("  Run evaluation first: python scripts/evaluate_comparison.py --exp_path " + exp_path)
    
    if not experiments:
        print("No experiments to compare!")
        return
    
    # Generate comparison plot
    if output_dir is None:
        output_dir = str(get_project_root() / "comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, 'comparison_curves.png')
    plot_results(experiments, plot_path)
    
    # Save comparison summary
    summary = {
        'experiments': [
            {
                'name': exp['name'],
                'path': exp['path'],
                'final_loss': exp['results'][-1]['val_loss'] if exp['results'] else None,
                'final_ppl': exp['results'][-1]['val_ppl'] if exp['results'] else None,
                'best_loss': min(r['val_loss'] for r in exp['results']) if exp['results'] else None,
                'best_ppl': min(r['val_ppl'] for r in exp['results']) if exp['results'] else None,
            }
            for exp in experiments
        ]
    }
    
    summary_path = os.path.join(output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(f"{'Model':<30} | {'Final Loss':>12} | {'Final PPL':>12} | {'Best PPL':>12}")
    print("-" * 80)
    for exp in summary['experiments']:
        final_loss = f"{exp['final_loss']:.4f}" if exp['final_loss'] else "N/A"
        final_ppl = f"{exp['final_ppl']:.2f}" if exp['final_ppl'] else "N/A"
        best_ppl = f"{exp['best_ppl']:.2f}" if exp['best_ppl'] else "N/A"
        print(f"{exp['name']:<30} | {final_loss:>12} | {final_ppl:>12} | {best_ppl:>12}")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints from comparison model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate single experiment
    python scripts/evaluate_comparison.py --exp_path flame/exp/gla_170M-wikitext103-...
    
    # Evaluate with explicit model
    python scripts/evaluate_comparison.py --exp_path flame/exp/my_exp --model gated_deltanet_340M
    
    # Evaluate specific steps only
    python scripts/evaluate_comparison.py --exp_path flame/exp/my_exp --steps 5000 10000 20000
    
    # Compare multiple experiments
    python scripts/evaluate_comparison.py --compare \\
        flame/exp/gla_170M-... \\
        flame/exp/gated_deltanet_340M-... \\
        flame/exp/mamba_340M-...
        """
    )
    
    # Main options
    parser.add_argument("--exp_path", type=str, help="Path to experiment folder")
    parser.add_argument("--model", type=str, help="Model ID from registry (for config lookup)")
    parser.add_argument("--config", type=str, help="Explicit path to model config")
    
    # Evaluation options
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                       choices=["wikitext-103", "wikitext-2"])
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs='+', help="Specific steps to evaluate")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    
    # Comparison mode
    parser.add_argument("--compare", type=str, nargs='+', 
                       help="Compare multiple experiment paths")
    parser.add_argument("--compare_output", type=str, 
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_experiments(args.compare, args.compare_output)
    elif args.exp_path:
        evaluate_experiment(
            args.exp_path,
            model_id=args.model,
            config_path=args.config,
            dataset=args.dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            output=args.output
        )
    else:
        parser.error("Either --exp_path or --compare is required")


if __name__ == "__main__":
    main()
