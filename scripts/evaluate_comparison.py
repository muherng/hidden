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
    
    # Evaluate on OpenWebText (for models trained on OpenWebText)
    python scripts/evaluate_comparison.py --exp_path flame/exp/gla_170M_openwebtext-... --dataset openwebtext
    
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
    
    # GPT2 small special case: match based on dataset and dropout pattern
    # Exp names look like: gpt2_small_wikitext_103_drop0.1_281519 or gpt2_small_openwebtext_drop0.1_281519
    if "gpt2_small" in exp_name_lower:
        # Determine dataset
        if "wikitext" in exp_name_lower:
            dataset = "wikitext103"
        elif "openwebtext" in exp_name_lower:
            dataset = "openwebtext"
        else:
            dataset = "wikitext103"  # Default
        
        # Determine dropout (look for drop0.1, drop01, nodrop, etc.)
        if "nodrop" in exp_name_lower or "drop0.0" in exp_name_lower or "drop0_0" in exp_name_lower:
            dropout = "nodrop"
        else:
            dropout = "drop01"  # Default to dropout=0.1
        
        model_id = f"gpt2_small_{dataset}_{dropout}"
        if model_id in registry["models"]:
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
    """Find all available checkpoint steps.
    
    Supports both flame format (checkpoint/step-{step}) and HuggingFace format (checkpoint-{step}).
    """
    # Try flame format first
    checkpoint_dir = os.path.join(exp_path, 'checkpoint')
    steps = []
    
    if os.path.exists(checkpoint_dir):
        # Flame format: checkpoint/step-{step}
        for name in os.listdir(checkpoint_dir):
            if name.startswith('step-'):
                try:
                    step = int(name.split('-')[1])
                    steps.append(step)
                except ValueError:
                    continue
    
    # Try HuggingFace format: checkpoint-{step} directories in exp_path
    if not steps:
        for name in os.listdir(exp_path):
            if name.startswith('checkpoint-'):
                try:
                    step = int(name.split('-')[1])
                    steps.append(step)
                except ValueError:
                    continue
    
    if not steps:
        print(f"No checkpoints found in {exp_path}")
        return []
    
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
    """Convert checkpoint to HuggingFace format.
    
    Supports both flame DCP format and HuggingFace format (no conversion needed).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Check if already in HuggingFace format (from HuggingFace Trainer)
    hf_checkpoint = os.path.join(exp_path, f'checkpoint-{step}')
    if os.path.exists(hf_checkpoint):
        # Already in HuggingFace format, just return it
        print(f"  Using HuggingFace checkpoint at {hf_checkpoint}")
        return hf_checkpoint
    
    # Try flame DCP format
    checkpoint_dir = os.path.join(exp_path, f'checkpoint/step-{step}')
    hf_path = os.path.join(exp_path, f'hf-{step}')
    
    if not os.path.exists(checkpoint_dir):
        print(f"  No checkpoint found for step {step}")
        return None
    
    # Check if already converted
    if os.path.exists(hf_path) and os.path.exists(os.path.join(hf_path, 'model.safetensors')):
        print(f"  HF checkpoint already exists at {hf_path}")
        return hf_path
    
    print(f"  Converting step {step} from DCP to HuggingFace format...")
    
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


def evaluate_checkpoint(model_path, token_ids=None, dataset_name='wikitext-103', seq_len=512, batch_size=8, device='cuda'):
    """Evaluate a single checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        token_ids: Pre-tokenized validation data (optional). If provided, dataset_name is ignored.
        dataset_name: Dataset name (only used if token_ids is None)
        seq_len: Sequence length
        batch_size: Batch size
        device: Device to run on
    """
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
    
    # Load validation data (only if not pre-provided)
    if token_ids is None:
        if dataset_name == "wikitext-103":
            data = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            text = " ".join(data["text"])
        elif dataset_name == "wikitext-2":
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            text = " ".join(data["text"])
        elif dataset_name == "openwebtext":
            # ========================================================================
            # VALIDATION SET SELECTION FOR OPENWEBTEXT
            # ========================================================================
            # OpenWebText doesn't have a standard validation split. We use a reserved
            # validation set approach:
            # 
            # Approach:
            # - Training skips the first N samples (specified by skip_samples in model registry)
            # - Validation uses a subset of those first N samples (first 1,000 samples)
            # - This ensures no overlap between training and validation data
            # 
            # Configuration:
            # - skip_samples is set in model_registry.yaml for OpenWebText models (typically 10,000)
            # - Validation uses first 1,000 samples (subset of skipped samples for faster evaluation)
            # - Since training skips 10,000 samples, using first 1,000 is safe and non-overlapping
            # ========================================================================
            
            # Use first 1,000 samples for validation (subset of the 10,000 skipped during training)
            VALIDATION_SIZE = 1_000  # Using subset of skipped samples for faster validation
            
            print(f"Loading OpenWebText validation subset (first {VALIDATION_SIZE:,} samples)...")
            print(f"  Note: Training skips first 10,000 samples, so these are reserved for validation")
            
            full_data = load_dataset("openwebtext", "plain_text", split="train", streaming=True)
            
            # Take the first N samples (which training skips)
            val_texts = []
            for i, sample in enumerate(full_data):
                if len(val_texts) >= VALIDATION_SIZE:
                    break  # Stop after collecting reserved validation samples
                val_texts.append(sample["text"])
            
            if len(val_texts) < VALIDATION_SIZE:
                raise ValueError(
                    f"Only collected {len(val_texts)} validation samples, expected {VALIDATION_SIZE}. "
                    f"This may indicate the dataset is smaller than expected."
                )
            
            print(f"  Collected {len(val_texts):,} validation samples")
            # Join all validation texts directly
            text = " ".join(val_texts)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: wikitext-103, wikitext-2, openwebtext")
        
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
                       seq_len=512, batch_size=8, steps=None, output=None, cleanup=True):
    """
    Evaluate all checkpoints for an experiment.
    
    Args:
        dataset: Evaluation dataset. Use 'openwebtext' for models trained on OpenWebText,
                 'wikitext-103' or 'wikitext-2' for WikiText-trained models.
    """
    """Evaluate all checkpoints for an experiment."""
    
    # Auto-detect model if not provided
    if model_id is None:
        model_id = detect_model_from_exp_path(exp_path)
        if model_id:
            print(f"Auto-detected model: {model_id}")
        else:
            print("Warning: Could not auto-detect model type. Please specify --model")
    
    # For GPT2 small models (HuggingFace), config_path is not needed
    is_gpt2_small = model_id and model_id.startswith('gpt2_small')
    
    # Get config path (only needed for flame models)
    if config_path is None and model_id and not is_gpt2_small:
        flame_dir = str(get_project_root() / "flame")
        config_path = get_model_config_path(model_id, flame_dir)
    
    if config_path is None and not is_gpt2_small:
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
    
    # Load and tokenize validation data ONCE before the loop
    print(f"\nLoading validation data for {dataset}...")
    token_ids = None
    if dataset == "openwebtext":
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        # Use first 1,000 samples for validation (subset of the 10,000 skipped during training)
        VALIDATION_SIZE = 1_000  # Using subset of skipped samples for faster validation
        print(f"Loading OpenWebText validation subset (first {VALIDATION_SIZE:,} samples)...")
        print(f"  Note: Training skips first 10,000 samples, so these are reserved for validation")
        
        full_data = load_dataset("openwebtext", "plain_text", split="train", streaming=True)
        val_texts = []
        for i, sample in enumerate(full_data):
            if len(val_texts) >= VALIDATION_SIZE:
                break
            val_texts.append(sample["text"])
        
        if len(val_texts) < VALIDATION_SIZE:
            raise ValueError(
                f"Only collected {len(val_texts)} validation samples, expected {VALIDATION_SIZE}."
            )
        
        print(f"  Collected {len(val_texts):,} validation samples")
        text = " ".join(val_texts)
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e7)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"  Tokenized into {len(token_ids):,} tokens")
    elif dataset in ["wikitext-103", "wikitext-2"]:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        if dataset == "wikitext-103":
            data = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        else:
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = " ".join(data["text"])
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e7)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"  Tokenized into {len(token_ids):,} tokens")
    
    results = []
    
    for step in tqdm(eval_steps, desc="Evaluating checkpoints"):
        print(f"\n--- Step {step} ---")
        
        # Convert checkpoint (or use existing HuggingFace checkpoint)
        if is_gpt2_small:
            # GPT2 small uses HuggingFace format directly
            hf_path = os.path.join(exp_path, f'checkpoint-{step}')
            if not os.path.exists(hf_path):
                print(f"  Checkpoint not found at {hf_path}")
                continue
        else:
            # Flame models need conversion from DCP
            hf_path = convert_checkpoint(exp_path, step, config_path)
            if hf_path is None:
                print(f"  Skipping step {step} (conversion failed)")
                continue
        
        # Evaluate (pass pre-tokenized data)
        try:
            loss, ppl = evaluate_checkpoint(hf_path, token_ids=token_ids, seq_len=seq_len, batch_size=batch_size)
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
    
    # Clean up hf-* directories after evaluation if requested
    if cleanup:
        print("\n" + "=" * 60)
        print("Cleaning up HuggingFace checkpoint directories...")
        print("=" * 60)
        cleanup_hf_checkpoints(exp_path=exp_path)
    
    return results


def cleanup_hf_checkpoints(exp_path=None, cleanup_all=False):
    """
    Clean up HuggingFace checkpoint directories (hf-*) to free disk space.
    These can be regenerated from original checkpoints if needed.
    
    Args:
        exp_path: Path to specific experiment directory. If None and cleanup_all=False, does nothing.
        cleanup_all: If True, clean up all hf-* directories in all experiments.
    """
    import shutil
    
    if cleanup_all:
        # Clean up all hf-* directories across all experiments
        flame_dir = str(get_project_root() / "flame")
        exp_base = os.path.join(flame_dir, "exp")
        if not os.path.exists(exp_base):
            print(f"Experiment directory not found: {exp_base}")
            return
        
        hf_dirs = []
        for root, dirs, files in os.walk(exp_base):
            for d in dirs:
                if d.startswith("hf-"):
                    hf_dirs.append(os.path.join(root, d))
        
        if not hf_dirs:
            print("No hf-* directories found to clean up.")
            return
        
        print(f"Found {len(hf_dirs)} hf-* directories to clean up.")
        total_size = 0
        for hf_dir in hf_dirs:
            try:
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(hf_dir)
                          for filename in filenames)
                total_size += size
                shutil.rmtree(hf_dir)
            except Exception as e:
                print(f"Warning: Could not delete {hf_dir}: {e}")
        
        print(f"Cleaned up {len(hf_dirs)} directories, freed ~{total_size / (1024**3):.2f} GB")
    
    elif exp_path:
        # Clean up hf-* directories for a specific experiment
        if not os.path.exists(exp_path):
            print(f"Experiment path not found: {exp_path}")
            return
        
        hf_dirs = []
        for item in os.listdir(exp_path):
            if item.startswith("hf-") and os.path.isdir(os.path.join(exp_path, item)):
                hf_dirs.append(os.path.join(exp_path, item))
        
        if not hf_dirs:
            print(f"No hf-* directories found in {exp_path}")
            return
        
        print(f"Found {len(hf_dirs)} hf-* directories to clean up in {exp_path}")
        total_size = 0
        for hf_dir in hf_dirs:
            try:
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(hf_dir)
                          for filename in filenames)
                total_size += size
                shutil.rmtree(hf_dir)
            except Exception as e:
                print(f"Warning: Could not delete {hf_dir}: {e}")
        
        print(f"Cleaned up {len(hf_dirs)} directories, freed ~{total_size / (1024**3):.2f} GB")
    else:
        print("No cleanup path specified. Use --cleanup-all or --exp_path with --cleanup")


def keep_best_and_final_checkpoints(exp_path, results=None):
    """
    Keep only the best (lowest perplexity) and final (highest step) checkpoints.
    Deletes all other checkpoint-* and hf-* directories to save disk space.
    
    Args:
        exp_path: Path to experiment directory
        results: List of evaluation results (from evaluate_experiment). 
                 If None, will try to load from eval_results.json.
    """
    import shutil
    
    # Load results if not provided
    if results is None:
        results_file = os.path.join(exp_path, 'eval_results.json')
        if not os.path.exists(results_file):
            print(f"Error: No eval_results.json found at {results_file}")
            print("Please run evaluation first before cleaning up checkpoints.")
            return
        with open(results_file, 'r') as f:
            data = json.load(f)
            results = data.get('results', [])
    
    if not results:
        print("No evaluation results found. Cannot determine best checkpoint.")
        return
    
    # Find best (lowest perplexity) and final (highest step) checkpoints
    best_result = min(results, key=lambda r: r['val_ppl'])
    final_result = max(results, key=lambda r: r['step'])
    
    best_step = best_result['step']
    final_step = final_result['step']
    
    print(f"\n" + "=" * 60)
    print("Keeping Best and Final Checkpoints")
    print("=" * 60)
    print(f"Best checkpoint:  step {best_step} (PPL: {best_result['val_ppl']:.2f})")
    print(f"Final checkpoint: step {final_step} (PPL: {final_result['val_ppl']:.2f})")
    
    keep_steps = {best_step, final_step}
    
    # Find all checkpoint directories (both flame and HuggingFace formats)
    deleted_count = 0
    freed_bytes = 0
    
    # Check for HuggingFace format: checkpoint-{step}
    for item in os.listdir(exp_path):
        if item.startswith('checkpoint-'):
            try:
                step = int(item.split('-')[1])
                if step not in keep_steps:
                    ckpt_path = os.path.join(exp_path, item)
                    if os.path.isdir(ckpt_path):
                        size = sum(os.path.getsize(os.path.join(dp, f))
                                  for dp, dn, fn in os.walk(ckpt_path) for f in fn)
                        shutil.rmtree(ckpt_path)
                        freed_bytes += size
                        deleted_count += 1
                        print(f"  Deleted: {item}")
            except (ValueError, IndexError):
                continue
    
    # Check for flame format: checkpoint/step-{step}
    checkpoint_dir = os.path.join(exp_path, 'checkpoint')
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith('step-'):
                try:
                    step = int(item.split('-')[1])
                    if step not in keep_steps:
                        step_path = os.path.join(checkpoint_dir, item)
                        if os.path.isdir(step_path):
                            size = sum(os.path.getsize(os.path.join(dp, f))
                                      for dp, dn, fn in os.walk(step_path) for f in fn)
                            shutil.rmtree(step_path)
                            freed_bytes += size
                            deleted_count += 1
                            print(f"  Deleted: checkpoint/{item}")
                except (ValueError, IndexError):
                    continue
    
    # Clean up hf-* directories for non-kept steps
    for item in os.listdir(exp_path):
        if item.startswith('hf-'):
            try:
                step = int(item.split('-')[1])
                if step not in keep_steps:
                    hf_path = os.path.join(exp_path, item)
                    if os.path.isdir(hf_path):
                        size = sum(os.path.getsize(os.path.join(dp, f))
                                  for dp, dn, fn in os.walk(hf_path) for f in fn)
                        shutil.rmtree(hf_path)
                        freed_bytes += size
                        deleted_count += 1
                        print(f"  Deleted: {item}")
            except (ValueError, IndexError):
                continue
    
    print("=" * 60)
    if deleted_count > 0:
        print(f"Deleted {deleted_count} checkpoint directories, freed ~{freed_bytes / (1024**3):.2f} GB")
    else:
        print("No checkpoints needed to be deleted.")
    print(f"Kept: step {best_step}" + (f" and step {final_step}" if final_step != best_step else " (also final)"))
    print("=" * 60)


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
    plt.ylim(0, 50)
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
    # Evaluate single experiment (WikiText-103, default)
    python scripts/evaluate_comparison.py --exp_path flame/exp/gla_170M-wikitext103-...
    
    # Evaluate on OpenWebText (for OpenWebText-trained models)
    python scripts/evaluate_comparison.py --exp_path flame/exp/gla_170M_openwebtext-... --dataset openwebtext
    
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
                       choices=["wikitext-103", "wikitext-2", "openwebtext"],
                       help="Dataset to evaluate on. Use 'openwebtext' for models trained on OpenWebText.")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs='+', help="Specific steps to evaluate")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--cleanup", action="store_true",
                       help="Delete hf-* checkpoint directories after evaluation to free disk space")
    parser.add_argument("--cleanup-all", action="store_true",
                       help="Delete all hf-* checkpoint directories across all experiments (frees ~151GB)")
    parser.add_argument("--keep_best_final", action="store_true",
                       help="After evaluation, delete all checkpoints except best (lowest PPL) and final")
    
    # Comparison mode
    parser.add_argument("--compare", type=str, nargs='+', 
                       help="Compare multiple experiment paths")
    parser.add_argument("--compare_output", type=str, 
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # Handle cleanup operations
    if args.cleanup_all:
        cleanup_hf_checkpoints(cleanup_all=True)
        return
    
    if args.compare:
        compare_experiments(args.compare, args.compare_output)
    elif args.exp_path:
        results = evaluate_experiment(
            args.exp_path,
            model_id=args.model,
            config_path=args.config,
            dataset=args.dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            output=args.output,
            cleanup=args.cleanup
        )
        # Keep only best and final checkpoints if requested
        if args.keep_best_final and results:
            keep_best_and_final_checkpoints(args.exp_path, results)
    else:
        parser.error("Either --exp_path or --compare is required")


if __name__ == "__main__":
    main()
