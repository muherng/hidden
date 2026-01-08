#!/usr/bin/env python3
"""
Evaluate all GLA checkpoints and generate validation curve.
Matches the eval frequency of TransformerScanModel for fair comparison.

Usage:
    python scripts/evaluate_all_checkpoints.py \
        --exp_path flame/exp/gla-170M-wikitext103-comparison-XXXXX \
        --config flame/configs/gla_170M.json
"""

import argparse
import json
import math
import os
import sys

# IMPORTANT: Import fla first to register model types with transformers
import fla  # noqa: F401

import torch
from torch.distributed.checkpoint import load as dcp_load
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '/data/lingo/morrisyau/hidden')


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


def convert_checkpoint(exp_path, step, config_path, tokenizer_path='gpt2'):
    """Convert DCP checkpoint to HuggingFace format using working DCP load method."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    checkpoint_dir = os.path.join(exp_path, f'checkpoint/step-{step}')
    hf_path = os.path.join(exp_path, f'hf-{step}')
    
    # Check if already converted
    if os.path.exists(hf_path) and os.path.exists(os.path.join(hf_path, 'model.safetensors')):
        print(f"  HF checkpoint already exists at {hf_path}")
        return hf_path
    
    print(f"  Converting step {step} to HuggingFace format...")
    
    os.makedirs(hf_path, exist_ok=True)
    
    # Load config
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate all GLA checkpoints")
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Path to experiment folder")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config")
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                        choices=["wikitext-103", "wikitext-2"])
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GLA Checkpoint Evaluation")
    print("=" * 60)
    print(f"Experiment: {args.exp_path}")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print("=" * 60)
    
    # Find checkpoints
    steps = find_checkpoints(args.exp_path)
    if not steps:
        print("No checkpoints found!")
        return
    
    print(f"Found {len(steps)} checkpoints: {steps}")
    
    results = []
    
    for step in tqdm(steps, desc="Evaluating checkpoints"):
        print(f"\n--- Step {step} ---")
        
        # Convert checkpoint
        hf_path = convert_checkpoint(
            args.exp_path, step, args.config
        )
        
        if hf_path is None:
            print(f"  Skipping step {step} (conversion failed)")
            continue
        
        # Evaluate
        try:
            loss, ppl = evaluate_checkpoint(
                hf_path, args.dataset, args.seq_len, args.batch_size
            )
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
    output_path = args.output or os.path.join(args.exp_path, 'eval_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
    
    # Plot validation curve
    if results:
        plt.figure(figsize=(10, 6))
        
        steps_plot = [r['step'] for r in results]
        ppls = [r['val_ppl'] for r in results]
        
        plt.subplot(1, 2, 1)
        plt.plot(steps_plot, [r['val_loss'] for r in results], 'b-o')
        plt.xlabel('Step')
        plt.ylabel('Validation Loss')
        plt.title('GLA Validation Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(steps_plot, ppls, 'r-o')
        plt.xlabel('Step')
        plt.ylabel('Validation Perplexity')
        plt.title('GLA Validation Perplexity')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(args.exp_path, 'eval_curve.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
