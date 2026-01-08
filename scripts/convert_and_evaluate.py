#!/usr/bin/env python3
"""
Convert DCP checkpoint and evaluate on WikiText validation.
Works around the dcp_to_torch_save metadata issue.

Usage:
    python scripts/convert_and_evaluate.py \
        --exp_path flame/exp/gla-170M-wikitext103-comparison-258221 \
        --step 30000 \
        --config flame/configs/gla_170M.json
"""

import argparse
import math
import os
import sys

# IMPORTANT: Import fla first to register model types with transformers
import fla  # noqa: F401 - registers GLA, DeltaNet, etc. with AutoConfig/AutoModel

import torch
from torch.distributed.checkpoint import load as dcp_load
from tqdm import tqdm

sys.path.insert(0, '/data/lingo/morrisyau/hidden')


def convert_checkpoint(exp_path, step, config_path, tokenizer_path='gpt2'):
    """Convert DCP checkpoint to HuggingFace format."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    checkpoint_dir = os.path.join(exp_path, f'checkpoint/step-{step}')
    output_dir = os.path.join(exp_path, f'hf-{step}')
    
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, 'model.safetensors')):
        print(f"HF checkpoint already exists at {output_dir}")
        return output_dir
    
    print(f"Converting checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    print("Loading config...")
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    config.save_pretrained(output_dir)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)
    
    # Create model and get state dict structure
    print("Creating model from config...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Create state dict with the expected structure for DCP
    # DCP saves with 'model' key wrapping the actual model state
    state_dict = {'model': model.state_dict()}
    
    # Load checkpoint using DCP
    print(f"Loading DCP checkpoint from {checkpoint_dir}...")
    try:
        dcp_load(state_dict, checkpoint_id=checkpoint_dir)
        print("DCP load successful!")
    except Exception as e:
        print(f"DCP load failed: {e}")
        raise
    
    # Load state dict into model
    print("Loading weights into model...")
    model.load_state_dict(state_dict['model'])
    
    # Save in HuggingFace format
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    
    print("Conversion complete!")
    return output_dir


def evaluate_model(model_path, dataset_name='wikitext-103', seq_len=512, batch_size=8, device='cuda'):
    """Evaluate model on validation set."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    print(f"\nLoading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Load validation data
    print(f"Loading {dataset_name} validation set...")
    if dataset_name == "wikitext-103":
        data = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    else:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    text = " ".join(data["text"])
    tokenizer.model_max_length = int(1e7)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Validation tokens: {len(token_ids):,}")
    
    # Create samples
    samples = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        samples.append(token_ids[i:i+seq_len])
    print(f"Evaluation samples: {len(samples)}")
    
    # Compute perplexity
    total_loss = 0.0
    total_tokens = 0
    
    print("Computing perplexity...")
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size)):
            batch_samples = samples[i:i+batch_size]
            input_ids = torch.tensor(batch_samples, dtype=torch.long, device=device)
            
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="wikitext-103")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--convert_only", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GLA Checkpoint Conversion & Evaluation")
    print("=" * 60)
    
    # Convert checkpoint
    hf_path = convert_checkpoint(
        args.exp_path, 
        args.step, 
        args.config,
        args.tokenizer
    )
    
    if args.convert_only:
        print(f"\nConversion complete. Model saved to: {hf_path}")
        return
    
    # Evaluate
    loss, ppl = evaluate_model(
        hf_path,
        args.dataset,
        args.seq_len,
        args.batch_size,
        args.device
    )
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Step: {args.step}")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Perplexity: {ppl:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

