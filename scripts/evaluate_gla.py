#!/usr/bin/env python3
"""
Evaluate GLA model on WikiText-103 validation set.
Computes perplexity to match TransformerScanModel evaluation.

Usage:
    python scripts/evaluate_gla.py --model_path exp/gla-170M-wikitext103-comparison-XXXXX/hf
    
    # Or with checkpoint step
    python scripts/evaluate_gla.py --model_path exp/gla-170M-wikitext103-comparison-XXXXX --step 62900
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, '/data/lingo/morrisyau/hidden')


def load_model(model_path, step=None, device='cuda'):
    """Load GLA model from HuggingFace format or convert from DCP checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Check if HF format exists
    hf_path = os.path.join(model_path, 'hf') if step is None else os.path.join(model_path, f'hf-{step}')
    if not os.path.exists(hf_path):
        hf_path = model_path  # Assume it's already the HF path
    
    if not os.path.exists(hf_path):
        raise ValueError(f"Model path not found: {hf_path}")
    
    print(f"Loading model from: {hf_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    return model, tokenizer


def create_eval_dataset(tokenizer, seq_len=512, dataset_name="wikitext-103"):
    """Create evaluation dataset matching TransformerScanModel format."""
    from datasets import load_dataset
    
    if dataset_name == "wikitext-103":
        data = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    elif dataset_name == "wikitext-2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Concatenate all text
    text = " ".join(data["text"])
    tokenizer.model_max_length = int(1e7)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"Validation set: {len(token_ids):,} tokens")
    
    # Create samples
    samples = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        samples.append(token_ids[i:i+seq_len])
    
    print(f"Created {len(samples)} evaluation samples of length {seq_len}")
    
    return samples


def compute_perplexity(model, samples, batch_size=16, device='cuda'):
    """Compute perplexity over evaluation samples."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    # Create batches
    num_batches = (len(samples) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
            batch_samples = samples[i:i+batch_size]
            
            # Prepare input
            input_ids = torch.tensor(batch_samples, dtype=torch.long, device=device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Accumulate loss
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Evaluate GLA model on WikiText validation set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (HF format or experiment folder)")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (if loading from experiment folder)")
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                        choices=["wikitext-103", "wikitext-2"],
                        help="Dataset to evaluate on")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length (should match training)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GLA Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Seq Length: {args.seq_len}")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.step, args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Create evaluation dataset
    samples = create_eval_dataset(tokenizer, args.seq_len, args.dataset)
    
    # Compute perplexity
    avg_loss, perplexity = compute_perplexity(
        model, samples, args.batch_size, args.device
    )
    
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.2f}")
    print("=" * 60)
    
    return avg_loss, perplexity


if __name__ == "__main__":
    main()

