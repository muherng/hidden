#!/usr/bin/env python3
"""
Diagnose whether there's a loss computation difference between
the model's built-in loss (fuse_cross_entropy) and manual cross-entropy.

Usage:
    python scripts/diagnose_loss.py --checkpoint flame/exp/gated_deltanet_170M-.../hf-15000
"""

import argparse
import torch
import torch.nn.functional as F
import math

# Import fla to register model types
from fla.models.gated_deltanet import GatedDeltaNetForCausalLM
from fla.models.gla import GLAForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def diagnose_loss(checkpoint_path, num_batches=50, batch_size=8, seq_len=512):
    print("=" * 70)
    print("Diagnosing Loss Computation Difference")
    print("=" * 70)
    
    print(f"\nLoading model from: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load validation data
    print("Loading WikiText-103 validation data...")
    data = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = " ".join(data["text"])
    tokenizer.model_max_length = int(1e7)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # Create samples (non-overlapping windows)
    samples = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        samples.append(token_ids[i:i+seq_len])
    
    print(f"Created {len(samples)} validation samples of length {seq_len}")
    num_batches = min(num_batches, len(samples) // batch_size)
    
    # Method 1: Model's built-in loss (uses fuse_cross_entropy)
    print("\n--- Method 1: Model's built-in forward(labels=...) ---")
    total_loss_builtin = 0.0
    total_tokens_builtin = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_samples = samples[i*batch_size:(i+1)*batch_size]
            input_ids = torch.tensor(batch_samples, dtype=torch.long, device='cuda')
            
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            batch_tokens = input_ids.numel()
            total_loss_builtin += loss.item() * batch_tokens
            total_tokens_builtin += batch_tokens
    
    avg_loss_builtin = total_loss_builtin / total_tokens_builtin
    ppl_builtin = math.exp(avg_loss_builtin)
    print(f"  Average loss: {avg_loss_builtin:.4f}")
    print(f"  Perplexity: {ppl_builtin:.2f}")
    
    # Method 2: Manual cross-entropy on logits
    print("\n--- Method 2: Manual cross-entropy on logits ---")
    total_loss_manual = 0.0
    total_tokens_manual = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_samples = samples[i*batch_size:(i+1)*batch_size]
            input_ids = torch.tensor(batch_samples, dtype=torch.long, device='cuda')
            
            # Get logits without computing loss
            outputs = model(input_ids=input_ids, labels=None)
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous().float()  # Convert to float32
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute cross-entropy manually
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            
            batch_tokens = shift_labels.numel()
            total_loss_manual += loss.item() * batch_tokens
            total_tokens_manual += batch_tokens
    
    avg_loss_manual = total_loss_manual / total_tokens_manual
    ppl_manual = math.exp(avg_loss_manual)
    print(f"  Average loss: {avg_loss_manual:.4f}")
    print(f"  Perplexity: {ppl_manual:.2f}")
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Built-in loss:    {avg_loss_builtin:.4f} (PPL: {ppl_builtin:.2f})")
    print(f"Manual CE loss:   {avg_loss_manual:.4f} (PPL: {ppl_manual:.2f})")
    print(f"Difference:       {abs(avg_loss_builtin - avg_loss_manual):.6f}")
    
    if abs(avg_loss_builtin - avg_loss_manual) < 0.01:
        print("\n✓ Loss computations are CONSISTENT - NOT a loss computation issue")
        print("  The train-val gap is likely due to missing dropout/regularization")
    else:
        print("\n✗ SIGNIFICANT DIFFERENCE - Loss computation IS an issue!")
        print("  The fuse_cross_entropy might be computing something different")
    print("=" * 70)
    
    return avg_loss_builtin, avg_loss_manual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose loss computation differences")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to HF checkpoint (e.g., flame/exp/.../hf-15000)")
    parser.add_argument("--num_batches", type=int, default=50,
                        help="Number of batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()
    
    diagnose_loss(args.checkpoint, args.num_batches, args.batch_size, args.seq_len)
