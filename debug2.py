#!/usr/bin/env python
import torch
import math
import argparse
import numpy as np

from transformers import (
    GPT2LMHeadModel,
    PreTrainedModel,
    GPT2Config,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback
)

# Import your model and prefix scan functions from your full script.
# Adjust the import as necessary depending on your file structure.
from tree_model6 import TransformerScanModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    
    # Configuration and parameters.
    chunk_size = 32
    n_chunks = 9  # Use 3 chunks for testing.
    batch = 1

    # For testing purposes, you can hard-code the checkpoint path.
    checkpoint = 'tree_model/tree_20250321_133251/checkpoint-730800/model.safetensors'

    
    # Load tokenizer and config.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )
    
    # Instantiate and load your model using the from_pretrained method.
    model = TransformerScanModel.from_pretrained(
        checkpoint, 
        config=config, 
        chunk_size=chunk_size,
        T1_num_layers=2,  # match training parameters
        T2_num_layers=2
    )
    #model.to(device)
    model.eval()

    # Create a dummy input sequence. (Total length = n_chunks * chunk_size)
    input_ids = torch.randint(0, config.vocab_size, (batch, n_chunks * chunk_size))
    print("Input shape:", input_ids.shape)

    # Compute sequential prefix states.
    P_seq = model.compute_sequential_prefix(input_ids, debug=True)
    print(f"\nSequential prefix scan (P_seq) shape: {P_seq.shape}")
    for i in range(P_seq.size(1)):
        norm_val = torch.norm(P_seq[:, i]).item()
        print(f"  P_seq[{i}] norm = {norm_val:.6f}")

    # Prepare chunk states for vectorized prefix scan.
    # Reshape input_ids into chunks: (batch, n_chunks, chunk_size)
    input_chunks = input_ids.view(batch, n_chunks, chunk_size)
    chunks = []
    for i in range(n_chunks):
        # Compute T0 embeddings for each chunk.
        emb = model.T0(input_chunks[:, i, :])
        chunks.append(emb)
    
    # Define dummy (should be the identity).
    dummy = (torch.zeros_like(chunks[0]), True)
    
    # Compute vectorized prefix states.
    P_vec = model.vectorized_prefix_scan(chunks, dummy, debug=True)
    print(f"\nVectorized prefix scan (P_vec) shape: {P_vec.shape}")
    for i in range(P_vec.size(1)):
        norm_val = torch.norm(P_vec[:, i]).item()
        print(f"  P_vec[{i}] norm = {norm_val:.6f}")
    
    # Compare the sequential and vectorized results.
    diff = torch.abs(P_seq - P_vec)
    print(f"\nOverall prefix diff: max = {diff.max().item():.6f}, mean = {diff.mean().item():.6f}")
    for i in range(P_seq.size(1)):
        seq_flat = P_seq[:, i].reshape(-1)
        vec_flat = P_vec[:, i].reshape(-1)
        diff_flat = torch.abs(seq_flat - vec_flat)
        print(f"\nPrefix index {i}: max diff = {diff_flat.max().item():.6f}, mean diff = {diff_flat.mean().item():.6f}")
        print(f"  P_seq[{i}][:10] = {seq_flat[:10].tolist()}")
        print(f"  P_vec[{i}][:10] = {vec_flat[:10].tolist()}")

if __name__ == "__main__":
    main()