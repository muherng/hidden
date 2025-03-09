import math
import os
import json
import datetime
import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

from transformers import (
    PreTrainedModel,
    GPT2Config,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutput

import datasets
from types import SimpleNamespace

def build_override_tensor(current_state, L, module2):
    """
    Build an override tensor of shape (1, L, hidden_dim) for module2.
    It uses module2.s_token as the default but overrides the first token
    with the provided current_state.
    """
    # Get default s_token expanded.
    override_tensor = module2.s_token.unsqueeze(0).unsqueeze(0).expand(1, L, module2.hidden_dim).clone()
    override_tensor[0, 0, :] = current_state
    return override_tensor

def autoregressive_evaluation_batch_debug(composite_model, batch, text_run, state_run, device):
    """
    Debug version: Performs autoregressive evaluation over a batch of samples in parallel,
    printing out intermediate shapes and values.
    """
    # Print overall batch information.
    #print("Batch keys:", batch.keys())
    #print("input_ids shape:", batch["input_ids"].shape)
    #print("s_mask shape:", batch["s_mask"].shape)
    
    # Move batch tensors to device.
    input_ids = batch["input_ids"].to(device)
    s_mask = batch["s_mask"].to(device)
    
    # If the tensors are 1-dimensional, assume it's a single sample and add a batch dimension.
    if input_ids.dim() == 1:
        #print("Detected 1D input_ids; unsqueezing to add batch dimension.")
        input_ids = input_ids.unsqueeze(0)
        s_mask = s_mask.unsqueeze(0)
    
    B = input_ids.shape[0]
    
    # Print raw inputs and masks for each sample.
    for i in range(B):
        sample_input = input_ids[i]
        sample_mask = s_mask[i]
        #print(f"\nSample {i} raw input_ids (first 20):", sample_input[:20].tolist())
        #print(f"Sample {i} raw s_mask (first 20):", sample_mask[:20].tolist())
        # Extract text tokens (non-s tokens) and print them.
        text_tokens = sample_input[~sample_mask]
        #print(f"Sample {i} extracted {text_tokens.numel()} text tokens:", text_tokens.tolist())
    
    # Build a list of text tokens per sample.
    text_tokens_list = []
    for i in range(B):
        tokens = input_ids[i][~s_mask[i]].tolist()
        text_tokens_list.append(tokens)
    
    # Print a summary for the first sample.
    if text_tokens_list[0]:
        print("\nExtracted text tokens for first sample (first 10 tokens):", text_tokens_list[0][:10])
    else:
        print("\nExtracted text tokens for first sample: []")
    
    T = len(text_tokens_list[0])
    print("\nNumber of text tokens in sample 0:", T)
    
    # Step 1: Compute the initial state for each sample in parallel using module1.
    dummy_input_ids = torch.full((B, 1), -1, dtype=torch.long, device=device)
    dummy_s_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
    with torch.no_grad():
        out1 = composite_model.module1(
            input_ids=dummy_input_ids,
            s_mask=dummy_s_mask,
            labels=None,
            return_hidden=True,
            mse_loss=False
        )
    # current_state shape: (B, hidden_dim)
    current_state = out1.hidden_states[:, 0, :]
    #print("\nInitial current_state shape:", current_state.shape)
    
    total_loss = 0.0
    token_count = 0
    pointer = 0   # pointer into each sample’s text tokens
    k = text_run  # block size

    while pointer < T:
        block_size = min(k, T - pointer)
        #print(f"\n--- Processing block starting at pointer {pointer} with block_size {block_size} ---")
        max_seq_len = block_size + 1  # Each sequence is: [s-token] + prefix of block
        
        batch_input_ids_block = []
        batch_s_mask_block = []
        batch_labels_block = []
        sample_indices = []  # to know which sequence came from which sample
        
        # Build sequences for each sample in the batch.
        for i in range(B):
            block_text = text_tokens_list[i][pointer:pointer + block_size]
            #print(f"  Sample {i} block text tokens (len {len(block_text)}):", block_text)
            for j in range(len(block_text)):
                # Sequence: [s-token placeholder] + prefix of block up to token j.
                seq = [-1] + block_text[:j+1]
                pad_length = max_seq_len - len(seq)
                padded_seq = seq + [0] * pad_length  # Use 0 as pad token.
                batch_input_ids_block.append(padded_seq)
                
                # Build s_mask: first token True (s-token), others False.
                s_mask_seq = [True] + [False] * (len(seq) - 1) + [False] * pad_length
                batch_s_mask_block.append(s_mask_seq)
                
                # Build labels: only the last non-padded token is active.
                labels_seq = [-100] * (len(seq) - 1) + [block_text[j]] + [-100] * pad_length
                batch_labels_block.append(labels_seq)
                sample_indices.append(i)
        
        # Convert lists to tensors.
        batch_input_ids_block = torch.tensor(batch_input_ids_block, dtype=torch.long, device=device)
        batch_s_mask_block = torch.tensor(batch_s_mask_block, dtype=torch.bool, device=device)
        batch_labels_block = torch.tensor(batch_labels_block, dtype=torch.long, device=device)
        
        #print("\nbatch_input_ids_block shape:", batch_input_ids_block.shape)
        #print("batch_s_mask_block shape:", batch_s_mask_block.shape)
        #print("batch_labels_block shape:", batch_labels_block.shape)
        #print("Unique label values in batch_labels_block:", torch.unique(batch_labels_block).tolist())
        
        # Build the 4D attention mask.
        L = max_seq_len
        causal_mask = torch.tril(torch.ones((L, L), device=device)).unsqueeze(0).unsqueeze(0)
        padding_mask = (batch_input_ids_block != 0).float().unsqueeze(1).unsqueeze(2)
        attn_mask = (1.0 - causal_mask * padding_mask) * -1e9
        #print("Attention mask for first sequence (slice):", attn_mask[0, 0])
        
        # Build the override tensor for each sequence using the current state.
        override_list = []
        for idx in sample_indices:
            override_tensor = build_override_tensor(current_state[idx].unsqueeze(0), L, composite_model.module2)
            override_list.append(override_tensor.squeeze(0))
        override_tensor = torch.stack(override_list, dim=0)
        #print("override_tensor shape:", override_tensor.shape)
        
        # Run module2 on the entire block batch.
        with torch.no_grad():
            out2 = composite_model.module2(
                input_ids=batch_input_ids_block,
                s_mask=batch_s_mask_block,
                override_s=override_tensor,
                labels=batch_labels_block,
                attention_mask=attn_mask,
                mse_loss=False
            )
        #print("Module2 output ce_loss for block:", out2.ce_loss.item())
        
        num_predictions = len(sample_indices)
        block_loss = out2.ce_loss.item() * num_predictions
        total_loss += block_loss
        token_count += num_predictions

        # Update state if a full block was processed and tokens remain.
        full_block_indices = []
        for i in range(B):
            if block_size == k and (pointer + k < T):
                full_block_indices.append(i)
        if full_block_indices:
            state_input_ids = []
            state_s_mask = []
            for i in full_block_indices:
                block_text = text_tokens_list[i][pointer:pointer + k]
                seq = [-1] + block_text + [-1]
                state_input_ids.append(seq)
                state_s_mask.append([True] + [False] * k + [True])
            state_input_ids = torch.tensor(state_input_ids, dtype=torch.long, device=device)
            state_s_mask = torch.tensor(state_s_mask, dtype=torch.bool, device=device)
            L_state = state_input_ids.shape[1]
            override_state = []
            for i in full_block_indices:
                override_tensor_state = build_override_tensor(current_state[i].unsqueeze(0), L_state, composite_model.module2)
                override_state.append(override_tensor_state.squeeze(0))
            override_state = torch.stack(override_state, dim=0)
            #print("\nState update for samples:", full_block_indices)
            #print("state_input_ids shape:", state_input_ids.shape)
            with torch.no_grad():
                out_state = composite_model.module2(
                    input_ids=state_input_ids,
                    s_mask=state_s_mask,
                    override_s=override_state,
                    labels=torch.full((len(full_block_indices), L_state), -100, dtype=torch.long, device=device),
                    attention_mask=None,
                    mse_loss=False,
                    return_hidden=True
                )
            new_states = out_state.hidden_states[:, -1, :]
            #print("New state shape:", new_states.shape)
            for j, i in enumerate(full_block_indices):
                current_state[i] = new_states[j]
        
        pointer += k
        #print('total_loss:', total_loss)
    return total_loss, token_count


def run_autoregressive_evaluation_debug(composite_model, eval_loader, text_run, state_run, device):
    """
    Runs the debug version of the parallel autoregressive evaluation over the entire evaluation dataset.
    """
    composite_model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in eval_loader:
        print("\n====== New Batch ======")
        loss, tokens = autoregressive_evaluation_batch_debug(composite_model, batch, text_run, state_run, device)
        total_loss += loss
        total_tokens += tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    print(f"\nAutoregressive Evaluation (Debug Parallel Batch) -> Avg CE Loss per token: {avg_loss:.4f}, Perplexity: {ppl:.4f}")
    return avg_loss, ppl
