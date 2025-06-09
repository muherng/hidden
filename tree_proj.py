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
import csv

from data.associative_recall import AssociativeRecallDataset

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

TOL = 1e-6  # Tolerance to determine if a tensor is the dummy

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
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutput

from types import SimpleNamespace
from typing import Optional

# import matplotlib.pyplot as plt

def pad_to_chunk(x, chunk_size, pad_token_id, device):
    # x: (batch, seq_length). Pad x along dim=1 so that seq_length becomes a multiple of chunk_size.
    L = x.size(1)
    remainder = L % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        pad_ids = torch.full((x.size(0), pad_len), pad_token_id, dtype=torch.long, device=device)
        return torch.cat([x, pad_ids], dim=1)
    return x

def get_offset(x, chunk_size):
    # Given the current sequence x (batch, seq_length), determine the offset within the last chunk.
    L = x.size(1)
    last_chunk_start = (L // chunk_size) * chunk_size
    offset = L - last_chunk_start - 1  # in [0, chunk_size-1)
    return offset


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.best_training_loss = float('inf')
        self.best_eval_loss = float('inf')
        self.last_eval_loss = None
        self.best_eval_accuracy = 0.0
        self.last_eval_accuracy = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 100 != 0:
            return
        if logs is None:
            return
        # Retrieve epoch from the state, if available
        epoch = getattr(state, "epoch", None)
        if "loss" in logs:
            current_loss = logs["loss"]
            if isinstance(current_loss, torch.Tensor):
                current_loss = current_loss.item()
            if current_loss < self.best_training_loss:
                self.best_training_loss = current_loss
        else:
            current_loss = None
        if "eval_loss" in logs:
            current_eval_loss = logs["eval_loss"]
            if isinstance(current_eval_loss, torch.Tensor):
                current_eval_loss = current_eval_loss.item()
            self.last_eval_loss = current_eval_loss
            if current_eval_loss < self.best_eval_loss:
                self.best_eval_loss = current_eval_loss
        else:
            current_eval_loss = self.last_eval_loss
        if "eval_accuracy" in logs:
            eval_accuracy = logs["eval_accuracy"]
            if isinstance(eval_accuracy, torch.Tensor):
                eval_accuracy = eval_accuracy.item()  
            self.last_eval_accuracy = eval_accuracy
            if eval_accuracy > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_accuracy 
        else:
            eval_accuracy = self.last_eval_accuracy 

        out_str = f"Step {state.global_step}: "
        if epoch is not None: 
            out_str += f"Epoch {epoch} | "
        if current_loss is not None:
            out_str += f"Training Loss: {current_loss:.4f} (Best: {self.best_training_loss:.4f})"
        if current_eval_loss is not None:
            out_str += f" | Eval Loss: {current_eval_loss:.4f} (Best: {self.best_eval_loss:.4f}, Acc: {eval_accuracy:.4f})"
        print(out_str)

def compute_metrics(eval_pred):
        logits = eval_pred.predictions["logits"]
        labels = eval_pred.label_ids
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        labels = torch.tensor(labels)

        # Flatten if needed
        if preds.ndim > 1:
            preds = preds.view(-1)
            labels = labels.view(-1)

        mask = labels != -100  # Ignore padding or masked labels
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return {
            "eval_accuracy": accuracy,
        }

# -----------------------------------------------------------------------------
# Model Definition: Transformer Scan with Binary Tree Aggregation using GPT-2 Blocks
# -----------------------------------------------------------------------------
class T0(nn.Module):
    """
    T0: Initial embedding module.
    Matches GPT2LMHeadModel's initial embedding behavior.
    """
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids):
        # Get sequence length from input_ids
        seq_len = input_ids.size(1)
        
        # Get token embeddings
        token_emb = self.wte(input_ids)  # (batch, seq_len, hidden_dim)
        
        # Get position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        pos_emb = self.wpe(position_ids)
        
        # Combine embeddings and apply dropout
        embeddings = token_emb + pos_emb
        embeddings = self.drop(embeddings)
        
        return embeddings

class T1(nn.Module):
    """
    T1: Aggregation module.
    Uses GPT-2 blocks (without a causal mask) to aggregate two sequences.
    """
    def __init__(self, config, chunk_size=64, num_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.project = nn.Linear(2 * chunk_size, chunk_size, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x, attention_mask=None, use_cache=False, output_attentions=False)[0]
        x = self.ln_f(x)
        x = self.project(x.transpose(-1,-2)).transpose(-1,-2)  # Project from 2*chunk_size to chunk_size
        return x

# class Combine(nn.Module):
#     """
#     Combines two chunks by projecting from 2*chunk_size to chunk_size.
#     """
#     def __init__(self, config, num_layers=1):
#         super().__init__()
#         self.projection = nn.Linear(2 * config.chunk_size, config.chunk_size, bias=False)

#     def forward(self, x, y):
#         # If left is dummy, return right.
#         if x[1]:
#             return y
#         # If right is dummy, return left.
#         if y[1]:
#             return x

#         # Both are real: combine via T1.
#         cat = torch.cat([x[0], y[0]], dim=1)  # shape: (batch, 2*chunk_size, hidden_dim)
#         out = self.T1(cat)
#         if isinstance(out, tuple):
#             out = out[0]
#         out = out[:, -self.chunk_size:, :]
#         return (out, False)
#         return x
    
class T2(nn.Module):
    """
    T2: Autoregressive prediction module.
    Uses GPT-2 blocks with a causal mask.
    When chunk_size=1, this should be equivalent to GPT2LMHeadModel's transformer blocks.
    """
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.drop = nn.Dropout(config.resid_pdrop)  # Add residual dropout
        
    def forward(self, x, causal_mask=None, past_key_values=None):
        # If no causal mask provided, create one
        if causal_mask is None:
            seq_length = x.size(1)
            causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).unsqueeze(0).unsqueeze(0)
            causal_mask = (1.0 - causal_mask) * -10000.0
            
        new_past = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, present = block(x, attention_mask=causal_mask, use_cache=True, output_attentions=False, layer_past=past)
            new_past.append(present)
        
        x = self.ln_f(x)
        x = self.drop(x)  # Apply residual dropout after layer norm
        return x, tuple(new_past)

def parallel_fold(S, op):
    """
    Given a list S of tensors, combine them in left-to-right order using op.
    (Here, op is a function that takes two tensors and returns their combination.)
    """
    result = S[0]
    for i in range(1, len(S)):
        # Concatenate result and S[i] along the token dimension and apply op.
        result = op(result, S[i])
    return result

class TransformerScanModel(nn.Module):
    """
    TransformerScanModel using binary tree forward/backward aggregation.
    
    This implementation assumes 8 chunks (for demonstration). The forward pass:
      - Computes T0 embeddings for each chunk.
      - Computes a binary tree using T1 in two passes:
          * Forward pass: compute only the necessary internal nodes.
          * Backward pass: compute missing prefix states.
      - Constructs final prefix states:
            [dummy, s[0:0], s[0:1], s[0:2], s[0:3], s[0:4], s[0:5], s[0:6]]
      - Runs T2 for autoregressive next-token prediction.
    """
    def __init__(self, config, chunk_size, T1_num_layers=1, T2_num_layers=2):
        super().__init__()
        self.config = config
        self.chunk_size = chunk_size
        self.vocab_size = config.vocab_size
        self.T0 = T0(config)
        self.T1 = T1(config, chunk_size=chunk_size, num_layers=T1_num_layers)
        self.T2 = T2(config, num_layers=T2_num_layers)
        self.T2_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def resize_token_embeddings(self, new_vocab_size):
        # Resize the embedding layer in T0
        self.T0.wte = nn.Embedding(new_vocab_size, self.config.n_embd)
        self.vocab_size = new_vocab_size
        self.T2_head = nn.Linear(self.config.n_embd, new_vocab_size, bias=False)
    
    def get_causal_mask(self, seq_length, device):
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).unsqueeze(0).unsqueeze(0)
        mask = (1.0 - mask) * -10000.0
        return mask

    def forward(self, input_ids, labels=None, parallel=False, **kwargs):
        """
        input_ids: (batch, seq_length), where seq_length is a multiple of chunk_size.
        Computes prefix states via a vectorized Blelloch scan and uses them for autoregressive prediction.
        """
        batch_size, seq_length = input_ids.shape
        num_chunks = seq_length // self.chunk_size
        
        # First compute embeddings for the entire sequence
        all_embeddings = self.T0(input_ids)  # (batch, seq_length, hidden_dim)
        
        # Then reshape into chunks
        chunks = all_embeddings.view(batch_size, num_chunks, self.chunk_size, -1)
        
        # Create level0 list from chunks
        level0 = [chunks[:, i, :, :] for i in range(num_chunks)]
        # Each element in level0: (batch, chunk_size, hidden_dim)
        
        # Define dummy state.
        dummy = (torch.zeros_like(level0[0]), True)
        # Compute prefix states using the vectorized scan.
        # P: (batch, num_chunks+1, chunk_size, hidden_dim)
        # parallel = False
        if parallel:
            P = self.vectorized_prefix_scan(level0, dummy, debug=False)
        else: 
            P,_ = self.compute_sequential_prefix(input_ids, debug=False)
        debug = False
        if debug: 
            P1 = self.vectorized_prefix_scan(level0, dummy, debug=False)
            P2,_ = self.compute_sequential_prefix(input_ids, debug=False)
            exit = False
            for i in range(num_chunks):
                if not torch.allclose(P1[:,i,:,:], P2[:,i,:,:], atol=1e-4):
                    print(f'Failed at chunk {i}')
                    exit = True
            if exit:     
                raise ValueError(f"Assert failed: vectorized prefix scan does not match sequential prefix scan")

        expected_dummy = dummy[0]
        actual_dummy = P[:, 0, :, :]
        assert torch.allclose(actual_dummy, expected_dummy, atol=1e-4), (
            "Assert failed: P[:,0,:,:] does not match dummy"
        )
        if debug: 
            expected = level0[0][:,:self.chunk_size,:]
            actual = P[:, 1, :, :]
            assert torch.allclose(actual, expected, atol=1e-4), (
                "Assert failed: P[:,1,:,:] does not match level0[0][:,:self.chunk_size,:]"
            )
        
        loss_fn = nn.CrossEntropyLoss()
        all_logits = []
        
        # --- Process chunk 0: standard autoregressive prediction ---
        T2_input_0 = level0[0][:, :self.chunk_size, :]  # (batch, chunk_size, hidden_dim)
        causal_mask_0 = self.get_causal_mask(T2_input_0.size(1), T2_input_0.device)
        T2_out_0,_ = self.T2(T2_input_0, causal_mask=causal_mask_0)  # (batch, chunk_size, hidden_dim)
        logits_chunk0 = self.T2_head(T2_out_0)  # (batch, chunk_size, vocab_size)
        all_logits.append(logits_chunk0)
        
        # --- Process chunks 1 to num_chunks-1 in parallel ---
        if num_chunks > 1:
            t2_inputs = []  # to store T2 inputs for each chunk (i from 1 to num_chunks-1)
            targets = []    # to store corresponding target tokens
            for i in range(1, num_chunks):
                # For chunk i, the T2 input is:
                # Concatenate the prefix state P[:, i, :, :] (aggregation of chunks 0...i-1)
                # with the T0 embedding of chunk i (excluding its last token).
                prefix = P[:, i, :, :]  # (batch, chunk_size, hidden_dim)
                current_emb = level0[i][:, :self.chunk_size, :]  # (batch, chunk_size-1, hidden_dim)
                t2_input = torch.cat([prefix, current_emb], dim=1)  # (batch, 2*chunk_size-1, hidden_dim)
                t2_inputs.append(t2_input)
            
            # Stack T2 inputs along a new dimension: shape becomes (batch, num_chunks-1, 2*chunk_size, hidden_dim)
            T2_inputs = torch.stack(t2_inputs, dim=1)
            # Reshape to merge the batch and chunk dimensions: (batch*(num_chunks-1), 2*chunk_size, hidden_dim)
            T2_inputs = T2_inputs.view(-1, T2_inputs.size(2), T2_inputs.size(3))
            seq_len_t2 = T2_inputs.size(1)  # this equals 2*chunk_size
            causal_mask = self.get_causal_mask(seq_len_t2, T2_inputs.device)
            #print('T2 inputs and causal mask shape: ', (T2_inputs.shape, causal_mask.shape))
            T2_out,_ = self.T2(T2_inputs, causal_mask=causal_mask)  # (batch*(num_chunks-1), 2*chunk_size, hidden_dim)
            # For each T2 input, we take the last chunk_size tokens to collect all the logits.
            # That is, we take only the last chunk_size tokens.
            T2_out = T2_out[:, - self.chunk_size:, :]  # (batch*(num_chunks-1), chunk_size, hidden_dim)
            logits_chunks = self.T2_head(T2_out)  # (batch*(num_chunks-1), chunk_size, vocab_size)
            # Reshape back to (batch, num_chunks-1, chunk_size, vocab_size)
            logits_chunks = logits_chunks.view(batch_size, num_chunks - 1, self.chunk_size, self.vocab_size)
            all_logits.append(logits_chunks)
        
            all_logits_1_flat = all_logits[1].view(batch_size, -1, self.vocab_size)
            final_logits = torch.cat([all_logits[0], all_logits_1_flat], dim=1)  # (batch, num_chunks*chunk_size, vocab_size)
        #print('final logits shape:', final_logits.shape)
        else: 
            # If num_chunks == 1, we only have the first chunk.
            final_logits = all_logits[0]
        outputs = {}
        outputs["logits"] = final_logits
        return outputs
    
    
    def combine(self, x, y):
        """
        Combine two states x and y, where each is a tuple (value, is_dummy).
        If one operand is dummy, return the other. Otherwise, combine the values.
        """
        # If left is dummy, return right.
        if x[1]:
            return y
        # If right is dummy, return left.
        if y[1]:
            return x

        # Both are real: combine via T1.
        cat = torch.cat([x[0], y[0]], dim=1)  # shape: (batch, 2*chunk_size, hidden_dim)
        out = self.T1(cat)
        if isinstance(out, tuple):
            out = out[0]
        # out = out[:, -self.chunk_size:, :]
        return (out, False)
    
    def vectorized_prefix_scan(self, states, dummy, debug=False):
        n = len(states)
        M = 2 ** math.ceil(math.log2(n)) if n > 0 else 1
        batch = states[0].size(0)
        device = states[0].device

        # Stack real states and build upsweep mask for padded positions
        state_tensor = torch.stack(states, dim=1)
        upsweep_mask = torch.zeros(batch, M, dtype=torch.bool, device=device).detach()
        if M > n:
            pad_tensor = dummy[0].unsqueeze(1).expand(batch, M - n, self.chunk_size, -1)
            state_tensor = torch.cat([state_tensor, pad_tensor], dim=1)
            upsweep_mask[:, n:] = True
        T = state_tensor
        if debug:
            print("DEBUG: Initial States")
            #print("  T0(chunk0):", states[0])
            #print("  T0(chunk1):", states[1])
            print("  state_tensor shape:", state_tensor.shape)

        # Upsweep (unchanged)
        L_levels = int(math.log2(M))
        #print("L_levels: ", L_levels)
        for d in range(L_levels):
            step = 2 ** (d + 1)
            num_groups = M // step
            T = T.view(batch, num_groups, step, self.chunk_size, -1)
            left_idx = (2 ** d) - 1
            right_idx = step - 1

            if debug:
                print("\n[Upsweep Level {}]".format(d))
                print("  step:", step, "num_groups:", num_groups)
                print("  left_idx:", left_idx, "(should correspond to chunk0)")
                print("  right_idx:", right_idx, "(position to store aggregated value)")
                print("  T shape:", T.shape)
                # Extract and print the groups
                #print("  Group 0, left element (chunk0):", T[0, 0, left_idx])
                #print("  Group 0, right element (chunk1):", T[0, 0, right_idx])

            left = T[:, :, left_idx]
            right = T[:, :, right_idx]

            if debug:
                for i in range(batch):
                    assert torch.allclose(left[i, 0], states[0][i], atol=1e-4), (
                        f"Upsweep Level {d}: For sample {i}, left block does not equal T0(chunk0)."
                    )
                print("  [Assert passed] Upsweep: Left block equals T0(chunk0) for all samples.")
            temp = torch.cat([left, right], dim=2).view(batch * num_groups, 2 * self.chunk_size, -1)

            if debug:
                print("  Concatenated (chunk0 || chunk1) shape:", temp.shape)
            out = self.T1(temp)
            if isinstance(out, tuple): out = out[0]
            if debug:
                print("  T1 output shape:", out.shape)
                print("  T1 output (for group 0):", out[0])
            out = out.view(batch, num_groups, 2 * self.chunk_size, -1)[:, :, -self.chunk_size:]
            if debug:
                print("  Trimmed T1 output (should replace chunk1):", out)
            T[:, :, right_idx] = out
            T = T.view(batch, M, self.chunk_size, -1)

            if debug:
                for i in range(batch):
                    assert torch.allclose(T[i, 0, :, :], states[0][i], atol=1e-4), (
                        f"After upsweep Level {d}: For sample {i}, T[:,0,:,:] does not equal T0(chunk0)."
                    )
                print("  [Assert passed] Upsweep: T[:,0,:,:] unchanged and equals T0(chunk0) for all samples.")
        


        # Downsweep: allocate new tree D and dummy-mask
        D = torch.zeros_like(T)
        downsweep_mask = torch.zeros(batch, M, dtype=torch.bool, device=device).detach()
        # Initialize root to dummy
        D[:, M - 1] = dummy[0]
        downsweep_mask[:, M - 1] = True

        # Parallel downsweep levels
        for d in reversed(range(L_levels)):
            step = 2 ** (d + 1)
            num_groups = M // step
            T_view = T.view(batch, num_groups, step, self.chunk_size, -1)
            D_view = D.view(batch, num_groups, step, self.chunk_size, -1)
            upsweep_mask_view = upsweep_mask.view(batch, num_groups, step).detach().clone()
            downsweep_mask_view = downsweep_mask.view(batch, num_groups, step).detach().clone()

            left_idx = (2 ** d) - 1
            right_idx = step - 1

            parent_val = D_view[:, :, right_idx]
            parent_dummy = downsweep_mask_view[:, :, right_idx]
            left_old_val = T_view[:, :, left_idx]
            left_dummy = upsweep_mask_view[:, :, left_idx]

            # Propagate parent down to left child
            D_view[:, :, left_idx] = parent_val
            downsweep_mask_view = downsweep_mask_view.clone()
            downsweep_mask_view[:, :, left_idx] = parent_dummy

            # Compute combined only for real-real pairs
            concat = torch.cat([parent_val, left_old_val], dim=2).view(batch * num_groups, 2 * self.chunk_size, -1)
            combined = self.T1(concat)
            if isinstance(combined, tuple): combined = combined[0]
            # combined = combined.view(batch, num_groups, 2 * self.chunk_size, -1)[:, :, -self.chunk_size:] T1 SHOULD DO THIS NOW

            # Merge according to dummy flags
            # If parent is dummy -> result = left_old_val
            # If left is dummy -> result = parent_val
            # Else -> result = combined
            result = torch.where(parent_dummy.unsqueeze(-1).unsqueeze(-1).detach(), left_old_val, combined)
            result = torch.where(left_dummy.unsqueeze(-1).unsqueeze(-1).detach(), parent_val, result)
            D_view[:, :, right_idx] = result
            downsweep_mask_view = downsweep_mask_view.clone()
            downsweep_mask_view[:, :, right_idx] = parent_dummy & left_dummy

        # Return leaves: exclusive prefixes (including dummy at index 0)
        return D[:, :n]

    def compute_sequential_prefix(self, input_ids, debug=False):
        """
        Computes prefix states sequentially using the binary-counter method.
        For an input with n chunks, returns a tensor of shape 
            (batch, n+1, chunk_size, hidden_dim)
        corresponding to:
            [ dummy, A0, T1(A0, A1), T1(T1(A0, A1), A2), T1(T1(A0, A1), T1(A2, A3)), ... ]
        Dummy states have an explicit boolean flag True.
        """
        batch_size, seq_length = input_ids.shape
        n = seq_length // self.chunk_size
        
        # First compute embeddings for the entire sequence
        all_embeddings = self.T0(input_ids)  # (batch, seq_length, hidden_dim)
        
        # Then reshape into chunks
        chunks = all_embeddings.view(batch_size, n, self.chunk_size, -1)
        
        # Create level0 list from chunks and wrap each state as (value, False)
        level0 = [(chunks[:, i, :, :], False) for i in range(n)]
        
        # Create dummy as (tensor, True)
        dummy = (torch.zeros_like(level0[0][0]), True)
        prefix_list = []
        # L will be our binary counter; its length is O(log n)
        L = [None] * (n.bit_length() + 1)
        for i in range(n):
            s = level0[i]  # (A[i], False)
            current = s
            j = 0
            # Carry update: while the j-th bit is 1, combine.
            while ((i >> j) & 1) == 1:
                current = self.combine(L[j], current)
                L[j] = None
                j += 1
            L[j] = current
            # Now, fold the non-None entries of L in descending order (MSB-to-LSB).
            if i == 0:
                prefix = s
            else:
                prefix = None
                for k in reversed(range(len(L))):
                    if L[k] is not None:
                        if prefix is None:
                            prefix = L[k]
                        else:
                            prefix = self.combine(prefix, L[k])
            prefix_list.append(prefix)
            if debug:
                print(f"Sequential: computed prefix for chunk {i}, flag: {prefix[1]}")
        # Concatenate dummy and all prefix values (ignoring flags for output)
        P_seq = torch.cat([dummy[0].unsqueeze(1)] + [p[0].unsqueeze(1) for p in prefix_list], dim=1)
        P_seq = P_seq[:,:-1,:,:]
        return P_seq, L

    def forward_inference(
        self,
        input_ids: torch.Tensor,
        L: Optional[list] = None,
        chunks_processed: int = 0,
        prefix_val: Optional[torch.Tensor] = None,
        past_key_values=None
    ):
        batch_size, total_len = input_ids.shape
        chunk = self.chunk_size
        num_full = total_len // chunk
        remainder_len = total_len % chunk

        # INITIAL PROMPT: fold all full chunks at once
        if L is None:
            if num_full > 0:
                P_seq, L = self.compute_sequential_prefix(input_ids[:, : num_full * chunk])
                chunks_processed = num_full
                prefix_val = P_seq[:, -1, :, :]  # last prefix of the prompt
            else:
                L = []
                chunks_processed = 0
                prefix_val = torch.zeros(batch_size, chunk, self.config.n_embd, device=input_ids.device)

        # INCREMENTAL STEP: update only on chunk boundary
        elif num_full > chunks_processed:
            print('update chunk: ', num_full)
            assert num_full == chunks_processed + 1, "Only one new chunk per token"
            start = chunks_processed * chunk
            state = (self.T0(input_ids[:, start : start + chunk]), False)

            j = 0
            while ((chunks_processed >> j) & 1) == 1:
                state = self.combine(L[j], state)
                L[j] = None
                j += 1
            if j >= len(L):
                L.extend([None] * (j + 1 - len(L)))
            L[j] = state
            chunks_processed += 1

            # Recompute prefix_val by folding non‑None L in MSB→LSB
            prefix = None
            for entry in reversed(L):
                if entry is not None:
                    prefix = entry if prefix is None else self.combine(prefix, entry)
            prefix_val = prefix[0]

        # Build T2 input using cached prefix_val
        # If we just crossed into a new full chunk, send the entire prefix chunk into T2 (cold start)
        if num_full > chunks_processed - 1:  # i.e. we just incremented chunks_processed above
            # prefix_val shape = (batch, chunk, hidden_dim)
            t2_out, past_key_values = self.T2(prefix_val, causal_mask=None, past_key_values=None)

        # Otherwise, only feed the newest single token into T2 using the cached KV
        else:
            # Take the last timestep of prefix_val as the new input token
            last_token = prefix_val[:, -1:, :]  # shape = (batch, 1, hidden_dim)
            t2_out, past_key_values = self.T2(last_token, causal_mask=None, past_key_values=past_key_values)

        next_logits = self.T2_head(t2_out[:, -1, :])

        return next_logits, L, chunks_processed, prefix_val, past_key_values


    @classmethod
    def from_pretrained(cls, checkpoint_path, config, chunk_size, device="cpu", **kwargs):
        # Instantiate the model on CPU first.
        model = cls(config, chunk_size, **kwargs)

        # If checkpoint_path is a directory, locate the weight file.
        if os.path.isdir(checkpoint_path):
            # Try "model.safetensors" first.
            potential_file = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(potential_file):
                checkpoint_file = potential_file
            else:
                # Fallback to "pytorch_model.bin".
                potential_file = os.path.join(checkpoint_path, "pytorch_model.bin")
                if os.path.exists(potential_file):
                    checkpoint_file = potential_file
                else:
                    raise FileNotFoundError("No valid model weights file found in the checkpoint directory.")
        else:
            checkpoint_file = checkpoint_path

        # Import and use the safetensors loader without a device parameter.
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_file)  # Loads on CPU by default.

        # If the checkpoint is a dict with "model_state_dict", extract it.
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        
        # Finally, move the model to the desired device.
        model.to(device)
        return model

# FINAL UNEMBEDDING LAYER?!

class TreeModel(TransformerScanModel):
    """Tree model with direct output supervision."""
    
    def __init__(self, config, chunk_size= 32, T1_num_layers = 1, T2_num_layers = 1):
        super().__init__(config, chunk_size=chunk_size, T1_num_layers = T1_num_layers, T2_num_layers = T2_num_layers)
    
    def forward(self, input_ids,
        labels=None,
        **kwargs):
        outputs = super().forward(input_ids, labels=None, output_hidden_states=True, return_dict=True)
        # loss = 0
        loss_fct = nn.CrossEntropyLoss()

        # final logits
        # print("input_ids: " + str(input_ids.shape))
        # print("labels: " + str(labels.shape))
        final_logits = outputs["logits"]
        # print("final_logits: " + str(final_logits.shape))
        # compute loss
        if labels is not None:
            # compute loss
            # print("final_logits view: " + str(final_logits.view(-1, self.config.vocab_size).shape))
            # print("labels view: " + str(labels.view(-1).shape))
            loss = loss_fct(final_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # outputs["loss"] = loss 
        return loss, outputs # 
    
# -----------------------------------------------------------------------------
# Main Training Code (unchanged)
# -----------------------------------------------------------------------------
def main():
    print("Training Transformer Scan with binary tree aggregation via projection on MQAR.", flush=True)
    parser = argparse.ArgumentParser(
        description='Train a GPT2-based Transformer Scan LM with binary tree aggregation on MQAR.'
    )
    parser.add_argument('--seq_len', type=int, default=64*8,
                        help='Number of tokens per sample (must be a multiple of chunk_size).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--chunk_size', type=int, default=64, help='Chunk size (to be safe, use powers of 2).')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension.')
    parser.add_argument('--T1_n_layers', type=int, default=6, help='Number of layers for the combining transformer.')
    parser.add_argument('--T2_n_layers', type=int, default=6, help='Number of layers for the readout transformer.')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of heads for each transformer.')
    # parser.add_argument('--train_mode', type=str, default='parallel', choices=['parallel', 'blelloch', 'sequential'],
    #                     help='Training mode: parallel or sequential.')
    parser.add_argument('--vocab_size', type=int, default=8192, help='Vocabulary size.')
    parser.add_argument('--num_train_examples', type=int, default=100000, help='Number of training examples.')
    parser.add_argument('--num_valid_examples', type=int, default=3000, help='Number of validation examples.')
    parser.add_argument('--num_kv_pairs', type=int, default=8, help='Number of key-value pairs.')
    parser.add_argument('--power_a', type=float, default=0.1, help='Parameter for gap size in data distribution.')
    args = parser.parse_args()

    if args.seq_len % args.chunk_size != 0:
        raise ValueError("seq_len must be a multiple of chunk_size.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    output_dir = f"out/mqar_{args.seq_len}/tree_proj_{args.n_embd}/chunk_size_{args.chunk_size}/kv_{args.num_kv_pairs}/a_{args.power_a}/tree_{timestamp}"
    print("Output directory:", output_dir)
    print("Arguments:", args)
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = AssociativeRecallDataset(
        vocab_size=args.vocab_size,
        num_examples=args.num_train_examples,
        input_seq_len=args.seq_len,
        num_kv_pairs=args.num_kv_pairs,
        power_a=args.power_a,
        seed=args.seed
    )
    valid_dataset = AssociativeRecallDataset(
        vocab_size=args.vocab_size,
        num_examples=args.num_valid_examples,
        input_seq_len=args.seq_len,
        num_kv_pairs=args.num_kv_pairs,
        power_a=args.power_a,
        seed=args.seed + 10  # Different seed for validation set
    )

    config = GPT2Config(
        vocab_size=args.vocab_size, #tokenizer.vocab_size
        n_positions=1024,
        n_embd=args.n_embd,
        n_layer=2, #6,
        n_head=1, #12,
        dropout=0.1
    )
    model = TreeModel(config, chunk_size=args.chunk_size,
                                 T1_num_layers=args.T1_n_layers, T2_num_layers=args.T2_n_layers) #, train_mode=args.train_mode
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,          
        warmup_steps=1000,           
        weight_decay=0.01,             
        fp16=False,
        seed=args.seed,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        max_grad_norm=1.0,
        logging_dir="./logs",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[PrintLossCallback()]
    )
    from transformers import ProgressCallback
    trainer.remove_callback(ProgressCallback)
    
    trainer.train()
    eval = trainer.evaluate()
    args_dict = vars(args)
    # Merge dictionaries
    combined = {**args_dict, **eval}

    path = os.path.join("results", "proj")
    os.makedirs(path, exist_ok=True)
    filename = f"results/proj/output_s{args.seq_len}_n{args.n_embd}_a{args.power_a}.csv"
    file_exists = os.path.exists(filename)

    # Append to CSV, write header only if file doesn't exist
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=combined.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(combined)

    print("Out :", combined)
    # csv_path = os.path.join(output_dir, "acc.csv")

if __name__ == "__main__":
    main()
