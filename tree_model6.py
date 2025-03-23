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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

import datasets
from types import SimpleNamespace

import matplotlib.pyplot as plt

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


# -----------------------------------------------------------------------------
# Dataset and Data Collation (same as before)
# -----------------------------------------------------------------------------
class WikiTextDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        #self.data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.data = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = " ".join(self.data["text"])
        self.tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        self.samples = []
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            self.samples.append(self.token_ids[i:i+seq_len])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": sample, "labels": sample}

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

# -----------------------------------------------------------------------------
# Trainer Callbacks (unchanged)
# -----------------------------------------------------------------------------
class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.best_training_loss = float('inf')
        self.best_eval_loss = float('inf')
        self.last_eval_loss = None

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
            training_perplexity = math.exp(current_loss) if current_loss < 100 else float('inf')
        else:
            current_loss = None
            training_perplexity = None
        if "eval_loss" in logs:
            current_eval_loss = logs["eval_loss"]
            if isinstance(current_eval_loss, torch.Tensor):
                current_eval_loss = current_eval_loss.item()
            self.last_eval_loss = current_eval_loss
            if current_eval_loss < self.best_eval_loss:
                self.best_eval_loss = current_eval_loss
            eval_perplexity = math.exp(current_eval_loss) if current_eval_loss < 100 else float('inf')
        else:
            current_eval_loss = self.last_eval_loss
            eval_perplexity = math.exp(current_eval_loss) if current_eval_loss is not None and current_eval_loss < 100 else float('inf')
        out_str = f"Step {state.global_step}: "
        if epoch is not None: 
            out_str += f"Epoch {epoch} | "
        if current_loss is not None:
            out_str += f"Training Loss: {current_loss:.4f} (Best: {self.best_training_loss:.4f}, Perp: {training_perplexity:.4f})"
        if current_eval_loss is not None:
            out_str += f" | Eval Loss: {current_eval_loss:.4f} (Best: {self.best_eval_loss:.4f}, Perp: {eval_perplexity:.4f})"
        print(out_str)

class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        print('evaluate')
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if not hasattr(self, 'eval_steps'):
            self.eval_steps = []
            self.eval_ppls = []
        current_step = self.state.global_step if hasattr(self.state, 'global_step') else 0
        eval_loss = metrics.get("eval_loss", None)
        eval_ppl = np.exp(eval_loss) if eval_loss is not None and eval_loss < 100 else float('inf')
        self.eval_steps.append(current_step)
        self.eval_ppls.append(eval_ppl)
        plt.figure()
        plt.plot(self.eval_steps, self.eval_ppls, marker='o')
        plt.xlabel("Global Step")
        plt.ylabel("Evaluation Perplexity")
        plt.title("Evaluation Perplexity Over Time")
        plt.ylim(0, min(400, max(self.eval_ppls)*1.1))
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/eval_ppl_{timestamp}.png")
        plt.close()
        return metrics

# -----------------------------------------------------------------------------
# Model Definition: Transformer Scan with Binary Tree Aggregation using GPT-2 Blocks
# -----------------------------------------------------------------------------
class T0(nn.Module):
    """
    T0: Initial embedding module.
    """
    def __init__(self, config, chunk_size):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(chunk_size, config.n_embd)
        self.chunk_size = chunk_size

    def forward(self, input_ids):
        token_emb = self.wte(input_ids)  # (batch, chunk_size, hidden_dim)
        positions = torch.arange(self.chunk_size, device=input_ids.device).unsqueeze(0)
        pos_emb = self.wpe(positions)
        return token_emb + pos_emb

class T1(nn.Module):
    """
    T1: Aggregation module.
    Uses GPT-2 blocks (without a causal mask) to aggregate two sequences.
    """
    def __init__(self, config, num_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    def forward(self, x):
        for block in self.blocks:
            x = block(x, attention_mask=None, use_cache=False, output_attentions=False)[0]
        x = self.ln_f(x)
        return x

class T2(nn.Module):
    """
    T2: Autoregressive prediction module.
    Uses GPT-2 blocks with a causal mask.
    """
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    def forward(self, x, causal_mask):
        for block in self.blocks:
            x = block(x, attention_mask=causal_mask, use_cache=False, output_attentions=False)[0]
        x = self.ln_f(x)
        return x

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
        self.T0 = T0(config, chunk_size)
        self.T1 = T1(config, num_layers=T1_num_layers)
        self.T2 = T2(config, num_layers=T2_num_layers)
        self.T2_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def get_causal_mask(self, seq_length, device):
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).unsqueeze(0).unsqueeze(0)
        mask = (1.0 - mask) * -10000.0
        return mask

    def vectorized_prefix_scan(self, states, dummy, debug=False):
        """
        Performs a fully vectorized Blelloch scan over a list of T0 states.
        Args:
            states: list of length n of tensors, each of shape (batch, chunk_size, hidden_dim)
            dummy: tensor of shape (batch, chunk_size, hidden_dim) to act as the identity.
        Returns:
            A tensor P of shape (batch, n+1, chunk_size, hidden_dim) where:
              P[:, 0, ...] = dummy and for i>=1, P[:, i, ...] = prefix sum of states[0:i]
        """
        batch_size = states[0].size(0)
        n = len(states)
        m = 1
        while m < n:
            m *= 2

        if debug:
            print(f"n={n}, padded length m={m}")

        # Stack states to shape (batch, n, chunk_size, hidden_dim) and pad along dim=1 to length m.
        X = torch.stack(states, dim=1)  # (batch, n, chunk_size, hidden_dim)
        pad = dummy.unsqueeze(1).expand(batch_size, m - n, self.chunk_size, -1)
        X = torch.cat([X, pad], dim=1)  # (batch, m, chunk_size, hidden_dim)

        if debug:
            print("After padding, X shape:", X.shape)

        def combine(x, y):
            # x and y are expected to be of shape (batch, G, chunk_size, hidden_dim)
            # Concatenate along the token dimension (dim=2) to get (batch, G, 2*chunk_size, hidden_dim).
            cat = torch.cat([x, y], dim=2)  # (batch, G, 2*chunk_size, hidden_dim)
            if debug:
                print("In combine, after cat:", cat.shape)
            B, G, L, H = cat.shape
            cat = cat.view(B * G, L, H)
            out = self.T1(cat)  # (B*G, L, H)
            out = out[:, -self.chunk_size:, :]  # (B*G, chunk_size, H)
            out = out.view(B, G, self.chunk_size, H)
            if debug:
                print("In combine, after T1 and reshape:", out.shape)
            return out

        L = int(math.log2(m))
        # Upsweep phase.
        for d in range(L):
            group_size = 2 ** (d + 1)
            num_groups = m // group_size
            # Reshape X to (batch, num_groups, group_size, chunk_size, hidden_dim)
            X = X.view(batch_size, num_groups, group_size, self.chunk_size, -1)
            left_idx = group_size // 2 - 1
            right_idx = group_size - 1
            left = X[:, :, left_idx, :, :]    # (batch, num_groups, chunk_size, hidden_dim)
            right = X[:, :, right_idx, :, :]    # (batch, num_groups, chunk_size, hidden_dim)
            if debug:
                print(f"Upsweep d={d}: left shape {left.shape}, right shape {right.shape}")
            combined = combine(left, right)     # (batch, num_groups, chunk_size, hidden_dim)
            # Update the right element in each group.
            X[:, :, right_idx, :, :] = combined
            # Reshape back to (batch, m, chunk_size, hidden_dim)
            X = X.view(batch_size, m, self.chunk_size, -1)
            if debug:
                print(f"After upsweep d={d}, X shape: {X.shape}")

        # Set the root (last element) to dummy.
        X[:, m - 1, :, :] = dummy

        # Downsweep phase.
        for d in reversed(range(L)):
            group_size = 2 ** (d + 1)
            num_groups = m // group_size
            X = X.view(batch_size, num_groups, group_size, self.chunk_size, -1)
            left_idx = group_size // 2 - 1
            right_idx = group_size - 1
            if debug:
                print(f"Downsweep d={d}, X shape: {X.shape}")
            # Save a copy of left branch.
            left = X[:, :, left_idx, :, :].clone()  # (batch, num_groups, chunk_size, hidden_dim)
            # Move the right branch to the left.
            X[:, :, left_idx, :, :] = X[:, :, right_idx, :, :]
            # Now update the right branch with combine(left, X[:, :, right_idx, :, :]).
            new_val = combine(left, X[:, :, right_idx, :, :])
            X[:, :, right_idx, :, :] = new_val
            X = X.view(batch_size, m, self.chunk_size, -1)
            if debug:
                print(f"After downsweep d={d}, X shape: {X.shape}")

        # Now, by Blelloch scan, for i>=1, X[:, i-1, :, :] holds the prefix sum of states[0:i].
        # Construct the final prefix tensor P: P[:, 0, :, :] = dummy and for i>=1, P[:, i, :, :] = X[:, i-1, :, :]
        P = [dummy.unsqueeze(1)]
        for i in range(1, n + 1):
            P.append(X[:, i - 1, :, :].unsqueeze(1))
        P = torch.cat(P, dim=1)  # (batch, n+1, chunk_size, hidden_dim)
        if debug:
            print("Final prefix P shape:", P.shape)
        return P

    def forward(self, input_ids, labels=None):
        """
        input_ids: (batch, seq_length), where seq_length is a multiple of chunk_size.
        Computes prefix states via a vectorized Blelloch scan and uses them for autoregressive prediction.
        """
        batch_size, seq_length = input_ids.shape
        num_chunks = seq_length // self.chunk_size
        
        # Split input into chunks: shape (batch, num_chunks, chunk_size)
        chunks = input_ids.view(batch_size, num_chunks, self.chunk_size)
        
        # Compute T0 embeddings for each chunk.
        level0 = [self.T0(chunks[:, i, :]) for i in range(num_chunks)]
        # Each element in level0: (batch, chunk_size, hidden_dim)
        
        # Define dummy state.
        dummy = torch.zeros_like(level0[0])
        # Compute prefix states using the vectorized scan.
        # P: (batch, num_chunks+1, chunk_size, hidden_dim)
        P = self.vectorized_prefix_scan(level0, dummy, debug=False)
        
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_logits = []
        
        # --- Process chunk 0: standard autoregressive prediction ---
        T2_input_0 = level0[0][:, :self.chunk_size - 1, :]  # (batch, chunk_size-1, hidden_dim)
        causal_mask_0 = self.get_causal_mask(T2_input_0.size(1), T2_input_0.device)
        T2_out_0 = self.T2(T2_input_0, causal_mask=causal_mask_0)  # (batch, chunk_size-1, hidden_dim)
        logits_chunk0 = self.T2_head(T2_out_0)  # (batch, chunk_size-1, vocab_size)
        all_logits.append(logits_chunk0)
        target_chunk0 = chunks[:, 0, 1:]  # (batch, chunk_size-1)
        loss_chunk0 = loss_fn(logits_chunk0.reshape(-1, self.vocab_size),
                              target_chunk0.reshape(-1))
        total_loss += loss_chunk0
        
        # --- Process chunks 1 to num_chunks-1 in parallel ---
        if num_chunks > 1:
            t2_inputs = []  # to store T2 inputs for each chunk (i from 1 to num_chunks-1)
            targets = []    # to store corresponding target tokens
            for i in range(1, num_chunks):
                # For chunk i, the T2 input is:
                # Concatenate the prefix state P[:, i, :, :] (aggregation of chunks 0...i-1)
                # with the T0 embedding of chunk i (excluding its last token).
                prefix = P[:, i, :, :]  # (batch, chunk_size, hidden_dim)
                current_emb = level0[i][:, :self.chunk_size - 1, :]  # (batch, chunk_size-1, hidden_dim)
                t2_input = torch.cat([prefix, current_emb], dim=1)  # (batch, 2*chunk_size-1, hidden_dim)
                t2_inputs.append(t2_input)
                targets.append(chunks[:, i, 1:])  # (batch, chunk_size-1)
            
            # Stack T2 inputs along a new dimension: shape becomes (batch, num_chunks-1, 2*chunk_size-1, hidden_dim)
            T2_inputs = torch.stack(t2_inputs, dim=1)
            # Reshape to merge the batch and chunk dimensions: (batch*(num_chunks-1), 2*chunk_size-1, hidden_dim)
            T2_inputs = T2_inputs.view(-1, T2_inputs.size(2), T2_inputs.size(3))
            seq_len_t2 = T2_inputs.size(1)  # this equals 2*chunk_size-1
            causal_mask = self.get_causal_mask(seq_len_t2, T2_inputs.device)
            T2_out = self.T2(T2_inputs, causal_mask=causal_mask)  # (batch*(num_chunks-1), 2*chunk_size-1, hidden_dim)
            # For each T2 input, we only want to predict the tokens corresponding to the current chunk.
            # That is, we take only the last (chunk_size-1) tokens.
            T2_out = T2_out[:, - (self.chunk_size - 1):, :]  # (batch*(num_chunks-1), chunk_size-1, hidden_dim)
            logits_chunks = self.T2_head(T2_out)  # (batch*(num_chunks-1), chunk_size-1, vocab_size)
            # Reshape back to (batch, num_chunks-1, chunk_size-1, vocab_size)
            logits_chunks = logits_chunks.view(batch_size, num_chunks - 1, self.chunk_size - 1, self.vocab_size)
            all_logits.append(logits_chunks)
            
            # Similarly, stack targets for chunks 1 to num_chunks-1: shape (batch, num_chunks-1, chunk_size-1)
            targets = torch.stack(targets, dim=1)
            loss_chunks = loss_fn(logits_chunks.reshape(-1, self.vocab_size),
                                  targets.reshape(-1))
            total_loss += loss_chunks*(num_chunks-1)
            #print("Chunk0 logits shape:", logits_chunk0.shape, "target shape:", target_chunk0.shape, "loss_chunk0:", loss_chunk0.item())
            #print("Parallel T2 inputs shape:", T2_inputs.shape)
            #print("T2_out shape:", T2_out.shape)
            #print("Logits_chunks shape:", logits_chunks.shape, "Targets shape:", targets.shape, "loss_chunks:", loss_chunks.item())
        
        total_loss = total_loss / num_chunks
        total_loss = total_loss
        return CausalLMOutput(loss=total_loss, logits=all_logits[-1])
    
    def sequential_inference(self, input_ids):
        """
        Performs sequential inference, processing the input chunk-by-chunk in a way that exactly 
        replicates the training-time forward-backward (parallel scan) computation.
        
        This method uses an online (binary-counter) algorithm to update a small list L of 
        intermediate prefix states. L has at most O(log N) elements and, at each new chunk,
        only O(log N) combine operations (each via T1) are performed.
        
        For each chunk i:
          - Compute s_i = T0(chunk_i)
          - Update L: while the lowest-level slot is already occupied (as indicated by the bits of i),
            combine that slot with the new value and carry upward.
          - Then compute the overall prefix by combining the remaining non-None elements of L (in order
            from high level to low level). This prefix exactly matches the one computed in training.
          - Use the prefix to form T2’s input (concatenated with the current chunk’s T0 embedding, 
            excluding its last token) for next-token prediction.
        
        Returns:
            A list of outputs (logits) for each chunk.
        
        Time complexity: O(log N) per chunk.
        Space complexity: O(log N) extra storage.
        """
        batch_size, seq_length = input_ids.shape
        num_chunks = seq_length // self.chunk_size
        # Split input into chunks: shape (batch, num_chunks, chunk_size)
        chunks = input_ids.view(batch_size, num_chunks, self.chunk_size)
        outputs = []
        # L will hold at most O(log(num_chunks)) states.
        # We use a list L where L[j] will hold the combined state for a block of size 2^j.
        L = [None] * (num_chunks.bit_length() + 1)  # Sufficient size for our binary counter.

        # Process chunks sequentially.
        for i in range(num_chunks):
            # Compute T0 embedding for chunk i.
            s = self.T0(chunks[:, i, :])  # (batch, chunk_size, hidden_dim)
            # Use binary counter logic to merge with previously stored states.
            # 'carry' the new value upward.
            x = s
            j = 0
            # While the j-th bit of i is set, combine L[j] with x and clear that slot.
            while (i >> j) & 1:
                # L[j] is not None.
                x = self.T1(torch.cat([L[j], x], dim=1))[:, -self.chunk_size:, :]  # Combine operation.
                L[j] = None
                j += 1
            # Store the merged result in L[j].
            L[j] = x

            # Now, compute the overall prefix for chunks 0...i.
            # We want the prefix that conditions chunk i.
            # By definition, for i==0, the prefix is s_0.
            # For i > 0, the prefix is the combination of the non-None entries in L,
            # taken in order from the highest index down to 0.
            prefix = None
            for k in reversed(range(len(L))):
                if L[k] is not None:
                    if prefix is None:
                        prefix = L[k]
                    else:
                        prefix = self.T1(torch.cat([L[k], prefix], dim=1))[:, -self.chunk_size:, :]
            # For chunk 0, we use its own embedding; for subsequent chunks, T2 input is:
            # prefix concatenated with T0(chunk_i) excluding its last token.
            if i == 0:
                T2_input = s[:, :self.chunk_size - 1, :]
            else:
                T2_input = torch.cat([prefix, s[:, :self.chunk_size - 1, :]], dim=1)
            causal_mask = self.get_causal_mask(T2_input.size(1), T2_input.device)
            T2_out = self.T2(T2_input, causal_mask=causal_mask)
            logits = self.T2_head(T2_out)
            outputs.append(logits)
            # At this point, any state that was merged is discarded (L slots that got overwritten)
            # so the maximum number of stored states is O(log(num_chunks)).
        return outputs
    
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
    

# -----------------------------------------------------------------------------
# Main Training Code (unchanged)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train a GPT2-based Transformer Scan LM with binary tree aggregation on WikiText-2.'
    )
    parser.add_argument('--seq_len', type=int, default=32*16,
                        help='Number of tokens per sample (must be a multiple of chunk_size).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size.')
    args = parser.parse_args()
    
    if args.seq_len % args.chunk_size != 0:
        raise ValueError("seq_len must be a multiple of chunk_size.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    output_dir = f"./tree_model/tree_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    train_dataset = WikiTextDataset(split="train", tokenizer=tokenizer, seq_len=args.seq_len)
    valid_dataset = WikiTextDataset(split="validation", tokenizer=tokenizer, seq_len=args.seq_len)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1
    )
    model = TransformerScanModel(config, chunk_size=args.chunk_size,
                                 T1_num_layers=2, T2_num_layers=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
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

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        callbacks=[PrintLossCallback()]
    )
    from transformers import ProgressCallback
    trainer.remove_callback(ProgressCallback)
    
    trainer.train()

if __name__ == "__main__":
    main()
