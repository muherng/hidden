#!/usr/bin/env python
import math
import os
import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
import datasets

# Define a global debug flag (set to True to enable debug printing)
DEBUG_ATTENTION = True

def debug_attention_window(attn_probs, window, threshold=1e-5):
    """
    Checks that for each token (query) in the attention probabilities, only keys within the allowed
    window have non-negligible probability. 

    Args:
      attn_probs: Tensor of shape (batch, num_heads, seq_len, seq_len) with attention probabilities.
      window: The allowed window size (an integer).
      threshold: Any probability below this value is considered effectively zero.
    """
    B, H, L, _ = attn_probs.shape
    # Create an (L, L) matrix where diff[i,j] = i - j.
    idx = torch.arange(L, device=attn_probs.device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # shape: (L, L)
    # Allowed if 0 <= (i - j) < window.
    allowed_mask = ((diff >= 0) & (diff < window)).float()  # 1 if allowed, else 0.
    # Expand allowed_mask to match attn_probs shape.
    allowed_mask = allowed_mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, L, L)
    # Create a mask of disallowed positions (1 where NOT allowed)
    disallowed = 1 - allowed_mask
    # Compute the maximum attention probability in disallowed positions.
    max_disallowed = (attn_probs * disallowed).max().item()
    if max_disallowed > threshold:
        print(f"DEBUG WARNING: Max attention prob outside allowed window = {max_disallowed:.6f} "
              f"(allowed window: {window} tokens)")
    else:
        print(f"DEBUG: Attention window check passed (max disallowed = {max_disallowed:.6f}).")

# =============================================================================
# 1. Dataset
# =============================================================================

class WikiTextDataset(Dataset):
    """
    Loads WikiText-2 raw text, tokenizes it, and splits it into non-overlapping
    chunks of fixed length (seq_len). For language modeling, labels are identical
    to the input_ids.
    """
    def __init__(self, split, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Load raw WikiText-2.
        data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = " ".join(data["text"])
        tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Create non-overlapping chunks.
        self.samples = []
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            self.samples.append(self.token_ids[i:i+seq_len])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": sample, "labels": sample}

# =============================================================================
# 2. Custom Attention Module with Windowed Mask
# =============================================================================

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class CustomGPT2Attention(GPT2Attention):
    """
    A custom GPT2Attention module that, after computing the usual query–key scores
    (and adding GPT‑2’s built‑in causal mask), further masks out keys that are too far
    in the past. Here we allow positions with
         0 ≤ i - j < window,
    meaning each token at position i can attend to tokens in positions
         max(0, i - window + 1) through i.
    (The built‑in causal mask already masks out future tokens.)
    """
    def __init__(self, config, window):
        super().__init__(config)
        self.window = window

    def _split_heads(self, tensor, num_heads, head_dim):
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)

    def _merge_heads(self, tensor, num_heads, head_dim):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
        return tensor.view(*new_shape)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                use_cache=False, output_attentions=False):
        # Compute query, key, value.
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute raw attention scores.
        attn_scores = torch.matmul(query, key.transpose(-1, -2))
        attn_scores = attn_scores / (float(self.head_dim) ** 0.5)

        # Add GPT-2's default causal mask.
        nd, ns = attn_scores.size(-2), attn_scores.size(-1)
        causal_mask = self.bias[:, :, ns-nd:ns, :ns]
        attn_scores = attn_scores + causal_mask

        # ----- Apply Custom Window Mask -----
        # Create a matrix diff of shape (seq_len, seq_len) where diff[i, j] = i - j.
        seq_len = nd  # current sequence length
        idx = torch.arange(seq_len, device=attn_scores.device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # shape: (seq_len, seq_len)
        # Allowed positions: those where 0 ≤ diff < window.
        # Disallowed positions get a penalty of -1e9.
        window_mask = torch.where((diff >= 0) & (diff < self.window),
                                  torch.tensor(0.0, device=attn_scores.device),
                                  torch.tensor(-1e9, device=attn_scores.device))
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        attn_scores = attn_scores + window_mask
        # -------------------------------------

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # ---- Debug: Check that disallowed positions are effectively zero ----
        # Only run this check once per forward pass (or only on first batch) to avoid too many prints.
        if DEBUG_ATTENTION and not hasattr(self, "_debug_checked"):
            debug_attention_window(attn_probs, self.window, threshold=1e-5)
            self._debug_checked = True
        # -----------------------------------------------------------------------

        if output_attentions:
            return attn_output, present, attn_probs
        else:
            return attn_output, present

# =============================================================================
# 3. Custom GPT-2 Model That Uses Our Custom Attention
# =============================================================================

class WindowedGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, window):
        super().__init__(config)
        self.window = window
        # Replace each block’s attention module with our custom version.
        for block in self.transformer.h:
            orig_attn = block.attn
            custom_attn = CustomGPT2Attention(self.config, window)
            custom_attn.load_state_dict(orig_attn.state_dict(), strict=False)
            block.attn = custom_attn

# =============================================================================
# 4. Optional Callback for Logging Loss
# =============================================================================

class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.best_training_loss = float('inf')
        self.best_eval_loss = float('inf')
        self.last_eval_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
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
        if current_loss is not None:
            out_str += f"Training Loss: {current_loss:.4f} (Best: {self.best_training_loss:.4f}, Perp: {training_perplexity:.4f})"
        if current_eval_loss is not None:
            out_str += f" | Eval Loss: {current_eval_loss:.4f} (Best: {self.best_eval_loss:.4f}, Perp: {eval_perplexity:.4f})"
        print(out_str)

# =============================================================================
# 5. Main Training Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GPT-2 LM with a windowed attention mask (custom attention) on WikiText-2."
    )
    parser.add_argument('--seq_len', type=int, default=128, help='Number of tokens per sample.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--window', type=int, default=3, help='Attention window size (max tokens in the past allowed).')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./gpt2_lm_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the GPT-2 tokenizer and set a padding token.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)
    tokenizer.pad_token = tokenizer.eos_token  # use EOS as the pad token

    # Create datasets.
    train_dataset = WikiTextDataset(split="train", tokenizer=tokenizer, seq_len=args.seq_len)
    valid_dataset = WikiTextDataset(split="validation", tokenizer=tokenizer, seq_len=args.seq_len)

    # Define a small GPT-2 configuration.
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.seq_len,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.3
    )
    model = WindowedGPT2LMHeadModel(config, window=args.window)
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=False,
        seed=args.seed,
        lr_scheduler_type="cosine",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer
    )
    trainer.add_callback(PrintLossCallback())
    trainer.train()

if __name__ == "__main__":
    main()
