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

from foobar import foobar 
foobar()

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

# -----------------------------------------------------------------------------
# Dataset and Data Collation (same as before)
# -----------------------------------------------------------------------------
class WikiTextDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        self.data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
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

class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

    def forward(self, input_ids, labels=None):
        """
        input_ids: (batch, seq_length), where seq_length is a multiple of chunk_size.
        Assumes exactly 8 chunks for this binary tree implementation.
        """
        batch_size, seq_length = input_ids.shape
        num_chunks = seq_length // self.chunk_size
        assert num_chunks == 8, "This binary tree implementation currently supports 8 chunks."
        
        # Split into chunks.
        chunks = input_ids.view(batch_size, num_chunks, self.chunk_size)
        
        # Level 0: T0 embeddings for each chunk.
        level0 = [ self.T0(chunks[:, i, :]) for i in range(num_chunks) ]  # each: (batch, chunk_size, hidden_dim)
        
        # Forward pass: compute internal nodes.
        forward_levels = {}
        forward_levels[0] = level0
        
        # Level 1: compute s[0:1], s[2:3], s[4:5], s[6:7]
        level1 = []
        for i in [0, 2, 4, 6]:
            inp = torch.cat([level0[i], level0[i+1]], dim=1)  # (batch, 2*chunk_size, hidden_dim)
            out = self.T1(inp)[:, -self.chunk_size:, :]         # (batch, chunk_size, hidden_dim)
            level1.append(out)
        forward_levels[1] = level1
        
        # Level 2: compute s[0:3] from level1[0] and level1[1]
        inp = torch.cat([level1[0], level1[1]], dim=1)
        level2 = [ self.T1(inp)[:, -self.chunk_size:, :] ]
        forward_levels[2] = level2
        
        # Level 3: compute s[0:5] from level2[0] and level1[2]
        inp = torch.cat([level2[0], level1[2]], dim=1)
        level3 = [ self.T1(inp)[:, -self.chunk_size:, :] ]
        forward_levels[3] = level3
        
        # Backward pass: compute missing prefix states.
        # s[0:2] = T1( s[0:1] and s[2,2] )
        b0_inp = torch.cat([level1[0], level0[2]], dim=1)
        b0 = self.T1(b0_inp)[:, -self.chunk_size:, :]
        # s[0:4] = T1( s[0:3] and s[4,4] )
        b1_inp = torch.cat([level2[0], level0[4]], dim=1)
        b1 = self.T1(b1_inp)[:, -self.chunk_size:, :]
        # s[0:6] = T1( s[0:5] and s[6,6] )
        b2_inp = torch.cat([level3[0], level0[6]], dim=1)
        b2 = self.T1(b2_inp)[:, -self.chunk_size:, :]
        
        # Assemble final prefix states.
        # Output prefix states: [dummy, s[0:0], s[0:1], s[0:2], s[0:3], s[0:4], s[0:5], s[0:6]]
        dummy_prefix = torch.zeros_like(level0[0])
        prefix_states = []
        prefix_states.append(dummy_prefix)  # For chunk 0.
        prefix_states.append(level0[0])       # s[0:0] for chunk 1.
        prefix_states.append(level1[0])       # s[0:1] for chunk 2.
        prefix_states.append(b0)              # s[0:2] for chunk 3.
        prefix_states.append(level2[0])       # s[0:3] for chunk 4.
        prefix_states.append(b1)              # s[0:4] for chunk 5.
        prefix_states.append(level3[0])       # s[0:5] for chunk 6.
        prefix_states.append(b2)              # s[0:6] for chunk 7.
        
        # Now compute loss with T2.
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_logits = []
        
        # Process chunk 0: standard autoregressive prediction.
        T2_input = level0[0][:, :self.chunk_size - 1, :]  # (batch, chunk_size-1, hidden_dim)
        causal_mask = self.get_causal_mask(T2_input.size(1), T2_input.device)
        T2_out = self.T2(T2_input, causal_mask=causal_mask)
        logits_chunk0 = self.T2_head(T2_out)
        all_logits.append(logits_chunk0)
        target_chunk0 = chunks[:, 0, 1:]
        loss_chunk0 = loss_fn(logits_chunk0.reshape(-1, self.vocab_size),
                              target_chunk0.reshape(-1))
        total_loss += loss_chunk0
        
        # Process chunks 1 to 7.
        for i in range(1, num_chunks):
            prefix = prefix_states[i]  # (batch, chunk_size, hidden_dim)
            current_emb = level0[i][:, :self.chunk_size - 1, :]  # (batch, chunk_size-1, hidden_dim)
            T2_input = torch.cat([prefix, current_emb], dim=1)    # (batch, 2*chunk_size-1, hidden_dim)
            causal_mask = self.get_causal_mask(T2_input.size(1), T2_input.device)
            T2_out = self.T2(T2_input, causal_mask=causal_mask)
            # Predict only the tokens for the current chunk.
            T2_out = T2_out[:, - (self.chunk_size - 1):, :]
            logits_chunk = self.T2_head(T2_out)
            all_logits.append(logits_chunk)
            target_chunk = chunks[:, i, 1:]
            loss_chunk = loss_fn(logits_chunk.reshape(-1, self.vocab_size),
                                 target_chunk.reshape(-1))
            total_loss += loss_chunk
        
        total_loss = total_loss / num_chunks
        return CausalLMOutput(loss=total_loss, logits=all_logits[-1])

# -----------------------------------------------------------------------------
# Main Training Code (unchanged)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train a GPT2-based Transformer Scan LM with binary tree aggregation on WikiText-2.'
    )
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Number of tokens per sample (must be a multiple of chunk_size).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
                                 T1_num_layers=1, T2_num_layers=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=10,
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
        logging_dir="./logs"
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
