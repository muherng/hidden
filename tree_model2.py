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

# Dummy function as in your boilerplate.
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
# Dataset and Data Collation
# -----------------------------------------------------------------------------
class WikiTextDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len):
        """
        Creates samples by concatenating the WikiText raw text and splitting it
        into non-overlapping chunks of length `seq_len`. For language modeling,
        both the input and the labels are the same.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Load WikiText raw text dataset.
        self.data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = " ".join(self.data["text"])
        self.tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        self.samples = []
        # Create non-overlapping chunks.
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            self.samples.append(self.token_ids[i:i+seq_len])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # For LM training the labels are the same as the input_ids.
        sample = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": sample, "labels": sample}

def collate_fn(batch):
    """
    Collate function to stack samples.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

# -----------------------------------------------------------------------------
# Trainer Callbacks
# -----------------------------------------------------------------------------
class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.best_training_loss = float('inf')
        self.best_eval_loss = float('inf')
        self.last_eval_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 10 != 0:
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
        # Call the superclass evaluate() to get the evaluation metrics.
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Initialize internal lists if they don't exist.
        if not hasattr(self, 'eval_steps'):
            self.eval_steps = []
            self.eval_ppls = []

        # Extract current step and perplexity.
        current_step = self.state.global_step if hasattr(self.state, 'global_step') else 0
        eval_loss = metrics.get("eval_loss", None)
        eval_ppl = np.exp(eval_loss) if eval_loss is not None and eval_loss < 100 else float('inf')

        # Append metrics to the lists.
        self.eval_steps.append(current_step)
        self.eval_ppls.append(eval_ppl)

        # Plotting logic.
        plt.figure()
        plt.plot(self.eval_steps, self.eval_ppls, marker='o')
        plt.xlabel("Global Step")
        plt.ylabel("Evaluation Perplexity")
        plt.title("Evaluation Perplexity Over Time")
        plt.ylim(0, min(400, max(self.eval_ppls)*1.1))  # dynamic scaling
        
        # Ensure the 'plots' directory exists.
        os.makedirs("plots", exist_ok=True)

        # Save the plot with timestamp.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"plots/eval_ppl_{timestamp}.png")
        plt.close()

        return metrics

# -----------------------------------------------------------------------------
# Model Definition: Transformer Scan using GPT-2 Blocks
# -----------------------------------------------------------------------------
class T0(nn.Module):
    """
    T0: Initial embedding module.
    Applies token embedding and positional embedding (for a fixed chunk length).
    """
    def __init__(self, config, chunk_size):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(chunk_size, config.n_embd)
        self.chunk_size = chunk_size

    def forward(self, input_ids):
        # input_ids: (batch, chunk_size)
        token_emb = self.wte(input_ids)  # (batch, chunk_size, hidden_dim)
        positions = torch.arange(self.chunk_size, device=input_ids.device).unsqueeze(0)  # (1, chunk_size)
        pos_emb = self.wpe(positions)  # (1, chunk_size, hidden_dim)
        return token_emb + pos_emb

class T1(nn.Module):
    """
    T1: Aggregation module.
    Uses one or more GPT-2 blocks (without a causal mask) to aggregate two chunks.
    """
    def __init__(self, config, num_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    def forward(self, x):
        # x: (batch, 2*chunk_size, hidden_dim)
        for block in self.blocks:
            # No causal mask (full attention)
            x = block(x, attention_mask=None, use_cache=False, output_attentions=False)[0]
        x = self.ln_f(x)
        return x

class T2(nn.Module):
    """
    T2: Autoregressive prediction module.
    Uses a stack of GPT-2 blocks with a causal mask.
    """
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    def forward(self, x, causal_mask):
        # x: (batch, seq_len, hidden_dim)
        for block in self.blocks:
            x = block(x, attention_mask=causal_mask, use_cache=False, output_attentions=False)[0]
        x = self.ln_f(x)
        return x

class TransformerScanModel(nn.Module):
    """
    Overall model combining T0, T1, and T2.
    
    The input sequence is split into fixed-length chunks.
    For each chunk:
      - T0 produces initial embeddings.
      - A prefix state is computed iteratively using T1.
      - T2 takes as input the previous prefix (if any) concatenated with the current chunk (except the last token)
        to produce outputs used for next-token prediction.
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
        # Optionally, you can tie the weights of T2_head with T0.wte (if desired).
        # self.T2_head.weight = self.T0.wte.weight

    def get_causal_mask(self, seq_length, device):
        # Generate a causal mask of shape (1, 1, seq_length, seq_length)
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).unsqueeze(0).unsqueeze(0)
        mask = (1.0 - mask) * -10000.0
        return mask

    def forward(self, input_ids, labels=None):
        #print("in forward")
        """
        input_ids: (batch, seq_length), where seq_length is a multiple of chunk_size.
        The loss is computed over each chunk’s prediction.
        """
        batch_size, seq_length = input_ids.shape
        assert seq_length % self.chunk_size == 0, "Sequence length must be a multiple of chunk_size"
        num_chunks = seq_length // self.chunk_size

        # Reshape tokens into chunks: (batch, num_chunks, chunk_size)
        chunks = input_ids.view(batch_size, num_chunks, self.chunk_size)

        # Compute T0 embeddings for each chunk.
        # s_list[i] = T0(chunk_i), shape: (batch, chunk_size, hidden_dim)
        s_list = []
        for i in range(num_chunks):
            chunk = chunks[:, i, :]  # (batch, chunk_size)
            s_i = self.T0(chunk)      # (batch, chunk_size, hidden_dim)
            s_list.append(s_i)

        # Compute prefix states iteratively.
        prefix_states = [s_list[0]]  # For chunk 0, prefix state is just s_list[0]
        for i in range(1, num_chunks):
            # Concatenate previous prefix with current chunk embedding.
            agg_input = torch.cat([prefix_states[i-1], s_list[i]], dim=1)  # (batch, 2*chunk_size, hidden_dim)
            agg_output = self.T1(agg_input)  # (batch, 2*chunk_size, hidden_dim)
            # Select the last chunk_size tokens.
            new_prefix = agg_output[:, -self.chunk_size:, :]  # (batch, chunk_size, hidden_dim)
            prefix_states.append(new_prefix)

        total_loss = 0.0
        loss_fn = nn.CrossEntropyLoss()
        all_logits = []

        # Process chunk 0: standard autoregressive next-token prediction.
        T2_input = s_list[0][:, :self.chunk_size - 1, :]  # (batch, chunk_size - 1, hidden_dim)
        seq_len_T2 = T2_input.size(1)
        causal_mask = self.get_causal_mask(seq_len_T2, T2_input.device)
        T2_out = self.T2(T2_input, causal_mask=causal_mask)  # (batch, seq_len_T2, hidden_dim)
        logits_chunk0 = self.T2_head(T2_out)  # (batch, seq_len_T2, vocab_size)
        all_logits.append(logits_chunk0)
        # Target for chunk 0: tokens 1:chunk_size from chunk 0.
        target_chunk0 = chunks[:, 0, 1:]  # (batch, chunk_size - 1)
        loss_chunk0 = loss_fn(logits_chunk0.reshape(-1, self.vocab_size),
                            target_chunk0.reshape(-1))
        total_loss += loss_chunk0

        for i in range(1, num_chunks):
            prefix = prefix_states[i - 1]  # (batch, chunk_size, hidden_dim)
            current_emb = s_list[i][:, :self.chunk_size - 1, :]  # (batch, chunk_size - 1, hidden_dim)
            T2_input = torch.cat([prefix, current_emb], dim=1)  # (batch, 2*chunk_size - 1, hidden_dim)
            seq_len_T2 = T2_input.size(1)
            causal_mask = self.get_causal_mask(seq_len_T2, T2_input.device)
            T2_out = self.T2(T2_input, causal_mask=causal_mask)  # (batch, seq_len_T2, hidden_dim)
            T2_out = T2_out[:, - (self.chunk_size - 1):, :]  # (batch, chunk_size - 1, hidden_dim)
            logits_chunk = self.T2_head(T2_out)  # (batch, chunk_size - 1, vocab_size)
            all_logits.append(logits_chunk)
            target_chunk = chunks[:, i, 1:]  # (batch, chunk_size - 1)
            loss_chunk = loss_fn(logits_chunk.reshape(-1, self.vocab_size),
                                target_chunk.reshape(-1))
            total_loss += loss_chunk

        total_loss = total_loss / num_chunks

        # Return a CausalLMOutput; here we return the loss and the logits of the last chunk.
        return CausalLMOutput(loss=total_loss, logits=all_logits[-1])

# -----------------------------------------------------------------------------
# Main Training Code
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train a GPT2-based Transformer Scan LM on WikiText-2.'
    )
    parser.add_argument('--seq_len', type=int, default=8*64,
                        help='Number of tokens per sample (must be a multiple of chunk_size).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # Optionally add an argument for chunk_size; defaulting to 8.
    parser.add_argument('--chunk_size', type=int, default=64, help='Chunk size.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Create a unique output directory.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./tree_model/tree_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the GPT2 tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    # Create training and validation datasets.
    train_dataset = WikiTextDataset(split="train", tokenizer=tokenizer, seq_len=args.seq_len)
    valid_dataset = WikiTextDataset(split="validation", tokenizer=tokenizer, seq_len=args.seq_len)

    # Create a GPT2 configuration.
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,  # match your tokenizer
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1,
        # layer_norm_epsilon defaults to 1e-5
    )
    # Instantiate our custom model.
    model = TransformerScanModel(config, chunk_size=args.chunk_size,
                                 T1_num_layers=4, T2_num_layers=4)
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
