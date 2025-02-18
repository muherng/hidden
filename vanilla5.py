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

# -----------------------------------------------------------------------------
# 1. Custom Dataset: Interleaved s tokens
# -----------------------------------------------------------------------------

class TextWithSTokenDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len, k):
        """
        For each contiguous chunk of text tokens (length seq_len), after every k text tokens,
        insert an s token marker (token id -1). Labels are set to the original token for text positions
        and -100 for s token positions.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.k = k

        self.data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = " ".join(self.data["text"])
        self.tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.vocab_size = tokenizer.vocab_size

        self.samples = []
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            a_tokens = self.token_ids[i:i + seq_len]
            input_ids_list = []  # For text tokens: token id; for s tokens: -1.
            s_mask_list = []     # True if position is an s token.
            labels_list = []     # For text tokens: token id; for s tokens: -100.
            for j, token in enumerate(a_tokens):
                input_ids_list.append(token)
                s_mask_list.append(False)
                labels_list.append(token)
                if (j + 1) % self.k == 0 and (j + 1) < len(a_tokens):
                    input_ids_list.append(-1)  # marker for s token.
                    s_mask_list.append(True)
                    labels_list.append(-100)   # ignore in loss.
            sample = {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "s_mask": torch.tensor(s_mask_list, dtype=torch.bool),
                "labels": torch.tensor(labels_list, dtype=torch.long)
            }
            self.samples.append(sample)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# -----------------------------------------------------------------------------
# 2. No-Extra-Layer S Token GPT Model (Module)
# -----------------------------------------------------------------------------

class NoExtraLayerSTokenGPTConfig(GPT2Config):
    def __init__(self, vocab_size=1, n_positions=1024, n_embd=256, n_layer=4, n_head=4,
                 dropout=0.1, s_token_learnable=False, **kwargs):
        super().__init__(vocab_size=vocab_size,
                         n_positions=n_positions,
                         n_embd=n_embd,
                         n_layer=n_layer,
                         n_head=n_head,
                         resid_pdrop=dropout,
                         embd_pdrop=dropout,
                         attn_pdrop=dropout,
                         **kwargs)
        self.s_token_learnable = s_token_learnable

class NoExtraLayerSTokenGPTModel(PreTrainedModel):
    config_class = NoExtraLayerSTokenGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.n_embd
        self.vocab_size = config.vocab_size

        # Standard token embedding.
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        # s token: fixed continuous vector.
        self.s_token = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=config.s_token_learnable)
        # Positional embeddings.
        self.position_embedding = nn.Embedding(config.n_positions, self.hidden_dim)
        # Transformer blocks.
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        # Decoder.
        self.decoder = nn.Linear(self.hidden_dim, self.vocab_size)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, s_mask: torch.BoolTensor,
                labels: torch.LongTensor = None, return_hidden: bool = False,
                override_s: torch.Tensor = None, attention_mask: torch.Tensor = None):
        """
        If override_s is provided, use that for positions where s_mask is True.
        If attention_mask is provided, pass it to each transformer block.
        """
        batch_size, seq_len = input_ids.shape
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        if override_s is not None:
            s_token_vector = override_s
        else:
            s_token_vector = self.s_token.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.hidden_dim)
        embeddings = torch.where(s_mask.unsqueeze(-1), s_token_vector, text_embeddings)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=embeddings.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = embeddings + self.position_embedding(position_ids)
        # Pass through transformer blocks with the custom attention mask if provided.
        for block in self.h:
            hidden_states = block(hidden_states, use_cache=False, attention_mask=attention_mask)[0]
        hidden_states = self.ln_f(hidden_states)
        logits = self.decoder(hidden_states)
        output = CausalLMOutput(logits=logits)
        if return_hidden:
            output.hidden_states = hidden_states
        if labels is not None and not return_hidden:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, self.vocab_size),
                                     shift_labels.reshape(-1),
                                     ignore_index=-100)
            output.loss = loss
        return output

# -----------------------------------------------------------------------------
# 3. Helper: Compute Custom Attention Mask
# -----------------------------------------------------------------------------

def compute_custom_attention_mask(s_mask):
    """
    Given s_mask of shape (B, L) (bool tensor), returns a tensor of shape (B, 1, L, L)
    to be added to the attention scores in module2 so that for each position i,
    only positions from the most recent s token (inclusive) to i-1 are allowed.
    We use a large negative number (-1e9) for disallowed positions.
    """
    B, L = s_mask.size()
    device = s_mask.device
    # Create an index tensor for positions.
    indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
    # For each sample, compute the index of the most recent s token for each position.
    # Set positions with no s token to 0.
    s_mask_idx = torch.where(s_mask, indices, torch.tensor(-1, device=device))
    # Use cummax to get the last s token index at each position.
    _, last_s = torch.cummax(s_mask_idx, dim=1)  # (B, L)
    # Create matrices of indices: row indices for i, col indices for j.
    i_indices = torch.arange(L, device=device).unsqueeze(1).expand(L, L)  # (L, L)
    j_indices = torch.arange(L, device=device).unsqueeze(0).expand(L, L)  # (L, L)
    # For each row i, get the allowed minimum index from last_s.
    # We need to expand last_s to (B, L, 1).
    last_s_expanded = last_s.unsqueeze(2)  # (B, L, 1)
    # Now, for each sample b and for each row i and column j, allowed if:
    # j < i (causal) and j >= last_s[b, i].
    # Compute the causal mask: allowed if j < i.
    causal = (j_indices < i_indices).to(device)  # (L, L), bool
    # Expand causal to (B, L, L)
    causal = causal.unsqueeze(0).expand(B, L, L)
    # For each sample, create allowed mask: j_indices >= last_s (broadcasted along dimension 1)
    allowed = (j_indices.unsqueeze(0).expand(B, L, L) >= last_s_expanded)
    # Final allowed mask: allowed if both conditions hold.
    final_allowed = causal & allowed
    # Set mask: 0 for allowed positions, -1e9 for disallowed.
    attn_mask = torch.where(final_allowed, torch.tensor(0.0, device=device), torch.tensor(-1e9, device=device))
    # Expand to shape (B, 1, L, L)
    attn_mask = attn_mask.unsqueeze(1)
    return attn_mask

# -----------------------------------------------------------------------------
# 4. Composite Two-Stage Model
# -----------------------------------------------------------------------------

class TwoStageModel(nn.Module):
    def __init__(self, module1: NoExtraLayerSTokenGPTModel, module2: NoExtraLayerSTokenGPTModel):
        """
        module1 processes input p and outputs p' (with continuous s' tokens).
        module2 takes p'' as input where s token positions are replaced with s' tokens from module1.
        """
        super().__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, input_ids: torch.LongTensor, s_mask: torch.BoolTensor, labels: torch.LongTensor = None):
        # Module1 pass: get hidden states.
        out1 = self.module1(input_ids=input_ids, s_mask=s_mask, labels=labels, return_hidden=True)
        # Compute custom attention mask from s_mask.
        custom_attn_mask = compute_custom_attention_mask(s_mask)
        # Module2 pass: use module1's hidden states to override s token positions, and pass the custom attention mask.
        out2 = self.module2(input_ids=input_ids, s_mask=s_mask, override_s=out1.hidden_states, labels=labels,
                            attention_mask=custom_attn_mask)
        return out2

# -----------------------------------------------------------------------------
# 5. Collate Function
# -----------------------------------------------------------------------------

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    s_mask = torch.stack([item["s_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "s_mask": s_mask, "labels": labels}

# -----------------------------------------------------------------------------
# 6. PrintLossCallback for Logging
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 7. Custom Trainer Subclass (Override compute_loss)
# -----------------------------------------------------------------------------

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=inputs["input_ids"].device, dtype=torch.float)
        return (loss, outputs) if return_outputs else loss

# -----------------------------------------------------------------------------
# 8. Main Training Script with Timestamp Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a two-stage transformer model: module1 and module2 with custom attention masking.'
    )
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Number of text tokens per sample (before inserting s tokens).')
    parser.add_argument('--k', type=int, default=3,
                        help='Insert an s token every k text tokens.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Create a unique output directory using a timestamp and input parameter info.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./two_stage_model", f"seq{args.seq_len}_k{args.k}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    # Create datasets.
    train_dataset = TextWithSTokenDataset(split="train", tokenizer=tokenizer, seq_len=args.seq_len, k=args.k)
    valid_dataset = TextWithSTokenDataset(split="validation", tokenizer=tokenizer, seq_len=args.seq_len, k=args.k)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    # Instantiate module1.
    config1 = NoExtraLayerSTokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,       # You could also reduce this if necessary.
        n_layer=4,        # Or reduce the number of layers.
        n_head=4,
        dropout=0.3,      # Increase dropout here.
        s_token_learnable=False,
    )
    module1 = NoExtraLayerSTokenGPTModel(config1)
    # Instantiate module2.
    config2 = NoExtraLayerSTokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,       # You could also reduce this if necessary.
        n_layer=4,        # Or reduce the number of layers.
        n_head=4,
        dropout=0.3,      # Increase dropout here.
        s_token_learnable=False,
    )
    module2 = NoExtraLayerSTokenGPTModel(config2)
    # Build composite two-stage model.
    composite_model = TwoStageModel(module1, module2)
    composite_model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,           # Lower learning rate.
        warmup_steps=1000,            # Increase warmup steps.
        weight_decay=0.1,             # Increase weight decay.
        fp16=False,
        seed=args.seed,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    trainer = CustomTrainer(
        model=composite_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    trainer.add_callback(PrintLossCallback())
    trainer.train()

if __name__ == "__main__":
    main()
