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

from transformers import (
    PreTrainedModel,
    GPT2Config,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutput

import datasets

# -----------------------------------------------------------------------------
# 1. Custom Dataset that Interleaves a Fixed "s Token"
# -----------------------------------------------------------------------------

class TextWithSTokenDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len, k):
        """
        Args:
          split: one of "train", "validation", or "test"
          tokenizer: a Hugging Face tokenizer (e.g. GPT2Tokenizer)
          seq_len: number of text tokens per sample (before inserting s tokens)
          k: after every k text tokens, insert one s token.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.k = k

        # Load WikiText dataset (raw text)
        self.data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = " ".join(self.data["text"])
        # Set a very high max_length since we do our own chunking.
        self.tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.vocab_size = tokenizer.vocab_size

        self.samples = []
        # Slide over the token stream in chunks of length seq_len.
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            a_tokens = self.token_ids[i:i + seq_len]
            input_ids_list = []     # For text tokens: store the token id; for s tokens: use -1.
            s_mask_list = []        # True if the position is an s token.
            labels_list = []        # For text tokens: token id; for s tokens: -100 to ignore in loss.
            for j, token in enumerate(a_tokens):
                # Append text token.
                input_ids_list.append(token)
                s_mask_list.append(False)
                labels_list.append(token)
                # Insert an s token after every k text tokens (except at the very end).
                if (j + 1) % self.k == 0 and (j + 1) < len(a_tokens):
                    input_ids_list.append(-1)  # Marker for s token.
                    s_mask_list.append(True)
                    labels_list.append(-100)   # Ignore s token in loss.
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
# 2. No-Extra-Layer S Token GPT Model
# -----------------------------------------------------------------------------

class NoExtraLayerSTokenGPTConfig(GPT2Config):
    def __init__(self,
                 vocab_size=1,      # will be set to the actual vocab size
                 n_positions=1024,
                 n_embd=256,
                 n_layer=4,
                 n_head=4,
                 dropout=0.1,
                 s_token_learnable=False,  # If True, s_token can be updated (optional)
                 **kwargs):
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

        # Learned embedding for text tokens.
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        # For s tokens, we assume the continuous vector is already in the hidden space.
        # Here we initialize it as zeros; you can modify the initialization as desired.
        self.s_token = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=config.s_token_learnable)
        # Positional embeddings.
        self.position_embedding = nn.Embedding(config.n_positions, self.hidden_dim)
        # Transformer blocks.
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        # Final decoder.
        self.decoder = nn.Linear(self.hidden_dim, self.vocab_size)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, s_mask: torch.BoolTensor, labels: torch.LongTensor = None):
        batch_size, seq_len = input_ids.shape
        # For text tokens, perform the standard embedding lookup.
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        # For s tokens, use the fixed continuous s token vector.
        # Note: we assume self.s_token is already in the hidden space.
        s_token_vector = self.s_token.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.hidden_dim)
        # Choose embeddings based on s_mask.
        embeddings = torch.where(s_mask.unsqueeze(-1), s_token_vector, text_embeddings)
        
        # Add positional embeddings.
        position_ids = torch.arange(seq_len, dtype=torch.long, device=embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = embeddings + self.position_embedding(position_ids)
        
        # Pass through transformer blocks.
        for block in self.h:
            hidden_states = block(hidden_states, use_cache=False)[0]
        hidden_states = self.ln_f(hidden_states)
        
        # Compute logits.
        logits = self.decoder(hidden_states)
        
        # If labels are provided, compute loss.
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            return CausalLMOutput(loss=loss, logits=logits)
        else:
            return CausalLMOutput(logits=logits)

# -----------------------------------------------------------------------------
# 3. Collate Function
# -----------------------------------------------------------------------------

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    s_mask = torch.stack([item["s_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "s_mask": s_mask, "labels": labels}

# -----------------------------------------------------------------------------
# 4. Training Script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a GPT-2 LM with interleaved s tokens (no extra layers for s tokens).'
    )
    parser.add_argument('--seq_len', type=int, default=128, help='Number of text tokens per sample (before inserting s tokens).')
    parser.add_argument('--k', type=int, default=10000, help='Insert an s token every k text tokens (set high to disable insertion).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./no_extra_s_token_model/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load GPT2 tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    # Create datasets.
    train_dataset = TextWithSTokenDataset(split="train", tokenizer=tokenizer, seq_len=args.seq_len, k=args.k)
    valid_dataset = TextWithSTokenDataset(split="validation", tokenizer=tokenizer, seq_len=args.seq_len, k=args.k)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Prepare model configuration.
    config = NoExtraLayerSTokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1,
        s_token_learnable=False,  # s token is fixed
    )
    model = NoExtraLayerSTokenGPTModel(config).to(device)

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
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    main()
