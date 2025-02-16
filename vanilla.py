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
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    TrainingArguments,
    Trainer,
    set_seed,
)
import datasets

# -----------------------------------------------------------------------------
# 1. Standard WikiText Dataset
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
# 2. Training Script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a standard GPT2 LM on WikiText-2 without noise tokens.'
    )
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Number of tokens per sample.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Create a unique output directory.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./gpt2_lm_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the GPT2 tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    # Create training and validation datasets.
    train_dataset = WikiTextDataset(split="train", tokenizer=tokenizer, seq_len=args.seq_len)
    valid_dataset = WikiTextDataset(split="validation", tokenizer=tokenizer, seq_len=args.seq_len)

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    # Load a standard GPT2 LM head model.
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,  # match your tokenizer
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        # You can adjust additional hyperparameters here if needed.
    )
    model = GPT2LMHeadModel(config)
    #model = GPT2LMHeadModel.from_pretrained("gpt2")
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
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
