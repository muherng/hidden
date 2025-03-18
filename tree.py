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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

def main():
    parser = argparse.ArgumentParser(
        description='Train a standard GPT2 LM on WikiText-2 without noise tokens.'
    )
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Number of tokens per sample.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
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

     # Load a standard GPT2 LM head model.
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,  # match your tokenizer
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1
        # You can adjust additional hyperparameters here if needed.
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(tokenizer.vocab_size)
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


