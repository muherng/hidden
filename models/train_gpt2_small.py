#!/usr/bin/env python3
"""
GPT2 Small (125M) Training Script using HuggingFace Trainer.

This script trains a canonical GPT2 small model on WikiText-103 or OpenWebText.
It's designed to work with the comparison pipeline:
- train_comparison.py launches this script for GPT2 small models
- evaluate_comparison.py evaluates the checkpoints

Checkpoints are saved in HuggingFace format for direct evaluation.

Usage:
    python models/train_gpt2_small.py --dataset wikitext-103 --dropout 0.1
    python models/train_gpt2_small.py --dataset openwebtext --skip_samples 10000 --dropout 0.1
"""

import argparse
import datetime
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)

import datasets


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def get_cache_dir():
    """Get the cache directory for tokenized datasets."""
    cache_dir = get_project_root() / "cache" / "tokenized"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# -----------------------------------------------------------------------------
# Dataset Classes
# -----------------------------------------------------------------------------
class WikiTextDataset(Dataset):
    """WikiText-103 or WikiText-2 dataset for language modeling with caching."""
    
    def __init__(self, dataset_name, split, tokenizer, seq_len, num_workers=8, use_cache=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []
        
        # Try to load from cache
        if use_cache:
            cache_file = get_cache_dir() / f"{dataset_name}_{split}_seq{seq_len}.pt"
            if cache_file.exists():
                print(f"  Loading cached tokenized data from {cache_file}")
                cached_data = torch.load(cache_file)
                self.samples = cached_data["samples"]
                print(f"  {split}: loaded {len(self.samples):,} samples from cache")
                return
        
        # Load dataset from HuggingFace
        print(f"  Loading {dataset_name} {split} split...")
        if dataset_name == "wikitext-103":
            data = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        elif dataset_name == "wikitext-2":
            data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        else:
            raise ValueError(f"Unknown WikiText dataset: {dataset_name}")
        
        # Join all text
        print(f"  Joining text...")
        text = " ".join(data["text"])
        
        # Tokenize
        print(f"  Tokenizing (this may take a few minutes for wikitext-103)...")
        self.tokenizer.model_max_length = int(1e7)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"  Tokenized: {len(token_ids):,} tokens")
        
        # Create non-overlapping samples
        print(f"  Creating samples...")
        for i in range(0, len(token_ids) - seq_len, seq_len):
            self.samples.append(token_ids[i:i+seq_len])
        
        print(f"  {split}: {len(token_ids):,} tokens -> {len(self.samples):,} samples")
        
        # Save to cache
        if use_cache:
            cache_file = get_cache_dir() / f"{dataset_name}_{split}_seq{seq_len}.pt"
            print(f"  Saving to cache: {cache_file}")
            torch.save({"samples": self.samples}, cache_file)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": sample, "labels": sample}


class OpenWebTextDataset(Dataset):
    """OpenWebText dataset for language modeling with memory-efficient processing and caching."""
    
    def __init__(self, split, tokenizer, seq_len, skip_samples=0, num_workers=8, use_cache=True, max_samples=None):
        """
        Args:
            split: 'train' or 'validation'
            tokenizer: GPT2 tokenizer
            seq_len: Sequence length
            skip_samples: Number of samples to skip (for validation set separation)
                         Training should skip first N samples (e.g., 10,000)
                         Validation uses first N samples (e.g., first 1,000 of 10,000)
            num_workers: Number of workers for tokenization
            use_cache: Whether to use cached tokenized data
            max_samples: Maximum number of samples to collect (None = no limit)
                        For Chinchilla scaling: 103,700 steps * 64 batch = ~6.6M samples
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []
        
        tokenizer.model_max_length = int(1e7)
        
        # Try to load from cache
        max_str = f"_max{max_samples}" if max_samples else ""
        cache_name = f"openwebtext_{split}_skip{skip_samples}_seq{seq_len}{max_str}"
        if use_cache:
            cache_file = get_cache_dir() / f"{cache_name}.pt"
            if cache_file.exists():
                print(f"  Loading cached tokenized data from {cache_file}")
                cached_data = torch.load(cache_file)
                self.samples = cached_data["samples"]
                print(f"  {split}: loaded {len(self.samples):,} samples from cache")
                return
        
        # Load OpenWebText with streaming for efficiency
        print(f"  Loading OpenWebText ({split} split)...")
        full_data = datasets.load_dataset("openwebtext", "plain_text", split="train", streaming=True)
        
        # Process in batches, creating samples on-the-fly to minimize memory usage
        BATCH_SIZE = 5_000  # Process 5k documents at a time
        token_buffer = []  # Buffer for incomplete sequences
        
        if split == "validation":
            # Validation uses first 1,000 samples (subset of skipped training samples)
            VALIDATION_SIZE = 1_000
            print(f"  Collecting first {VALIDATION_SIZE:,} samples for validation...")
            
            batch_texts = []
            for i, sample in enumerate(full_data):
                if i >= VALIDATION_SIZE:
                    break
                batch_texts.append(sample["text"])
            
            if len(batch_texts) < VALIDATION_SIZE:
                print(f"  Warning: Only collected {len(batch_texts)} samples")
            
            # Tokenize validation batch (small enough to fit in memory)
            print(f"  Tokenizing {len(batch_texts):,} validation documents...")
            text = " ".join(batch_texts)
            all_token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            # Create samples
            for i in range(0, len(all_token_ids) - seq_len, seq_len):
                self.samples.append(all_token_ids[i:i+seq_len])
            
            print(f"  {split}: {len(all_token_ids):,} tokens -> {len(self.samples):,} samples")
            
        elif split == "train":
            # Training skips first skip_samples, then processes in batches
            # Memory-efficient: create samples as we go, don't accumulate all tokens
            print(f"  Skipping first {skip_samples:,} samples (reserved for validation)...")
            if max_samples:
                print(f"  Collecting up to {max_samples:,} samples...")
            print(f"  Processing in batches of {BATCH_SIZE:,} documents...")
            
            batch_texts = []
            samples_seen = 0
            batches_processed = 0
            total_tokens = 0
            done = False
            
            for sample in full_data:
                if done:
                    break
                    
                samples_seen += 1
                if samples_seen <= skip_samples:
                    continue  # Skip validation samples
                
                batch_texts.append(sample["text"])
                
                # Process batch when full
                if len(batch_texts) >= BATCH_SIZE:
                    text = " ".join(batch_texts)
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    
                    # Add to buffer and extract complete samples
                    token_buffer.extend(tokens)
                    while len(token_buffer) >= seq_len:
                        self.samples.append(token_buffer[:seq_len])
                        token_buffer = token_buffer[seq_len:]
                        
                        # Check if we've reached max_samples
                        if max_samples and len(self.samples) >= max_samples:
                            done = True
                            break
                    
                    total_tokens += len(tokens)
                    batches_processed += 1
                    batch_texts = []  # Clear for next batch
                    
                    if batches_processed % 20 == 0:
                        print(f"    Processed {batches_processed * BATCH_SIZE:,} documents, "
                              f"{total_tokens:,} tokens, {len(self.samples):,} samples...")
            
            # Process remaining documents (only if not done)
            if batch_texts and not done:
                text = " ".join(batch_texts)
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_buffer.extend(tokens)
                while len(token_buffer) >= seq_len:
                    self.samples.append(token_buffer[:seq_len])
                    token_buffer = token_buffer[seq_len:]
                    if max_samples and len(self.samples) >= max_samples:
                        break
                total_tokens += len(tokens)
            
            print(f"  Processed {samples_seen - skip_samples:,} documents")
            print(f"  {split}: {total_tokens:,} tokens -> {len(self.samples):,} samples")
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Save to cache
        if use_cache:
            cache_file = get_cache_dir() / f"{cache_name}.pt"
            print(f"  Saving to cache: {cache_file}")
            torch.save({"samples": self.samples}, cache_file)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": sample, "labels": sample}


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


# -----------------------------------------------------------------------------
# Trainer Callbacks
# -----------------------------------------------------------------------------
class PrintLossCallback(TrainerCallback):
    """Callback to print training progress."""
    
    def __init__(self):
        self.best_training_loss = float('inf')
        self.best_eval_loss = float('inf')
        self.last_eval_loss = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 100 != 0:
            return
        if logs is None:
            return
        
        epoch = getattr(state, "epoch", None)
        
        # Training loss
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
        
        # Eval loss
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
        
        # Build output string
        out_str = f"Step {state.global_step}: "
        if epoch is not None:
            out_str += f"Epoch {epoch:.2f} | "
        if current_loss is not None:
            out_str += f"Train Loss: {current_loss:.4f} (Best: {self.best_training_loss:.4f}, PPL: {training_perplexity:.2f})"
        if current_eval_loss is not None:
            out_str += f" | Eval Loss: {current_eval_loss:.4f} (Best: {self.best_eval_loss:.4f}, PPL: {eval_perplexity:.2f})"
        
        print(out_str, flush=True)


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main(args):
    set_seed(args.seed)
    
    # Check CUDA availability with detailed diagnostics
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA compiled: {torch.version.cuda}")
    print(f"PyTorch built with CUDA: {torch.backends.cuda.is_built()}")
    
    # Try explicit CUDA initialization
    try:
        torch.cuda.init()
        print("CUDA init() succeeded")
    except Exception as e:
        print(f"CUDA init() failed: {e}")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available! Training on CPU will be very slow.")
        print("Diagnostic info:")
        print(f"  torch.backends.cudnn.is_available(): {torch.backends.cudnn.is_available()}")
        import os
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')[:200]}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # Generate experiment name and output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dropout_str = f"drop{args.dropout}" if args.dropout > 0 else "nodrop"
    dataset_short = args.dataset.replace("-", "_")
    exp_name = f"gpt2_small_{dataset_short}_{dropout_str}_{timestamp}"
    
    # Save to flame/exp/ directory for consistency with other models
    project_root = get_project_root()
    output_dir = project_root / "flame" / "exp" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment: {exp_name}")
    print(f"Output directory: {output_dir}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)
    
    # Load datasets (with caching for WikiText)
    print("\nLoading datasets...")
    if args.dataset == "wikitext-103":
        train_dataset = WikiTextDataset(
            dataset_name="wikitext-103",
            split="train",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_workers=args.tokenize_workers,
            use_cache=True,
        )
        eval_dataset = WikiTextDataset(
            dataset_name="wikitext-103",
            split="validation",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_workers=args.tokenize_workers,
            use_cache=True,
        )
    elif args.dataset == "wikitext-2":
        train_dataset = WikiTextDataset(
            dataset_name="wikitext-2",
            split="train",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_workers=args.tokenize_workers,
            use_cache=True,
        )
        eval_dataset = WikiTextDataset(
            dataset_name="wikitext-2",
            split="validation",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_workers=args.tokenize_workers,
            use_cache=True,
        )
    elif args.dataset == "openwebtext":
        # Calculate max samples needed for training
        # Add 10% buffer to ensure we have enough samples for shuffling
        if args.total_steps:
            max_train_samples = int(args.total_steps * args.batch_size * 1.1)
        else:
            max_train_samples = None  # No limit if using epochs
        
        train_dataset = OpenWebTextDataset(
            split="train",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            skip_samples=args.skip_samples,
            num_workers=args.tokenize_workers,
            use_cache=True,
            max_samples=max_train_samples,
        )
        eval_dataset = OpenWebTextDataset(
            split="validation",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            skip_samples=0,  # Validation uses first samples
            num_workers=args.tokenize_workers,
            use_cache=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Eval samples: {len(eval_dataset):,}")
    
    # Create GPT2 small config (125M parameters)
    # Standard GPT2 small: 12 layers, 768 hidden, 12 heads
    config = GPT2Config(
        vocab_size=50257,  # GPT2 tokenizer vocab size
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,  # 4x hidden_size
        activation_function="gelu_new",
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    
    # Ensure we use standard attention (not flash attention)
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    
    # Create model
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"\nModel: GPT2 Small")
    print(f"Parameters: {count_params(model)/1e6:.2f}M")
    print(f"Dropout: {args.dropout}")
    
    # Calculate training setup
    if args.total_steps:
        max_steps = args.total_steps
        num_train_epochs = 9999  # Will be limited by max_steps
    else:
        max_steps = -1
        num_train_epochs = args.epochs
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        
        # Training parameters
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        
        # Steps/epochs
        max_steps=max_steps if max_steps > 0 else -1,
        num_train_epochs=num_train_epochs if max_steps <= 0 else 9999,
        
        # Optimizer
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,  # Keep all checkpoints
        
        # Logging
        logging_steps=args.logging_steps,
        logging_dir=str(output_dir / "logs"),
        report_to=[],  # No external logging (no wandb)
        
        # Other
        fp16=False,  # Use full precision for reproducibility
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Total steps: {max_steps if max_steps > 0 else 'N/A (using epochs)'}")
    print(f"  Epochs: {num_train_epochs if max_steps <= 0 else 'N/A (using steps)'}")
    print(f"  Adam betas: ({args.adam_beta1}, {args.adam_beta2})")
    print(f"  BF16: {training_args.bf16}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        callbacks=[PrintLossCallback()],
    )
    
    # Remove default progress callback for cleaner output
    from transformers import ProgressCallback
    trainer.remove_callback(ProgressCallback)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Final model saved to: {final_path}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GPT2 Small (125M) on WikiText-103 or OpenWebText"
    )
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                       choices=["wikitext-103", "wikitext-2", "openwebtext"],
                       help="Dataset to train on")
    parser.add_argument("--skip_samples", type=int, default=0,
                       help="Number of samples to skip for OpenWebText (for validation set)")
    parser.add_argument("--tokenize_workers", type=int, default=8,
                       help="Number of workers for tokenization")
    
    # Model
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate (0.0 or 0.1)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Per-device batch size")
    parser.add_argument("--seq_len", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Warmup steps")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="Adam beta2 (0.999 for HuggingFace default, 0.95 for nanoGPT)")
    
    # Duration
    parser.add_argument("--total_steps", type=int, default=None,
                       help="Total training steps (overrides epochs)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs (used if total_steps not set)")
    
    # Checkpointing & logging
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=5000,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log every N steps")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
