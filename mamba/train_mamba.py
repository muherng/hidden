import torch
import os
import json
import argparse
import numpy as np
import random
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from associative_recall import AssociativeRecallDataset

class CustomMambaConfig(MambaConfig):
    """Custom MambaConfig that adds to_dict method required by Trainer."""
    def to_dict(self):
        """Convert config to dictionary format."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "ssm_cfg": self.ssm_cfg,
            "rms_norm": self.rms_norm,
            "residual_in_fp32": self.residual_in_fp32,
            "fused_add_norm": self.fused_add_norm,
            "pad_vocab_size_multiple": self.pad_vocab_size_multiple
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="mamba_models")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--input_seq_len", type=int, default=64)
    parser.add_argument("--num_kv_pairs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use_bfloat16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_wandb", type=bool, default=False)
    return parser.parse_args()

def save_model_with_shared_tensors(model, output_dir, _internal_call=False):
    """Custom save method that handles shared tensors correctly"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model using torch.save instead of safetensors
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save the config
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)

def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Custom loss computation for Mamba model using direct supervision for associative recall
    """
    # Get model outputs - Mamba only needs input_ids
    outputs = model(input_ids=inputs['input_ids'])
    
    # Get logits from the last hidden state
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    
    # Use the actual labels from the dataset for direct supervision
    labels = inputs['labels']
    
    # Compute loss only on non-ignored tokens (where labels != -100)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    if return_outputs:
        return (loss, outputs)
    return loss

def compute_metrics(eval_pred):
    """Compute metrics for evaluation using direct supervision."""
    logits, labels = eval_pred
    
    # Compute loss only on non-ignored tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Compute error rate only on non-ignored tokens
    predictions = torch.argmax(logits, dim=-1)
    mask = labels != -100
    num_errors = ((predictions != labels) & mask).sum().item()
    total_tokens = mask.sum().item()
    error_rate = num_errors / total_tokens if total_tokens > 0 else 1.0
    
    return {
        "eval_loss": float(loss),
        "eval_error_rate": float(error_rate),
        "eval_num_errors": int(num_errors)
    }

def collate_fn(batch):
    """Collate function to properly format batches for training."""
    print("\nCollate function:")
    print(f"Batch size: {len(batch)}")
    print(f"Batch type: {type(batch)}")
    print(f"First item type: {type(batch[0])}")
    print(f"First item keys: {list(batch[0].keys())}")
    print(f"First item shapes: {[(k, v.shape) for k, v in batch[0].items()]}")
    print(f"First item contents types: {[(k, type(v)) for k, v in batch[0].items()]}")
    print(f"First item memory addresses: {[(k, id(v)) for k, v in batch[0].items()]}")
    print(f"First item device: {[(k, v.device) for k, v in batch[0].items()]}")
    print(f"First item requires_grad: {[(k, v.requires_grad) for k, v in batch[0].items()]}")
    print(f"First item is_leaf: {[(k, v.is_leaf) for k, v in batch[0].items()]}")
    print(f"First item grad_fn: {[(k, v.grad_fn) for k, v in batch[0].items()]}")
    print(f"First item storage: {[(k, v.storage().data_ptr()) for k, v in batch[0].items()]}")
    
    try:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        print(f"Stacked shapes - input_ids: {input_ids.shape}, labels: {labels.shape}")
        return {"input_ids": input_ids, "labels": labels}
    except KeyError as e:
        print(f"KeyError in collate_fn: {e}")
        print("Available keys in first item:", list(batch[0].keys()))
        print("Full batch contents:")
        for i, item in enumerate(batch):
            print(f"Item {i}:")
            print(f"  Keys: {list(item.keys())}")
            print(f"  Types: {[(k, type(v)) for k, v in item.items()]}")
            print(f"  Shapes: {[(k, v.shape) for k, v in item.items()]}")
        raise

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_dataset = AssociativeRecallDataset(
        vocab_size=args.vocab_size,
        num_examples=args.num_examples,
        input_seq_len=args.input_seq_len,
        seed=args.seed,
        num_kv_pairs=args.num_kv_pairs,
        power_a=1.0  # Set to 1 for uniformly distributed gaps between keys and values
    )
    
    valid_dataset = AssociativeRecallDataset(
        vocab_size=args.vocab_size,
        num_examples=args.num_examples // 10,  # Use 10% of training size for validation
        input_seq_len=args.input_seq_len,
        seed=args.seed + 10,  # Different seed for validation set
        num_kv_pairs=args.num_kv_pairs,
        power_a=1.0  # Set to 1 for uniformly distributed gaps between keys and values
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(valid_dataset)}")
    
    # Initialize Mamba model
    model_name = "state-spaces/mamba-370m"
    print(f"\nLoading model from {model_name}...")
    
    config = CustomMambaConfig(
        vocab_size=args.vocab_size,
        d_model=2560,
        n_layer=24,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8
    )
    
    model = MambaLMHeadModel(
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32,
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        report_to="none" if args.disable_wandb else "wandb",
        bf16=args.use_bfloat16,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,  # Disable multiprocessing to debug data loading
        dataloader_pin_memory=False,  # Disable pin memory to debug data loading
    )
    
    # Create custom data loaders
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    eval_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize trainer with custom data loaders
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,  # Use collate_fn directly as data_collator
    )
    
    # Override the trainer's get_train_dataloader and get_eval_dataloader methods
    trainer.get_train_dataloader = lambda: train_dataloader
    trainer.get_eval_dataloader = lambda: eval_dataloader
    
    # Add the custom compute_loss method to the trainer
    trainer.compute_loss = compute_loss.__get__(trainer)
    
    # Override the save_model method
    original_save_model = trainer.save_model
    def custom_save_model(output_dir, _internal_call=False):
        save_model_with_shared_tensors(model, output_dir, _internal_call)
    trainer.save_model = custom_save_model
    
    # Initialize wandb if not disabled
    if not args.disable_wandb:
        print("\n=== Initializing Weights & Biases ===")
        try:
            if wandb.api.api_key is None:
                print("WARNING: No wandb API key found. Please run 'wandb login' first.")
                print("Continuing without wandb logging...")
            else:
                wandb_run = wandb.init(
                    project="mamba-ar",
                    config={
                        "model": model_name,
                        "vocab_size": args.vocab_size,
                        "input_seq_len": args.input_seq_len,
                        "num_kv_pairs": args.num_kv_pairs,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                    }
                )
                print(f"Wandb run initialized successfully with ID: {wandb_run.id}")
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            print("Continuing without wandb logging...")
    
    # Train the model
    print("\n=== Starting Training ===")
    trainer.train()
    
    # Save final model
    print("\n=== Saving Final Model ===")
    trainer.save_model(args.output_dir)
    
    # Close wandb run if it was initialized
    if not args.disable_wandb and 'wandb_run' in locals():
        print("\n=== Finalizing Weights & Biases ===")
        wandb_run.finish()
        print("Wandb run completed and synced")

if __name__ == "__main__":
    main() 