import torch
import os
import json
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    TrainerCallback,
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from utils.data_loaders import ChunkedDataset
from utils.data_collators import (
    DataCollatorForLanguageModelingWithNextTokenSupervision,
    DataCollatorForLanguageModelingWithDirectSupervision
)
from permutation_task import PermutationTask, compute_parity
from utils.model_utils import setup_tokenizer
import torch.nn as nn

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="mamba_models")
    parser.add_argument("--num_items", type=int, default=3, choices=[3, 5])
    parser.add_argument("--supervision_type", type=str, default="direct_state", 
                       choices=["direct_state", "direct_topic", "next_token"])
    parser.add_argument("--no_pretrain", action="store_true", default=False)
    parser.add_argument("--from_checkpoint", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_bfloat16", action="store_true", default=False)
    parser.add_argument("--save_all_checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_determinism", action="store_true", default=False)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--is_parity_cur", action="store_true", default=False)
    parser.add_argument("--disable_wandb", type=bool, default=False, help="Set to False to enable wandb logging")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--num_stories", type=int, default=10000)
    parser.add_argument("--generate_dataset", action="store_true", default=False)
    parser.add_argument("--eval_lengths", action="store_true", default=False)
    parser.add_argument("--eval_ratio", type=float, default=0.01)
    return parser.parse_args()

class DatasetItem:
    def __init__(self, input_ids, attention_mask, state_seq):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.state_seq = state_seq
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def keys(self):
        return ['input_ids', 'attention_mask', 'state_seq']

class PermutationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, state_seq):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.state_seq = state_seq
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = DatasetItem(
            input_ids=self.input_ids[idx],
            attention_mask=self.attention_mask[idx],
            state_seq=self.state_seq[idx]
        )
        return item

def setup_data_collator(args, tokenizer, state_tokens, parity=None):
    """Set up the data collator based on the supervision type."""
    class MambaDataCollator:
        def __init__(self, tokenizer, max_len, state_tokens, label_key="state_seq"):
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.state_tokens = state_tokens
            self.label_key = label_key
            
        def __call__(self, features):
            # Create batch from features - only include input_ids and state_seq
            batch = {
                'input_ids': torch.stack([f['input_ids'] for f in features]),
            }
            
            # Create labels for direct supervision
            label_seq = [
                " ".join([self.state_tokens[tuple(item.tolist())] for item in f[self.label_key]])
                for f in features
            ]
            labels = self.tokenizer(
                label_seq,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )["input_ids"]
            
            # Make labels same shape as input_ids
            batch["labels"] = labels[:, :batch["input_ids"].size(1)]
            
            # For Mamba, we need to separate input_ids and labels
            # as the model only accepts input_ids
            return {
                'input_ids': batch['input_ids'],
                'labels': batch['labels']
            }

    return MambaDataCollator(tokenizer, args.max_len, state_tokens, "state_seq")

def prepare_dataset(args, debug=False):
    """Prepare the dataset for training."""
    print(f"Loading dataset from {args.data_dir}")
    
    # Load the full dataset
    dataset = ChunkedDataset(
        data_dir=args.data_dir,
        max_len=args.max_len,
        chunk_size=1,  # Process one story at a time
        debug=debug
    )
    
    # Calculate train size
    train_size = int(len(dataset) * (1 - args.eval_ratio))
    
    # Split dataset
    if args.full_determinism:
        # Use deterministic split
        indices = list(range(len(dataset)))
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        train_dataset = [dataset[i] for i in train_indices]
        eval_dataset = [dataset[i] for i in eval_indices]
    else:
        # Use random split
        indices = list(range(len(dataset)))
        if not args.data_determinism:
            random.shuffle(indices)
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        train_dataset = [dataset[i] for i in train_indices]
        eval_dataset = [dataset[i] for i in eval_indices]
    
    # Set up tokenizer
    task = PermutationTask(num_items=args.num_items)
    state_tokens = {state.permutation: state.to_string() for state in task.states}
    action_tokens = {action.permutation: action.to_string() for action in task.actions}
    tokenizer = setup_tokenizer("mamba", state_tokens, action_tokens)
    print(f"\nVocabulary size: {len(tokenizer)} tokens")
    
    # Tokenize stories and convert to tensors
    def process_batch(batch):
        # Tokenize stories
        stories = [item['story'] for item in batch]
        tokenized = tokenizer(
            stories,
            padding='max_length',
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt"
        )
        
        # Convert state sequences to tensors
        state_seqs = torch.stack([torch.tensor(item['state_seq']) for item in batch])
        
        return PermutationDataset(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            state_seq=state_seqs
        )
    
    train_dataset = process_batch(train_dataset)
    eval_dataset = process_batch(eval_dataset)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

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

def evaluate_model(model, eval_dataset, batch_size, data_collator=None):
    """Simple evaluation function that computes metrics on the eval dataset."""
    model.eval()
    total_loss = 0
    total_errors = 0
    total_tokens = 0
    
    # Get the device the model is on
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Create a simple dataloader with the data collator
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator if data_collator is not None else None
    )
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to the same device as the model
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass - Mamba only needs input_ids
            outputs = model(input_ids=batch['input_ids'])
            logits = outputs.logits
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            total_loss += loss.item()
            
            # Compute error rate
            predictions = torch.argmax(shift_logits, dim=-1)
            num_errors = (predictions != shift_labels).sum().item()
            total_errors += num_errors
            total_tokens += shift_labels.numel()
    
    # Compute average metrics
    avg_loss = total_loss / len(eval_dataloader)
    error_rate = total_errors / total_tokens if total_tokens > 0 else 1.0
    
    model.train()
    return {
        "eval_loss": avg_loss,
        "eval_error_rate": error_rate,
        "eval_num_errors": total_errors
    }

class PrintMetricsCallback(TrainerCallback):
    def __init__(self, trainer, eval_steps=100):
        self.eval_steps = eval_steps
        self.trainer = trainer
        print("PrintMetricsCallback initialized with trainer")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Print training metrics and run evaluation periodically."""
        if logs is not None and "loss" in logs:
            print(f"Step {state.global_step}: loss = {logs['loss']:.4f}")
            
            if state.global_step % self.eval_steps == 0:
                print(f"\n=== Evaluation at step {state.global_step} ===")
                try:
                    metrics = evaluate_model(
                        self.trainer.model,
                        self.trainer.eval_dataset,
                        self.trainer.args.per_device_eval_batch_size,
                        data_collator=self.trainer.data_collator
                    )
                    print("Metrics:", metrics)
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    print("Trainer state:", self.trainer is not None)
                    if self.trainer is not None:
                        print("Model state:", self.trainer.model is not None)
                        print("Eval dataset state:", self.trainer.eval_dataset is not None)

def setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator):
    """Set up the trainer with the appropriate arguments."""
    print("Setting up trainer")
    
    # Print debug information about the datasets
    print("\n=== Dataset Information ===")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    if len(eval_dataset) > 0:
        print("First eval sample:", eval_dataset[0])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,  # Log more frequently to get training loss
        save_steps=500 if not args.save_all_checkpoints else args.save_all_checkpoints,
        save_total_limit=None if args.save_all_checkpoints else 1,
        report_to="none" if args.disable_wandb else "wandb",
        dataloader_num_workers=1,
        bf16=args.use_bfloat16,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=args.full_determinism,
    )
    
    print("\n=== Training Arguments ===")
    print(f"Output directory: {args.output_dir}")
    
    print("\nCreating trainer with compute_metrics function")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Add metrics printing callback with trainer reference
    callback = PrintMetricsCallback(trainer=trainer, eval_steps=100)
    trainer.add_callback(callback)
    print("Callback added to trainer")
    
    # Override the save_model method
    original_save_model = trainer.save_model
    def custom_save_model(output_dir, _internal_call=False):
        save_model_with_shared_tensors(model, output_dir, _internal_call)
    trainer.save_model = custom_save_model
    
    return trainer

def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Custom loss computation for Mamba model
    """
    # Get model outputs - Mamba only needs input_ids
    outputs = model(input_ids=inputs['input_ids'])
    
    # Get logits from the last hidden state
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    
    # For direct supervision, shift both logits and labels by one position
    shift_logits = logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
    shift_labels = inputs['labels'][..., 1:].contiguous()  # a b c d e -> b c d e
    
    # Compute loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    
    if return_outputs:
        return (loss, outputs)
    return loss

def save_checkpoint(model, tokenizer, output_dir, checkpoint_name):
    """Save a checkpoint with the given name."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{checkpoint_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    save_model_with_shared_tensors(model, checkpoint_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save tokenizer config
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_type": "mamba",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "eos_token": "[EOS]"
    }
    with open(os.path.join(checkpoint_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"Saved checkpoint to {checkpoint_dir}")

def load_checkpoint(model, tokenizer, checkpoint_dir):
    """Load a checkpoint from the given directory."""
    print(f"Loading checkpoint from {checkpoint_dir}")
    
    # Print some model parameters before loading
    print("\nModel parameters before loading checkpoint:")
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
            break  # Just print first weight parameter as example
    
    # Load model state dict
    state_dict = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    
    # Print same parameters after loading
    print("\nModel parameters after loading checkpoint:")
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
            break  # Just print first weight parameter as example
    
    # Check if tokenizer files exist
    tokenizer_config_path = os.path.join(checkpoint_dir, "tokenizer_config.json")
    tokenizer_file_path = os.path.join(checkpoint_dir, "tokenizer.json")
    
    if os.path.exists(tokenizer_config_path) and os.path.exists(tokenizer_file_path):
        # Load tokenizer config
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        
        # Load tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            checkpoint_dir,
            tokenizer_file=tokenizer_file_path,
            **tokenizer_config
        )
    else:
        print("Tokenizer files not found in checkpoint. Using current tokenizer.")
        # Keep using the current tokenizer since it's already set up correctly
        pass
    
    return model, tokenizer

def compute_metrics(eval_pred, compute_result=True):
    """Compute metrics for evaluation."""
    print("\n=== compute_metrics called ===")
    print("eval_pred type:", type(eval_pred))
    print("compute_result:", compute_result)
    
    try:
        logits, labels = eval_pred
        print(f"[compute_metrics] Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        print(f"[compute_metrics] After shift - Logits shape: {shift_logits.shape}, Labels shape: {shift_labels.shape}")
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        print(f"[compute_metrics] Computed loss: {float(loss):.4f}")
        
        # Compute error rate (percentage of incorrect predictions)
        predictions = torch.argmax(shift_logits, dim=-1)
        num_errors = (predictions != shift_labels).sum().item()
        total_tokens = shift_labels.numel()
        error_rate = num_errors / total_tokens if total_tokens > 0 else 1.0
        print(f"[compute_metrics] Num errors: {num_errors}, Num total: {total_tokens}, Error rate: {error_rate:.4f}")
        
        return {
            "eval_loss": float(loss),
            "eval_error_rate": float(error_rate),
            "eval_num_errors": int(num_errors)
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"eval_loss": 0.0, "eval_error_rate": 1.0, "eval_num_errors": 0}

def main():
    """Main function."""
    args = parse_arguments()
    print('args.max_len:', args.max_len)
    root_output = args.output_dir
    args.output_dir = args.output_dir + f"/mamba_{args.max_len}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up dataset directory
    dataset_dir = os.path.join("datasets_new", f"permutation_{args.num_items}_{args.max_len}")
    
    # Generate dataset if needed
    if not os.path.exists(dataset_dir) or args.generate_dataset:
        print(f"Dataset for max_len={args.max_len} not found. Generating dataset...")
        task = PermutationTask(num_items=args.num_items)
        num_steps = max(args.max_len, 16)
        _stories, _states = task.simulate(
            steps=num_steps,
            num_stories=args.num_stories,
            story_so_far="",
            states_so_far=[task.init_state],
            write_dir=dataset_dir,
            train_ratio=0.99
        )
    else:
        print(f"Loading dataset from {dataset_dir}")
    
    args.data_dir = dataset_dir
    
    # Handle checkpoint loading
    if args.from_checkpoint is not None:
        # from_checkpoint is the max_len of the previous checkpoint
        checkpoint_root = root_output + f"/mamba_{args.from_checkpoint}"
        print('checkpoint_root:', checkpoint_root)
        if os.path.exists(checkpoint_root):
            ckpt_subdirs = [d for d in os.listdir(checkpoint_root) if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()]
            print('ckpt_subdirs:', ckpt_subdirs)
            if ckpt_subdirs:
                latest_ckpt = max(ckpt_subdirs, key=lambda d: int(d.split("-")[-1]))
                args.from_checkpoint = os.path.join(checkpoint_root, latest_ckpt)
            else:
                print('set to None because no checkpoint found')
                args.from_checkpoint = None
        else:
            print('set to None because no checkpoint found 2')
            args.from_checkpoint = None
    print('args.from_checkpoint:', args.from_checkpoint)
    
    # Set up determinism
    if args.full_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set up task and tokens
    task = PermutationTask(num_items=args.num_items)
    state_tokens = {state.permutation: state.to_string() for state in task.states}
    action_tokens = {action.permutation: action.to_string() for action in task.actions}
    parity = None
    
    if args.is_parity_cur:
        parity = {state: compute_parity(state) for state in state_tokens}
    
    # Set up tokenizer
    tokenizer = setup_tokenizer("mamba", state_tokens, action_tokens)
    
    # Load or initialize Mamba model
    model_name = "state-spaces/mamba-370m"  # Using 370M parameter model
    print(f"\nLoading model from {model_name}...")
    
    # Create config with correct vocabulary size
    config = MambaConfig(
        vocab_size=len(tokenizer),
        d_model=2560,  # From mamba-130m config
        n_layer=24,    # From mamba-130m config
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8
    )
    print('vocab_size:', config.vocab_size)
    
    # Load model with new config
    model = MambaLMHeadModel(
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32,
    )
    
    # Load from checkpoint if specified
    if args.from_checkpoint:
        model, tokenizer = load_checkpoint(model, tokenizer, args.from_checkpoint)
    
    # Set up data collator
    data_collator = setup_data_collator(args, tokenizer, state_tokens, parity)
    
    # Handle length generalization evaluation
    if args.eval_lengths:
        args.eval_lengths = str([int(8*i) + 10 for i in range(1,20)])[1:-1]
        if not args.from_checkpoint:
            raise ValueError("--eval_lengths needs a trained model supplied with --from_checkpoint")

        lengths = [int(x) for x in args.eval_lengths.split(",")]
        results = {}
        error_rates = {}

        for L in lengths:
            print(f"\n### Evaluating sequence length {L} ###")
            original_max_len = args.max_len
            args.max_len = L

            args.data_dir = os.path.join("datasets_new", f"permutation_{args.num_items}_{L}")
            if not os.path.exists(args.data_dir):
                print("  ↳ dataset missing – generating once")
                tmp_task = PermutationTask(num_items=args.num_items)
                tmp_task.simulate(
                    steps=max(L,16),
                    num_stories=args.num_stories,
                    story_so_far="",
                    states_so_far=[tmp_task.init_state],
                    write_dir=args.data_dir,
                    train_ratio=0.5
                )

            train_dataset, eval_dataset = prepare_dataset(args, debug=args.debug)
            trainer = setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator)
            
            # Use evaluate_model which already works with Mamba
            metrics = evaluate_model(model, eval_dataset, args.batch_size, data_collator)
            print("\n=== Evaluation metrics ===")
            print("metrics:", metrics)
            
            if "eval_loss" in metrics:
                results[L] = metrics["eval_loss"]
            if "eval_error_rate" in metrics:
                error_rates[L] = metrics["eval_error_rate"]
            
            # Restore original max_len
            args.max_len = original_max_len

        # ---- PLOT: eval‑loss vs. sequence length ----
        xs = sorted(results.keys())
        ys = [results[x] for x in xs]

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Sequence length")
        plt.ylabel("Evaluation loss")
        plt.ylim(0, 20)
        plt.title("Length‑generalisation performance")
        plt.grid(True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create plot_data directory
        plot_data_dir = os.path.join(args.output_dir, "plot_data")
        os.makedirs(plot_data_dir, exist_ok=True)

        # Create plots directory
        plots_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Save raw data
        data = {
            "model": "mamba",
            "lengths": xs,
            "losses": ys,
            "error_rates": [error_rates[x] for x in xs] if error_rates else None,
            "timestamp": timestamp
        }
        data_path = os.path.join(plot_data_dir, f"length_generalization_data_{timestamp}.json")
        with open(data_path, "w") as f:
            json.dump(data, f)
        print(f"\nData saved → {data_path}")

        plot_path = os.path.join(plots_dir, f"length_generalisation_loss_{timestamp}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"\nLoss plot saved → {plot_path}")

        # ---- PLOT: error rate vs. sequence length ----
        if error_rates:
            xs = sorted(error_rates.keys())
            ys = [error_rates[x] for x in xs]

            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Sequence length")
            plt.ylabel("Error rate")
            plt.ylim(0, 1)
            plt.title(f"Mamba Length‑generalisation error rate")
            plt.grid(True)

            plot_path = os.path.join(plots_dir, f"length_generalisation_error_{timestamp}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            print(f"\nError rate plot saved → {plot_path}")

        print("\n======  Summary  ======")
        for L in sorted(results.keys()):
            print(f"L={L:>3}  |  eval_loss = {results[L]:.4f}", end="")
            if L in error_rates:
                print(f"  |  error_rate = {error_rates[L]:.4f}")
            else:
                print()
        return

    # Regular training path
    train_dataset, eval_dataset = prepare_dataset(args, debug=args.debug)
    trainer = setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator)

    # Add the custom compute_loss method to the trainer
    trainer.compute_loss = compute_loss.__get__(trainer)

    # Initialize wandb if not disabled
    if not args.disable_wandb:
        print("\n=== Initializing Weights & Biases ===")
        try:
            # First check if we're logged in
            if wandb.api.api_key is None:
                print("WARNING: No wandb API key found. Please run 'wandb login' first.")
                print("Continuing without wandb logging...")
            else:
                wandb_run = wandb.init(
                    project="state-tracking",
                    entity="morrisyau95",  # Updated to match your actual username
                    config={
                        "model": model_name,
                        "num_items": args.num_items,
                        "supervision_type": args.supervision_type,
                        "max_len": args.max_len,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "learning_rate": trainer.args.learning_rate,
                        "early_stopping": args.early_stopping,
                    }
                )
                print(f"Wandb run initialized successfully with ID: {wandb_run.id}")
                print(f"View run at: {wandb_run.get_url()}")
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            print("Continuing without wandb logging...")

    # Train the model
    trainer.train()
    
    # Evaluate after training
    print("\n=== Final Evaluation ===")
    metrics = evaluate_model(model, eval_dataset, args.batch_size, data_collator)
    print("Final metrics:", metrics)
    
    # Save final checkpoint
    save_checkpoint(model, tokenizer, args.output_dir, "final")
    
    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Close wandb run if it was initialized
    if not args.disable_wandb and 'wandb_run' in locals():
        print("\n=== Finalizing Weights & Biases ===")
        wandb_run.finish()
        print("Wandb run completed and synced")

if __name__ == "__main__":
    main() 