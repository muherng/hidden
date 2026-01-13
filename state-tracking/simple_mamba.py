import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import os
import json

def create_tokenizer():
    """Create a minimal tokenizer for the Mamba model."""
    # Create a minimal vocabulary dictionary
    minimal_vocab = {
        "[PAD]": 0,
        "[EOS]": 1,
        "[UNK]": 2,
    }
    
    # Build a fast tokenizer
    tk_model = models.WordLevel(vocab=minimal_vocab, unk_token="[UNK]")
    tk = Tokenizer(tk_model)
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.decoder = decoders.WordPiece()
    
    # Create the Hugging Face Fast tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]"
    )
    return tokenizer

def prepare_dataset(tokenizer, max_length=128):
    """Load and prepare a small text dataset."""
    # Load a small dataset (using tiny_shakespeare as an example)
    dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
    
    def tokenize_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=False  # Don't return special tokens mask
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

class MambaDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        # Only keep input_ids
        batch = {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features])
        }
        return batch

def save_model(model, tokenizer, output_dir):
    """Save model and tokenizer using PyTorch's native format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save config as JSON
    config_dict = {
        "d_model": model.config.d_model,
        "n_layer": model.config.n_layer,
        "vocab_size": model.config.vocab_size,
        "ssm_cfg": model.config.ssm_cfg,
        "rms_norm": model.config.rms_norm,
        "residual_in_fp32": model.config.residual_in_fp32,
        "fused_add_norm": model.config.fused_add_norm,
        "pad_vocab_size_multiple": model.config.pad_vocab_size_multiple,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Load model
    model_name = "state-spaces/mamba-130m"
    print(f"\nLoading model from {model_name}...")
    model = MambaLMHeadModel.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float16,
    )
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = prepare_dataset(tokenizer)
    
    # Create custom data collator
    data_collator = MambaDataCollator(tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./mamba_trained",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",  # Disable wandb
        save_safetensors=False,  # Disable safetensors
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the model using our custom save function
    print("\nSaving model...")
    save_model(model, tokenizer, "./mamba_trained_final")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 