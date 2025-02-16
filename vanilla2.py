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
    TrainerCallback,
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
            input_ids_list = []     # For text tokens, store the token id; for s tokens, use -1.
            s_mask_list = []        # True if the position is an s token.
            labels_list = []        # For text tokens, store the token id; for s tokens, use -100.
            for j, token in enumerate(a_tokens):
                # Append text token.
                input_ids_list.append(token)
                s_mask_list.append(False)
                labels_list.append(token)
                # Insert an s token after every k text tokens (except at the very end).
                if (j + 1) % self.k == 0 and (j + 1) < len(a_tokens):
                    input_ids_list.append(-1)  # Marker for s token.
                    s_mask_list.append(True)
                    labels_list.append(-100)   # Do not predict s tokens.
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
# 2. Custom GPT2-Based Model that Uses a Fixed s Token Vector
# -----------------------------------------------------------------------------

class STokenGPTConfig(GPT2Config):
    def __init__(self,
                 vocab_size=1,    # will be set to the actual vocab size
                 n_positions=1024,
                 n_embd=256,
                 n_layer=4,
                 n_head=4,
                 input_dim=200,    # dimension for token embeddings
                 dropout=0.1,     # dropout probability for regularization
                 s_token_learnable=False,  # whether the s token vector is learnable
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
        self.input_dim = input_dim
        self.s_token_learnable = s_token_learnable

class STokenGPTModel(PreTrainedModel):
    config_class = STokenGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.hidden_dim = config.n_embd
        self.vocab_size = config.vocab_size

        # Learned embedding for text tokens.
        self.token_embedding = nn.Embedding(self.vocab_size, self.input_dim)
        # Define the s token vector. It can be fixed or learnable.
        self.s_token = nn.Parameter(torch.ones(self.input_dim) / math.sqrt(self.input_dim),
                                    requires_grad=config.s_token_learnable)
        # Project from input_dim (either text embedding or s token) to hidden_dim.
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        # Positional embeddings.
        self.position_embedding = nn.Embedding(config.n_positions, self.hidden_dim)
        # Transformer blocks.
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        # Project back to input_dim and then decode to vocabulary logits.
        self.output_projection = nn.Linear(self.hidden_dim, self.input_dim)
        self.decoder = nn.Linear(self.input_dim, self.vocab_size)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, s_mask: torch.BoolTensor, labels: torch.LongTensor = None):
        batch_size, seq_len = input_ids.shape
        # For text tokens (input_ids >= 0) we use the learned embedding.
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        # For s tokens (marked by s_mask=True), use the fixed s token vector.
        # Expand self.s_token to (batch_size, seq_len, input_dim).
        s_token_vector = self.s_token.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.input_dim)
        # Choose embeddings based on s_mask.
        embeddings = torch.where(s_mask.unsqueeze(-1), s_token_vector, text_embeddings)
        # Project to the transformer hidden dimension.
        hidden_states = self.input_proj(embeddings)
        # Add positional embeddings.
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        # Pass through transformer blocks.
        for block in self.h:
            hidden_states = block(hidden_states, use_cache=False)[0]
        hidden_states = self.ln_f(hidden_states)
        # Project back to input_dim and decode to vocabulary logits.
        x = self.output_projection(hidden_states)
        logits = self.decoder(x)
        return CausalLMOutput(logits=logits)

# -----------------------------------------------------------------------------
# 3. Custom Trainer (Using standard compute_loss logic)
# -----------------------------------------------------------------------------

class STokenTrainer(Trainer):
    def __init__(self, *args, train_loader=None, valid_loader=None, **kwargs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Do not pass train_loader/valid_loader to the parent
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_loader is not None:
            return self.train_loader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self.valid_loader is not None:
            return self.valid_loader
        return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, L, vocab_size)

        # Shift logits and labels for next-token prediction.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="mean"
        )
        return (loss, outputs) if return_outputs else loss

# -----------------------------------------------------------------------------
# 4. Training Script with Modified Callbacks
# -----------------------------------------------------------------------------

def collate_fn(batch):
    """
    Collate function to stack samples.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    s_mask = torch.stack([item["s_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_ids": input_ids,
        "s_mask": s_mask,
        "labels": labels,
    }

# Callback that prints the current and best training/validation losses and perplexities.
class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.best_training_loss = float('inf')
        self.best_eval_loss = float('inf')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Update best training loss if available.
        if "loss" in logs:
            current_loss = logs["loss"]
            if current_loss < self.best_training_loss:
                self.best_training_loss = current_loss

        # Update best eval loss if available.
        if "eval_loss" in logs:
            current_eval_loss = logs["eval_loss"]
            if current_eval_loss < self.best_eval_loss:
                self.best_eval_loss = current_eval_loss

        # Compute current perplexities.
        if "loss" in logs:
            training_loss = logs["loss"]
            training_perplexity = math.exp(training_loss) if training_loss < 100 else float('inf')
        else:
            training_loss = float('nan')
            training_perplexity = float('nan')

        if "eval_loss" in logs:
            eval_loss = logs["eval_loss"]
            eval_perplexity = math.exp(eval_loss) if eval_loss < 100 else float('inf')
        else:
            eval_loss = float('nan')
            eval_perplexity = float('nan')

        print(
            f"Step {state.global_step}: "
            f"Training Loss: {training_loss:.4f} (Best: {self.best_training_loss:.4f}, Perp: {training_perplexity:.4f}) | "
            f"Validation Loss: {eval_loss:.4f} (Best: {self.best_eval_loss:.4f}, Perp: {eval_perplexity:.4f})"
        )

def main():
    parser = argparse.ArgumentParser(
        description='Train a transformer with interleaved fixed s tokens (like a padding token).'
    )
    # Dataset/model parameters.
    parser.add_argument('--seq_len', type=int, default=128, help='Number of text tokens per sample (before inserting s tokens).')
    parser.add_argument('--k', type=int, default=3, help='Insert an s token every k text tokens.')
    parser.add_argument('--input_dim', type=int, default=200, help='Dimension of token embeddings.')
    # Model hyperparameters.
    parser.add_argument('--model_emb', type=int, default=256, help='Transformer hidden dimension.')
    parser.add_argument('--model_layers', type=int, default=4, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability.')
    parser.add_argument('--s_token_learnable', action='store_true', help='If set, the s token vector is learnable.')
    # Training hyperparameters.
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./s_token_trainer/{args.model_emb}_{args.model_layers}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load GPT2 tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    # Create datasets.
    train_dataset = TextWithSTokenDataset(split="train", tokenizer=tokenizer,
                                          seq_len=args.seq_len, k=args.k)
    valid_dataset = TextWithSTokenDataset(split="validation", tokenizer=tokenizer,
                                          seq_len=args.seq_len, k=args.k)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    # Prepare model configuration.
    config = STokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=args.model_emb,
        n_layer=args.model_layers,
        n_head=args.n_head,
        input_dim=args.input_dim,
        dropout=args.dropout,
        s_token_learnable=args.s_token_learnable,
        layer_norm_epsilon=1e-5
    )
    model = STokenGPTModel(config).to(device)

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
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=False,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        save_total_limit=3,
        dataloader_num_workers=2,
        seed=args.seed,
        max_grad_norm=args.gradient_clip,
        lr_scheduler_type="cosine"
    )

    trainer = STokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        train_loader=train_loader,
        valid_loader=valid_loader
    )
    trainer.add_callback(PrintLossCallback())

    trainer.train()

if __name__ == "__main__":
    main()
