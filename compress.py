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
    def __init__(self, split, tokenizer, seq_len, text_run, state_run):
        """
        For each contiguous chunk of tokenized text (length seq_len), after every k text tokens,
        insert an s token marker (token id -1). Labels are set to the original token for text positions
        and -100 for s token positions.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_run = text_run
        self.state_run = state_run

        # Load the dataset (using WikiText-103 for a larger corpus)
        self.data = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        # Pre-tokenize the dataset using .map() with batched processing and multiple processes.
        def tokenize_fn(examples):
            # Tokenize a batch of texts; note that 'text' is a key in each example.
            return tokenizer(examples["text"], add_special_tokens=False)
        
        self.data = self.data.map(
            tokenize_fn, 
            batched=True,         # Process batches of examples for efficiency.
            num_proc=4,           # Adjust number of processes based on your CPU.
            remove_columns=["text"]  # Remove the raw text after tokenization.
        )

        # Flatten the list of token ids from all examples.
        token_ids_list = self.data["input_ids"]
        self.token_ids = [token for sublist in token_ids_list for token in sublist]
        self.vocab_size = tokenizer.vocab_size

        # Create samples by splitting token_ids into chunks of seq_len tokens,
        # interleaving s tokens as defined by text_run and state_run.
        self.samples = []
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            a_tokens = self.token_ids[i:i + seq_len]
            input_ids_list = []
            s_mask_list = []
            labels_list = []
            for j, token in enumerate(a_tokens):
                input_ids_list.append(token)
                s_mask_list.append(False)
                labels_list.append(token)
                if (j + 1) % self.text_run == 0 and (j + 1) < len(a_tokens):
                    for _ in range(state_run):
                        input_ids_list.append(-1)  # Marker for s token.
                        s_mask_list.append(True)
                        labels_list.append(-100)   # Ignore index for loss.
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

def compute_custom_attention_mask(s_mask, mode='sliding', window=None):
    """
    Computes an attention mask of shape (B, 1, L, L).

    Args:
        s_mask (torch.BoolTensor): Boolean tensor of shape (B, L). For mode='default'
            this is used to determine the most recent s token per position. In sliding mode,
            s_mask is ignored.
        mode (str): Either 'default' (the original behavior) or 'sliding'.
        window (int, optional): In sliding mode, each token i can attend only to tokens 
            in the range where 0 <= (i - j) < window (i.e. tokens from i-window+1 to i, inclusive).
            If None in sliding mode, defaults to full causal attention (i.e. window=L).

    Returns:
        torch.Tensor: An attention mask tensor of shape (B, 1, L, L) with 0 for allowed
            positions and -1e9 for disallowed positions.
    """
    B, L = s_mask.size()
    device = s_mask.device

    if mode == 'sliding':
        # If no window is specified, default to full causal attention.
        if window is None:
            window = L
        # Create matrices of position indices.
        i_indices = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
        j_indices = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
        # Compute difference (i - j). Self attention is allowed when diff == 0.
        diff = i_indices - j_indices  # shape: (L, L)
        # Allowed if 0 <= diff < window.
        allowed = (diff >= 0) & (diff < window)
        attn_mask = torch.where(allowed, torch.tensor(0.0, device=device),
                                torch.tensor(-1e9, device=device))
        # Expand to (B, 1, L, L)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)
        return attn_mask

    elif mode == 'stagger':
        # Original implementation based on s_mask.
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        # Mark positions where s_mask is True; use -1 for others.
        s_mask_idx = torch.where(s_mask, indices, torch.tensor(-1, device=device))
        # Compute the most recent s token index at each position using cummax.
        _, last_s = torch.cummax(s_mask_idx, dim=1)
        # Create matrices of indices.
        i_indices = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
        j_indices = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
        last_s_expanded = last_s.unsqueeze(2)  # Shape (B, L, 1)
        causal = (j_indices < i_indices).to(device).unsqueeze(0).expand(B, L, L)
        allowed = (j_indices.unsqueeze(0).expand(B, L, L) >= last_s_expanded)
        final_allowed = causal & allowed
        attn_mask = torch.where(final_allowed, torch.tensor(0.0, device=device),
                                torch.tensor(-1e9, device=device))
        attn_mask = attn_mask.unsqueeze(1)
        return attn_mask

    else:
        raise ValueError("Invalid mode for compute_custom_attention_mask. "
                         "Choose 'default' or 'stagger'.")


# -----------------------------------------------------------------------------
# 4. Composite Two-Stage Model
# -----------------------------------------------------------------------------

class TwoStageModel(nn.Module):
    def __init__(self, module1: NoExtraLayerSTokenGPTModel, module2: NoExtraLayerSTokenGPTModel, window=None):
        """
        module1 processes input p and outputs p' (with continuous s' tokens).
        module2 takes p'' as input where s token positions are replaced with s' tokens from module1.
        """
        super().__init__()
        self.module1 = module1
        self.module2 = module2
        self.window = window

    def forward(self, input_ids: torch.LongTensor, s_mask: torch.BoolTensor, labels: torch.LongTensor = None):
        # Module1 pass: get hidden states.
        out1 = self.module1(input_ids=input_ids, s_mask=s_mask, labels=labels, return_hidden=True)
        # Compute custom attention mask from s_mask.
        custom_attn_mask = compute_custom_attention_mask(s_mask, mode='sliding', window=self.window)
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
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Number of text tokens per sample (before inserting s tokens).')
    parser.add_argument('--mode', type=str, default='one_stage',
                        help='either two module training or just single module(2) options [one_stage,two_stage]')
    parser.add_argument('--text_run', type=int, default=9,
                        help='Insert an s token every text_run text tokens.')
    parser.add_argument('--state_run', type=int, default=1,
                        help='Insert state_run s tokens every text_run text tokens.')
    parser.add_argument('--window', type=int, default=8,
                        help='length of sliding causal attention window for module2.')
    parser.add_argument('--hidden', type=int, default=768, help='hidden dimension of both modules.')
    parser.add_argument('--layers', type=int, default=12, help='number of layers of both modules.')
    parser.add_argument('--heads', type=int, default=12, help='number of heads of both modules.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()
    
    #if mode is one stage, insert 0 state tokens which is equivalent to just running module2
    if args.mode == 'one_stage': 
        args.text_run = args.seq_len+1
        args.state_run = 0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Create a unique output directory using a timestamp and input parameter info.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./two_stage_model", f"seq{args.seq_len}_k{args.text_run}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    # Create datasets.
    train_dataset = TextWithSTokenDataset(split="train", tokenizer=tokenizer, seq_len=args.seq_len, text_run=args.text_run, state_run=args.state_run)
    valid_dataset = TextWithSTokenDataset(split="validation", tokenizer=tokenizer, seq_len=args.seq_len, text_run=args.text_run, state_run=args.state_run)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=8)

    # Instantiate module1.
    config1 = NoExtraLayerSTokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=args.hidden,       
        n_layer=args.layers,        
        n_head=args.heads,
        dropout=0.1,      
        s_token_learnable=False,
    )
    module1 = NoExtraLayerSTokenGPTModel(config1)
    # Instantiate module2.
    config2 = NoExtraLayerSTokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=args.hidden,       
        n_layer=args.layers,        
        n_head=args.heads,
        dropout=0.1,      
        s_token_learnable=False,
    )
    module2 = NoExtraLayerSTokenGPTModel(config2)
    # Build composite two-stage model.
    composite_model = TwoStageModel(module1, module2, window=args.window)
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
        learning_rate=5e-4,          
        warmup_steps=1000,           
        weight_decay=0.01,             
        fp16=False,
        seed=args.seed,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        max_grad_norm=1.0
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
