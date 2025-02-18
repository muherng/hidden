import math
import os
import datetime
import argparse

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
    TrainerCallback
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutput

import datasets

# -----------------------------------------------------------------------------
# 1. Dataset: WikiTextDatasetWithS
# -----------------------------------------------------------------------------

class WikiTextDatasetWithS(Dataset):
    """
    Loads WikiText-2 raw text, tokenizes it, and splits it into non-overlapping
    chunks of fixed length (seq_len). For language modeling, labels are identical to
    input_ids. Returns a dummy s_mask (all False) since for windowed attention we do not use s tokens.
    """
    def __init__(self, split, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = " ".join(data["text"])
        #print(f"DEBUG: Loaded text length: {len(text)}")
        tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        #print(f"DEBUG: Number of tokens: {len(self.token_ids)}")
        self.vocab_size = tokenizer.vocab_size

        self.samples = []
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            sample_tokens = self.token_ids[i:i+seq_len]
            if len(sample_tokens) > 0:
                self.samples.append(sample_tokens)
        #print(f"DEBUG: Number of samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_tokens = self.samples[idx]
        sample = torch.tensor(sample_tokens, dtype=torch.long)
        s_mask = torch.zeros_like(sample, dtype=torch.bool)
        ret = {"input_ids": sample, "labels": sample, "s_mask": s_mask}
        #print(f"DEBUG: __getitem__ idx {idx} returns keys: {list(ret.keys())}")
        return ret

# -----------------------------------------------------------------------------
# 2. Model: No-Extra-Layer S Token GPT Model (for both module1 and module2)
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

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.s_token = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=config.s_token_learnable)
        self.position_embedding = nn.Embedding(config.n_positions, self.hidden_dim)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        self.decoder = nn.Linear(self.hidden_dim, self.vocab_size)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, s_mask: torch.BoolTensor,
                labels: torch.LongTensor = None, return_hidden: bool = False,
                override_s: torch.Tensor = None, attention_mask: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        if override_s is not None:
            s_token_vector = override_s
        else:
            s_token_vector = self.s_token.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.hidden_dim)
        embeddings = torch.where(s_mask.unsqueeze(-1), s_token_vector, text_embeddings)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=embeddings.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = embeddings + self.position_embedding(position_ids)
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
# 3. Helper: Compute Window Attention Mask
# -----------------------------------------------------------------------------

def compute_window_attention_mask(seq_len, window, device):
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    allowed = (j < i) & (j >= (i - window))
    mask = torch.where(allowed, torch.tensor(0.0, device=device), torch.tensor(-1e9, device=device))
    return mask

# -----------------------------------------------------------------------------
# 4. Helper: Compute Original Custom Attention Mask (from s_mask)
# -----------------------------------------------------------------------------

def compute_custom_attention_mask(s_mask):
    B, L = s_mask.size()
    device = s_mask.device
    indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    s_mask_idx = torch.where(s_mask, indices, torch.tensor(-1, device=device))
    _, last_s = torch.cummax(s_mask_idx, dim=1)
    i_indices = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
    j_indices = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
    last_s_expanded = last_s.unsqueeze(2)
    causal = (j_indices < i_indices).to(device)
    causal = causal.unsqueeze(0).expand(B, L, L)
    allowed = (j_indices.unsqueeze(0).expand(B, L, L) >= last_s_expanded)
    final_allowed = causal & allowed
    attn_mask = torch.where(final_allowed, torch.tensor(0.0, device=device), torch.tensor(-1e9, device=device))
    attn_mask = attn_mask.unsqueeze(1)
    return attn_mask

# -----------------------------------------------------------------------------
# 5. Composite Two-Stage Model
# -----------------------------------------------------------------------------

class TwoStageModel(nn.Module):
    def __init__(self, module1: NoExtraLayerSTokenGPTModel, module2: NoExtraLayerSTokenGPTModel):
        super().__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, input_ids: torch.LongTensor, s_mask: torch.BoolTensor, labels: torch.LongTensor = None, window: int = 0):
        out1 = self.module1(input_ids=input_ids, s_mask=s_mask, labels=labels, return_hidden=True)
        device = input_ids.device
        if window > 0:
            custom_attn_mask = compute_window_attention_mask(input_ids.size(1), window, device)
            custom_attn_mask = custom_attn_mask.unsqueeze(0)
        else:
            custom_attn_mask = compute_custom_attention_mask(s_mask)
        out2 = self.module2(input_ids=input_ids, s_mask=s_mask, override_s=out1.hidden_states, labels=labels,
                            attention_mask=custom_attn_mask)
        return out2

# -----------------------------------------------------------------------------
# 6. Collate Function
# -----------------------------------------------------------------------------

def collate_fn(batch):
    #for i, item in enumerate(batch):
    #    print(f"DEBUG: Batch item {i} keys: {list(item.keys())}")
    input_ids = torch.stack([item["input_ids"] for item in batch])
    s_mask = torch.stack([item["s_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "s_mask": s_mask, "labels": labels}

# -----------------------------------------------------------------------------
# 7. PrintLossCallback with on_evaluate
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
            print(f"Step {state.global_step}: Training Loss: {current_loss:.4f} (Best: {self.best_training_loss:.4f}, Perp: {training_perplexity:.4f})")
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print("DEBUG: Evaluation metrics:", metrics)
        if "eval_loss" in metrics:
            current_eval_loss = metrics["eval_loss"]
            if isinstance(current_eval_loss, torch.Tensor):
                current_eval_loss = current_eval_loss.item()
            self.last_eval_loss = current_eval_loss
            if current_eval_loss < self.best_eval_loss:
                self.best_eval_loss = current_eval_loss
            eval_perplexity = math.exp(current_eval_loss) if current_eval_loss < 100 else float('inf')
            print(f"Step {state.global_step}: Eval Loss: {current_eval_loss:.4f} (Best: {self.best_eval_loss:.4f}, Perp: {eval_perplexity:.4f})")
        else:
            print(f"Step {state.global_step}: Eval Loss not reported.")

# -----------------------------------------------------------------------------
# 8. Custom Trainer Subclass
# -----------------------------------------------------------------------------

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=inputs["input_ids"].device, dtype=torch.float)
        return (loss, outputs) if return_outputs else loss

# -----------------------------------------------------------------------------
# 9. Compute Metrics Function
# -----------------------------------------------------------------------------

def compute_metrics(eval_pred):
    # eval_pred is assumed to be a dict with key "loss"
    loss = eval_pred["loss"]
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return {"eval_loss": loss}

# -----------------------------------------------------------------------------
# 10. Main Training Script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a two-stage transformer model with a custom window attention mask for module2.'
    )
    parser.add_argument('--seq_len', type=int, default=128, help='Number of text tokens per sample.')
    parser.add_argument('--k', type=int, default=3, help='Insert an s token every k text tokens.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--window', type=int, default=0, help='If >0, use a sliding window of this size for attention in module2.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./two_stage_model", f"seq{args.seq_len}_k{args.k}_win{args.window}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    train_dataset = WikiTextDatasetWithS(split="train", tokenizer=tokenizer, seq_len=args.seq_len)
    valid_dataset = WikiTextDatasetWithS(split="validation", tokenizer=tokenizer, seq_len=args.seq_len)

    # Use num_workers=0 for easier debugging.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    config1 = NoExtraLayerSTokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.3,
        s_token_learnable=False,
    )
    module1 = NoExtraLayerSTokenGPTModel(config1)
    config2 = NoExtraLayerSTokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.3,
        s_token_learnable=False,
    )
    module2 = NoExtraLayerSTokenGPTModel(config2)
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
        learning_rate=args.learning_rate,
        warmup_steps=1000,
        weight_decay=0.1,
        fp16=False,
        seed=args.seed,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        remove_unused_columns=False
    )

    # Wrap composite_model to always pass the window argument.
    class ModelWrapper(nn.Module):
        def __init__(self, model, window):
            super().__init__()
            self.model = model
            self.window = window
        def forward(self, **inputs):
            return self.model(input_ids=inputs["input_ids"],
                              s_mask=inputs["s_mask"],
                              labels=inputs["labels"],
                              window=self.window)
    wrapped_model = ModelWrapper(composite_model, args.window)
    wrapped_model.to(device)

    trainer = CustomTrainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(PrintLossCallback())

    # Explicitly evaluate and print metrics before training
    eval_metrics = trainer.evaluate()
    print("Explicit Evaluation Metrics:", eval_metrics)

    trainer.train()

if __name__ == "__main__":
    main()
