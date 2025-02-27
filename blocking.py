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
from types import SimpleNamespace

# -----------------------------------------------------------------------------
# 1. Custom Dataset: Interleaved s tokens
# -----------------------------------------------------------------------------
class TextWithSTokenDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len, text_run, state_run):
        """
        For each contiguous chunk of tokenized text (length seq_len), we create a sample with blocks that
        always begin and end with state_run s tokens.  Specifically, we:
        
         - Prepend state_run s tokens at the very start.
         - Insert text tokens.
         - After every text_run text tokens, insert state_run s tokens.
         - At the end, if there is an incomplete block, append state_run s tokens.
        
        For text tokens the label is the token id, and for s tokens (marked with token id -1) the label is -100.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_run = text_run
        self.state_run = state_run

        # Load the dataset (using WikiText-103 for a larger corpus)
        #self.data = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        self.data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Pre-tokenize the dataset.
        def tokenize_fn(examples):
            return tokenizer(examples["text"], add_special_tokens=False)
        
        self.data = self.data.map(
            tokenize_fn, 
            batched=True,
            num_proc=4,
            remove_columns=["text"]
        )

        # Flatten token ids.
        token_ids_list = self.data["input_ids"]
        self.token_ids = [token for sublist in token_ids_list for token in sublist]
        self.vocab_size = tokenizer.vocab_size

        # Create samples.
        self.samples = []
        # We step through the flattened tokens in chunks of length seq_len.
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            a_tokens = self.token_ids[i:i + seq_len]
            input_ids_list = []
            s_mask_list = []
            labels_list = []
            
            # Prepend state_run s tokens at the beginning of the sample.
            for _ in range(self.state_run):
                input_ids_list.append(-1)
                s_mask_list.append(True)
                labels_list.append(-100)
            
            text_count = 0
            for j, token in enumerate(a_tokens):
                # Add a text token.
                input_ids_list.append(token)
                s_mask_list.append(False)
                labels_list.append(token)
                text_count += 1

                # After text_run text tokens, insert state_run s tokens.
                if text_count == self.text_run:
                    for _ in range(self.state_run):
                        input_ids_list.append(-1)
                        s_mask_list.append(True)
                        labels_list.append(-100)
                    text_count = 0

            # If there's an incomplete block at the end, add trailing state_run s tokens.
            if text_count > 0:
                for _ in range(self.state_run):
                    input_ids_list.append(-1)
                    s_mask_list.append(True)
                    labels_list.append(-100)
            
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
#
# This is essentially a GPT-2 style model that has a standard token embedding for text tokens
# and a separate s token parameter (which may or may not be learnable).
#
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
        # s token: fixed continuous vector (or learnable if specified).
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
                override_s: torch.Tensor = None, attention_mask: torch.Tensor = None, mse_loss=False):
        """
        If override_s is provided, it is used at positions where s_mask is True.
        An attention_mask (if provided) is used in each transformer block.
        """
        batch_size, seq_len = input_ids.shape
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        if override_s is not None:
            s_token_vector = override_s
        else:
            s_token_vector = self.s_token.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.hidden_dim)
        embeddings = torch.where(s_mask.unsqueeze(-1), s_token_vector, text_embeddings)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=embeddings.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_encoded_embeddings = embeddings + self.position_embedding(position_ids)
        # Copy for target of MSE loss.
        embed_copy = pos_encoded_embeddings.clone()
        hidden_states = pos_encoded_embeddings
        for block in self.h:
            hidden_states = block(hidden_states, use_cache=False, attention_mask=attention_mask)[0]
        hidden_states = self.ln_f(hidden_states)
        logits = self.decoder(hidden_states)
        output = CausalLMOutput(logits=logits)
        if return_hidden:
            output.hidden_states = hidden_states
        if labels is not None and not return_hidden:
            # Shift for next-token prediction.
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            ce_loss = F.cross_entropy(shift_logits.reshape(-1, self.vocab_size),
                                      shift_labels.reshape(-1),
                                      ignore_index=-100)
            if mse_loss: 
                # Compute MSE loss on positions where label is -100 (i.e. s tokens).
                mask = (shift_labels == -100).unsqueeze(-1)
                if mask.any():
                    mask_expanded = mask.expand(-1, -1, self.hidden_dim)
                    mse_loss_val = F.mse_loss(embed_copy[:, 1:][mask_expanded], hidden_states[:, :-1][mask_expanded])
                else:
                    mse_loss_val = torch.tensor(0.0, device=hidden_states.device)
                regular = 0.5
                output.ce_loss = ce_loss
                output.mse_loss = mse_loss_val
                output.loss = (1 - regular) * ce_loss + regular * mse_loss_val 
            else: 
                output.ce_loss = ce_loss
                output.mse_loss = torch.tensor(0.0, device=hidden_states.device)
                output.loss = ce_loss
        return output

# -----------------------------------------------------------------------------
# 3. Helper: Compute Custom Attention Mask
# -----------------------------------------------------------------------------
def compute_custom_attention_mask(s_mask, mode='sliding', window=None):
    """
    Computes an attention mask of shape (B, 1, L, L) for a given s_mask.
    In sliding mode, each token attends only to tokens within a specified window.
    """
    B, L = s_mask.size()
    device = s_mask.device

    if mode == 'sliding':
        if window is None:
            window = L
        i_indices = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
        j_indices = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
        diff = i_indices - j_indices
        allowed = (diff >= 0) & (diff < window)
        attn_mask = torch.where(allowed, torch.tensor(0.0, device=device),
                                torch.tensor(-1e9, device=device))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)
        return attn_mask
    elif mode == 'stagger':
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        s_mask_idx = torch.where(s_mask, indices, torch.tensor(-1, device=device))
        _, last_s = torch.cummax(s_mask_idx, dim=1)
        i_indices = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
        j_indices = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
        last_s_expanded = last_s.unsqueeze(2)
        causal = (j_indices < i_indices).to(device).unsqueeze(0).expand(B, L, L)
        allowed = (j_indices.unsqueeze(0).expand(B, L, L) >= last_s_expanded)
        final_allowed = causal & allowed
        attn_mask = torch.where(final_allowed, torch.tensor(0.0, device=device),
                                torch.tensor(-1e9, device=device))
        attn_mask = attn_mask.unsqueeze(1)
        return attn_mask 
    elif mode == 'ladder':
        if window is None:
            raise ValueError("Window parameter must be specified for ladder mode")
        i_indices = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
        j_indices = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
        block_start = (torch.arange(L, device=device) // window) * window
        allowed = (j_indices >= block_start.unsqueeze(1)) & (j_indices <= i_indices)
        attn_mask = torch.where(allowed, torch.tensor(0.0, device=device),
                                torch.tensor(-1e9, device=device))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)
        return attn_mask
    else:
        raise ValueError("Invalid mode for compute_custom_attention_mask. "
                         "Choose 'default' or 'stagger'.")

def compute_block_mask(L, state_run, device):
    """
    Returns a block attention mask of shape (L, L) for a block of length L,
    where positions before index (state_run-1) are fully masked (set to -1e9)
    and for positions i >= state_run-1, only tokens j with j in [state_run-1, i] are unmasked.
    """
    # Create row and column indices.
    i = torch.arange(L, device=device).unsqueeze(1)  # shape (L, 1)
    j = torch.arange(L, device=device).unsqueeze(0)  # shape (1, L)
    # For rows with i < state_run-1, we want to mask all positions.
    # For i >= state_run-1, allowed positions are j in [state_run-1, i].
    allowed = (i >= (state_run - 1)) & (j >= (state_run - 1)) & (j <= i)
    mask = torch.where(allowed, 0.0, -1e9)
    return mask

# -----------------------------------------------------------------------------
# 4. Helper: Create Blocks for Module2
# -----------------------------------------------------------------------------
def create_blocks_vectorized(hidden, input_ids, s_mask, labels, text_run, state_run):
    """
    Vectorized version of block creation.
    
    Args:
        hidden: Tensor of shape (B, L, hidden_dim) from module1.
        input_ids: Tensor of shape (B, L) with original token ids.
        s_mask: Tensor of shape (B, L) indicating s token positions.
        labels: Tensor of shape (B, L) with labels.
        text_run: int, number of text tokens per block (in between s token groups).
        state_run: int, number of s tokens at the beginning and end of each block.
    
    Returns:
        A tuple of tensors:
          blocks_input_ids: (B, n_blocks, L_block)
          blocks_s_mask: (B, n_blocks, L_block)
          blocks_labels: (B, n_blocks, L_block)
          blocks_hidden: (B, n_blocks, L_block, hidden_dim)
    """
    L_block = text_run + 2 * state_run  # Each block has state_run tokens at start and end.
    step = text_run + state_run         # Blocks slide with overlap of state_run tokens.
    blocks_hidden = hidden.unfold(dimension=1, size=L_block, step=step)
    blocks_input_ids = input_ids.unfold(dimension=1, size=L_block, step=step)
    blocks_s_mask = s_mask.unfold(dimension=1, size=L_block, step=step)
    blocks_labels = labels.unfold(dimension=1, size=L_block, step=step)
    return blocks_input_ids, blocks_s_mask, blocks_labels, blocks_hidden

class TwoStageModel(nn.Module):
    def __init__(self, module1: NoExtraLayerSTokenGPTModel, module2: NoExtraLayerSTokenGPTModel,
                 window=None, text_run=None, state_run=None):
        """
        module1 processes input p and outputs p' (with continuous s tokens).
        module2 takes blocks from p'' (original text tokens interleaved with module1’s s outputs).
        """
        super().__init__()
        self.module1 = module1
        self.module2 = module2
        self.window = window
        self.text_run = text_run
        self.state_run = state_run

    def forward(self, input_ids: torch.LongTensor, s_mask: torch.BoolTensor, labels: torch.LongTensor = None):
        # Module1 pass: get hidden states.
        out1 = self.module1(input_ids=input_ids, s_mask=s_mask, labels=labels,
                            return_hidden=True, mse_loss=False)
        # Use module1's hidden states as override for s token positions.
        # --- Vectorized block extraction via unfold ---
        # (Assuming a fixed block length and step as follows:)
        L_block = self.text_run + 2 * self.state_run  # block length
        step = self.text_run + self.state_run           # sliding step
        blocks_hidden = out1.hidden_states.unfold(dimension=1, size=L_block, step=step)
        blocks_input_ids = input_ids.unfold(dimension=1, size=L_block, step=step)
        blocks_s_mask = s_mask.unfold(dimension=1, size=L_block, step=step)
        blocks_labels = labels.unfold(dimension=1, size=L_block, step=step)
        # blocks_* have shape (B, n_blocks, L_block) (for blocks_hidden, extra dim hidden_dim)
        B, n_blocks, L_block = blocks_input_ids.shape
        flat_input_ids = blocks_input_ids.reshape(B * n_blocks, L_block)
        flat_s_mask = blocks_s_mask.reshape(B * n_blocks, L_block)
        flat_labels = blocks_labels.reshape(B * n_blocks, L_block)
        flat_override_s = blocks_hidden.reshape(B * n_blocks, L_block, -1)
        
        # --- New Block Mask ---
        # Build a block_mask of shape (L_block, L_block) that is the same for all blocks.
        block_mask = compute_block_mask(L_block, self.state_run, device=input_ids.device)
        # Expand to shape (B*n_blocks, 1, L_block, L_block) as expected by module2.
        block_mask = block_mask.unsqueeze(0).unsqueeze(1).expand(B * n_blocks, 1, L_block, L_block)
        
        # Process all blocks at once.
        out2 = self.module2(
            input_ids=flat_input_ids.to(input_ids.device),
            s_mask=flat_s_mask.to(input_ids.device),
            override_s=flat_override_s.to(input_ids.device),
            labels=flat_labels.to(input_ids.device),
            attention_mask=block_mask,
            mse_loss=True
        )
        # Here, out2.loss is computed over all blocks at once.
        return SimpleNamespace(loss=out2.loss, ce_loss=out2.ce_loss, mse_loss=out2.mse_loss, logits=out2.logits)


# -----------------------------------------------------------------------------
# 6. Collate Function
# -----------------------------------------------------------------------------
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    s_mask = torch.stack([item["s_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "s_mask": s_mask, "labels": labels}

# -----------------------------------------------------------------------------
# 7. PrintLossCallback for Logging
# -----------------------------------------------------------------------------
class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.best_training_loss = float('inf')
        self.best_eval_loss = float('inf')
        self.last_eval_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 10 != 0:
            return
        if logs is None:
            return
        current_loss = logs.get("loss", None)
        if current_loss is not None:
            if isinstance(current_loss, torch.Tensor):
                current_loss = current_loss.item()
            if current_loss < self.best_training_loss:
                self.best_training_loss = current_loss
        current_eval_loss = logs.get("eval_loss", None)
        if current_eval_loss is not None:
            if isinstance(current_eval_loss, torch.Tensor):
                current_eval_loss = current_eval_loss.item()
            self.last_eval_loss = current_eval_loss
            if current_eval_loss < self.best_eval_loss:
                self.best_eval_loss = current_eval_loss
        ce_loss = logs.get("eval_ce_loss", None)
        if isinstance(ce_loss, torch.Tensor):
            ce_loss = ce_loss.item()
        mse_loss = logs.get("eval_mse_loss", None)
        if isinstance(mse_loss, torch.Tensor):
            mse_loss = mse_loss.item()
        out_str = f"Step {state.global_step}: "
        if current_loss is not None:
            out_str += f"Loss: {current_loss:.4f} "
        if current_eval_loss is not None:
            out_str += f"| Eval Loss: {current_eval_loss:.4f} "
        if ce_loss is not None:
            out_str += f"| Eval CE Loss: {ce_loss:.4f}, Eval PPL Loss: {math.exp(ce_loss):.4f} "
        if mse_loss is not None:
            out_str += f"| Eval MSE Loss: {mse_loss:.4f}"
        print(out_str)

# -----------------------------------------------------------------------------
# 8. Custom Trainer Subclass (Override compute_loss)
# -----------------------------------------------------------------------------
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        # Retrieve ce_loss and mse_loss if present; otherwise default to a zero tensor.
        ce_loss = getattr(outputs, "ce_loss", torch.tensor(0.0, device=loss.device))
        mse_loss = getattr(outputs, "mse_loss", torch.tensor(0.0, device=loss.device))
        extra = {"ce_loss": ce_loss, "mse_loss": mse_loss}
        return (loss, {"loss": loss, **extra})
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss, outputs = self.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        # Build the log dictionary.
        logs = {"loss": loss.item()}
        ce_loss = outputs.get("ce_loss", torch.tensor(0.0))
        mse_loss = outputs.get("mse_loss", torch.tensor(0.0))
        logs["ce_loss"] = ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss
        logs["mse_loss"] = mse_loss.item() if torch.is_tensor(mse_loss) else mse_loss
        self.log(logs)
        return loss.detach()
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            ce_loss = getattr(outputs, "ce_loss", torch.tensor(0.0, device=loss.device))
            mse_loss = getattr(outputs, "mse_loss", torch.tensor(0.0, device=loss.device))
            extra = {
                "ce_loss": ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss,
                "mse_loss": mse_loss.item() if torch.is_tensor(mse_loss) else mse_loss
            }
            if prediction_loss_only:
                return (loss, None, inputs.get("labels"), extra)
            logits = outputs.logits
        return (loss, logits, inputs.get("labels"), extra)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        losses, ce_losses, mse_losses = [], [], []
        for inputs in eval_dataloader:
            loss, logits, labels, extra = self.prediction_step(
                self.model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
            )
            losses.append(loss.item())
            ce_losses.append(extra.get("ce_loss", 0))
            mse_losses.append(extra.get("mse_loss", 0))
        mean_ce = np.mean(ce_losses) if ce_losses else 0.0
        mean_mse = np.mean(mse_losses) if mse_losses else 0.0
        metrics = {
            "eval_loss": np.mean(losses),
            "eval_ce_loss": mean_ce,
            "eval_mse_loss": mean_mse,
            "eval_ppl": np.exp(mean_ce) if mean_ce > 0 else float('inf')
        }
        print(f"Evaluation metrics: {metrics}")
        return metrics


# -----------------------------------------------------------------------------
# 9. Main Training Script with Timestamp Logic
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train a two-stage transformer model with custom blocking logic.'
    )
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Number of text tokens per sample (before inserting s tokens).')
    parser.add_argument('--mode', type=str, default='two_stage',
                        help='Either one_stage or two_stage training mode.')
    parser.add_argument('--text_run', type=int, default=9,
                        help='Insert an s token every text_run text tokens.')
    parser.add_argument('--state_run', type=int, default=1,
                        help='Insert state_run s tokens every text_run text tokens.')
    parser.add_argument('--window', type=int, default=8,
                        help='Length of sliding causal attention window for module2.')
    parser.add_argument('--hidden', type=int, default=768, help='Hidden dimension of both modules.')
    parser.add_argument('--layers', type=int, default=12, help='Number of layers of both modules.')
    parser.add_argument('--heads', type=int, default=12, help='Number of heads of both modules.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()
    
    # For one-stage mode (i.e. module2 only), we disable s tokens.
    if args.mode == 'one_stage': 
        args.text_run = args.seq_len + 1
        args.state_run = 0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./two_stage_model", f"seq{args.seq_len}_k{args.text_run}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    # Create datasets.
    train_dataset = TextWithSTokenDataset(split="train", tokenizer=tokenizer, seq_len=args.seq_len,
                                           text_run=args.text_run, state_run=args.state_run)
    valid_dataset = TextWithSTokenDataset(split="validation", tokenizer=tokenizer, seq_len=args.seq_len,
                                           text_run=args.text_run, state_run=args.state_run)

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
    composite_model = TwoStageModel(module1, module2, window=args.window,
                                    text_run=args.text_run, state_run=args.state_run)
    composite_model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
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
