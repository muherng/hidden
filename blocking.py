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
        For each contiguous chunk of tokenized text (length seq_len), we create a sample that,
        after interleaving state tokens, contains an exact integer number of complete blocks.
        
        We do this by:
          - Taking a chunk of seq_len tokens.
          - Trimming it so that only an integer number of text-run groups is used.
          - Prepending state_run state tokens.
          - After every text_run text tokens, inserting state_run state tokens.
        
        For text tokens, the label is the token id; for state tokens (token id -1) the label is -100.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_run = text_run
        self.state_run = state_run

        # Load dataset (using WikiText-2 for a smaller corpus here).
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
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            a_tokens = self.token_ids[i:i + seq_len]
            # Keep only an integer number of complete text groups.
            n_groups = len(a_tokens) // self.text_run
            a_tokens = a_tokens[: n_groups * self.text_run]

            input_ids_list = []
            s_mask_list = []
            labels_list = []

            # Prepend state_run s tokens.
            for _ in range(self.state_run):
                input_ids_list.append(-1)
                s_mask_list.append(True)
                labels_list.append(-100)

            text_count = 0
            for token in a_tokens:
                # Add a text token.
                input_ids_list.append(token)
                s_mask_list.append(False)
                labels_list.append(token)
                text_count += 1
                # After every text_run text tokens, insert state_run s tokens.
                if text_count == self.text_run:
                    for _ in range(self.state_run):
                        input_ids_list.append(-1)
                        s_mask_list.append(True)
                        labels_list.append(-100)
                    text_count = 0

            # (Now the sample length is exactly:
            #   state_run + n_groups*(text_run + state_run)
            # which is an exact multiple of the block length = text_run + 2*state_run.)

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
                 dropout=0.1, s_token_learnable=False, state_run=None, text_run=None, **kwargs):
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
        self.state_run = state_run
        self.text_run = text_run 

class NoExtraLayerSTokenGPTModel(PreTrainedModel):
    config_class = NoExtraLayerSTokenGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.state_run = config.state_run
        self.text_run = config.text_run 

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
                override_s: torch.Tensor = None, attention_mask: torch.Tensor = None, mse_loss: bool = False):
        """
        Forward pass with two different loss computations.
        
        If mse_loss is True and self.state_run > 0 then we assume block mode:
        - Let L_block = text_run + 2*state_run.
        - Compute Cross Entropy (CE) loss over the region:
            indices [state_run-1, ..., state_run+text_run-2] predict targets at indices [state_run, ..., state_run+text_run-1].
        - Compute MSE loss over the region:
            indices [state_run+text_run-1, ..., L_block-2] predict targets at indices [state_run+text_run, ..., L_block-1].
        - Combine losses with mixing ratio.
        
        Otherwise (for example when self.state_run==0), we do standard shift-by-one next-token prediction.
        """
        batch_size, seq_len = input_ids.shape
        # Get embeddings for text tokens.
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        if override_s is not None:
            s_token_vector = override_s
        else:
            s_token_vector = self.s_token.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.hidden_dim)
        # Replace positions where s_mask is True with s_token vector.
        embeddings = torch.where(s_mask.unsqueeze(-1), s_token_vector, text_embeddings)
        # Add positional embeddings.
        position_ids = torch.arange(seq_len, dtype=torch.long, device=embeddings.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_encoded_embeddings = embeddings + self.position_embedding(position_ids)
        # Copy for MSE target.
        embed_copy = pos_encoded_embeddings.clone()
        hidden_states = pos_encoded_embeddings
        # Pass through transformer blocks.
        for block in self.h:
            hidden_states = block(hidden_states, use_cache=False, attention_mask=attention_mask)[0]
        hidden_states = self.ln_f(hidden_states)
        logits = self.decoder(hidden_states)
        output = CausalLMOutput(logits=logits)
        if return_hidden:
            output.hidden_states = hidden_states
        if labels is not None and not return_hidden:
            # If state_run==0 or we are not in block mode, use standard next-token prediction.
            if self.state_run == 0 or not mse_loss:
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                ce_loss = F.cross_entropy(shift_logits.reshape(-1, self.vocab_size),
                                        shift_labels.reshape(-1),
                                        ignore_index=-100)
                output.ce_loss = ce_loss
                output.mse_loss = torch.tensor(0.0, device=hidden_states.device)
                output.loss = ce_loss
            else:
                # Custom block loss.
                # Assume block length L_block = text_run + 2*state_run.
                L_block = logits.size(1)
                sr = self.state_run      # number of state tokens per block group.
                tr = self.text_run       # number of text tokens.
                # Boundary: the CE region spans indices [sr-1, sr+tr-2] (predictions)
                # and targets are at indices [sr, sr+tr-1].
                B = sr + tr
                ce_logits = logits[:, sr - 1 : B - 1, :]  # shape: (batch, tr, vocab)
                ce_targets = labels[:, sr : B]             # shape: (batch, tr)
                ce_loss = F.cross_entropy(
                    ce_logits.reshape(-1, self.vocab_size),
                    ce_targets.reshape(-1),
                    ignore_index=-100
                )
                # MSE region: predictions from indices [B-1, L_block-2] to predict targets at [B, L_block-1].
                mse_pred = hidden_states[:, B - 1 : L_block - 1, :]
                mse_target = embed_copy[:, B : L_block, :]
                mse_loss_val = F.mse_loss(mse_pred, mse_target)
                regular = 0.0
                output.ce_loss = ce_loss
                output.mse_loss = mse_loss_val
                #output.loss = (1 - regular) * ce_loss + regular * mse_loss_val
                output.loss = ce_loss
        return output


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
        #print('n_blocks:', n_blocks)
        flat_input_ids = blocks_input_ids.reshape(B * n_blocks, L_block)
        flat_s_mask = blocks_s_mask.reshape(B * n_blocks, L_block)
        flat_labels = blocks_labels.reshape(B * n_blocks, L_block)
        flat_override_s = blocks_hidden.reshape(B * n_blocks, L_block, -1)
        
        # --- New Block Mask ---
        # Build a block_mask of shape (L_block, L_block) that is the same for all blocks.
        block_mask = compute_block_mask(L_block, self.state_run, device=input_ids.device)
        # Expand to shape (B*n_blocks, 1, L_block, L_block) as expected by module2.
        block_mask = block_mask.unsqueeze(0).unsqueeze(1).expand(B * n_blocks, 1, L_block, L_block)
        #print('flat_override_s:', flat_override_s.shape)
        print('flat_override_s:', flat_override_s[:,0,0])
        
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

class StateTokenCheckCallback(TrainerCallback):
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance
        self.initial_state_token = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Save the initial module1.s_token
        model = kwargs.get("model", None)
        if model is None:
            return
        # Assuming the composite model has module1 as an attribute
        self.initial_state_token = model.module1.s_token.clone().detach()
        print("Saved initial module1.s_token.")

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None or self.initial_state_token is None:
            return
        current_token = model.module1.s_token.detach()
        max_diff = (current_token - self.initial_state_token).abs().max().item()
        if max_diff > self.tolerance:
            print(f"Warning: module1.s_token has changed (max diff = {max_diff:.2e}) at step {state.global_step}.")


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
        n_positions=2048,
        n_embd=args.hidden,       
        n_layer=args.layers,        
        n_head=args.heads,
        dropout=0.1,      
        s_token_learnable=False,
        state_run = args.state_run,
        text_run = args.text_run
    )
    module1 = NoExtraLayerSTokenGPTModel(config1)
    # Instantiate module2.
    config2 = NoExtraLayerSTokenGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=2048,
        n_embd=args.hidden,       
        n_layer=args.layers,        
        n_head=args.heads,
        dropout=0.1,      
        s_token_learnable=False,
        state_run = args.state_run,
        text_run = args.text_run
    )
    module2 = NoExtraLayerSTokenGPTModel(config2)
    # Build composite two-stage model.
    composite_model = TwoStageModel(module1, module2, window=args.window,
                                    text_run=args.text_run, state_run=args.state_run)
    composite_model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=10000,
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
    
    #debugging
    state_token_check = StateTokenCheckCallback(tolerance=1e-6)
    trainer.add_callback(state_token_check)
    
    trainer.train()

if __name__ == "__main__":
    main()
