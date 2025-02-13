import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedModel, GPT2Config, GPT2Tokenizer, TrainingArguments, Trainer, TrainerCallback
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutput
from transformers.optimization import AdamW, get_scheduler
import datasets
import argparse
import datetime
import os
import json
import warnings

warnings.filterwarnings("ignore", message="`tokenizer` is deprecated", category=FutureWarning)

# Generate a unique timestamp and determine device
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ==============================================================================
# Helper: Custom Attention Mask
# ==============================================================================
def compute_custom_attention_mask(noise_mask):
    """
    Vectorized custom attention mask.
    
    For each query position t in a sequence:
      - If no noise token has been seen (i.e. last_noise[t] == -1), allow keys 0 <= k < t.
      - Otherwise, allow keys k such that last_noise[t] <= k < t.
    
    Allowed positions get 0; disallowed positions get -1e9.
    
    Args:
      noise_mask: Tensor of shape (B, L) with booleans indicating noise positions.
      
    Returns:
      A tensor of shape (B, L, L) with attention mask scores.
    """
    B, L = noise_mask.shape
    device = noise_mask.device
    # Create index tensor: shape (B, L)
    indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    # For positions that are not noise, set to -1.
    noise_indices = torch.where(noise_mask, indices, torch.full((B, L), -1, device=device))
    # For each position t, compute the maximum index so far (i.e. the most recent noise token index)
    last_noise = torch.cummax(noise_indices, dim=1)[0]  # shape (B, L); values are -1 if no noise seen.
    # For each query position t, define lower bound as:
    #    lower_bound[t] = last_noise[t] if last_noise[t] != -1 else 0.
    lower_bound = torch.maximum(last_noise, torch.zeros_like(last_noise))
    
    # Build matrices of query indices and key indices.
    query_idx = torch.arange(L, device=device).unsqueeze(0).unsqueeze(-1)  # shape (1, L, 1)
    key_idx = torch.arange(L, device=device).unsqueeze(0).unsqueeze(1)       # shape (1, 1, L)
    # Expand lower_bound to shape (B, L, 1)
    lower_bound_exp = lower_bound.unsqueeze(-1)  # (B, L, 1)
    # Allowed positions: key index >= lower_bound and key index < query index.
    allowed = (key_idx >= lower_bound_exp) & (key_idx < query_idx)
    # Build mask: allowed positions 0, disallowed positions -1e9.
    mask = torch.where(allowed, torch.tensor(0.0, device=device), torch.tensor(-1e9, device=device))
    return mask  # shape (B, L, L)




# ==============================================================================
# 1. Custom Dataset with Interleaved Noise Tokens and Learned Text Token IDs
# ==============================================================================
#
# Each sample is constructed from a contiguous chunk of tokenized text.
# For every k text tokens, we insert a noise token.
# The input is of the form:
#    p = {a₀, a₁, …, aₖ, s₀, aₖ₊₁, …, a_T}
# where aᵢ are discrete tokens and sⱼ are random noise vectors.
# The labels are the discrete token ids for aᵢ positions and -100 for noise.
# (Next-token prediction is done by shifting logits and labels.)

class TextWithNoiseDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len, k, input_dim):
        """
        Args:
          split: one of "train", "validation", or "test"
          tokenizer: a Hugging Face tokenizer (e.g. GPT2Tokenizer)
          seq_len: number of text tokens per sample (before inserting noise)
          k: after every k text tokens, insert one noise token.
          input_dim: dimension of each token’s continuous representation.
                     (Text tokens will be embedded via a learned embedding.)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.k = k
        self.input_dim = input_dim

        # Load dataset (using WikiText-2 raw text here)
        self.data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = " ".join(self.data["text"])
        self.tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.vocab_size = tokenizer.vocab_size

        self.samples = []
        # Slide over the token stream in chunks of length seq_len.
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            a_tokens = self.token_ids[i:i+seq_len]
            input_ids_list = []     # For text tokens: token id; for noise: -1.
            noise_mask_list = []    # True for noise positions.
            noise_vectors_list = [] # For noise positions: fixed random noise vector; else zeros.
            labels_list = []        # For text tokens: token id; for noise: -100.
            for j, token in enumerate(a_tokens):
                # Add text token.
                input_ids_list.append(token)
                noise_mask_list.append(False)
                noise_vectors_list.append(torch.zeros(input_dim))
                labels_list.append(token)
                # After every k text tokens (except at the end), insert a noise token.
                if (j + 1) % self.k == 0 and (j + 1) < len(a_tokens):
                    input_ids_list.append(-1)  # marker for noise
                    noise_mask_list.append(True)
                    noise_vectors_list.append(torch.randn(input_dim))
                    labels_list.append(-100)   # ignore in loss
            sample = {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "noise_mask": torch.tensor(noise_mask_list, dtype=torch.bool),
                "noise_vectors": torch.stack(noise_vectors_list),
                "labels": torch.tensor(labels_list, dtype=torch.long)
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==============================================================================
# 2. Modified GPT2-Based Model with Learned Token Embedding and Hidden Return Option
# ==============================================================================
#
# The model embeds discrete tokens via a learned embedding.
# For positions flagged as noise, it uses the provided noise vector.
# After transformer blocks, the hidden states are projected to an intermediate space
# (of dimension input_dim) before decoding.
#
# An optional flag 'custom_attention_mask' (if provided) is used in each block,
# and if return_hidden is True, the intermediate representation is returned.
 
class VectorGPTConfig(GPT2Config):
    def __init__(self,
                 vocab_size=1,    # will be set to ntokens
                 n_positions=64,
                 n_embd=256,
                 n_layer=4,
                 n_head=4,
                 input_dim=10,    # dimension of token representation
                 ntokens=1,       # vocabulary size
                 **kwargs):
        super().__init__(vocab_size=vocab_size,
                         n_positions=n_positions,
                         n_embd=n_embd,
                         n_layer=n_layer,
                         n_head=n_head,
                         **kwargs)
        self.input_dim = input_dim
        self.ntokens = ntokens


class VectorGPTModel(PreTrainedModel):
    config_class = VectorGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.hidden_dim = config.n_embd
        self.ntokens = config.ntokens

        self.token_embedding = nn.Embedding(self.ntokens, self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.position_embedding = nn.Embedding(config.n_positions, self.hidden_dim)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        self.output_projection = nn.Linear(self.hidden_dim, self.input_dim)
        self.decoder = nn.Linear(self.input_dim, self.ntokens)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, noise_mask: torch.BoolTensor,
                noise_vectors: torch.FloatTensor, labels: torch.LongTensor = None,
                return_hidden: bool = False, custom_attention_mask: torch.Tensor = None):
        """
        Args:
           input_ids: (B, L) LongTensor; discrete tokens (≥0) for text and -1 for noise.
           noise_mask: (B, L) BoolTensor; True for noise positions.
           noise_vectors: (B, L, input_dim) FloatTensor; for noise positions, the provided vector.
           labels: (B, L) LongTensor; token id for text tokens, -100 for noise.
           return_hidden: if True, return the intermediate representation (B, L, input_dim).
           custom_attention_mask: Optional (B, L, L) tensor; if provided, unsqueezed to (B, 1, L, L)
                                  and passed to each transformer block.
        """
        batch_size, seq_len = input_ids.shape
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        embeddings = torch.where(noise_mask.unsqueeze(-1), noise_vectors, text_embeddings)
        hidden_states = self.input_proj(embeddings)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        if custom_attention_mask is not None:
            extended_mask = custom_attention_mask.unsqueeze(1)  # shape (B, 1, L, L)
        else:
            extended_mask = None
        for block in self.h:
            outputs = block(hidden_states, attention_mask=extended_mask, use_cache=False)
            hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        x = self.output_projection(hidden_states)
        logits = self.decoder(x)
        if return_hidden:
            return CausalLMOutput(logits=logits, hidden_states=x)
        else:
            return CausalLMOutput(logits=logits)


# ==============================================================================
# 3. Custom Trainer Class with Two-Stage Forward Pass and Custom Attention Mask
# ==============================================================================
#
# In compute_loss:
#  - First forward pass (p → p'): run with return_hidden=True to obtain s' vectors.
#  - Compute a custom attention mask (using the updated logic) from noise_mask.
#  - Second forward pass (p'' → output): use s' in place of the original noise vectors,
#    and pass in the custom attention mask.
#  - Compute next-token prediction loss (using shifted logits/labels) only on the second pass.
#
# Note: Only text token positions (where label != -100) contribute to the loss.

class VectorGPTTrainer(Trainer):
    def __init__(self, *args, train_loader=None, valid_loader=None, custom_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_args = custom_args
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def get_train_dataloader(self):
        return self.train_loader

    def get_eval_dataloader(self, eval_dataset=None):
        if self.valid_loader is not None:
            return self.valid_loader
        else:
            return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        noise_mask = inputs["noise_mask"]
        noise_vectors = inputs["noise_vectors"]
        labels = inputs["labels"]

        # First pass: run to get s' vectors.
        outputs1 = model(input_ids=input_ids,
                         noise_mask=noise_mask,
                         noise_vectors=noise_vectors,
                         labels=labels,
                         return_hidden=True)
        s_prime = outputs1.hidden_states  # shape: (B, L, input_dim)

        # Compute the custom attention mask.
        custom_attn_mask = compute_custom_attention_mask(noise_mask)  # shape: (B, L, L)

        # Second pass: use s' in place of original noise vectors.
        outputs2 = model(input_ids=input_ids,
                         noise_mask=noise_mask,
                         noise_vectors=s_prime,
                         labels=labels,
                         custom_attention_mask=custom_attn_mask)
        logits2 = outputs2.logits  # (B, L, ntokens)

        # Shift logits and labels for next-token prediction.
        shift_logits = logits2[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss_sum = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum"
        )
        valid_tokens = (shift_labels.reshape(-1) != -100).sum().float()
        mean_loss = loss_sum / valid_tokens if valid_tokens > 0 else loss_sum
        return (mean_loss, outputs2) if return_outputs else mean_loss


# ==============================================================================
# 4. Training Script: Putting Everything Together
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a VectorGPT model with two-stage forward pass and custom attention mask.'
    )
    parser.add_argument('--seq_len', type=int, default=128, help='Number of text tokens per sample (before inserting noise).')
    parser.add_argument('--k', type=int, default=3, help='Insert a noise token every k text tokens.')
    parser.add_argument('--input_dim', type=int, default=200, help='Dimension of each token representation (and noise vector).')
    parser.add_argument('--model_emb', type=int, default=256, help='Transformer hidden dimension.')
    parser.add_argument('--model_layers', type=int, default=4, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    args = parser.parse_args()
    print('args:', args)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e7)

    train_dataset = TextWithNoiseDataset(split="train", tokenizer=tokenizer,
                                          seq_len=args.seq_len, k=args.k, input_dim=args.input_dim)
    valid_dataset = TextWithNoiseDataset(split="validation", tokenizer=tokenizer,
                                          seq_len=args.seq_len, k=args.k, input_dim=args.input_dim)

    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        noise_mask = torch.stack([item["noise_mask"] for item in batch])
        noise_vectors = torch.stack([item["noise_vectors"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "noise_mask": noise_mask, "noise_vectors": noise_vectors, "labels": labels}

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    config = VectorGPTConfig(
        n_positions=1000,
        n_embd=args.model_emb,
        n_layer=args.model_layers,
        n_head=args.n_head,
        input_dim=args.input_dim,
        ntokens=tokenizer.vocab_size
    )
    model = VectorGPTModel(config)

    output_dir = f"./vector_gpt_trainer/{args.model_emb}_{args.model_layers}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=300,
        fp16=True,
        save_total_limit=3,
        seed=42,
        report_to="tensorboard"
    )

    custom_args = {"ntoken": tokenizer.vocab_size}

    trainer = VectorGPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=None,
        custom_args=custom_args,
        train_loader=train_loader,
        valid_loader=valid_loader
    )

    class SaveLossCallback(TrainerCallback):
        def __init__(self, output_file="losses.json"):
            self.output_file = output_file
            self.losses = {"training_loss": [], "validation_loss": []}
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.losses["training_loss"].append(logs["loss"])
                if "eval_loss" in logs:
                    self.losses["validation_loss"].append(logs["eval_loss"])
                with open(self.output_file, "w") as f:
                    json.dump(self.losses, f, indent=4)

    trainer.add_callback(SaveLossCallback(f"./results/losses_{timestamp}.json"))

    class PrintLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if "loss" in logs:
                loss = logs["loss"]
                perplexity = math.exp(loss) if loss < 100 else float('inf')
                print(f"Step {state.global_step}: Training Loss: {loss:.4f} | Perplexity: {perplexity:.4f}")
            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                eval_perplexity = math.exp(eval_loss) if eval_loss < 100 else float('inf')
                print(f"Step {state.global_step}: Validation Loss: {eval_loss:.4f} | Perplexity: {eval_perplexity:.4f}")

    trainer.add_callback(PrintLossCallback())

    trainer.train()
