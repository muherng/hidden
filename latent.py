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
# 1. Custom Dataset with Interleaved Noise Tokens and Learned Text Token IDs
# ==============================================================================
#
# For each contiguous sequence of text tokens (length seq_len), we interleave a
# noise token every k text tokens (except at the end). For text token positions, we
# store the token id (to be later embedded by a learned embedding). For noise token
# positions we store a special marker (here, -1), a boolean flag (True), and a fixed
# random noise vector (sampled once during dataset creation). Note that the labels
# here are simply the discrete token ids for text tokens and -100 for noise tokens.
#
# We will perform next-token prediction (i.e. the target for position t is the token
# at position t+1), so later in the loss computation we shift the labels.

class TextWithNoiseDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len, k, input_dim):
        """
        Args:
          split: one of "train", "validation", or "test"
          tokenizer: a Hugging Face tokenizer (e.g. GPT2Tokenizer)
          seq_len: number of text tokens per sample (before inserting noise)
          k: after every k text tokens, insert one noise token.
          input_dim: the dimension of each token’s continuous representation.
                     (Text tokens will be embedded into this space via a learned embedding.)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.k = k
        self.input_dim = input_dim

        # Load the dataset (using WikiText-2 raw text here)
        self.data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        # Concatenate all texts and tokenize
        text = " ".join(self.data["text"])
        # Override the tokenizer's max length to avoid warnings (we manually chunk later)
        self.tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.vocab_size = tokenizer.vocab_size

        # Build samples from the token stream
        self.samples = []
        # Slide over the token stream in chunks of length seq_len.
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            a_tokens = self.token_ids[i:i+seq_len]
            input_ids_list = []     # For text positions, store the token id; for noise positions, use -1.
            noise_mask_list = []    # True if the position is a noise token.
            noise_vectors_list = [] # For noise positions, store a fixed random noise vector; else, a dummy (zeros).
            labels_list = []        # For text tokens, store the token id; for noise, use -100.
            for j, token in enumerate(a_tokens):
                # Text token position
                input_ids_list.append(token)
                noise_mask_list.append(False)
                noise_vectors_list.append(torch.zeros(input_dim))  # dummy vector (not used)
                labels_list.append(token)
                # Insert a noise token after every k text tokens (except at the end).
                if (j + 1) % self.k == 0 and (j + 1) < len(a_tokens):
                    input_ids_list.append(-1)            # marker for noise (won't be used in embedding lookup)
                    noise_mask_list.append(True)
                    noise_vectors_list.append(torch.randn(input_dim))  # fixed random noise vector
                    labels_list.append(-100)             # noise token: do not predict
            sample = {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),       # shape: (L,)
                "noise_mask": torch.tensor(noise_mask_list, dtype=torch.bool),       # shape: (L,)
                "noise_vectors": torch.stack(noise_vectors_list),                    # shape: (L, input_dim)
                "labels": torch.tensor(labels_list, dtype=torch.long)                # shape: (L,)
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==============================================================================
# 2. Modified GPT2-Based Model with Learned Token Embedding
# ==============================================================================
#
# This model uses a learned embedding (nn.Embedding) to convert text token IDs
# into continuous vectors. For noise tokens (flagged by noise_mask), we override the
# learned embedding with the fixed noise vector provided by the dataset.
# The combined sequence is then projected to the transformer's hidden dimension,
# processed with GPT2 blocks, and finally decoded into logits over tokens.

class VectorGPTConfig(GPT2Config):
    def __init__(self,
                 vocab_size=1,    # will be set to ntokens
                 n_positions=64,
                 n_embd=256,
                 n_layer=4,
                 n_head=4,
                 input_dim=10,    # dimension of token representation (learned for text tokens)
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

        # Learned token embedding for text tokens.
        self.token_embedding = nn.Embedding(self.ntokens, self.input_dim)
        # A projection to map the input_dim (from embedding or noise) to hidden_dim.
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        # Positional embeddings.
        self.position_embedding = nn.Embedding(config.n_positions, self.hidden_dim)
        # GPT2 transformer blocks.
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        # Project hidden states back to input_dim.
        self.output_projection = nn.Linear(self.hidden_dim, self.input_dim)
        # Decoder: maps input_dim to logits over tokens.
        self.decoder = nn.Linear(self.input_dim, self.ntokens)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, noise_mask: torch.BoolTensor,
                noise_vectors: torch.FloatTensor, labels: torch.LongTensor = None):
        """
        Args:
           input_ids: LongTensor of shape (batch_size, seq_len) where text token positions
                      contain valid token ids (>=0) and noise positions are marked with -1.
           noise_mask: BoolTensor of shape (batch_size, seq_len) that is True for noise positions.
           noise_vectors: FloatTensor of shape (batch_size, seq_len, input_dim). For noise positions,
                          this contains the fixed noise vector; for text positions, a dummy value.
           labels: LongTensor of shape (batch_size, seq_len), with -100 for positions to ignore.
        """
        batch_size, seq_len = input_ids.shape
        # For text positions, look up the learned embedding.
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        # For positions flagged as noise, select the provided noise vector.
        embeddings = torch.where(noise_mask.unsqueeze(-1), noise_vectors, text_embeddings)
        # Project the combined embeddings to the hidden dimension.
        hidden_states = self.input_proj(embeddings)  # (B, seq_len, hidden_dim)
        # Add positional embeddings.
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        # Process through GPT2 blocks.
        for block in self.h:
            outputs = block(hidden_states, use_cache=False)
            hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        # Project back to input_dim and decode to logits over tokens.
        x = self.output_projection(hidden_states)
        logits = self.decoder(x)
        return CausalLMOutput(logits=logits)


# ==============================================================================
# 3. Custom Trainer Class
# ==============================================================================
#
# In this updated compute_loss we perform the standard next-token prediction loss
# by shifting the logits and labels by one timestep. That is, the model's output at time t
# is compared with the target token at time t+1. This way, the model is only penalized
# for predicting a text token (and not the inserted noise).
#
# The loss is computed as the sum over valid tokens divided by the number of valid tokens,
# ensuring the per-token loss (and hence perplexity) is correctly normalized.

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
        # 'inputs' is a dict with keys: "input_ids", "noise_mask", "noise_vectors", "labels".
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, L, ntokens)

        # For next-token prediction, shift logits and labels:
        shift_logits = logits[:, :-1, :]   # all but the last timestep
        shift_labels = labels[:, 1:]         # all but the first timestep

        # Compute the sum loss over valid (non-ignored) tokens.
        loss_sum = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum"
        )
        # Count valid tokens.
        valid_tokens = (shift_labels.reshape(-1) != -100).sum().float()
        mean_loss = loss_sum / valid_tokens if valid_tokens > 0 else loss_sum
        return (mean_loss, outputs) if return_outputs else mean_loss


# ==============================================================================
# 4. Training Script: Putting Everything Together
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a VectorGPT model on text with interleaved noise (learned token embeddings).'
    )
    # Dataset/model parameters.
    parser.add_argument('--seq_len', type=int, default=128, help='Number of text tokens per sample (before inserting noise).')
    parser.add_argument('--k', type=int, default=3, help='Insert a noise token every k text tokens.')
    parser.add_argument('--input_dim', type=int, default=200, help='Dimension of each token representation (and noise vector).')
    # Model hyperparameters.
    parser.add_argument('--model_emb', type=int, default=256, help='Transformer hidden dimension.')
    parser.add_argument('--model_layers', type=int, default=4, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads.')
    # Training hyperparameters.
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    args = parser.parse_args()
    print('args:', args)

    # 1. Load a tokenizer (using GPT2's tokenizer).
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Optionally, add special tokens if needed.
    tokenizer.model_max_length = int(1e7)

    # 2. Create the training and validation datasets.
    train_dataset = TextWithNoiseDataset(split="train", tokenizer=tokenizer,
                                          seq_len=args.seq_len, k=args.k, input_dim=args.input_dim)
    valid_dataset = TextWithNoiseDataset(split="validation", tokenizer=tokenizer,
                                          seq_len=args.seq_len, k=args.k, input_dim=args.input_dim)

    # 3. Define a collate_fn to batch the samples.
    def collate_fn(batch):
        # Each sample is a dict with keys: "input_ids", "noise_mask", "noise_vectors", "labels".
        input_ids = torch.stack([item["input_ids"] for item in batch])
        noise_mask = torch.stack([item["noise_mask"] for item in batch])
        noise_vectors = torch.stack([item["noise_vectors"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "input_ids": input_ids,
            "noise_mask": noise_mask,
            "noise_vectors": noise_vectors,
            "labels": labels
        }

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Prepare the model configuration.
    config = VectorGPTConfig(
        n_positions=1000,         # must be >= max sequence length (including noise tokens)
        n_embd=args.model_emb,
        n_layer=args.model_layers,
        n_head=args.n_head,
        input_dim=args.input_dim,
        ntokens=tokenizer.vocab_size  # vocabulary size
    )
    model = VectorGPTModel(config)

    # 5. Set up training arguments.
    output_dir = f"./vector_gpt_trainer/{args.model_emb}_{args.model_layers}_{timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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

    # 6. Instantiate the custom trainer.
    trainer = VectorGPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # used for saving, etc.
        eval_dataset=valid_dataset,
        tokenizer=None,  # not used here
        custom_args=custom_args,
        train_loader=train_loader,
        valid_loader=valid_loader
    )

    # 7. (Optional) Add callbacks for logging and saving losses.
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

    # 8. Start training!
    trainer.train()
