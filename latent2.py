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
# Each sample is constructed from a contiguous chunk of tokenized text.
# For every k text tokens, a noise token is inserted.
#
# The input is of the form:
#    p = {a_0, a_1, ..., a_k, s_0, a_{k+1}, ..., a_{2k}, s_1, ..., a_T}
#
# where a_i are discrete tokens and s_j are random noise vectors.
# The labels are the discrete token ids for the a_i positions and -100 for noise.
#
# (Note: Next-token prediction is performed by shifting the logits and labels by one.)

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
        # Concatenate all texts and tokenize
        text = " ".join(self.data["text"])
        # Override max length to avoid warnings (we chunk manually)
        self.tokenizer.model_max_length = int(1e7)
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.vocab_size = tokenizer.vocab_size

        self.samples = []
        # Slide over the token stream in chunks of length seq_len.
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            a_tokens = self.token_ids[i:i+seq_len]
            input_ids_list = []     # For text positions: the token id; for noise: -1.
            noise_mask_list = []    # True if position is a noise token.
            noise_vectors_list = [] # For noise positions: fixed random noise vector; otherwise, zeros.
            labels_list = []        # For text tokens: token id; for noise: -100.
            for j, token in enumerate(a_tokens):
                # Discrete text token.
                input_ids_list.append(token)
                noise_mask_list.append(False)
                noise_vectors_list.append(torch.zeros(input_dim))  # dummy vector (unused)
                labels_list.append(token)
                # After every k text tokens (except at the end), insert a noise token.
                if (j + 1) % self.k == 0 and (j + 1) < len(a_tokens):
                    input_ids_list.append(-1)            # marker for noise (won't be used for embedding lookup)
                    noise_mask_list.append(True)
                    noise_vectors_list.append(torch.randn(input_dim))  # random noise vector
                    labels_list.append(-100)             # ignored in loss
            sample = {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),  # (L,)
                "noise_mask": torch.tensor(noise_mask_list, dtype=torch.bool),  # (L,)
                "noise_vectors": torch.stack(noise_vectors_list),               # (L, input_dim)
                "labels": torch.tensor(labels_list, dtype=torch.long)           # (L,)
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
# The model embeds discrete tokens using a learned embedding.
# For positions flagged as noise, it uses the provided noise vector.
# After processing through the transformer blocks, the hidden states are
# projected to an intermediate continuous space (dimension input_dim) and then decoded.
#
# The forward method accepts a flag 'return_hidden'. When True, it returns
# the intermediate continuous representation (to be used in the second pass).

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

        # Learned embedding for text tokens.
        self.token_embedding = nn.Embedding(self.ntokens, self.input_dim)
        # Project the (embedded or noise) input to the hidden dimension.
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        # Positional embeddings.
        self.position_embedding = nn.Embedding(config.n_positions, self.hidden_dim)
        # Transformer blocks.
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        # Project hidden states back to intermediate space.
        self.output_projection = nn.Linear(self.hidden_dim, self.input_dim)
        # Decoder: from intermediate space to logits over tokens.
        self.decoder = nn.Linear(self.input_dim, self.ntokens)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, noise_mask: torch.BoolTensor,
                noise_vectors: torch.FloatTensor, labels: torch.LongTensor = None,
                return_hidden: bool = False):
        """
        Args:
           input_ids: (B, L) LongTensor; text token positions contain valid token ids (>=0),
                      and noise positions are marked with -1.
           noise_mask: (B, L) BoolTensor; True for noise positions.
           noise_vectors: (B, L, input_dim) FloatTensor; for noise positions, the provided noise vector.
           labels: (B, L) LongTensor; for text tokens, token id; for noise, -100.
           return_hidden: if True, return the intermediate representation 'x' (of shape (B, L, input_dim))
                          along with the logits.
        """
        batch_size, seq_len = input_ids.shape
        # Embed discrete tokens.
        text_embeddings = self.token_embedding(input_ids.clamp(min=0))
        # For noise positions, use the supplied vector.
        embeddings = torch.where(noise_mask.unsqueeze(-1), noise_vectors, text_embeddings)
        # Project to hidden dimension.
        hidden_states = self.input_proj(embeddings)  # (B, L, hidden_dim)
        # Add positional embeddings.
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        # Pass through transformer blocks.
        for block in self.h:
            outputs = block(hidden_states, use_cache=False)
            hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        # Project to intermediate space.
        x = self.output_projection(hidden_states)  # (B, L, input_dim)
        logits = self.decoder(x)  # (B, L, ntokens)
        if return_hidden:
            return CausalLMOutput(logits=logits, hidden_states=x)
        else:
            return CausalLMOutput(logits=logits)


# ==============================================================================
# 3. Custom Trainer Class with Two-Stage Forward Pass (Loss Computed Only on Second Pass)
# ==============================================================================
#
# In compute_loss:
#  - First, perform a forward pass with return_hidden=True to obtain s' vectors.
#  - Then, construct a second input by replacing the noise vectors with s' (i.e. p'').
#  - Run the second forward pass and compute the next-token prediction loss (with shifting)
#    while masking out the noise positions.
#  - The loss is computed only on the second pass.

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
        # Unpack inputs.
        input_ids = inputs["input_ids"]
        noise_mask = inputs["noise_mask"]
        noise_vectors = inputs["noise_vectors"]
        labels = inputs["labels"]

        # ---------- First Forward Pass ----------
        # Run the model to obtain intermediate hidden representations (s') at noise positions.
        outputs1 = model(input_ids=input_ids,
                         noise_mask=noise_mask,
                         noise_vectors=noise_vectors,
                         labels=labels,
                         return_hidden=True)
        # s' vectors are obtained as hidden_states from outputs1.
        s_prime = outputs1.hidden_states  # shape: (B, L, input_dim)

        # ---------- Second Forward Pass ----------
        # Create new input p'' by replacing noise vectors with s' from the first pass.
        outputs2 = model(input_ids=input_ids,
                         noise_mask=noise_mask,
                         noise_vectors=s_prime,  # use s' instead of original random noise
                         labels=labels)
        logits2 = outputs2.logits  # (B, L, ntokens)

        # For next-token prediction, shift logits and labels.
        shift_logits = logits2[:, :-1, :]   # drop the last timestep
        shift_labels = labels[:, 1:]          # drop the first timestep

        # Compute cross-entropy loss over valid tokens only.
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
        description='Train a VectorGPT model with a two-stage forward pass (loss computed only on second pass).'
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
    tokenizer.model_max_length = int(1e7)

    # 2. Create the training and validation datasets.
    train_dataset = TextWithNoiseDataset(split="train", tokenizer=tokenizer,
                                          seq_len=args.seq_len, k=args.k, input_dim=args.input_dim)
    valid_dataset = TextWithNoiseDataset(split="validation", tokenizer=tokenizer,
                                          seq_len=args.seq_len, k=args.k, input_dim=args.input_dim)

    # 3. Define a collate_fn to batch the samples.
    def collate_fn(batch):
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
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=None,
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
