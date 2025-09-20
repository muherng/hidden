import os
import argparse
import json
import random

import torch
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
# Import base Mamba config and define a subclass that implements `to_dict`, which the
# Hugging Face `Trainer` expects for logging integrations (e.g. WandB). Without this
# method `Trainer` raises `AttributeError: 'MambaConfig' object has no attribute 'to_dict'`.
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig


class CustomMambaConfig(MambaConfig):
    """Extend `MambaConfig` with a `to_dict` method for compatibility with HF Trainer."""

    def to_dict(self):  # type: ignore[override]
        """Return a serialisable representation of the config."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "ssm_cfg": self.ssm_cfg,
            "rms_norm": self.rms_norm,
            "residual_in_fp32": self.residual_in_fp32,
            "fused_add_norm": self.fused_add_norm,
            "pad_vocab_size_multiple": self.pad_vocab_size_multiple,
        }

    # Custom loss: call model with input_ids only, then compute xent against labels.
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(input_ids=inputs["input_ids"])  # logits
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        if return_outputs:
            return (loss, outputs)
        return loss

# ----- utility to save model with shared tensors ------------------------------


def save_model_with_shared_tensors(model: MambaLMHeadModel, output_dir: str):
    """Save model avoiding safetensors shared-weight crash (embed/lm_head)."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    if hasattr(model, "config") and hasattr(model.config, "to_dict"):
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

# ----- custom compute_loss for Trainer -------------------------------------------------

def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # noqa: D401
    """Cross-entropy loss for causal LM without passing labels to model."""
    labels = inputs.pop("labels")
    outputs = model(input_ids=inputs["input_ids"])  # Mamba returns logits
    logits = outputs.logits if hasattr(outputs, "logits") else outputs  # (B, L, V)

    # Shift so that tokens < t predict token at t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    if return_outputs:
        return (loss, outputs)
    return loss

os.environ["HF_NO_CONVERT_SLOW_TOKENIZERS"] = "1"  # don't auto-convert slow tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune Mamba on WikiText-103 for causal LM")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-370m")
    parser.add_argument("--output_dir", type=str, default="mamba_wt103")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bfloat16", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12, help="Number of Mamba layers to approximate model size (12 => ~130M params)")
    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load tokenizer (we rely on the one shipped with the checkpoint)
    print("Loading tokenizer …")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load WikiText-103
    print("Loading WikiText-103 … (this may take a while)")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Tokenize and chunk into blocks of context_length
    def tokenize_function(text_examples):
        return tokenizer(text_examples["text"], return_special_tokens_mask=False)

    print("Tokenising …")
    tokenized = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    def group_texts(examples):
        # concatenate and chunk
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        # drop remainder
        total_length = (total_length // args.context_length) * args.context_length
        result = {
            k: [t[i : i + args.context_length] for i in range(0, total_length, args.context_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True, num_proc=4)
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Build model config – if user passes an existing HF checkpoint (e.g. mamba-130m),
    # load its config then random-init weights. Otherwise use custom dimensions.
    print("Initializing custom Mamba config …")
    config = CustomMambaConfig(
        vocab_size=len(tokenizer),
        d_model=args.hidden_dim,
        n_layer=args.n_layers,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    )

    model = MambaLMHeadModel(config=config,
        dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32)

    # Patch forward to ignore labels kwarg (Trainer passes it during evaluation)
    original_forward = model.forward

    def forward_no_labels(*f_args, **f_kwargs):
        f_kwargs.pop("labels", None)
        return original_forward(*f_args, **f_kwargs)

    model.forward = forward_no_labels  # type: ignore[assignment]

    # Simple collate_fn that stacks tensors and omits attention_mask to avoid issues with Mamba
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        if "labels" in batch[0]:
            labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
        else:
            labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    # ----------------- periodic evaluation callback ----------------------------

    # -------- evaluation helper ---------------------------------------------------------

    def compute_perplexity(model, dataset, batch_size):
        model.eval()
        device = next(model.parameters()).device
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        total_loss = 0
        total_tokens = 0
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"])
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch["labels"][:, 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                total_tokens += (shift_labels != -100).numel()
        model.train()
        avg_nll = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_nll)).item()

    class PerplexityCallback(TrainerCallback):
        """Evaluate model every `eval_steps` and print perplexity."""

        def __init__(self, trainer_ref, eval_steps=1000):
            self.trainer = trainer_ref
            self.eval_steps = eval_steps
            self.best_loss = float("inf")

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.eval_steps == 0 and state.global_step != 0:
                ppl = compute_perplexity(self.trainer.model, self.trainer.eval_dataset, args.per_device_eval_batch_size)
                nll = torch.log(torch.tensor(ppl))
                print(f"\n=== Step {state.global_step}: val_nll={nll:.4f}, perplexity={ppl:.3f} ===\n")

                # Save best model so far
                if nll < self.best_loss:
                    self.best_loss = nll
                    best_dir = os.path.join(self.trainer.args.output_dir, "best")
                    print(f"New best model (nll {nll:.4f}); saving to {best_dir}")
                    self.trainer.save_model(best_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_steps=100,
        save_steps=500,
        logging_steps=100,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        save_total_limit=2,
        report_to=[] if args.disable_wandb else ["wandb"],
        bf16=args.use_bfloat16,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.add_callback(PerplexityCallback(trainer, eval_steps=1000))

    # override checkpoint saving to handle shared tensors
    trainer.save_model = lambda dir_path, _internal_call=False: save_model_with_shared_tensors(model, dir_path)

    # override compute_loss
    trainer.compute_loss = compute_loss.__get__(trainer)

    print("Starting training …")
    trainer.train()

    final_ppl = compute_perplexity(model, eval_dataset, args.batch_size)
    print(f"\nFinal validation perplexity: {final_ppl:.3f}")

    print("Saving model …")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training args
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
