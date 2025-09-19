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
    DataCollatorForLanguageModeling,
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig

os.environ["HF_NO_CONVERT_SLOW_TOKENIZERS"] = "1"  # don't auto-convert slow tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune Mamba on WikiText-103 for causal LM")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-370m")
    parser.add_argument("--output_dir", type=str, default="mamba_wt103")
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bfloat16", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=512)
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
    dataset = load_dataset("wikitext", "wikitext-103-v1")

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

    # Build a fresh Mamba model from scratch
    print("Initializing Mamba model from scratch …")
    config = MambaConfig(
        vocab_size=len(tokenizer),
        d_model=args.hidden_dim,
        n_layer=8,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    )

    model = MambaLMHeadModel(
        config=config,
        dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_steps=1000,
        save_steps=1000,
        logging_steps=50,
        report_to=[] if args.disable_wandb else ["wandb"],
        bf16=args.use_bfloat16,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training …")
    trainer.train()

    print("Saving model …")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training args
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
