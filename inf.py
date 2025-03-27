import torch
import time
import argparse
from transformers import GPT2Tokenizer, set_seed, GPT2Config
from datasets import load_dataset
from tree_model6 import TransformerScanModel

from transformers import (
    GPT2LMHeadModel,
    PreTrainedModel,
    GPT2Config,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=79000)
    parser.add_argument("--npositions", type=int, default=80000)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    prompt_capacity = args.npositions - args.max_new_tokens
    assert prompt_capacity > 0, "npositions must exceed max_new_tokens!"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.npositions
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.npositions,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )

    # Load raw Wikitext‑2
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw = "\n".join(ds["text"])

    # Tokenize → trim to exactly n_positions
    tokens = tokenizer(
        raw,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_capacity,
    ).input_ids[0]
    tokens = tokens[: prompt_capacity]

    input_ids = tokens.unsqueeze(0).repeat(args.batch_size, 1).to(device)

    # Build & randomly initialize TransformerScanModel
    model = TransformerScanModel(config=config, chunk_size=args.chunk_size, T1_num_layers=2, T2_num_layers=2)
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, config.initializer_range)
            if m.bias is not None: m.bias.data.zero_()
        elif isinstance(m, torch.nn.Embedding):
            m.weight.data.normal_(0.0, config.initializer_range)
        elif isinstance(m, torch.nn.LayerNorm):
            m.bias.data.zero_(); m.weight.data.fill_(1.0)
    model.apply(_init_weights)
    model = model.to(device).eval()

    # Autoregressive generation
    if True: 
        L = None
        chunks_processed = 0
        prefix_val = None
        past_key_values = None
        start = time.time()
        for i in range(args.max_new_tokens):
            if i % 100 == 0:
                print(f"Generating token {i}/{args.max_new_tokens}")
            if i == 1: 
                start = time.time()
            #next_logits, L = model.forward_inference(input_ids, L=L)
            next_logits, L, chunks_processed, prefix_val, past_key_values = model.forward_inference_fix(
                input_ids, L, chunks_processed, prefix_val, past_key_values = past_key_values
            )
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        total = time.time() - start
        print(f"Generated {args.max_new_tokens} tokens in {total:.2f}s ({total/args.max_new_tokens:.4f}s/token)")

        # Print first 200 chars of first example
        print(tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)[:200])

    # ---- Measure vanilla GPT‑2 inference (with caching) ----
    vanilla = GPT2LMHeadModel(config).to(device).eval()
    input_ids = input_ids.clone()
    past = None
    torch.cuda.synchronize(); start = time.time()
    with torch.no_grad():
        for i in range(args.max_new_tokens):
            if i%100 == 0: 
                print('token number: ', i)
            outputs = vanilla(input_ids[:, -1:], past_key_values=past)
            logits = outputs.logits
            past = outputs.past_key_values
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    torch.cuda.synchronize(); vanilla_time = time.time() - start

    print(f"Vanilla GPT‑2: {vanilla_time:.3f}s total — {vanilla_time/args.max_new_tokens:.6f}s/token")

if __name__ == "__main__":
    main()