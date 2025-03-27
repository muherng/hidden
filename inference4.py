import torch
import time
import argparse
from transformers import GPT2Tokenizer, set_seed
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
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64000)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # For testing purposes, you can hard-code the checkpoint path.
    args.checkpoint = 'tree_model/tree_20250321_133251/checkpoint-730800/model.safetensors'

    set_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024*64,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )

    model = TransformerScanModel.from_pretrained(
        args.checkpoint, 
        config=config, 
        chunk_size=args.chunk_size,
        T1_num_layers=2,  # match training parameters
        T2_num_layers=2
    ).to(device).eval()

    # Prepare batched prompts
    prompts = [args.prompt] * args.batch_size
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)

    # Initialize L as None (will be built on first call)
    L = None

    if False: 
        start = time.time()
        for i in range(args.max_new_tokens):
            if i%100 == 0: 
                print('token number: ', i)
            next_logits, L = model.forward_inference(input_ids, L=L)
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (batch,1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        total_time = time.time() - start
        print(f"Generated {args.max_new_tokens} tokens for batch size {args.batch_size} in {total_time:.3f}s "
            f"({total_time/args.max_new_tokens:.4f}s/token).")

        # Print a snippet of the first generated sequence
        output = tokenizer.decode(input_ids[0, :].tolist(), skip_special_tokens=True)
        print("Example output:", output[:200])

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