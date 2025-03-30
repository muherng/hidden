import torch
import time
import argparse
import matplotlib.pyplot as plt

from transformers import GPT2Tokenizer, set_seed, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from tree_model6 import TransformerScanModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=40000)
    parser.add_argument("--npositions", type=int, default=41000)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    prompt_capacity = args.npositions - args.max_new_tokens
    assert prompt_capacity > 0

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

    raw = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    tokens = tokenizer(raw, return_tensors="pt", add_special_tokens=False, truncation=True,
                       max_length=prompt_capacity).input_ids[0][:prompt_capacity]
    input_ids = tokens.unsqueeze(0).repeat(args.batch_size, 1).to(device)

    # Initialize TransformerScanModel
    model = TransformerScanModel(config=config, chunk_size=args.chunk_size, T1_num_layers=2, T2_num_layers=2)
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, config.initializer_range)
            if m.bias is not None: m.bias.data.zero_()
        elif isinstance(m, torch.nn.Embedding):
            m.weight.data.normal_(0.0, config.initializer_range)
        elif isinstance(m, torch.nn.LayerNorm):
            m.bias.data.zero_(); m.weight.data.fill_(1.0)
    model.apply(init_weights)
    model.to(device).eval()

    scan_times = []
    L = None
    chunks_processed = 0
    prefix_val = None
    past_key_values = None

    with torch.no_grad():
        for i in range(args.max_new_tokens):
            t0 = time.time()
            next_logits, L, chunks_processed, prefix_val, past_key_values = model.forward_inference(
                input_ids, L, chunks_processed, prefix_val, past_key_values=past_key_values
            )
            scan_times.append(time.time() - t0)
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if i % 1000 == 0:
                print(f"Scan model token {i}")

    print(f"TransformerScanModel average time/token: {sum(scan_times)/len(scan_times):.6f}s")

    # Vanilla GPT‑2
    vanilla = GPT2LMHeadModel(config).to(device).eval()
    vanilla_times = []
    past = None

    with torch.no_grad():
        for i in range(args.max_new_tokens):
            t0 = time.time()
            outputs = vanilla(input_ids[:, -1:], past_key_values=past)
            vanilla_times.append(time.time() - t0)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if i % 1000 == 0:
                print(f"Vanilla model token {i}")

    print(f"Vanilla GPT-2 average time/token: {sum(vanilla_times)/len(vanilla_times):.6f}s")

    # Plotting
    plt.figure()
    plt.plot(range(len(scan_times)), scan_times, label="TransformerScanModel")
    plt.plot(range(len(vanilla_times)), vanilla_times, label="Vanilla GPT-2")
    plt.xlabel("Token index")
    plt.ylabel("Time per token (s)")
    plt.title("Inference Speed vs Generation Length")
    plt.legend()
    plt.savefig("inference_speed.png", dpi=300)


if __name__ == "__main__":
    main()
