import gc
import torch
import time
import argparse
import matplotlib.pyplot as plt

from transformers import GPT2Tokenizer, set_seed, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from models.tree_model6 import TransformerScanModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=40000)
    parser.add_argument("--npositions", type=int, default=41000)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vanilla_device", type=str, default="cpu",
                        help="Device for vanilla GPT-2 (use 'cpu' to observe linear KV cache latency)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Disable Flash / memory-efficient SDPA so attention cost grows with sequence length
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)  # force math (O(L^2)) kernel

    prompt_capacity = args.npositions - args.max_new_tokens
    assert prompt_capacity > 0

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.npositions

    # Use the standard GPT-2 (124 M params) config – much larger than the toy model
    config = GPT2Config.from_pretrained("gpt2")
    # Extend context length to match experiment
    config.n_positions = args.npositions
    config.n_ctx = args.npositions
    config.attn_implementation = "eager"
    config._attn_implementation = "eager"

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
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            next_logits, L, chunks_processed, prefix_val, past_key_values = model.forward_inference(
                input_ids, L, chunks_processed, prefix_val, past_key_values=past_key_values
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            scan_times.append(time.time() - t0)

            next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if i % 1000 == 0:
                print(f"Scan model token {i}")

    print(f"TransformerScanModel average time/token: {sum(scan_times)/len(scan_times):.6f}s", flush=True)

    # Free memory from the first experiment before loading vanilla GPT-2
    print("[DEBUG] Deleting scan model variables...", flush=True)
    del model, L, prefix_val, past_key_values, input_ids
    print("[DEBUG] Running gc.collect()...", flush=True)
    gc.collect()
    if device.type == "cuda":
        print(f"[DEBUG] GPU memory before empty_cache: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        torch.cuda.empty_cache()
        print(f"[DEBUG] GPU memory after empty_cache: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Reset input_ids to the original prompt for a fair comparison
    print("[DEBUG] Creating input_ids...", flush=True)
    input_ids = tokens.unsqueeze(0).repeat(args.batch_size, 1).to(device)
    print(f"[DEBUG] input_ids shape: {input_ids.shape}", flush=True)

    # Vanilla GPT-2: Create with n_positions=1024 to avoid OOM from large buffers,
    # then manually resize the causal mask buffers in each attention layer.
    # Position embeddings stay at 1024 entries; we clamp position_ids during generation.
    print("[DEBUG] Creating GPT2LMHeadModel on CPU...", flush=True)
    vanilla_config = GPT2Config.from_pretrained("gpt2")
    vanilla_config.attn_implementation = "eager"
    vanilla_config._attn_implementation = "eager"
    max_pos_embed = vanilla_config.n_positions  # Save original (1024) for position_ids clamping
    vanilla = GPT2LMHeadModel(vanilla_config)
    
    # Resize the causal mask (bias buffer) in each attention layer to handle 40k tokens
    # This buffer is created during __init__ with size 1024x1024; we need to expand it
    print("[DEBUG] Resizing attention bias buffers...", flush=True)
    new_max_pos = args.npositions
    new_bias = torch.tril(torch.ones((new_max_pos, new_max_pos), dtype=torch.bool)).view(
        1, 1, new_max_pos, new_max_pos
    )
    for block in vanilla.transformer.h:
        block.attn.bias = new_bias
    
    # Update config for consistency
    vanilla_config.n_positions = args.npositions
    vanilla_config.n_ctx = args.npositions
    
    # Run vanilla GPT-2 on specified device (default: CPU to observe linear KV cache latency)
    vanilla_device = torch.device(args.vanilla_device)
    print(f"[DEBUG] Moving GPT2LMHeadModel to {vanilla_device}...", flush=True)
    vanilla = vanilla.to(vanilla_device)
    # Also move the bias tensors (they were assigned as attributes, not registered buffers)
    for block in vanilla.transformer.h:
        block.attn.bias = block.attn.bias.to(vanilla_device)
    print("[DEBUG] Setting to eval mode...", flush=True)
    vanilla = vanilla.eval()
    print("[DEBUG] GPT2LMHeadModel ready", flush=True)
    vanilla_times = []

    # Reset input_ids for vanilla on its device
    input_ids = tokens.unsqueeze(0).repeat(args.batch_size, 1).to(vanilla_device)
    seq_len = input_ids.shape[1]

    # PREFILL: Process full prompt first to build initial KV cache
    # This makes comparison with TransformerScanModel fair (both start with same context)
    print(f"[DEBUG] Prefill: processing {seq_len} prompt tokens...", flush=True)
    with torch.no_grad():
        # Clamp position_ids for full prompt
        position_ids = torch.arange(seq_len, device=vanilla_device).unsqueeze(0)
        position_ids = position_ids.clamp(max=max_pos_embed - 1)
        outputs = vanilla(input_ids, position_ids=position_ids, use_cache=True)
        past = outputs.past_key_values
    print(f"[DEBUG] Prefill complete, KV cache has {seq_len} entries", flush=True)

    # DECODE: Generate tokens one at a time, measuring latency
    # With KV cache, each token requires O(L) attention where L grows linearly
    # On GPU with small batch, this may be hidden by overhead. Try:
    #   --batch_size 32 (more work per attention op)
    #   --vanilla_device cpu (removes GPU parallelism)

    with torch.no_grad():
        for i in range(args.max_new_tokens):
            # Clamp position_id to max 1023 (position embedding limit)
            cur_pos = min(seq_len + i, max_pos_embed - 1)
            position_ids = torch.tensor([[cur_pos]], device=vanilla_device).expand(args.batch_size, -1)

            if vanilla_device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            outputs = vanilla(input_ids[:, -1:], past_key_values=past, position_ids=position_ids)
            
            if vanilla_device.type == "cuda":
                torch.cuda.synchronize()
            
            vanilla_times.append(time.time() - t0)

            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if i % 1000 == 0:
                print(f"Vanilla model token {i}, seq_len={seq_len + i}")

    print(f"Vanilla GPT-2 average time/token: {sum(vanilla_times)/len(vanilla_times):.6f}s")

    # Plotting
    plt.figure()
    start_idx = 100  # skip the first 100 tokens
    if len(scan_times) > start_idx:
        plt.plot(range(start_idx, len(scan_times)), scan_times[start_idx:], label="TransformerScanModel")
    if len(vanilla_times) > start_idx:
        plt.plot(range(start_idx, len(vanilla_times)), vanilla_times[start_idx:], label="Vanilla GPT-2")
    plt.xlabel("Token index")
    plt.ylabel("Time per token (s)")
    plt.title("Inference Speed vs Generation Length")
    plt.legend()
    plt.savefig("inference_speed.png", dpi=300)


if __name__ == "__main__":
    main()
