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
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--prompt_len", type=int, default=100,
                        help="Prompt length for both models (keep small for GPU memory)")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Disable Flash / memory-efficient SDPA so attention cost grows with sequence length
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)  # force math (O(L^2)) kernel

    # Config for both models
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config.from_pretrained("gpt2")
    config.attn_implementation = "eager"
    config._attn_implementation = "eager"
    # Extend context length for TransformerScanModel
    npositions = args.prompt_len + args.max_new_tokens + 1000  # buffer
    config.n_positions = npositions
    config.n_ctx = npositions

    # Load prompt tokens
    raw = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    tokens = tokenizer(raw, return_tensors="pt", add_special_tokens=False, truncation=True,
                       max_length=args.prompt_len).input_ids[0][:args.prompt_len]

    print(f"Running comparison: batch_size={args.batch_size}, prompt_len={args.prompt_len}, max_new_tokens={args.max_new_tokens}")

    # ============ TransformerScanModel ============
    print("Running TransformerScanModel...")
    input_ids = tokens.unsqueeze(0).repeat(args.batch_size, 1).to(device)

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
                print(f"  TransformerScanModel token {i}/{args.max_new_tokens}")

    print(f"TransformerScanModel average time/token: {sum(scan_times)/len(scan_times):.6f}s")

    # Free memory before loading vanilla GPT-2
    del model, L, prefix_val, past_key_values, input_ids
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ============ Vanilla GPT-2 ============
    print("Running Vanilla GPT-2...")
    
    # Create vanilla GPT-2 with small n_positions, then resize bias buffers
    vanilla_config = GPT2Config.from_pretrained("gpt2")
    vanilla_config.attn_implementation = "eager"
    vanilla_config._attn_implementation = "eager"
    max_pos_embed = vanilla_config.n_positions  # 1024 - for clamping position_ids
    vanilla = GPT2LMHeadModel(vanilla_config)
    
    # Resize causal mask buffers to handle longer sequences
    new_bias = torch.tril(torch.ones((npositions, npositions), dtype=torch.bool)).view(
        1, 1, npositions, npositions
    )
    for block in vanilla.transformer.h:
        block.attn.bias = new_bias
    
    vanilla_config.n_positions = npositions
    vanilla_config.n_ctx = npositions
    
    vanilla = vanilla.to(device)
    for block in vanilla.transformer.h:
        block.attn.bias = block.attn.bias.to(device)
    vanilla = vanilla.eval()

    # Same prompt as TransformerScanModel
    input_ids = tokens.unsqueeze(0).repeat(args.batch_size, 1).to(device)

    # Prefill: process prompt to build initial KV cache (same as TransformerScanModel)
    with torch.no_grad():
        position_ids = torch.arange(args.prompt_len, device=device).unsqueeze(0)
        position_ids = position_ids.clamp(max=max_pos_embed - 1).expand(args.batch_size, -1)
        outputs = vanilla(input_ids, position_ids=position_ids, use_cache=True)
        past = outputs.past_key_values

    vanilla_times = []

    with torch.no_grad():
        for i in range(args.max_new_tokens):
            seq_len = args.prompt_len + i
            cur_pos = min(seq_len, max_pos_embed - 1)
            position_ids = torch.tensor([[cur_pos]], device=device).expand(args.batch_size, -1)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            outputs = vanilla(input_ids[:, -1:], past_key_values=past, position_ids=position_ids)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            vanilla_times.append(time.time() - t0)

            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if i % 1000 == 0:
                print(f"  Vanilla GPT-2 token {i}/{args.max_new_tokens}, KV cache size={seq_len}")

    print(f"Vanilla GPT-2 average time/token: {sum(vanilla_times)/len(vanilla_times):.6f}s")

    # Plotting
    plt.figure(figsize=(10, 6))
    start_idx = 100  # skip warmup
    if len(scan_times) > start_idx:
        plt.plot(range(start_idx, len(scan_times)), scan_times[start_idx:], label="TransformerScanModel", alpha=0.8)
    if len(vanilla_times) > start_idx:
        plt.plot(range(start_idx, len(vanilla_times)), vanilla_times[start_idx:], label="Vanilla GPT-2 (KV cache)", alpha=0.8)
    plt.xlabel("Token index")
    plt.ylabel("Time per token (s)")
    plt.title(f"Inference Speed: batch_size={args.batch_size}, prompt_len={args.prompt_len}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("inference_speed.png", dpi=300)
    print("Plot saved to inference_speed.png")


if __name__ == "__main__":
    main()
