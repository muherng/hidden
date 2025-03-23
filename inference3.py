#!/usr/bin/env python
import torch
import time
import argparse
from transformers import GPT2Tokenizer, GPT2Config, set_seed
from tree_model6 import TransformerScanModel  # Adjust this import as needed

def pad_to_chunk(x, chunk_size, pad_token_id, device):
    # x: (batch, seq_length). Pad x along dim=1 so that seq_length becomes a multiple of chunk_size.
    L = x.size(1)
    remainder = L % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        pad_ids = torch.full((x.size(0), pad_len), pad_token_id, dtype=torch.long, device=device)
        return torch.cat([x, pad_ids], dim=1)
    return x

def get_offset(x, chunk_size):
    # Given the current sequence x (batch, seq_length), determine the offset within the last chunk.
    L = x.size(1)
    last_chunk_start = (L // chunk_size) * chunk_size
    offset = L - last_chunk_start - 1  # in [0, chunk_size-1)
    return offset

def compare_prefix_states(model, input_ids, chunk_size, debug=False):
    """
    Given a padded input_ids tensor (shape: [batch, seq_length], where seq_length is a multiple of chunk_size),
    this function computes the prefix states in two ways:
      (a) Using the vectorized_prefix_scan method.
      (b) Using a sequential binary-counter method identical to the one used in sequential_inference.
    
    It then prints, for each chunk index, summary statistics (L2 norm) and the maximum/mean absolute differences.
    The final prefix tensor P is expected to have shape (batch, num_chunks+1, chunk_size, hidden_dim),
    where P[:,0] is the dummy state and for i>=1, P[:,i] is the prefix state that conditions chunk i.
    """
    batch_size, seq_length = input_ids.shape
    num_chunks = seq_length // chunk_size
    # Split input into chunks.
    chunks = input_ids.view(batch_size, num_chunks, chunk_size)
    
    # Compute T0 outputs for each chunk.
    level0 = [model.T0(chunks[:, i, :]) for i in range(num_chunks)]
    dummy = torch.zeros_like(level0[0])
    
    # (a) Compute vectorized prefix using the balanced-tree (Blelloch) scan.
    P_vector = model.vectorized_prefix_scan(level0, dummy, debug=debug)
    # (b) Compute prefix sequentially using binary counter logic.
    L_slots = [None] * (num_chunks.bit_length() + 1)
    sequential_prefixes = []
    for i in range(num_chunks):
        s = level0[i]
        x = s
        j = 0
        while (i >> j) & 1:
            # Combine L_slots[j] and x.
            x = model.T1(torch.cat([L_slots[j], x], dim=1))[:, -chunk_size:, :]
            L_slots[j] = None
            j += 1
        L_slots[j] = x
        # Combine non-None entries in L_slots (from highest index downwards) to get the prefix.
        prefix = None
        for k in reversed(range(len(L_slots))):
            if L_slots[k] is not None:
                if prefix is None:
                    prefix = L_slots[k]
                else:
                    prefix = model.T1(torch.cat([L_slots[k], prefix], dim=1))[:, -chunk_size:, :]
        # For chunk 0, we use s; for subsequent chunks, we use the computed prefix.
        if i == 0:
            sequential_prefixes.append(s)
        else:
            sequential_prefixes.append(prefix)
    P_seq = torch.cat([dummy.unsqueeze(1)] + [p.unsqueeze(1) for p in sequential_prefixes], dim=1)
    # P_seq: (batch, num_chunks+1, chunk_size, hidden_dim)
    
    # Compare the two.
    diff = torch.abs(P_vector - P_seq)
    overall_max = diff.max().item()
    overall_mean = diff.mean().item()
    print(f"Prefix state comparison: overall max diff = {overall_max:.6f}, overall mean diff = {overall_mean:.6f}")
    
    for i in range(P_vector.size(1)):
        vec_state = P_vector[:, i, :, :]
        seq_state = P_seq[:, i, :, :]
        chunk_diff = torch.abs(vec_state - seq_state)
        norm_vec = torch.norm(vec_state, p=2).item()
        norm_seq = torch.norm(seq_state, p=2).item()
        max_diff = chunk_diff.max().item()
        mean_diff = chunk_diff.mean().item()
        print(f"Chunk {i}: vectorized L2 norm = {norm_vec:.6f}, sequential L2 norm = {norm_seq:.6f}, max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}")
    
    return P_vector, P_seq


def debug_compare_step(model, generated, args, tokenizer, tol=1e-3, debug=False):
    """
    Runs both the vectorized forward() and sequential_inference() prefix computations
    on the same padded input and compares the resulting prefix states.
    Returns the next-token logits from the forward() pass.
    """
    device = generated.device
    padded_generated = pad_to_chunk(generated, args.chunk_size, tokenizer.eos_token_id, device)
    # Compare prefix states using the compare_prefix_states() function.
    P_vector, P_seq = compare_prefix_states(model, padded_generated, args.chunk_size, debug=debug)
    
    # Additionally, print overall difference.
    diff = torch.abs(P_vector - P_seq)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"(Debug step) Overall prefix state diff: max = {max_diff:.6f}, mean = {mean_diff:.6f}")
    assert max_diff < tol, f"Prefix state difference exceeds tolerance: {max_diff} > {tol}"
    
    # Now run the forward() pass to get the next-token logits.
    outputs = model(padded_generated)
    offset = get_offset(generated, args.chunk_size)
    if outputs.logits.dim() == 3:
        token_logits_forward = outputs.logits[:, offset, :]  # (batch, vocab_size)
    elif outputs.logits.dim() == 4:
        token_logits_forward = outputs.logits[:, -1, offset, :]
    else:
        raise ValueError("Unexpected logits dimensions from forward()")
    
    return token_logits_forward

def main():
    parser = argparse.ArgumentParser(description="Inference with token-by-token logit and prefix state comparison")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--chunk_size", type=int, default=32, help="Chunk size used during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for prefix state differences")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    args = parser.parse_args()
    
    # For testing purposes, you can hard-code the checkpoint path.
    args.checkpoint = 'tree_model/tree_20250321_133251/checkpoint-730800/model.safetensors'
    
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # Load tokenizer and config.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )
    
    # Instantiate and load your model using the from_pretrained method.
    model = TransformerScanModel.from_pretrained(
        args.checkpoint, 
        config=config, 
        chunk_size=args.chunk_size,
        T1_num_layers=2,  # match training parameters
        T2_num_layers=2
    )
    model.to(device)
    model.eval()
    
    # Tokenize prompt.
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    generated = pad_to_chunk(input_ids, args.chunk_size, tokenizer.eos_token_id, device)
    
    print("Initial prompt:", tokenizer.decode(generated[0]))
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(args.max_new_tokens):
            # At a chunk boundary, add an extra token to break it.
            if generated.size(1) % args.chunk_size == 0:
                extra = torch.tensor([[tokenizer.eos_token_id]], device=device)
                generated = torch.cat([generated, extra], dim=1)
            
            # Run debug comparison: compare prefix states from both methods.
            try:
                token_logits_forward = debug_compare_step(model, generated, args, tokenizer, tol=args.tol, debug=args.debug)
            except AssertionError as e:
                print("AssertionError during debug compare:", e)
                break
            
            # Use the forward() logits for greedy decoding.
            offset = get_offset(generated, args.chunk_size)
            outputs = model(pad_to_chunk(generated, args.chunk_size, tokenizer.eos_token_id, device))
            if outputs.logits.dim() == 3:
                logits = outputs.logits[:, offset, :]
            elif outputs.logits.dim() == 4:
                logits = outputs.logits[:, -1, offset, :]
            else:
                raise ValueError("Unexpected logits dimensions in forward()")
            
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                print("EOS token generated; stopping.")
                break
    
    total_time = time.time() - start_time
    num_generated = generated.size(1) - input_ids.size(1)
    print("Final generated text:", tokenizer.decode(generated[0], clean_up_tokenization_spaces=True))
    print(f"Generated {num_generated} tokens in {total_time:.3f} seconds "
          f"({num_generated/total_time:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
