#!/usr/bin/env python
import torch
import time
import argparse
from transformers import GPT2Tokenizer, GPT2Config, set_seed
from tree_model6 import TransformerScanModel  # Replace with the proper import path for your model

def main():
    parser = argparse.ArgumentParser(description="Inference script for TransformerScanModel")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=512-32, help="Max number of tokens to generate")
    parser.add_argument("--chunk_size", type=int, default=32, help="Chunk size used during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    args = parser.parse_args()

    args.checkpoint = 'tree_model/tree_20250321_133251/checkpoint-730800/model.safetensors'

    set_seed(args.seed)
    
    # Load GPT-2 tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Create GPT-2 configuration (adjust parameters as needed).
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )
    
    model = TransformerScanModel(config, chunk_size=args.chunk_size,
                             T1_num_layers=2, T2_num_layers=2)
    model = TransformerScanModel.from_pretrained(
        args.checkpoint, 
        config=config, 
        chunk_size=args.chunk_size,
        T1_num_layers=2,  # match training parameters
        T2_num_layers=2
    )
    model.to(args.device)
    model.eval()

    # Tokenize the prompt.
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    
    # Ensure input length is a multiple of chunk_size (pad with EOS token if necessary).
    seq_len = input_ids.size(1)
    remainder = seq_len % args.chunk_size
    if remainder != 0:
        pad_length = args.chunk_size - remainder
        pad_ids = torch.full((1, pad_length), tokenizer.eos_token_id, dtype=torch.long, device=args.device)
        input_ids = torch.cat([input_ids, pad_ids], dim=1)
    
    print("Prompt:", tokenizer.decode(input_ids[0]))
    
    # Inference loop: generate tokens one by one.
    generated = input_ids
    max_new_tokens = args.max_new_tokens
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # If the generated sequence length is exactly a multiple of chunk_size,
            # append an extra token (e.g. the EOS token) to break the boundary.
            if generated.size(1) % args.chunk_size == 0:
                extra = torch.tensor([[tokenizer.eos_token_id]], device=args.device)
                generated = torch.cat([generated, extra], dim=1)
            
            # L is the length of the current sequence.
            L = generated.size(1)
            remainder = L % args.chunk_size
            pad_len = (args.chunk_size - remainder) if remainder != 0 else 0
            
            # Pad to the next multiple of chunk_size if needed.
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.eos_token_id, dtype=torch.long, device=args.device)
                padded_generated = torch.cat([generated, pad_ids], dim=1)
            else:
                padded_generated = generated
            
            # Run the model.
            outputs = model(padded_generated)
            
            # Determine the offset within the last chunk.
            last_chunk_start = (L // args.chunk_size) * args.chunk_size
            offset = L - last_chunk_start - 1  # offset in [0, chunk_size-1)
            
            # Handle different logits shapes.
            if outputs.logits.dim() == 3:
                # Shape: (batch, chunk_size-1, vocab_size)
                logits = outputs.logits[:, offset, :]  # (batch, vocab_size)
            elif outputs.logits.dim() == 4:
                # Shape: (batch, num_chunks-1, chunk_size-1, vocab_size)
                # Take the last chunk and then index the offset.
                logits = outputs.logits[:, -1, offset, :]  # (batch, vocab_size)
            else:
                raise ValueError("Unexpected logits dimensions")
            
            # Greedy decoding: choose the token with maximum logit.
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Expected shape: (batch, 1)
            
            # Append the predicted token.
            generated = torch.cat([generated, next_token], dim=1)
            
            # Optionally, stop if EOS token is generated.
            if next_token.item() == tokenizer.eos_token_id:
                break

    end_time = time.time()
    total_time = end_time - start_time
    num_generated = generated.size(1) - input_ids.size(1)
    print("Generated text:", tokenizer.decode(generated[0]))
    print(f"Generated {num_generated} tokens in {total_time:.3f} seconds "
          f"({num_generated/total_time:.2f} tokens/sec)")

if __name__ == "__main__":
    main()