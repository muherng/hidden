#!/usr/bin/env python3
"""
Verify parameter counts for fair comparison between:
1. TransformerScanModel (GPT2-base, T1=1, T2=12)
2. GLA-170M

Run: python scripts/verify_model_params.py
"""

import sys
sys.path.insert(0, '/data/lingo/morrisyau/hidden')

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)

print("=" * 60)
print("Model Parameter Comparison")
print("=" * 60)

# 1. TransformerScanModel
print("\n1. TransformerScanModel (your model)")
print("-" * 40)
try:
    from transformers import GPT2Config, GPT2Tokenizer
    from models.tree_model6 import TransformerScanModel
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")
    config.vocab_size = tokenizer.vocab_size
    
    model = TransformerScanModel(
        config, 
        chunk_size=512, 
        T1_num_layers=1, 
        T2_num_layers=12,
        train_mode="sequential"
    )
    
    total = count_params(model)
    print(f"   Config: GPT2-base (hidden=768, layers=12, heads=12)")
    print(f"   T1 layers: 1")
    print(f"   T2 layers: 12")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Total Parameters: {format_params(total)} ({total:,})")
except Exception as e:
    print(f"   Error loading TransformerScanModel: {e}")

# 2. GLA-170M
print("\n2. GLA-170M (comparison model)")
print("-" * 40)
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    import json
    
    # Load config
    with open('/data/lingo/morrisyau/hidden/flame/configs/gla_170M.json') as f:
        gla_config = json.load(f)
    
    print(f"   Config: hidden={gla_config['hidden_size']}, layers={gla_config['num_hidden_layers']}, heads={gla_config['num_heads']}")
    print(f"   Vocab size: {gla_config['vocab_size']}")
    
    # Try to load and count
    config = AutoConfig.from_pretrained('/data/lingo/morrisyau/hidden/flame/configs/gla_170M.json', trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    total = count_params(model)
    print(f"   Total Parameters: {format_params(total)} ({total:,})")
except Exception as e:
    print(f"   Error loading GLA model: {e}")
    print("   (This is expected if fla is not properly installed)")
    
    # Manual estimation
    h = 768  # hidden_size
    L = 12   # num_layers
    V = 50257  # vocab_size
    r = 4    # hidden_ratio
    
    # Rough estimate for GLA
    embed = V * h  # token embedding
    pos_embed = 0  # GLA typically uses RoPE (no position embedding)
    per_layer = (
        3 * h * h +  # Q, K, V projections (approx)
        h * h +      # output projection
        2 * h * (h * r)  # FFN (gate + up + down)
    )
    output = V * h  # output projection (if not tied)
    
    estimated = embed + L * per_layer + output
    print(f"   Estimated Parameters: {format_params(estimated)} ({estimated:,})")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("Both models should be approximately 160-180M parameters")
print("for a fair comparison.")
print("=" * 60)

