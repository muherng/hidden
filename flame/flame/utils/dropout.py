"""
Dropout utilities for fla models.

This module provides functions to add dropout to fla models (GLA, Gated DeltaNet, etc.)
using PyTorch forward hooks. This approach doesn't require modifying the fla source code.

Dropout locations match GPT2 Small for fair comparison:
1. Embedding dropout (embd_pdrop): After embeddings
2. Q/K/V dropout (attn_pdrop equivalent): After Q, K, V projections
3. Residual dropout (resid_pdrop): After attention output, before residual add
4. Residual dropout (resid_pdrop): After MLP output, before residual add

Usage:
    from flame.utils.dropout import add_dropout_to_model
    
    model = load_model(...)
    model = add_dropout_to_model(model, dropout_rate=0.1)
"""

import torch.nn as nn
from typing import Any, Tuple, Optional


def add_dropout_to_model(
    model: nn.Module, 
    dropout_rate: float = 0.1,
    enable_embedding_dropout: bool = True,
    enable_qkv_dropout: bool = True,
    enable_residual_dropout: bool = True,
) -> nn.Module:
    """
    Add dropout to an fla model using forward hooks.
    
    Dropout is applied at locations comparable to GPT2 Small:
    1. After embeddings (embedding dropout)
    2. After Q, K, V projections (attention dropout equivalent)
    3. After attention output (residual dropout)
    4. After MLP output (residual dropout)
    
    Args:
        model: An fla model (GLAForCausalLM, GatedDeltaNetForCausalLM, etc.)
        dropout_rate: Dropout probability (default: 0.1)
        enable_embedding_dropout: Whether to add embedding dropout (default: True)
        enable_qkv_dropout: Whether to add Q/K/V projection dropout (default: True)
        enable_residual_dropout: Whether to add residual dropout (default: True)
    
    Returns:
        The same model with dropout hooks registered (modified in-place)
    
    Note:
        - Dropout only applies during training (model.train())
        - Each location gets independent Dropout instances (different random masks)
    """
    if dropout_rate <= 0:
        # No dropout requested, return model unchanged
        return model
    
    # Find the model backbone and layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        backbone = model.model
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        backbone = model
        layers = model.layers
    else:
        raise ValueError(
            f"Cannot find layers in model. Expected model.model.layers or model.layers. "
            f"Got model type: {type(model)}"
        )
    
    num_hooks = 0
    
    # 1. Embedding dropout (after embeddings, like GPT2's embd_pdrop)
    if enable_embedding_dropout:
        if hasattr(backbone, 'embeddings'):
            emb_dropout = nn.Dropout(dropout_rate)
            backbone.embeddings.register_forward_hook(_make_dropout_hook(emb_dropout))
            num_hooks += 1
            print(f"[Dropout] Added embedding dropout (rate={dropout_rate})")
        else:
            print("[Dropout] Warning: No embeddings module found, skipping embedding dropout")
    
    # Process each layer
    for layer_idx, layer in enumerate(layers):
        # Check if this layer has attn and mlp modules
        if not hasattr(layer, 'attn') or not hasattr(layer, 'mlp'):
            print(f"[Dropout] Warning: Layer {layer_idx} missing attn or mlp, skipping")
            continue
        
        attn = layer.attn
        
        # 2. Q/K/V projection dropout (equivalent to GPT2's attn_pdrop)
        if enable_qkv_dropout:
            # Add dropout after Q projection
            if hasattr(attn, 'q_proj'):
                q_dropout = nn.Dropout(dropout_rate)
                attn.q_proj.register_forward_hook(_make_dropout_hook(q_dropout))
                num_hooks += 1
            
            # Add dropout after K projection
            if hasattr(attn, 'k_proj'):
                k_dropout = nn.Dropout(dropout_rate)
                attn.k_proj.register_forward_hook(_make_dropout_hook(k_dropout))
                num_hooks += 1
            
            # Add dropout after V projection
            if hasattr(attn, 'v_proj'):
                v_dropout = nn.Dropout(dropout_rate)
                attn.v_proj.register_forward_hook(_make_dropout_hook(v_dropout))
                num_hooks += 1
        
        # 3 & 4. Residual dropout (after attention and MLP outputs)
        if enable_residual_dropout:
            # Create separate Dropout instances for attention and MLP
            attn_dropout = nn.Dropout(dropout_rate)
            mlp_dropout = nn.Dropout(dropout_rate)
            
            # Register hooks on attention and MLP outputs
            layer.attn.register_forward_hook(_make_dropout_hook(attn_dropout))
            layer.mlp.register_forward_hook(_make_dropout_hook(mlp_dropout))
            num_hooks += 2
    
    # Summary
    qkv_hooks = 3 * len(layers) if enable_qkv_dropout else 0
    resid_hooks = 2 * len(layers) if enable_residual_dropout else 0
    emb_hooks = 1 if enable_embedding_dropout else 0
    
    print(f"[Dropout] Summary (rate={dropout_rate}):")
    print(f"  - Embedding dropout: {emb_hooks} hook(s)")
    print(f"  - Q/K/V dropout: {qkv_hooks} hooks ({len(layers)} layers × 3)")
    print(f"  - Residual dropout: {resid_hooks} hooks ({len(layers)} layers × 2)")
    print(f"  - Total: {num_hooks} dropout hooks")
    
    return model


def add_dropout_to_model_legacy(model: nn.Module, dropout_rate: float = 0.1) -> nn.Module:
    """
    Legacy function: Add only residual dropout (original behavior).
    
    This applies dropout only after attention and MLP outputs, matching the
    original implementation before GPT2 parity was added.
    
    Args:
        model: An fla model (GLAForCausalLM, GatedDeltaNetForCausalLM, etc.)
        dropout_rate: Dropout probability (default: 0.1)
    
    Returns:
        The same model with dropout hooks registered (modified in-place)
    """
    return add_dropout_to_model(
        model, 
        dropout_rate=dropout_rate,
        enable_embedding_dropout=False,
        enable_qkv_dropout=False,
        enable_residual_dropout=True,
    )


def _make_dropout_hook(dropout_module: nn.Dropout):
    """
    Create a forward hook function that applies dropout.
    
    The hook handles both:
    - Tuple outputs: (hidden_states, attentions, past_key_values) from attention
    - Tensor outputs: hidden_states from MLP or projections
    
    Args:
        dropout_module: nn.Dropout instance to apply
    
    Returns:
        A hook function compatible with register_forward_hook
    """
    def hook(module: nn.Module, input: Tuple[Any, ...], output: Any) -> Any:
        # Only apply dropout during training
        if not module.training:
            return output
        
        if isinstance(output, tuple):
            # Attention output: (hidden_states, attentions, past_key_values)
            # Apply dropout only to hidden_states (first element)
            hidden_states = output[0]
            hidden_states = dropout_module(hidden_states)
            return (hidden_states,) + output[1:]
        else:
            # MLP/projection output: just hidden_states tensor
            return dropout_module(output)
    
    return hook


def remove_dropout_hooks(model: nn.Module) -> nn.Module:
    """
    Remove all forward hooks from a model.
    
    This is useful if you want to disable dropout after training.
    Note: This removes ALL forward hooks, not just dropout hooks.
    
    Args:
        model: Model with hooks registered
    
    Returns:
        Model with hooks removed
    """
    for module in model.modules():
        module._forward_hooks.clear()
    
    print("[Dropout] Removed all forward hooks from model")
    return model
