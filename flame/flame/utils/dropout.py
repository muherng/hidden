"""
Dropout utilities for fla models.

This module provides functions to add dropout to fla models (GLA, Gated DeltaNet, etc.)
using PyTorch forward hooks. This approach doesn't require modifying the fla source code.

Usage:
    from flame.utils.dropout import add_dropout_to_model
    
    model = load_model(...)
    model = add_dropout_to_model(model, dropout_rate=0.1)
"""

import torch.nn as nn
from typing import Any, Tuple


def add_dropout_to_model(model: nn.Module, dropout_rate: float = 0.1) -> nn.Module:
    """
    Add dropout to an fla model using forward hooks.
    
    Dropout is applied after attention and MLP outputs in each transformer layer,
    before the residual connection. This matches the standard transformer dropout
    placement (as in GPT-2, BERT, etc.).
    
    Args:
        model: An fla model (GLAForCausalLM, GatedDeltaNetForCausalLM, etc.)
        dropout_rate: Dropout probability (default: 0.1)
    
    Returns:
        The same model with dropout hooks registered (modified in-place)
    
    Note:
        - Dropout only applies during training (model.train())
        - Each layer gets independent Dropout instances (different random masks)
        - Hooks are registered on `layer.attn` and `layer.mlp` modules
    """
    if dropout_rate <= 0:
        # No dropout requested, return model unchanged
        return model
    
    # Find the layers - fla models typically have model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError(
            f"Cannot find layers in model. Expected model.model.layers or model.layers. "
            f"Got model type: {type(model)}"
        )
    
    num_hooks = 0
    
    for layer_idx, layer in enumerate(layers):
        # Check if this layer has attn and mlp modules
        if not hasattr(layer, 'attn') or not hasattr(layer, 'mlp'):
            print(f"Warning: Layer {layer_idx} missing attn or mlp, skipping dropout")
            continue
        
        # Create separate Dropout instances for attention and MLP
        # Each instance maintains independent random state
        attn_dropout = nn.Dropout(dropout_rate)
        mlp_dropout = nn.Dropout(dropout_rate)
        
        # Register hooks
        layer.attn.register_forward_hook(_make_dropout_hook(attn_dropout))
        layer.mlp.register_forward_hook(_make_dropout_hook(mlp_dropout))
        num_hooks += 2
    
    print(f"[Dropout] Added {num_hooks} dropout hooks (rate={dropout_rate}) to {len(layers)} layers")
    
    return model


def _make_dropout_hook(dropout_module: nn.Dropout):
    """
    Create a forward hook function that applies dropout.
    
    The hook handles both:
    - Tuple outputs: (hidden_states, attentions, past_key_values) from attention
    - Tensor outputs: hidden_states from MLP
    
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
            # MLP output: just hidden_states tensor
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
