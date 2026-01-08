#!/usr/bin/env python3
"""
Custom checkpoint conversion script that handles DCP format properly.
This is a workaround for the metadata lookup issue in the default converter.

Usage:
    python scripts/convert_checkpoint.py \
        --checkpoint_dir flame/exp/gla-170M-wikitext103-comparison-258221/checkpoint/step-30000 \
        --config flame/configs/gla_170M.json \
        --output_dir flame/exp/gla-170M-wikitext103-comparison-258221/hf-30000
"""

import argparse
import os
import sys
import json

import torch
from safetensors.torch import save_file


def load_dcp_checkpoint(checkpoint_dir):
    """Load distributed checkpoint using torch.distributed.checkpoint."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
    from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.metadata import Metadata
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    # List checkpoint files
    files = os.listdir(checkpoint_dir)
    print(f"Files in checkpoint: {files}")
    
    # Try to load using the raw distcp file
    distcp_files = [f for f in files if f.endswith('.distcp')]
    metadata_file = os.path.join(checkpoint_dir, '.metadata')
    
    if not distcp_files:
        raise ValueError(f"No .distcp files found in {checkpoint_dir}")
    
    # Load metadata
    if os.path.exists(metadata_file):
        print("Loading metadata...")
        metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
        print(f"Metadata keys: {metadata.keys() if hasattr(metadata, 'keys') else type(metadata)}")
    
    # Try direct loading of the distcp file
    distcp_path = os.path.join(checkpoint_dir, distcp_files[0])
    print(f"Loading distcp file: {distcp_path}")
    
    # Load the raw checkpoint data
    state_dict = torch.load(distcp_path, map_location='cpu', weights_only=False)
    
    return state_dict


def load_dcp_with_fsspec(checkpoint_dir):
    """Alternative method using fsspec reader."""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.state_dict_loader import load
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
    
    print(f"Trying fsspec method for: {checkpoint_dir}")
    
    # Create empty state dict to load into
    state_dict = {}
    
    try:
        # Use the standard DCP load
        reader = FileSystemReader(checkpoint_dir)
        load(state_dict, reader)
        return state_dict
    except Exception as e:
        print(f"fsspec method failed: {e}")
        return None


def extract_model_state_dict(state_dict):
    """Extract model weights from the full checkpoint state dict."""
    if 'model' in state_dict:
        return state_dict['model']
    
    # Sometimes the state dict is nested differently
    model_keys = [k for k in state_dict.keys() if 'model' in k.lower()]
    if model_keys:
        print(f"Found model-related keys: {model_keys}")
    
    # If no 'model' key, assume the state dict IS the model weights
    # Filter out optimizer and other state
    model_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # Clean up key names if needed
            clean_key = k
            if clean_key.startswith('model.'):
                clean_key = clean_key[6:]  # Remove 'model.' prefix
            model_state[clean_key] = v
    
    return model_state if model_state else state_dict


def convert_checkpoint(checkpoint_dir, config_path, output_dir, tokenizer_path='gpt2'):
    """Convert DCP checkpoint to HuggingFace format."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and save config
    print(f"Loading config from: {config_path}")
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    config.save_pretrained(output_dir)
    
    # Load and save tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)
    
    # Try different loading methods
    state_dict = None
    
    # Method 1: Direct distcp load
    try:
        state_dict = load_dcp_checkpoint(checkpoint_dir)
        print("Loaded using direct method")
    except Exception as e:
        print(f"Direct method failed: {e}")
    
    # Method 2: fsspec reader
    if state_dict is None:
        state_dict = load_dcp_with_fsspec(checkpoint_dir)
        if state_dict:
            print("Loaded using fsspec method")
    
    if state_dict is None:
        raise RuntimeError("Failed to load checkpoint with all methods")
    
    print(f"State dict keys: {list(state_dict.keys())[:10]}...")
    
    # Extract model weights
    model_state = extract_model_state_dict(state_dict)
    print(f"Model state dict has {len(model_state)} keys")
    
    # Create model and load weights
    print("Creating model from config...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Try to load state dict
    try:
        model.load_state_dict(model_state, strict=True)
        print("Loaded state dict with strict=True")
    except Exception as e:
        print(f"Strict loading failed: {e}")
        print("Trying with strict=False...")
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
    
    # Save model
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    
    print("Conversion complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory (e.g., .../checkpoint/step-30000)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for HuggingFace model")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Tokenizer to use")
    args = parser.parse_args()
    
    convert_checkpoint(
        args.checkpoint_dir,
        args.config,
        args.output_dir,
        args.tokenizer
    )


if __name__ == "__main__":
    main()

