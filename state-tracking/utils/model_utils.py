"""
Common model utilities for setting up tokenizers and models.
"""
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GPT2LMHeadModel, 
    GPT2Config,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2Tokenizer,
    PreTrainedTokenizerFast
)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
import torch
import os
from torch.nn import CrossEntropyLoss

# Import custom model classes
from utils.models import (
    GPT2ModelWithLayerTargets,
    LlamaModelWithLayerTargets,
    PythiaModelWithLayerTargets,
    TreeModel,
    GPT2MaskedLMHeadModel,
    MambaModelWithLayerTargets
)

from utils.tree import TransformerScanModel

def setup_tokenizer(model_name, state_tokens, action_tokens):
    """
    Set up the tokenizer for the given model.
    
    For all models, build a minimal vocabulary containing only:
      - Basic tokens: [PAD], [EOS], [UNK]
      - The state tokens (from state_tokens)
      - The action tokens (from action_tokens)
    """
    print("Loading tokenizer")
    # Create a minimal vocabulary dictionary.
    minimal_vocab = {}
    # Basic tokens.
    minimal_vocab["[PAD]"] = 0
    minimal_vocab["[EOS]"] = 1
    minimal_vocab["[UNK]"] = 2
    next_index = 3
    
    # Add state tokens.
    for token in state_tokens.values():
        if token not in minimal_vocab:
            minimal_vocab[token] = next_index
            next_index += 1

    # Add action tokens.
    for token in action_tokens.values():
        if token not in minimal_vocab:
            minimal_vocab[token] = next_index
            next_index += 1
    
    # Build a fast tokenizer using the tokenizers library.
    tk_model = models.WordLevel(vocab=minimal_vocab, unk_token="[UNK]")
    tk = Tokenizer(tk_model)
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.decoder = decoders.WordPiece()
    
    # Create the Hugging Face Fast tokenizer.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]"
    )
    
    return tokenizer

def setup_model(tokenizer, model_name=None, checkpoint_path=None, use_bfloat16=False, 
                no_pretrain=False, output_dir=None, use_custom_models=False,
                layerwise_supervision_config=None, chunk_size=None,
                T1_num_layers=1, T2_num_layers=1):
    """
    Set up the model for training.
    """
    if model_name:
        # Determine model configuration and class
        if "tree" in model_name.lower():
            config = GPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_positions=4096,
                n_embd=768,
                n_layer=6, #this is overridden by T1_num_layers and T2_num_layers
                n_head=12,
                dropout=0.1
            )
            print('chunk size: ', chunk_size)
            if checkpoint_path is not None: 
                model = TreeModel.from_pretrained(checkpoint_path, config=config, chunk_size=chunk_size,
                                                  T1_num_layers=T1_num_layers, T2_num_layers=T2_num_layers)
            else: 
                model = TreeModel(config, chunk_size=chunk_size,
                                            T1_num_layers=T1_num_layers, T2_num_layers=T2_num_layers)
        elif "mamba" in model_name.lower():
            if checkpoint_path:
                print(f"Loading Mamba model from checkpoint: {checkpoint_path}")
                model = MambaModelWithLayerTargets.from_pretrained(
                    checkpoint_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                    layerwise_supervision_config=layerwise_supervision_config,
                )
            else:
                model = MambaModelWithLayerTargets.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                    layerwise_supervision_config=layerwise_supervision_config,
                )
        elif "gpt" in model_name.lower():
            if checkpoint_path:
                print(f"Loading model from checkpoint: {checkpoint_path}")
                # Load the base model directly from checkpoint
                model = GPT2LMHeadModel.from_pretrained(
                    checkpoint_path,
                    ignore_mismatched_sizes=True
                )
                # Modify the forward pass to handle direct supervision
                original_forward = model.forward
                def new_forward(input_ids, attention_mask=None, labels=None, **kwargs):
                    outputs = original_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,  # Don't pass labels to base model
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    
                    if labels is not None:
                        # Shift labels and logits for direct supervision
                        shift_logits = outputs.logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
                        shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
                        
                        # Compute loss
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        # Add loss to outputs
                        outputs.loss = loss
                        outputs['loss'] = loss
                        
                    return outputs
                model.forward = new_forward
            else:
                # For new models, use default configuration
                config = GPT2Config(
                    vocab_size=tokenizer.vocab_size,
                    n_positions=4096,
                    n_embd=768,
                    n_layer=6,
                    n_head=12,
                    dropout=0.1,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                model = GPT2LMHeadModel(config)
                # Modify the forward pass
                original_forward = model.forward
                def new_forward(input_ids, attention_mask=None, labels=None, **kwargs):
                    outputs = original_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,  # Don't pass labels to base model
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    
                    if labels is not None:
                        # Shift labels and logits for direct supervision
                        shift_logits = outputs.logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
                        shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
                        
                        # Compute loss
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        # Add loss to outputs
                        outputs.loss = loss
                        outputs['loss'] = loss
                        
                    return outputs
                model.forward = new_forward
            
            # Resize token embeddings to match tokenizer
            model.resize_token_embeddings(len(tokenizer))
        elif "pythia" in model_name.lower():
            config_class = AutoConfig
            model_class = PythiaModelWithLayerTargets if use_custom_models else GPTNeoXForCausalLM
        else:
            config_class = LlamaConfig
            model_class = LlamaModelWithLayerTargets if use_custom_models else LlamaForCausalLM
    else:
        model_class = AutoModelForCausalLM
        config_class = AutoConfig
    
    # Check for checkpoints in output directory (for training)
    if not checkpoint_path and output_dir and os.path.exists(output_dir):
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if len(subdirs) > 0:
            # Filter for subdirectories that match the expected pattern "checkpoint-NUMBER"
            valid_subdirs = []
            for subdir in subdirs:
                parts = subdir.split("-")
                if len(parts) >= 2 and parts[0] == "checkpoint" and parts[1].isdigit():
                    valid_subdirs.append(subdir)
            
            if valid_subdirs:
                min_subdir = min(valid_subdirs, key=lambda x: int(x.split("-")[1]))
                checkpoint_path = os.path.join(output_dir, min_subdir)
            else:
                print(f"Warning: Found subdirectories in {output_dir}, but none match the expected 'checkpoint-NUMBER' format.")
    
    # Load or create model
    if "tree" in model_name.lower():
        print('Tree model has custom checkpoint loading separate from other huggingface models')
    elif checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        if use_custom_models:
            if "gpt" in model_name.lower():
                # Load base model
                model = GPT2LMHeadModel.from_pretrained(checkpoint_path, ignore_mismatched_sizes=True)
                # Modify the forward pass
                original_forward = model.forward
                def new_forward(input_ids, attention_mask=None, labels=None, **kwargs):
                    outputs = original_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,  # Don't pass labels to base model
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    
                    if labels is not None:
                        # Shift labels and logits for direct supervision
                        shift_logits = outputs.logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
                        shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
                        
                        # Compute loss
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        # Add loss to outputs
                        outputs.loss = loss
                        outputs['loss'] = loss
                        
                    return outputs
                model.forward = new_forward
            elif "pythia" in model_name.lower():
                model = PythiaModelWithLayerTargets.from_pretrained(
                    checkpoint_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                    layerwise_supervision_config=layerwise_supervision_config,
                )
            else:
                model = LlamaModelWithLayerTargets.from_pretrained(
                    checkpoint_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                    layerwise_supervision_config=layerwise_supervision_config,
                )
        else:
            if "gpt" in model_name.lower():
                # Load base model
                model = GPT2LMHeadModel.from_pretrained(checkpoint_path, ignore_mismatched_sizes=True)
                # Modify the forward pass
                original_forward = model.forward
                def new_forward(input_ids, attention_mask=None, labels=None, **kwargs):
                    outputs = original_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,  # Don't pass labels to base model
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    
                    if labels is not None:
                        # Shift labels and logits for direct supervision
                        shift_logits = outputs.logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
                        shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
                        
                        # Compute loss
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        # Add loss to outputs
                        outputs.loss = loss
                        outputs['loss'] = loss
                        
                    return outputs
                model.forward = new_forward
            elif "pythia" in model_name.lower():
                model = GPTNeoXForCausalLM.from_pretrained(checkpoint_path)
            else:
                model = LlamaForCausalLM.from_pretrained(checkpoint_path)
    elif no_pretrain:
        # Initialize a new model with random weights
        print("Initializing model with random weights")
        import json
        if "gpt" in model_name.lower():
            # Create a custom config with the tokenizer's vocabulary size
            config = GPT2Config(
                vocab_size=len(tokenizer),  # Use the actual tokenizer size
                n_positions=4096,
                n_embd=768,
                n_layer=6,
                n_head=12,
                dropout=0.1,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            print("Config details:\n", json.dumps(config.to_dict(), indent=2))
            model = GPT2LMHeadModel(config)
            # Modify the forward pass
            original_forward = model.forward
            def new_forward(input_ids, attention_mask=None, labels=None, **kwargs):
                outputs = original_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,  # Don't pass labels to base model
                    output_hidden_states=True,
                    return_dict=True,
                    **kwargs
                )
                
                if labels is not None:
                    # Shift labels and logits for direct supervision
                    shift_logits = outputs.logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
                    shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
                    
                    # Compute loss
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    # Add loss to outputs
                    outputs.loss = loss
                    outputs['loss'] = loss
                    
                return outputs
            model.forward = new_forward
        elif "pythia" in model_name.lower():
            config = config_class.from_pretrained(model_name)
            print("Config details:\n", json.dumps(config.to_dict(), indent=2))
            if use_custom_models:
                config.torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
                model = PythiaModelWithLayerTargets(config, layerwise_supervision_config=layerwise_supervision_config)
            else:
                model = GPTNeoXForCausalLM(config)
        else:
            config = config_class.from_pretrained(model_name)
            print("Config details:\n", json.dumps(config.to_dict(), indent=2))
            if use_custom_models:
                config.torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
                model = LlamaModelWithLayerTargets(config, layerwise_supervision_config=layerwise_supervision_config)
            else:
                model = LlamaForCausalLM(config)
    else:
        # Load pre-trained model
        if use_custom_models:
            if "gpt" in model_name.lower():
                # Create a custom config with the tokenizer's vocabulary size
                config = GPT2Config(
                    vocab_size=len(tokenizer),  # Use the actual tokenizer size
                    n_positions=4096,
                    n_embd=768,
                    n_layer=6,
                    n_head=12,
                    dropout=0.1,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                model = GPT2LMHeadModel(config)
                # Modify the forward pass
                original_forward = model.forward
                def new_forward(input_ids, attention_mask=None, labels=None, **kwargs):
                    outputs = original_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,  # Don't pass labels to base model
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    
                    if labels is not None:
                        # Shift labels and logits for direct supervision
                        shift_logits = outputs.logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
                        shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
                        
                        # Compute loss
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        # Add loss to outputs
                        outputs.loss = loss
                        outputs['loss'] = loss
                        
                    return outputs
                model.forward = new_forward
            elif "pythia" in model_name.lower():
                model = PythiaModelWithLayerTargets.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                    layerwise_supervision_config=layerwise_supervision_config,
                )
            else:
                model = LlamaModelWithLayerTargets.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                    layerwise_supervision_config=layerwise_supervision_config,
                )
        else:
            if "gpt" in model_name.lower():
                # Create a custom config with the tokenizer's vocabulary size
                config = GPT2Config(
                    vocab_size=len(tokenizer),  # Use the actual tokenizer size
                    n_positions=4096,
                    n_embd=768,
                    n_layer=6,
                    n_head=12,
                    dropout=0.1,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                model = GPT2LMHeadModel(config)
                # Modify the forward pass
                original_forward = model.forward
                def new_forward(input_ids, attention_mask=None, labels=None, **kwargs):
                    outputs = original_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,  # Don't pass labels to base model
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    
                    if labels is not None:
                        # Shift labels and logits for direct supervision
                        shift_logits = outputs.logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
                        shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
                        
                        # Compute loss
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        # Add loss to outputs
                        outputs.loss = loss
                        outputs['loss'] = loss
                        
                    return outputs
                model.forward = new_forward
            elif "pythia" in model_name.lower():
                model = GPTNeoXForCausalLM.from_pretrained(model_name)
            else:
                model = LlamaForCausalLM.from_pretrained(model_name)
    
    # Resize token embeddings to match tokenizer
    print('len tokenizer: ', len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    
    # Set up precision
    if use_bfloat16:
        model = model.to(torch.bfloat16)
    
    # Enable gradient checkpointing for multi-GPU training (for training)
    if use_custom_models and torch.cuda.device_count() > 1:
        model.gradient_checkpointing_enable()
        model = torch.nn.DataParallel(model)
        model.module.gradient_checkpointing_enable()
    
    model = model.to("cuda")
    return model 