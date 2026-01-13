from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, LlamaForCausalLM, GPTNeoXForCausalLM
from transformers import AutoModelForCausalLM

from utils.tree import TransformerScanModel 



class GPT2MaskedLMHeadModel(GPT2LMHeadModel):
    """GPT2 model with masked language modeling head for selective loss computation."""
    
    def forward(self, input_ids, attention_mask=None, labels=None, labels_mask=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = outputs.logits
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
            shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            if labels_mask is not None:
                # torch gather
                total_loss = 0
                for i in range(shift_labels.size(0)):
                    loss = loss_fct(shift_logits[i].view(-1, shift_logits[i].size(-1)), shift_labels[i].view(-1))
                    total_loss += loss[labels_mask[i, 1:]].mean()
                total_loss /= shift_labels.size(0)
                outputs.logits = shift_logits
            else:
                total_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).mean()
            outputs.loss = total_loss
            outputs['loss'] = total_loss
        return outputs


class LlamaMaskedLMHeadModel(LlamaForCausalLM):
    """Llama model with masked language modeling head for selective loss computation."""
    
    def forward(self, input_ids, attention_mask=None, labels=None, labels_mask=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
            shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            if labels_mask is not None:
                # torch gather
                total_loss = 0
                for i in range(shift_labels.size(0)):
                    loss = loss_fct(shift_logits[i].view(-1, shift_logits[i].size(-1)), shift_labels[i].view(-1))
                    total_loss += loss[labels_mask[i, 1:]].mean()
                total_loss /= shift_labels.size(0)
                outputs.logits = shift_logits
            else:
                total_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).mean()
            outputs.loss = total_loss
            outputs['loss'] = total_loss
        return outputs


class ModelWithLayerTargetsMixin:
    """Mixin for models that need to compute loss on intermediate layer outputs."""
    
    def init_layer_targets(self, config, layerwise_supervision_config=None):
        self.layerwise_supervision_config = layerwise_supervision_config
        if self.layerwise_supervision_config is not None:
            self.layer_classifiers = nn.ModuleDict({
                layer_idx: nn.Linear(config.hidden_size, self.layerwise_supervision_config[layer_idx]["n_label_classes"])
                for layer_idx in self.layerwise_supervision_config
            })
    
    def forward_with_layer_targets(
        self, input_ids,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        loss = 0
        loss_fct = nn.CrossEntropyLoss()
        for i in range(len(hidden_states)):
            if f"layer_{i-1}_labels" in kwargs:
                print('layer', i-1)
                # move labels to correct device to enable model parallelism
                # layer_i_logits = self.lm_head(hidden_states[i])
                classification_logits = self.layer_classifiers[str(i-1)](hidden_states[i])
                # Shift so that tokens < n predict n
                shift_logits = classification_logits[..., :-1, :].contiguous()
                shift_labels = kwargs[f"layer_{i-1}_labels"][..., 1:].contiguous()
                # Flatten the tokens
                loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # final logits
        final_logits = outputs.logits
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        #print('labels: ', labels)
        return loss, outputs


class GPT2ModelWithLayerTargets(ModelWithLayerTargetsMixin, GPT2LMHeadModel):
    """GPT2 model with layer-wise supervision."""
    
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)
    
    def forward(self, *args, **kwargs):
        loss,outputs = self.forward_with_layer_targets(*args, **kwargs)
        outputs["loss"] = loss
        return outputs


class LlamaModelWithLayerTargets(ModelWithLayerTargetsMixin, LlamaForCausalLM):
    """Llama model with layer-wise supervision."""
    
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)
    
    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)


class PythiaModelWithLayerTargets(ModelWithLayerTargetsMixin, GPTNeoXForCausalLM):
    """Pythia model with layer-wise supervision."""
    
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)
    
    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)
    

class TreeModel(TransformerScanModel):
    """Tree model with direct output supervision."""
    
    def __init__(self, config, chunk_size= 32, T1_num_layers = 1, T2_num_layers = 1):
        super().__init__(config, chunk_size=chunk_size, T1_num_layers = T1_num_layers, T2_num_layers = T2_num_layers)
    
    def forward(self, input_ids,
        attention_mask=None,
        labels=None,
        **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=None, output_hidden_states=True, return_dict=True)
        loss = 0
        loss_fct = nn.CrossEntropyLoss()

        # final logits
        final_logits = outputs["logits"]
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Return the loss in the outputs dictionary
        outputs["loss"] = loss
        return outputs


class GPT2DirectSupervisionModel(GPT2LMHeadModel):
    """GPT2 model with direct supervision that handles label shifting correctly."""
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # First get the base model outputs
        outputs = super().forward(
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


class MambaModelWithLayerTargets(AutoModelForCausalLM):
    """Mamba model with layer-wise supervision support."""
    
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.layerwise_supervision_config = layerwise_supervision_config
        self.layer_outputs = {}
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Don't pass labels to base model
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        if labels is not None:
            # Shift labels and logits for direct supervision
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add loss to outputs
            outputs.loss = loss
            outputs['loss'] = loss
            
            # Store layer outputs if layerwise supervision is configured
            if self.layerwise_supervision_config:
                for layer_name in self.layerwise_supervision_config:
                    if hasattr(self, layer_name):
                        layer_output = getattr(self, layer_name)
                        self.layer_outputs[layer_name] = layer_output
        
        return outputs
