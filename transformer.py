import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2LayerNorm
from transformers.modeling_outputs import CausalLMOutput

# ----------------------------------------------------
# 1. Synthetic Dataset of Random Vectors
# ----------------------------------------------------
class RandomVectorDataset(Dataset):
    """
    Each sample is a sequence of length seq_len (30), where each vector has dimension vector_dim (200).
    We'll generate random floats in [0,1).
    """
    def __init__(self, num_samples=1000, seq_len=30, vector_dim=200):
        super().__init__()
        self.seq_len = seq_len
        self.vector_dim = vector_dim
        self.data = []
        for _ in range(num_samples):
            # shape: (seq_len, vector_dim)
            seq = torch.rand(seq_len, vector_dim)
            self.data.append(seq)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        We'll return a dictionary with 'inputs' and 'labels'.
        For autoregressive next-step prediction:
          inputs: shape (seq_len, vector_dim)
          labels: same shape (seq_len, vector_dim)
        """
        seq = self.data[idx]
        return {
            "inputs": seq,     # (30, 200)
            "labels": seq      # also (30, 200) - we'll do shift-by-1 in the model
        }

# ----------------------------------------------------
# 2. GPT2-Based Model for Next-Vector Prediction
# ----------------------------------------------------
class VectorGPTConfig(GPT2Config):
    """
    Custom config that reuses GPT2 internals.
    We're not really using 'vocab_size', but GPT2Config requires it.
    """
    def __init__(self,
                 vocab_size=1,  # dummy
                 n_positions=64,  # must be >= seq_len (30)
                 n_embd=256,     # hidden dimension
                 n_layer=4,      # number of transformer layers
                 n_head=4,       # number of attention heads
                 **kwargs):
        super().__init__(vocab_size=vocab_size,
                         n_positions=n_positions,
                         n_embd=n_embd,
                         n_layer=n_layer,
                         n_head=n_head,
                         **kwargs)


class VectorGPTModel(PreTrainedModel):
    """
    A GPT2-like model that:
      - Maps 200-d input vectors -> hidden_dim
      - Applies GPT2 blocks (decoder-only, causal)
      - Projects back to 200-d for next-step prediction
    """
    config_class = VectorGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.input_dim = 200
        self.hidden_dim = config.n_embd
        
        # 1. Input embedding: map 200 -> hidden_dim
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 2. Positional embeddings: (n_positions, hidden_dim)
        self.position_embedding = nn.Embedding(config.n_positions, self.hidden_dim)
        
        # 3. GPT2 blocks
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        
        # 4. Final layer norm
        self.ln_f = GPT2LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        
        # 5. Output projection: hidden_dim -> 200
        self.output_projection = nn.Linear(self.hidden_dim, self.input_dim)
        
        self.post_init()  # HF utility to initialize weights

    def forward(
        self,
        inputs: torch.FloatTensor,  # (batch_size, seq_len, 200)
        labels: torch.FloatTensor = None
    ):
        bsz, seq_len, _ = inputs.shape
        
        # 1. Create position ids [0..seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)  # (bsz, seq_len)
        
        # 2. Embed inputs + positional embeddings
        hidden_states = self.input_embedding(inputs)              # (bsz, seq_len, hidden_dim)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        
        # 3. Pass through GPT2 blocks (causal)
        for block in self.h:
            outputs = block(hidden_states, use_cache=False)
            hidden_states = outputs[0]
        
        hidden_states = self.ln_f(hidden_states)                  # (bsz, seq_len, hidden_dim)
        
        # 4. Project back to 200-d
        logits = self.output_projection(hidden_states)            # (bsz, seq_len, 200)
        
        loss = None
        if labels is not None:
            # Typical shift-by-1 for next-step prediction:
            # the output at t matches label at t+1
            pred = logits[:, :-1, :]                              # (bsz, seq_len-1, 200)
            tgt  = labels[:, 1:, :]                               # (bsz, seq_len-1, 200)
            
            mse_fn = nn.MSELoss()
            loss = mse_fn(pred, tgt)
        
        return CausalLMOutput(
            loss=loss,
            logits=logits
        )

# ----------------------------------------------------
# Quick Demo (Optional)
# ----------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic dataset
    dataset = RandomVectorDataset(num_samples=4, seq_len=30, vector_dim=200)
    
    # Just take one sample to see shapes
    sample = dataset[0]
    inputs = sample["inputs"].unsqueeze(0)  # (1, 30, 200)
    labels = sample["labels"].unsqueeze(0)  # (1, 30, 200)
    
    # Create a small model config & model
    config = VectorGPTConfig(
        n_positions=64,   # must be >= 30
        n_embd=128,
        n_layer=2,
        n_head=4
    )
    model = VectorGPTModel(config)
    
    # Forward pass
    out = model(inputs=inputs, labels=labels)
    print("Logits shape:", out.logits.shape)  # (1, 30, 200)
    print("Loss:", out.loss)                  # MSE over shifted positions
