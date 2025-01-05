import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutput

#imports for training
from transformers import TrainingArguments, Trainer
from transformers.optimization import AdamW, get_scheduler

from sklearn.model_selection import train_test_split

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
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
        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        
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
# 1. Custom Trainer Class
# ----------------------------------------------------
class VectorGPTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation: Ensure a scalar tensor is returned for loss.
        """
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)  # Forward pass with explicit labels
        
        # Extract the loss (already computed in the model)
        loss = outputs.loss  # This is a scalar tensor

        if return_outputs:
            return loss, outputs  # Return both loss and outputs if requested
        else:
            return loss  # Return only the scalar loss


    def create_optimizer(self):
        """
        Custom optimizer: PyTorch AdamW with decoupled weight decay.
        """
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Custom scheduler: Cosine annealing with warmup.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=optimizer if optimizer else self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

# ----------------------------------------------------
# 2. Training Script
# ----------------------------------------------------
if __name__ == "__main__":
    # 1. Synthetic dataset
    dataset = RandomVectorDataset(num_samples=1000, seq_len=30, vector_dim=200)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    # Train/Validation/Test split
    train_size = int(0.7 * len(dataset))  # 70% training
    valid_size = int(0.15 * len(dataset))  # 15% validation
    test_size = len(dataset) - train_size - valid_size  # 15% test

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 2. Model configuration and instantiation
    config = VectorGPTConfig(
        n_positions=64,  # must be >= 30 (seq_len)
        n_embd=256,      # hidden dimension
        n_layer=6,       # transformer layers
        n_head=8,        # attention heads
    )
    model = VectorGPTModel(config)

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir="./vector_gpt_trainer",  # Directory to save checkpoints
        overwrite_output_dir=True,         # Overwrite existing output dir
        eval_strategy="epoch",       # Evaluate at the end of each epoch
        save_strategy="epoch",             # Save checkpoints every epoch
        logging_dir="./logs",              # Directory for TensorBoard logs
        logging_steps=50,                  # Log every 50 steps
        save_total_limit=3,                # Keep only the last 3 checkpoints
        learning_rate=5e-4,                # Initial learning rate
        weight_decay=0.01,                 # Weight decay for AdamW
        adam_beta1=0.9,                    # First momentum parameter
        adam_beta2=0.98,                   # Second momentum parameter
        adam_epsilon=1e-6,                 # Epsilon for numerical stability
        warmup_steps=500,                  # Warmup steps for scheduler
        per_device_train_batch_size=16,    # Batch size per GPU during training
        per_device_eval_batch_size=16,     # Batch size per GPU during evaluation
        gradient_accumulation_steps=1,     # Accumulate gradients over 1 step
        fp16=True,                         # Use mixed precision (FP16)
        num_train_epochs=10,               # Total number of epochs
        report_to="tensorboard",           # Log to TensorBoard
        seed=42,                           # Set seed for reproducibility
    )

    # 4. Custom trainer
    trainer = VectorGPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=None,  # Not needed for vector-based tasks
    )

    # 5. Start training
    trainer.train()

    # 6. Optional: Evaluate after training
    results = trainer.evaluate()
    print("Evaluation Results:", results)

#TODO's train/test split, evaluation, logging, etc. 
#saving checkpoints. 


# ----------------------------------------------------
# Quick Demo (Optional)
# ----------------------------------------------------
""" if __name__ == "__main__":
    # Create a synthetic dataset
    dataset = RandomVectorDataset(num_samples=4, seq_len=30, vector_dim=200)
    
    # Just take one sample to see shapes
    sample = dataset[0]
    inputs = sample["inputs"].unsqueeze(0).to(device)  # (1, 30, 200)
    labels = sample["labels"].unsqueeze(0).to(device)  # (1, 30, 200)
    
    # Create a small model config & model
    config = VectorGPTConfig(
        n_positions=64,   # must be >= 30
        n_embd=128,
        n_layer=2,
        n_head=4
    )
    model = VectorGPTModel(config).to(device)
    
    # Forward pass
    out = model(inputs=inputs, labels=labels)
    print("Logits shape:", out.logits.shape)  # (1, 30, 200)
    print("Loss:", out.loss)                  # MSE over shifted positions """
