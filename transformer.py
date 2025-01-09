import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from transformers import PreTrainedModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutput

#imports for training
from transformers import TrainingArguments, Trainer
from transformers.optimization import AdamW, get_scheduler

from sklearn.model_selection import train_test_split

#synthetic dataset imports
from synthetic import RandomVectorDataset, FixedRotationDataset, LinearDynamicsDataset, RNN_TANH_Dataset
import argparse

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
                 input_dim=10,   # input vector dimension
                 **kwargs):
        super().__init__(vocab_size=vocab_size,
                         n_positions=n_positions,
                         n_embd=n_embd,
                         n_layer=n_layer,
                         n_head=n_head,
                         **kwargs)
        self.input_dim = input_dim


class VectorGPTModel(PreTrainedModel):
    """
    A GPT2-like model that:
      - Maps input_dim input vectors -> hidden_dim
      - Applies GPT2 blocks (decoder-only, causal)
      - Projects back to 200-d for next-step prediction
    """
    config_class = VectorGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.input_dim = config.input_dim
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
        
        return CausalLMOutput(
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
        outputs = model(**inputs)  # Forward pass without labels
        
        # Extract the logits
        logits = outputs.logits  # (bsz, seq_len, vocab_size)
        
        if labels is not None:
            # Create a mask for every other token, the size of the mask is seq_len - 1 because we are predicting the next token
            # TODO: adjust the mask for deep networks
            mask = torch.arange(labels.size(1)-1) % 2 == 1  # (seq_len - 1,)
            mask = mask.to(labels.device)  # Ensure mask is on the same device as labels
            #print('mask: ', mask)
            
            # Apply the mask to predictions and labels
            pred = logits[:, :-1, :]  # (bsz, seq_len, vocab_size)
            tgt = labels[:, 1:, :]  # (bsz, seq_len, vocab_size)
            
            pred = pred[:,mask,:]
            tgt = tgt[:,mask,:]

            huber_fn = nn.HuberLoss()
            loss = huber_fn(pred, tgt)
        else:
            loss = None

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
    #parse args
    parser = argparse.ArgumentParser(description='training a GPT model on synthetic data')
    parser.add_argument('--data', type=str, default='random',
                    help='options are [random, rotation, LDS, RNN_TANH]')
    parser.add_argument('--input_dim', type=int, default=10,
                    help='integer input dimension')
    parser.add_argument('--num_samples', type=int, default=10000,
                    help='number of sequences each of seq_len')
    parser.add_argument('--seq_len', type=int, default=4,
                    help='length of each sequence')
    
    args = parser.parse_args()
    print('args: ', args)
    input_dim = args.input_dim
    num_samples = args.num_samples
    seq_len = args.seq_len

    if args.data == 'random': 
        dataset = RandomVectorDataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)
    if args.data =='rotation':
        dataset = FixedRotationDataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)
    if args.data == 'LDS':
        dataset = LinearDynamicsDataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)
    if args.data == 'RNN_TANH': 
        dataset = RNN_TANH_Dataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)

    
    #TODO: RNN dataset is wrong order (seq_len, batch_size, input_dim) instead of (batch_size, seq_len, input_dim)
    print('number of samples in dataset: ', len(dataset.data))
    print('datapoint shape: ', dataset.data[0].shape)

    # Train/Validation/Test split
    valid_size = min(int(0.15 * len(dataset)), 100)
    test_size = min(int(0.15 * len(dataset)), 100)
    train_size = len(dataset) - valid_size - test_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    # 2. Model configuration and instantiation
    config = VectorGPTConfig(
        n_positions=64,  # must be >= 30 (seq_len)
        n_embd=768,      # hidden dimension
        n_layer=12,       # transformer layers
        n_head=12,        # attention heads
        input_dim=input_dim,  # input vector dimension
    )
    model = VectorGPTModel(config)

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir="./vector_gpt_trainer",  # Directory to save checkpoints
        save_prefix=f"{args.data}_{args.input_dim}_{args.num_samples}_{args.seq_len}",
        save_steps=500,                    # Save checkpoint every 500 steps
        overwrite_output_dir=True,         # Overwrite existing output dir
        eval_strategy="epoch",             # Evaluate at the end of each epoch
        save_strategy="epoch",             # Save checkpoints every epoch
        logging_dir="./logs",              # Directory for TensorBoard logs
        logging_steps=10,                  # Log every 10 steps for finer feedback
        save_total_limit=3,                # Keep only the last 3 checkpoints
        load_best_model_at_end=True,       # Load the best model based on validation loss
        metric_for_best_model="eval_loss", # Use validation loss for checkpoint selection
        greater_is_better=False,           # Lower loss is better
        learning_rate=3e-4,                # Lower learning rate for stability
        weight_decay=0.01,                 # Weight decay for regularization
        adam_beta1=0.9,                    # First momentum parameter
        adam_beta2=0.98,                   # Second momentum parameter
        adam_epsilon=1e-6,                 # Epsilon for numerical stability
        warmup_steps=300,                  # Reduced warmup steps
        per_device_train_batch_size=16,    # Batch size per GPU during training
        per_device_eval_batch_size=16,     # Batch size per GPU during evaluation
        gradient_accumulation_steps=1,     # Accumulate gradients over 1 step
        fp16=True,                         # Use mixed precision (FP16)
        max_grad_norm=1.0,                 # Gradient clipping
        num_train_epochs=1,                # Fewer epochs to prevent overfitting
        report_to="tensorboard",           # Log to TensorBoard
        seed=42,                           # Set seed for reproducibility
    )


    # 4. Custom trainer
    trainer = VectorGPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=None,  # Not needed for vector-based tasks
    )

    # 5. Start training
    trainer.train()

    # 6. Optional: Evaluate after training
    # Evaluate on test dataset
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)

"""     # 3. Training arguments for Vision Transformer
    training_args = TrainingArguments(
        output_dir="./vector_gpt_trainer",  # Directory to save checkpoints
        save_steps=1000,                    # Save checkpoint every 500 steps
        overwrite_output_dir=True,         # Overwrite existing output dir
        eval_strategy="epoch",             # Evaluate at the end of each epoch
        save_strategy="epoch",             # Save checkpoints every epoch
        logging_dir="./logs",              # Directory for TensorBoard logs
        logging_steps=10,                  # Log every 10 steps for finer feedback
        save_total_limit=5,                # Keep only the last 3 checkpoints
        load_best_model_at_end=True,       # Load the best model based on validation loss
        metric_for_best_model="eval_loss", # Use validation loss for checkpoint selection
        greater_is_better=False,           # Lower loss is better
        learning_rate=5e-5,                # Lower learning rate for stability
        weight_decay=0.01,                 # Weight decay for regularization
        adam_beta1=0.9,                    # First momentum parameter
        adam_beta2=0.999,                   # Second momentum parameter
        adam_epsilon=1e-8,                 # Epsilon for numerical stability
        warmup_steps=500,                  # Reduced warmup steps
        per_device_train_batch_size=32,    # Batch size per GPU during training
        per_device_eval_batch_size=32,     # Batch size per GPU during evaluation
        gradient_accumulation_steps=2,     # Accumulate gradients over 1 step
        fp16=True,                         # Use mixed precision (FP16)
        max_grad_norm=1.0,                 # Gradient clipping
        num_train_epochs=1,                # Fewer epochs to prevent overfitting
        report_to="tensorboard",           # Log to TensorBoard
        seed=42,                           # Set seed for reproducibility
    ) """
