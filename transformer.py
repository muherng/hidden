import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader
from transformers import PreTrainedModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutput

#imports for training
from transformers import TrainingArguments, Trainer
from transformers.optimization import AdamW, get_scheduler

from sklearn.model_selection import train_test_split

#synthetic dataset imports
from synthetic import RandomVectorDataset, FixedRotationDataset, LinearDynamicsDataset, LSTM_Dataset
import argparse
import os

import datetime
import numpy as np
from helper import evaluate, kl_loss

import warnings
warnings.filterwarnings(
    "ignore",
    message="`tokenizer` is deprecated and will be removed in version 5.0.0 for `VectorGPTTrainer.__init__`. Use `processing_class` instead.",
    category=FutureWarning
)

# Generate a unique timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
                 ntokens=1,      # number of tokens
                 **kwargs):
        super().__init__(vocab_size=vocab_size,
                         n_positions=n_positions,
                         n_embd=n_embd,
                         n_layer=n_layer,
                         n_head=n_head,
                         **kwargs)
        self.input_dim = input_dim
        self.ntokens = ntokens


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
        self.ntokens = config.ntokens
        
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

        self.decoder = nn.Linear(self.input_dim, self.ntokens)
        
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
    
    def __init__(self, *args, train_loader=None, valid_loader=None, custom_args=None, **kwargs):
        self.mask = custom_args['mask']
        self.mask_out = custom_args['mask_out']
        self.ntoken = custom_args['ntoken']
        self.print_interval = 100
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Load the original model and move it to the device
        model_tgt = torch.load(f'saved_models/LSTM/{input_dim}_{num_layers}model.pt', map_location=device)
        model_tgt.to(device)
        model_tgt.eval()
        self.model_tgt = model_tgt
        super().__init__(*args, **kwargs)
    
    def get_train_dataloader(self):
        """
        Returns the custom DataLoader for training.
        """
        return self.train_loader
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Returns the custom DataLoader for evaluation if provided,
        otherwise fall back to the default behavior.
        """
        if self.valid_loader is not None:
            return self.valid_loader
        else:
            return super().get_eval_dataloader(eval_dataset)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation: Ensure a scalar tensor is returned for loss.
        """
        inputs_copy = inputs.copy()
        labels = inputs.pop("labels", None)
        tokens = inputs.pop("tokens", None)
        outputs = model(**inputs)  # Forward pass without labels
        mask = self.mask
        mask_out = self.mask_out
        # Extract the logits
        logits = outputs.logits  # (bsz, seq_len, vocab_size)
        
        if labels is not None:
            #default mask is every other token 
            if mask is None: 
                print('warning: no mask provided, using default mask')
                mask = torch.arange(labels.size(1)-1) % 2 == 1  # (seq_len - 1,)
            #ensure mask is of the right length for the labels shifted by one
            mask = mask[:labels.size(1)-1].bool()
            mask = mask.to(labels.device)  # Ensure mask is on the same device as labels
        
            
            # Apply the mask to predictions and labels
            pred = logits[:, :-1, :]  # (bsz, seq_len, vocab_size)
            tgt = labels[:, 1:, :]  # (bsz, seq_len, vocab_size)
            #create a copy of prediction before masking 
            pred_copy = pred.clone()
            
            mask = mask.bool()
            pred_mask = pred[:,mask,:]
            tgt_mask = tgt[:,mask,:]   

            huber_fn = nn.HuberLoss(reduction="none")
            huber_loss = huber_fn(pred_mask, tgt_mask)
            huber_loss = huber_loss.mean()

            mask_out = mask_out.bool()
            out_tgt = self.model_tgt.decoder(tgt)
            out_pred = model.decoder(pred)
            
            #mask out the input embeddings
            out_tgt_masked = out_tgt[:,mask_out,:]
            out_pred_masked = out_pred[:,mask_out,:]
            reg = 'huber'
            if reg == 'kl':
                penalty = kl_loss(out_pred_masked, out_tgt_masked)
            if reg == 'huber': 
                penalty = huber_fn(out_pred_masked, out_tgt_masked).mean()
            regular = 0.5
            loss = (1-regular)*huber_loss + regular*penalty
            # Print the coordinates with the largest loss intermittently
            if self.state.global_step % self.print_interval == 0:
                print("train penalty loss: ", penalty)
                print("train huber loss: ", huber_loss)
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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Overrides evaluate to always compute evaluation loss using eval_loss.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        total_loss = 0.0
        total_kl = 0.0
        total_huber_loss = 0.0
        total_transf_perplexity = 0.0
        total_lstm_perplexity = 0.0
        nb_steps = 0
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                loss_final, kl, huber_loss, transf_perplexity, lstm_perplexity  = self.eval_loss(self.model, batch)
                #total_loss += loss.item()
                total_loss += loss_final.item()
                total_kl += kl.item()
                total_huber_loss += huber_loss.item()
                total_transf_perplexity += transf_perplexity.item()
                total_lstm_perplexity += lstm_perplexity.item()
            nb_steps += 1
        mean_loss = total_loss / nb_steps 
        mean_kl = total_kl / nb_steps 
        mean_huber_loss = total_huber_loss / nb_steps 
        mean_transf_perplexity = total_transf_perplexity / nb_steps 
        mean_lstm_perplexity = total_lstm_perplexity / nb_steps 
        print("Evaluation Loss:", mean_loss)
        print("Evaluation KL Loss:", mean_kl)
        print("Evaluation Huber Loss:", mean_huber_loss)
        print("Evaluation Transformer Perplexity:", mean_transf_perplexity)
        print("Evaluation LSTM Perplexity:", mean_lstm_perplexity)  
        # Save perplexities over time
        perplex_file = f"./results/perplexity_{timestamp}.pt"
        new_entry = torch.tensor([[mean_transf_perplexity, mean_lstm_perplexity]])
        if os.path.exists(perplex_file):
            past = torch.load(perplex_file, weights_only=True)
            updated = torch.cat((past, new_entry), dim=0)
        else:
            updated = new_entry
        torch.save(updated, perplex_file)
        if os.path.exists(perplex_file):
            # Load the tensor and convert to numpy array
            data = torch.load(perplex_file, weights_only=True).cpu().numpy()
            steps = range(data.shape[0])
            transformer_perp = data[:, 0]
            lstm_perp = data[:, 1]

            plt.figure(figsize=(10, 6))
            plt.plot(steps, transformer_perp, label="Transformer Perplexity", marker="v")
            plt.plot(steps, lstm_perp, label="LSTM Perplexity", marker="s")
            plt.xlabel("Evaluation Steps")
            plt.ylabel("Perplexity")
            plt.title("Transformer vs. LSTM Perplexity over Epochs")
            plt.legend()
            plt.ylim(0.0, 1000)
            plt.savefig(f"./plots/perplexity_plot_{timestamp}.png")
            plt.close()
        else:
            print("Perplexity file not found at", perplex_file)
        #return {f"{metric_key_prefix}_loss": mean_loss}
        # Return all necessary metrics so that they become part of the logged dictionary.
        return {
            f"{metric_key_prefix}_loss": mean_loss,
            "mean_transf_perplexity": mean_transf_perplexity,
            "mean_lstm_perplexity": mean_lstm_perplexity
        }

    def eval_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation: Ensure a scalar tensor is returned for loss.
        """
        model.eval()

        labels = inputs.pop("labels", None).to(device)
        tokens = inputs.pop("tokens").to(device)
        inputs["inputs"] = inputs["inputs"].to(device)
        model.to(device)
        outputs = model(**inputs)
        mask = self.mask
        mask_out = self.mask_out

        #The mask is applied to the inputs shifted by one.  We define mask_original to be the mask without shifting by one.  
        #We do this so that it's incredibly clear how the mask is applied to the generated inputs.  
        mask_original = torch.tensor([False] + mask.tolist()).bool() #mask unshifted by one 
        
        if labels is not None:
            #ensure mask is of the right length for the labels shifted by one
            mask = mask[:labels.size(1)-1].bool().to(device)
            
            mask_idx = torch.where(mask_original)[0]
            if len(mask_idx) > 0: 
                context_len = mask_idx[0].item()
            else: 
                raise ValueError('No context length found')
            
            # Ensure generated is on the same device as the model
            generated = inputs['inputs'][:,:context_len,:].to(device)

            # Iteratively generate the next tokens
            #resulting tensor is of len labels.size(1) + 1
            #example: context length is 4, label size is 10, generated size is 11
            #we ignore the first token in the generated tensor 
            #later we will also ignore the last token in the generated tensor
            for index in range(labels.size(1)+1):
                if index < context_len: 
                    continue
                #if last token is reached or mask is true, generate next token 
                if index == labels.size(1) or mask_original[index]: 
                    logits = self.model(generated).logits  # your model forward
                    next_token = logits[:, -1:, :]  # get the most recent token
                    generated = torch.cat((generated, next_token), dim=1)  # append the next token
                else: 
                    generated = torch.cat((generated, labels[:,index,:].unsqueeze(1).to(device)), dim=1)
            
            #ignore first token 
            generated = generated[:,1:,:]   
            
            # Apply the mask to predictions and labels
            pred = generated[:, :-1, :]  # (bsz, seq_len, vocab_size)
            tgt = labels[:, 1:, :]  # (bsz, seq_len, vocab_size)
            #create a copy of prediction before masking 
            pred_copy = pred.clone()
            
            #mask = torch.zeros(labels.size(1) - 1, dtype=torch.bool)
            mask = mask.bool()
            pred_mask = pred[:,mask,:]
            tgt_mask = tgt[:,mask,:]   

            huber_fn = nn.HuberLoss(reduction="none")
            huber_loss = huber_fn(pred_mask, tgt_mask)
            huber_loss = huber_loss.mean()

            mask_out = mask_out.bool()
            out_tgt = self.model_tgt.decoder(tgt.to(device))
            out_pred = model.decoder(pred)
            out_tgt_masked = out_tgt[:,mask_out,:]
            out_pred_masked = out_pred[:,mask_out,:]
            regularize = huber_fn(out_pred_masked, out_tgt_masked).mean()

            eval_mode = 'kl'
            if eval_mode == 'kl':
                kl = kl_loss(out_pred_masked, out_tgt_masked)
                #print('kl loss: ', kl)
                logits_logprob = F.log_softmax(out_pred_masked, dim=-1)
                penalty = F.nll_loss(
                    logits_logprob.view(-1, logits_logprob.size(-1)),
                    tokens.view(-1),
                    reduction="mean"
                )
                transf_perplexity = torch.exp(penalty)
                #print('perplexity of transformer: ', torch.exp(penalty))
                lstm_logits_logprob = F.log_softmax(out_tgt_masked, dim=-1)
                lstm_penalty = F.nll_loss(
                    lstm_logits_logprob.view(-1, lstm_logits_logprob.size(-1)),
                    tokens.view(-1),
                    reduction="mean"
                )
                lstm_perplexity = torch.exp(lstm_penalty)
                #print('perplexity of lstm: ', torch.exp(lstm_penalty))
                debug = False 
                if debug == True: 
                    model_obj = torch.load('saved_models/LSTM/100_2model.pt', map_location=device)
                    model_obj.to(device)
                    model_obj.eval()
                    #model_obj.decoder
                    # Extract initial hidden/cell states from the first 'context_len' tokens of inputs["inputs"]
                    # inputs["inputs"] is shape: (batch_size, context_len + ..., hidden_dim)
                    init_states = inputs["inputs"][:, :context_len, :]
                    # For a 2-layer LSTM, assume:
                    #   Layer 1 hidden state: init_states[:, 0, :]
                    #   Layer 1 cell state:   init_states[:, 1, :]
                    #   Layer 2 hidden state: init_states[:, 2, :]
                    #   Layer 2 cell state:   init_states[:, 3, :]
                    h0 = torch.stack([init_states[:, 0, :], init_states[:, 2, :]], dim=0)   # (2, batch_size, hidden_dim)
                    c0 = torch.stack([init_states[:, 1, :], init_states[:, 3, :]], dim=0)   # (2, batch_size, hidden_dim)
                    hidden0 = (h0, c0)
                    
                    # Prepare tokens for the LSTM:
                    # tokens: (batch_size, token_seq) => LSTM expects (token_seq, batch_size)
                    tokens_seq = tokens.t()  # shape: (token_seq, batch_size)
                    
                    # Pass tokens through the lstm model_obj's encoder and LSTM:
                    embedded = model_obj.encoder(tokens_seq)              # (token_seq, batch_size, embed_dim)
                    out_lstm, _ = model_obj.rnn(embedded, hidden0)          # (token_seq, batch_size, hidden_dim)
                    
                    # Project LSTM outputs using the pretrained decoder
                    logits_lstm = model_obj.decoder(out_lstm)               # (token_seq, batch_size, vocab_size)
                    # Transpose back to (batch_size, token_seq, vocab_size)
                    logits_lstm = logits_lstm.transpose(0, 1)
                    
                    # Apply softmax to convert to probability distributions
                    probs_lstm = torch.softmax(logits_lstm, dim=-1)
                    probs_target = torch.softmax(out_tgt_masked, dim=-1)
                    
                    # Adjust for off-by-one:
                    # Compare all but the final token in probs_lstm with all but the first token in probs_target.
                    diff = torch.abs(probs_lstm[:, :-1, :] - probs_target[:, 1:, :])
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    
                    print("DEBUG LSTM outputs (after softmax and off-by-one adjustment):")
                    print("Max difference between debug LSTM probs and out_tgt_masked probs:", max_diff)
                    print("Mean difference between debug LSTM probs and out_tgt_masked probs:", mean_diff)
                    print("DEBUG LSTM outputs (after softmax and off-by-one adjustment):")   


            regular = 0.5
            loss_final = (1-regular)*huber_loss + regular*regularize 
        else:
            loss_final = None

        if return_outputs:
            return loss_final, pred  # Return both loss and outputs if requested
        else:
            return loss_final, kl, huber_loss, transf_perplexity, lstm_perplexity  # Return only the scalar loss

# ----------------------------------------------------
# 2. Training Script
# ----------------------------------------------------
if __name__ == "__main__":
    # 1. Synthetic dataset
    #parse args
    parser = argparse.ArgumentParser(description='training a GPT model on synthetic data')
    parser.add_argument('--data', type=str, default='LSTM',
                    help='options are [random, rotation, LDS, RNN_TANH, RNN]')
    parser.add_argument('--input_dim', type=int, default=100,
                    help='integer input dimension')
    parser.add_argument('--num_samples', type=int, default=10000,
                    help='number of sequences each of seq_len')
    parser.add_argument('--seq_len', type=int, default=4,
                    help='length of each sequence')
    parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in data generating model')
    parser.add_argument('--model_emb', type=int, default=768,
                    help='dimension of embedding in transformer model')
    parser.add_argument('--model_layers', type=int, default=12,
                    help='layers in transformer model')
    parser.add_argument('--epochs', type=int, default=10,
                    help='epochs  to train transformer model')
    
    args = parser.parse_args()
    print('args: ', args)
    input_dim = args.input_dim
    num_samples = args.num_samples
    seq_len = args.seq_len
    num_layers = args.num_layers
    model_emb = args.model_emb
    model_layers = args.model_layers
    epochs = args.epochs

    if args.data == 'random': 
        dataset = RandomVectorDataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)
    if args.data =='rotation':
        dataset = FixedRotationDataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)
    if args.data == 'LDS':
        dataset = LinearDynamicsDataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)
    if args.data == 'RNN_TANH': 
        dataset = RNN_TANH_Dataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)
    if args.data == 'RNN': 
        dataset = RNN_Dataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, num_layers=num_layers, seed=42)
    if args.data == 'LSTM': 
        dataset = LSTM_Dataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, num_layers=num_layers, seed=42)

    
    #TODO: RNN dataset is wrong order (seq_len, batch_size, input_dim) instead of (batch_size, seq_len, input_dim)
    #print('number of samples in dataset: ', len(dataset.data))
    #print('datapoint shape: ', dataset.data[0].shape)
    #print('mask: ', dataset.mask)
    #print('mask_out: ', dataset.mask_out)  
    #print('tokens: ', dataset.token_data) 

    # Train/Validation/Test split
    valid_size = min(int(0.15 * len(dataset)), 1000)
    test_size = min(int(0.15 * len(dataset)), 1000)
    train_size = len(dataset) - valid_size - test_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    
    # Define a custom collate function
    def collate_fn(batch):
        inputs = torch.stack([item["inputs"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        tokens = torch.stack([item["tokens"] for item in batch])  # or torch.stack(...) if they are tensors
        
        return {
            "inputs": inputs,
            "labels": labels,
            "tokens": tokens
        }

    # Create a DataLoader for the train_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 2. Model configuration and instantiation
    config = VectorGPTConfig(
        n_positions=2000,  # must be >= 30 (seq_len)
        n_embd=model_emb,      # hidden dimension
        n_layer=model_layers,       # transformer layers
        n_head=12,        # attention heads
        input_dim=input_dim,  # input vector dimension
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        ntokens = dataset.ntoken
    )
    model = VectorGPTModel(config)
 
    # 3. Training arguments
    # Check if the directory exists, and create it if it does not
    output_dir = f"./vector_gpt_trainer/{args.model_emb}_{args.model_layers}_{timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=f"./vector_gpt_trainer/{args.model_emb}_{args.model_layers}_{timestamp}",  # Directory to save checkpoints
        save_steps=500,                    # Save checkpoint every 500 steps
        overwrite_output_dir=False,         # Overwrite existing output dir
        eval_strategy="steps",             # Evaluate at the end of each epoch
        eval_steps=100,                    # Evaluate every 100 steps
        save_strategy="steps",             # Save checkpoints every epoch
        logging_dir="./logs",              # Directory for TensorBoard logs
        logging_steps=100,                  # Log every 10 steps for finer feedback
        save_total_limit=3,                # Keep only the last 3 checkpoints
        load_best_model_at_end=True,       # Load the best model based on validation loss
        metric_for_best_model="eval_loss", # Use validation loss for checkpoint selection
        greater_is_better=False,           # Lower loss is better
        learning_rate=3e-4,                # Lower learning rate for stability 3e-4 original setting
        weight_decay=0.01,                 # Weight decay for regularization original 0.01
        adam_beta1=0.9,                    # First momentum parameter
        adam_beta2=0.98,                   # Second momentum parameter
        adam_epsilon=1e-6,                 # Epsilon for numerical stability
        warmup_steps=300,                  # Reduced warmup steps
        per_device_train_batch_size=16,    # Batch size per GPU during training
        per_device_eval_batch_size=16,     # Batch size per GPU during evaluation
        gradient_accumulation_steps=1,     # Accumulate gradients over 1 step
        fp16=True,                         # Use mixed precision (FP16)
        max_grad_norm=1.0,                 # Gradient clipping
        num_train_epochs=epochs,                # Fewer epochs to prevent overfitting
        report_to="tensorboard",           # Log to TensorBoard
        seed=42,                           # Set seed for reproducibility
    )


    custom_args = {'mask': dataset.mask, 'mask_out': dataset.mask_out, 'ntoken': dataset.ntoken}
    # 4. Custom trainer
    trainer = VectorGPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=None,  # Not needed for vector-based tasks
        custom_args = custom_args,
        train_loader = train_loader,
        valid_loader=valid_loader
    )

    import json
    from transformers import TrainerCallback

    class SaveLossCallback(TrainerCallback):
        def __init__(self, output_file="losses.json"):
            self.output_file = output_file
            self.losses = {"training_loss": [], "validation_loss": []}
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:  # Training loss
                    self.losses["training_loss"].append(logs["loss"])
                if "eval_loss" in logs:  # Validation loss
                    self.losses["validation_loss"].append(logs["eval_loss"])
            
            # Save to file after every logging step
            with open(self.output_file, "w") as f:
                json.dump(self.losses, f, indent=4)

    # Add the callback to the Trainer
    trainer.add_callback(SaveLossCallback(f"./results/output_{args.input_dim}_{args.num_layers}_{args.model_emb}_{args.model_layers}_{timestamp}losses.json"))

    from transformers import TrainerCallback
    import matplotlib.pyplot as plt

    class PlotLossCallback(TrainerCallback):
        def __init__(self, output_file, plot_file, plot_interval=100):
            super().__init__()
            self.output_file = output_file
            self.plot_file = plot_file
            self.plot_interval = plot_interval

        def on_log(self, args, state, control, logs=None, **kwargs):
            # Only plot every 'plot_interval' steps
            if state.global_step % self.plot_interval == 0 and state.global_step > 0:
                with open(self.output_file, "r") as f:
                    losses = json.load(f)

                training_loss = losses.get("training_loss", [])
                validation_loss = losses.get("validation_loss", [])
                transf_perplexity = losses.get("mean_transf_perplexity", [])
                lstm_perplexity = losses.get("mean_lstm_perplexity", [])

                plt.figure(figsize=(10, 6))
                #plt.plot(training_loss, label="Training Loss", marker="o")
                #if validation_loss:
                #    plt.plot(validation_loss, label="Validation Loss", marker="x")
                if transf_perplexity:
                    plt.plot(transf_perplexity, label="Transformer Perplexity", marker="v")
                if lstm_perplexity:
                    plt.plot(lstm_perplexity, label="LSTM Perplexity", marker="s")
                plt.xlabel("Steps")
                plt.ylabel("perplexity")
                plt.title("Transformer and LSTM Perplexity")
                plt.legend()
                # Option 1: Set fixed y-limits (adjust these numbers as needed)
                plt.ylim(0.0, 40000)

                plt.savefig(self.plot_file)
                plt.close()

    # Then add the callback to your Trainer:
    plot_callback = PlotLossCallback(
        f"./results/output_{args.input_dim}_{args.num_layers}_{args.model_emb}_{args.model_layers}_{timestamp}losses.json",
        f'./plots/output_{args.input_dim}_{args.num_layers}_{args.model_emb}_{args.model_layers}_{timestamp}loss_plot.png',
        plot_interval=200  # for example
    )

    class PrintLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if "loss" in logs:
                print(f"Step {state.global_step}: Training Loss: {logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"Step {state.global_step}: Validation Loss: {logs['eval_loss']:.4f}")

    # Add the callback to your Trainer
    trainer.add_callback(PrintLossCallback())

    # add plotting callback, right now plotting is handled manually in evaluate function.  
    #trainer.add_callback(plot_callback)

    # 5. Start training
    trainer.train()

    # 6. Optional: Evaluate after training
    # Evaluate on test dataset
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)

