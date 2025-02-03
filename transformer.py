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
from synthetic import RandomVectorDataset, FixedRotationDataset, LinearDynamicsDataset, RNN_TANH_Dataset, RNN_Dataset, LSTM_Dataset
import argparse
import os

import datetime
import numpy as np
from helper import evaluate, kl_loss

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
        self.print_interval = 1000
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Load the original model and move it to the device
        model_tgt = torch.load(f'saved_models/LSTM/{input_dim}_{num_layers}model.pt', map_location=device)
        model_tgt.to(device)
        self.model_tgt = model
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
            # Create a mask for every other token, the size of the mask is seq_len - 1 because we are predicting the next token
            # TODO: adjust the mask for deep networks
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
            loss = huber_fn(pred_mask, tgt_mask)
            loss = loss.mean()

            mask_out = mask_out.bool()
            out_tgt = self.model_tgt.decoder(tgt)
            out_pred = model.decoder(pred)
            
            #mask out the input embeddings
            out_tgt_masked = out_tgt[:,mask_out,:]
            out_pred_masked = out_pred[:,mask_out,:]
            kl = kl_loss(out_pred_masked, out_tgt_masked)
            regular = 0.5
            loss = (1-regular)*loss + regular*kl
            #additional_loss = huber_fn(out_pred_masked, out_tgt_masked).mean()
            # Print the coordinates with the largest loss intermittently
            #if False and self.state.global_step % self.print_interval == 0:
            #    print("kl loss: ", kl)
            #    with torch.no_grad(): 
            #        eval_loss = self.eval_loss(model,inputs_copy)
            #        print('Evaluation Loss: ', eval_loss)
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
        nb_steps = 0
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                loss = self.eval_loss(self.model, batch)
                total_loss += loss.item()
            nb_steps += 1
        mean_loss = total_loss / nb_steps if nb_steps > 0 else float("inf")
        print("Evaluation Loss:", mean_loss)
        return {f"{metric_key_prefix}_loss": mean_loss}

    def eval_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation: Ensure a scalar tensor is returned for loss.
        """

        labels = inputs.pop("labels", None).to(device)
        tokens = inputs.pop("tokens").to(device)
        print('labels: ', labels.shape)
        print('tokens shape: ', tokens.shape)
        #raise ValueError('stop here')
        mask = self.mask
        mask_out = self.mask_out
        # Extract the logits
        #logits = outputs.logits  # (bsz, seq_len, vocab_size)
        #TODO: make this unbelievably clear 
        mask_original = torch.tensor([False] + mask.tolist()).bool() #mask unshifted by one 
        
        if labels is not None:
            #labels = labels.to(device)
            #ensure mask is of the right length for the labels shifted by one
            mask = mask[:labels.size(1)-1].bool().to(device)
            #mask = mask.to(labels.device)  # Ensure mask is on the same device as labels
            
            mask_idx = torch.where(mask_original)[0]
            if len(mask_idx) > 0: 
                context_len = mask_idx[0].item()
            else: 
                raise ValueError('No context length found')
            
            generated = inputs['inputs'][:,:context_len,:].to(device)
            # Ensure generated is on the same device as the model
            #generated = generated.to(device)
            #generated = sliced_inputs.clone()  # Forward pass without labels

            # Iteratively generate the next tokens
            #resulting tensor is of len labels.size(1) + 1
            #example: context length is 4, label size is 10, generated size is 11
            #we ignore the first token in the generated tensor 
            #later we will also ignore the last token in the generated tensor
            for index in range(labels.size(1)+1):
                #print('index: ', index)
                #print('generated size: ', generated.size())
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
            loss = huber_fn(pred_mask, tgt_mask)
            loss = loss.mean()

            mask_out = mask_out.bool()
            out_tgt = self.model_tgt.decoder(tgt.to(device))
            out_pred = model.decoder(pred)
            out_tgt_masked = out_tgt[:,mask_out,:]
            out_pred_masked = out_pred[:,mask_out,:]
            #additional_loss = huber_fn(out_pred_masked, out_tgt_masked).mean()
            eval_mode = 'NLL'
            if eval_mode == 'kl':
                penalty = kl_loss(out_pred_masked, out_tgt_masked)
            if eval_mode == 'NLL':
                logits_logprob = F.log_softmax(out_pred_masked, dim=-1)
                penalty = F.nll_loss(
                    logits_logprob.view(-1, logits_logprob.size(-1)),
                    tokens.view(-1),
                    reduction="mean"
                )
                print('perplexity of transformer: ', torch.exp(penalty))
                lstm_logits_logprob = F.log_softmax(out_tgt_masked, dim=-1)
                lstm_penalty = F.nll_loss(
                    lstm_logits_logprob.view(-1, lstm_logits_logprob.size(-1)),
                    tokens.view(-1),
                    reduction="mean"
                )
                print('perplexity of lstm: ', torch.exp(lstm_penalty))

            regular = 0.5
            loss_final = (1-regular)*loss + regular*penalty 
            # Print the 10 largest values in out_tgt_masked and their coordinates
            values, indices = torch.topk(out_tgt_masked.view(-1), 10)
            coords = np.unravel_index(indices.cpu().numpy(), out_tgt_masked.shape)

            print("Top 10 coordinates in out_tgt_masked and corresponding out_pred_masked values:")
            for i, val in enumerate(values):
                b, p, d = coords[0][i], coords[1][i], coords[2][i]
                print(f"{i+1}) TGT[{b}, {p}, {d}] = {val.item():.4f}, PRED = {out_pred_masked[b,p,d].item():.4f}")
            
            print('penalty: ', penalty)
            print('huber loss: ', loss) 
        else:
            loss_final = None

        if return_outputs:
            return loss_final, pred  # Return both loss and outputs if requested
        else:
            return loss_final  # Return only the scalar loss

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
        print('RNN_TANH is now defunct')
        dataset = RNN_TANH_Dataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, seed=42)
    if args.data == 'RNN': 
        dataset = RNN_Dataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, num_layers=num_layers, seed=42)
    if args.data == 'LSTM': 
        dataset = LSTM_Dataset(num_samples=num_samples, seq_len=seq_len, vector_dim=input_dim, num_layers=num_layers, seed=42)

    
    #TODO: RNN dataset is wrong order (seq_len, batch_size, input_dim) instead of (batch_size, seq_len, input_dim)
    print('number of samples in dataset: ', len(dataset.data))
    print('datapoint shape: ', dataset.data[0].shape)
    print('mask: ', dataset.mask)
    print('mask_out: ', dataset.mask_out)  
    print('tokens: ', dataset.token_data) 

    # Train/Validation/Test split
    valid_size = min(int(0.15 * len(dataset)), 100)
    test_size = min(int(0.15 * len(dataset)), 100)
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
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Iterating over the DataLoader
    #for batch in train_loader:
    #    inputs = batch["inputs"]
    #    labels = batch["labels"]
    #    tokens = batch["tokens"]
        #print('tokens: ', tokens)
    # DataLoader
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    #valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    # 2. Model configuration and instantiation
    config = VectorGPTConfig(
        n_positions=2000,  # must be >= 30 (seq_len)
        n_embd=model_emb,      # hidden dimension
        n_layer=model_layers,       # transformer layers
        n_head=12,        # attention heads
        input_dim=input_dim,  # input vector dimension
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
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
        weight_decay=0.0,                 # Weight decay for regularization original 0.01
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

                plt.figure(figsize=(10, 6))
                plt.plot(training_loss, label="Training Loss", marker="o")
                if validation_loss:
                    plt.plot(validation_loss, label="Validation Loss", marker="x")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss")
                plt.legend()
                plt.grid(True)
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

    trainer.add_callback(plot_callback)

    # 5. Start training
    trainer.train()

    # 6. Optional: Evaluate after training
    # Evaluate on test dataset
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)

