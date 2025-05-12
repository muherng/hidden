from tree import TransformerScanModel
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
import argparse
import datetime
import math
import numpy as np

from transformers import (
    GPT2Config,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback
)

from data.associative_recall import multiquery_ar
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

TOL = 1e-6  # Tolerance to determine if a tensor is the dummy


class AssociativeRecallDataset(Dataset):
    def __init__(self, vocab_size, num_examples, input_seq_len, seed, **kwargs):
        """
        Dataset for the multi-query associative recall task.
        """
        self.inputs, self.labels = multiquery_ar(
            vocab_size=vocab_size,
            num_examples=num_examples,
            input_seq_len=input_seq_len,
            seed=seed,
            **kwargs
        )

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

# # -----------------------------------------------------------------------------
# # Trainer Callbacks
# # -----------------------------------------------------------------------------
# class PrintLossCallback(TrainerCallback):
#     def __init__(self):
#         self.best_training_loss = float('inf')
#         self.best_eval_loss = float('inf')
#         self.last_eval_loss = None
#         self.last_eval_accuracy = None
#         self.best_eval_accuracy = 0.0

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if state.global_step % 2 != 0:
#             return
#         if logs is None:
#             return

#         epoch = getattr(state, "epoch", None)
#         current_loss = logs.get("loss")
#         current_eval_loss = logs.get("eval_loss")
#         current_eval_accuracy = logs.get("eval_accuracy")

#         # Convert torch.Tensor to float
#         if isinstance(current_loss, torch.Tensor):
#             current_loss = current_loss.item()
#         if isinstance(current_eval_loss, torch.Tensor):
#             current_eval_loss = current_eval_loss.item()
#         if isinstance(current_eval_accuracy, torch.Tensor):
#             current_eval_accuracy = current_eval_accuracy.item()

#         if current_loss is not None and current_loss < self.best_training_loss:
#             self.best_training_loss = current_loss
#         if current_eval_loss is not None:
#             self.last_eval_loss = current_eval_loss
#             if current_eval_loss < self.best_eval_loss:
#                 self.best_eval_loss = current_eval_loss
#         if current_eval_accuracy is not None:
#             self.last_eval_accuracy = current_eval_accuracy
#             if current_eval_accuracy > self.best_eval_accuracy:
#                 self.best_eval_accuracy = current_eval_accuracy

#         out_str = f"Step {state.global_step}: "
#         if epoch is not None:
#             out_str += f"Epoch {epoch:.2f} | "
#         if current_loss is not None:
#             out_str += f"Training Loss: {current_loss:.4f} (Best: {self.best_training_loss:.4f})"
#         if current_eval_loss is not None:
#             out_str += f" | Eval Loss: {current_eval_loss:.4f} (Best: {self.best_eval_loss:.4f})"
#         if current_eval_accuracy is not None:
#             out_str += f" | Eval Accuracy: {current_eval_accuracy:.4f} (Best: {self.best_eval_accuracy:.4f})"
#         print(out_str)

class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.best_training_loss = float('inf')
        self.best_eval_loss = float('inf')
        self.last_eval_loss = None
        self.best_eval_accuracy = 0.0
        self.last_eval_accuracy = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 100 != 0:
            return
        if logs is None:
            return
        # Retrieve epoch from the state, if available
        epoch = getattr(state, "epoch", None)
        if "loss" in logs:
            current_loss = logs["loss"]
            if isinstance(current_loss, torch.Tensor):
                current_loss = current_loss.item()
            if current_loss < self.best_training_loss:
                self.best_training_loss = current_loss
        else:
            current_loss = None
        if "eval_loss" in logs:
            current_eval_loss = logs["eval_loss"]
            if isinstance(current_eval_loss, torch.Tensor):
                current_eval_loss = current_eval_loss.item()
            self.last_eval_loss = current_eval_loss
            if current_eval_loss < self.best_eval_loss:
                self.best_eval_loss = current_eval_loss
        else:
            current_eval_loss = self.last_eval_loss
        if "eval_accuracy" in logs:
            eval_accuracy = logs["eval_accuracy"]
            if isinstance(eval_accuracy, torch.Tensor):
                eval_accuracy = eval_accuracy.item()  
            self.last_eval_accuracy = eval_accuracy
            if eval_accuracy > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_accuracy 
        else:
            eval_accuracy = self.last_eval_accuracy 

        out_str = f"Step {state.global_step}: "
        if epoch is not None: 
            out_str += f"Epoch {epoch} | "
        if current_loss is not None:
            out_str += f"Training Loss: {current_loss:.4f} (Best: {self.best_training_loss:.4f})"
        if current_eval_loss is not None:
            out_str += f" | Eval Loss: {current_eval_loss:.4f} (Best: {self.best_eval_loss:.4f}, Acc: {eval_accuracy:.4f})"
        print(out_str)

def compute_metrics(eval_pred):
        logits = eval_pred.predictions["logits"]
        labels = eval_pred.label_ids
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        labels = torch.tensor(labels)

        # Flatten if needed
        if preds.ndim > 1:
            preds = preds.view(-1)
            labels = labels.view(-1)

        mask = labels != -100  # Ignore padding or masked labels
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return {
            "eval_accuracy": accuracy,
        }

# class CustomTrainer(Trainer):
#     def accuracy(self, eval_pred):
#         logits, labels = eval_pred
#         preds = torch.argmax(torch.tensor(logits), dim=-1)
#         labels = torch.tensor(labels)

#         # Flatten if needed
#         if preds.ndim > 1:
#             preds = preds.view(-1)
#             labels = labels.view(-1)

#         mask = labels != -100  # Ignore padding or masked labels
#         correct = (preds[mask] == labels[mask]).sum().item()
#         total = mask.sum().item()
#         accuracy = correct / total if total > 0 else 0.0
#         return {
#             "eval_accuracy": accuracy,
#         }

    # def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    #     # print('evaluate')
    #     metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    #     print("metrics" + str(metrics))
    #     if not hasattr(self, 'eval_steps'):
    #         self.eval_steps = []
    #         # self.eval_accuracy = []
    #     current_step = self.state.global_step if hasattr(self.state, 'global_step') else 0
    #     # eval_loss = metrics.get("eval_loss", None)
    #     # eval_ppl = np.exp(eval_loss) if eval_loss is not None and eval_loss < 100 else float('inf')

    #     self.eval_steps.append(current_step)
    #     # self.eval_ppls.append(eval_ppl)
    #     # plt.figure()
    #     # plt.plot(self.eval_steps, self.eval_ppls, marker='o')
    #     # plt.xlabel("Global Step")
    #     # plt.ylabel("Evaluation Perplexity")
    #     # plt.title("Evaluation Perplexity Over Time")
    #     # plt.ylim(0, min(400, max(self.eval_ppls)*1.1))
    #     # os.makedirs("plots", exist_ok=True)
    #     # plt.savefig(f"plots/eval_ppl_{timestamp}.png")
    #     # plt.close()
    #     return metrics

  
class TreeModel(TransformerScanModel):
    """Tree model with direct output supervision."""
    
    def __init__(self, config, chunk_size= 32, T1_num_layers = 1, T2_num_layers = 1):
        super().__init__(config, chunk_size=chunk_size, T1_num_layers = T1_num_layers, T2_num_layers = T2_num_layers)
    
    def forward(self, input_ids,
        labels=None,
        **kwargs):
        outputs = super().forward(input_ids, labels=None, output_hidden_states=True, return_dict=True)
        loss = 0
        loss_fct = nn.CrossEntropyLoss()

        # final logits
        final_logits = outputs["logits"]
        # compute loss
        if labels is not None:
            # compute loss
            # print("labels: " + str(labels))
            # print("final_logits: " + str(final_logits))
            # print("final_logits view: " + str(final_logits.view(-1, self.config.vocab_size)))
            # print("labels view: " + str(labels.view(-1)))
            loss += loss_fct(final_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # add to the total loss
            # outputs["loss"] += loss 
        return loss, outputs # 
    
# -----------------------------------------------------------------------------
# Main Training Code (unchanged)
# -----------------------------------------------------------------------------
def main():
    print("Training Transformer Scan with binary tree aggregation on MQAR.", flush=True)
    parser = argparse.ArgumentParser(
        description='Train a GPT2-based Transformer Scan LM with binary tree aggregation on MQAR.'
    )
    parser.add_argument('--seq_len', type=int, default=64*8,
                        help='Number of tokens per sample (must be a multiple of chunk_size).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--chunk_size', type=int, default=64, help='Chunk size (to be safe, use powers of 2).')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension.')
    # parser.add_argument('--train_mode', type=str, default='parallel', choices=['parallel', 'blelloch', 'sequential'],
    #                     help='Training mode: parallel or sequential.')
    parser.add_argument('--vocab_size', type=int, default=8192, help='Vocabulary size.')
    parser.add_argument('--num_train_examples', type=int, default=100000, help='Number of training examples.')
    parser.add_argument('--num_valid_examples', type=int, default=3000, help='Number of validation examples.')
    parser.add_argument('--num_kv_pairs', type=int, default=8, help='Number of key-value pairs.')
    args = parser.parse_args()

    if args.seq_len % args.chunk_size != 0:
        raise ValueError("seq_len must be a multiple of chunk_size.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    output_dir = f"out/mqar_{args.seq_len}/tree_model_{args.n_embd}/tree_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = AssociativeRecallDataset(
        vocab_size=args.vocab_size,
        num_examples=args.num_train_examples,
        input_seq_len=args.seq_len,
        num_kv_pairs=args.num_kv_pairs,
        seed=args.seed
    )
    valid_dataset = AssociativeRecallDataset(
        vocab_size=args.vocab_size,
        num_examples=args.num_valid_examples,
        input_seq_len=args.seq_len,
        num_kv_pairs=args.num_kv_pairs,
        seed=args.seed + 10  # Different seed for validation set
    )

    config = GPT2Config(
        vocab_size=args.vocab_size, #tokenizer.vocab_size
        n_positions=1024,
        n_embd=args.n_embd,
        n_layer=2, #6,
        n_head=1, #12,
        dropout=0.1
    )
    model = TreeModel(config, chunk_size=args.chunk_size,
                                 T1_num_layers=config.n_layer, T2_num_layers=config.n_layer) #, train_mode=args.train_mode
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,          
        warmup_steps=1000,           
        weight_decay=0.01,             
        fp16=False,
        seed=args.seed,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        max_grad_norm=1.0,
        logging_dir="./logs",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[PrintLossCallback()]
    )
    from transformers import ProgressCallback
    trainer.remove_callback(ProgressCallback)
    
    trainer.train()
    print(trainer.evaluate())

if __name__ == "__main__":
    main()