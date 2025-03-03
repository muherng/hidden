import os
import time
import math
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from nanogpt import GPTConfig, GPT  # assumes a model.py defining your GPT and GPTConfig classes
from compressed_gpt import CompressedGPT 
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import set_seed
import datetime
import warnings
import wandb
warnings.filterwarnings("ignore")

wikitext_train = None
wikitext_val = None

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model")
    # I/O settings
    parser.add_argument("--name", type=str, default="", help="run name")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--model", type=str, default="gpt2", help="model name", choices=["gpt2", "compressed"])
    parser.add_argument("--out_dir", type=str, default="../../out/", help="directory to save checkpoints")
    parser.add_argument("--data_dir", type=str, default="../../data/", help="data dir")
    parser.add_argument("--eval_interval", type=int, default=2000, help="iterations between evaluations")
    parser.add_argument("--log_interval", type=int, default=1, help="iterations between logging")
    parser.add_argument("--eval_iters", type=int, default=200, help="number of iterations for evaluation")
    parser.add_argument("--eval_only", action="store_true", help="run evaluation only and exit")
    parser.add_argument("--always_save_checkpoint", action="store_true", help="save checkpoint even if not the best loss")
    # Data settings
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="tokenizer name")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40, help="gradient accumulation steps")
    parser.add_argument("--batch_size", type=int, default=12, help="micro-batch size")
    parser.add_argument("--block_size", type=int, default=1024, help="sequence length")
    # Model settings
    parser.add_argument("--n_layer", type=int, default=12, help="number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--bias", action="store_true", help="use bias in layers")
    parser.add_argument("--offset", type=int, default=0, help="offset for the compressed model")
    # Optimizer settings
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="maximum learning rate")
    parser.add_argument("--max_iters", type=int, default=600000, help="total number of training iterations")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="weight decay coefficient")
    parser.add_argument("--beta1", type=float, default=0.9, help="adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping value")
    # LR decay settings
    parser.add_argument("--decay_lr", action="store_true", help="use learning rate decay")
    parser.add_argument("--warmup_iters", type=int, default=2000, help="number of warmup iterations")
    parser.add_argument("--lr_decay_iters", type=int, default=600000, help="iterations to decay LR")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="minimum learning rate")
    # Misc
    parser.add_argument("--init_from", type=str, choices=["scratch", "resume", "gpt2"], default="scratch", help="initialization method")
    parser.add_argument("--compile", action="store_true", help="compile the model with torch.compile")
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default="float16", help="data type")
    parser.add_argument("--nowandb", action="store_true", help="Debug mode (disable wandb)")
    return parser.parse_args()


def get_tokenizer(name = 'gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer, tokenizer.vocab_size

def load_wikitext_tokens(split, args):
    """
    Loads WikiText-2 (raw version) for the given split,
    concatenates all texts, tokenizes with GPT-2 tokenizer,
    and returns a 1D torch.LongTensor of token ids.
    """
    cache_dir = args.data_dir if hasattr(args, "data_dir") and args.data_dir else None
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
    # Concatenate all the texts with a double newline (to mimic paragraph boundaries)
    all_text = "\n\n".join(dataset["text"])
    tokenizer, vocab_size = get_tokenizer(args.tokenizer)
    token_ids = tokenizer.encode(all_text)
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long)

    n_chunks = tokens_tensor.numel() // args.block_size
    tokens_tensor = tokens_tensor[:n_chunks * args.block_size].view(n_chunks, args.block_size)
    
    return tokens_tensor

def prepare_data(args):
    global wikitext_train, wikitext_val
    print("Loading WikiText-2 train split...")
    wikitext_train = load_wikitext_tokens("train", args)
    print("Loading WikiText-2 validation split...")
    wikitext_val = load_wikitext_tokens("validation", args)


def get_batch(split, args, device):
    """
    A simple data loader that uses numpy memmap to load the dataset.
    """
    data = wikitext_train if split == "train" else wikitext_val
    if data is None:
        raise ValueError("Data not loaded; please call prepare_data(args) before training.")
    
    ix = torch.randint(0, data.size(0), (args.batch_size,))
    X = data[ix]  # [batch_size, block_size]

    # For Y, shift the tokens by one. For the last token, you can wrap around or set a special value.
    Y = torch.zeros_like(X)
    Y[:, :-1] = X[:, 1:]
    Y[:, -1] = X[:, 0]  # or you could use a padding token
    return X.to(device), Y.to(device)



def estimate_metrics(model, args, device, ctx, splits = ['train', 'val']):
    """
    Estimate loss on both train and validation sets.
    """
    out = {'loss': {}, 'perplexity': {}, 'accuracy': {}, 'bits/token': {}}
    model.eval()
    for split in splits:
        losses = torch.zeros(args.eval_iters)
        perpl_s = torch.zeros(args.eval_iters)
        acc_s = torch.zeros(args.eval_iters)
        bpt_s = torch.zeros(args.eval_iters)

        for k in range(args.eval_iters):
            X, Y = get_batch(split, args, device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            perpl_s[k] = torch.exp(loss).item()
            acc_s[k] = logits.argmax(dim=-1).eq(Y).float().mean().item()
            bpt_s[k] = loss.item() / math.log(2)  # converting nats to bits

        out['loss'][split] = losses.mean().item()
        out['perplexity'][split] = perpl_s.mean().item()
        out['accuracy'][split] = acc_s.mean().item()
        out['bits/token'][split] = bpt_s.mean().item()

    model.train()
    return out

def get_lr(it, args):
    """
    Cosine learning rate decay with linear warmup.
    """
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> Using device: {device}")

    run_name = f"{args.model}_offset{args.offset}_{args.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    args.out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.nowandb:    
        run = wandb.init(
            entity="sharut",
            project="transformer-compression", 
            config=args,
            dir=args.out_dir,
            name=run_name
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Determine vocab size from metadata if available
    prepare_data(args)
    meta_vocab_size = get_tokenizer(args.tokenizer)[1]

    # Build or load the model
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size+1, # add one for the scratch token
        bias=args.bias,
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
        dropout=args.dropout,
        offset = args.offset
    )
    model_map = {"gpt2": GPT, "compressed": CompressedGPT}

    iter_num = 0
    best_val_loss = float("inf")

    if args.init_from == "scratch":
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = model_map[args.model](gptconf)
        # model = GPT(gptconf)
    
    elif args.init_from == "resume":
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        print(f"Resuming training from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_args.update({k: checkpoint["model_args"][k] for k in model_args})
        gptconf = GPTConfig(**model_args)
        model = model_map[args.model](gptconf)
        # model = GPT(gptconf)
        model.load_state_dict(checkpoint["model"])
        iter_num = checkpoint.get("iter_num", 0)
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    
    elif args.init_from == "gpt2":
        print("Initializing from OpenAI GPT-2 weights")
        # override dropout if needed
        override_args = dict(dropout=args.dropout)
        model = model_map[args.model].from_pretrained("gpt2", override_args)
        # model = GPT.from_pretrained("gpt2", override_args)
        # update model_args from the loaded config
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)
    else:
        raise ValueError(f"Unknown init_from option {args.init_from}")

    # Optionally crop the block size
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
        model_args["block_size"] = args.block_size

    model.to(device)


    # Optimizer and GradScaler (if using fp16)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device.type)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # Optionally compile the model (requires PyTorch 2.0+)
    if args.compile:
        print("Compiling the model (this may take a minute)...")
        model = torch.compile(model)

    # Training loop
    X, Y = get_batch("train", args, device)
    t0 = time.time()
    local_iter_num = 0  # number of iterations since training started

    while iter_num < args.max_iters:
        # Update learning rate
        lr = get_lr(iter_num, args) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate and checkpoint periodically
        if iter_num % args.eval_interval == 0:
            metrics = estimate_metrics(model, args, device, ctx)
            tr_log = ' | '.join([f"train_{k} {v['train']:.4f}" for k, v in metrics.items()])
            val_log = ' | '.join([f"val_{k} {v['val']:.4f}" for k, v in metrics.items()])

            if not args.nowandb:
                wandb.log({**metrics, "lr": lr})
            
            print(f"Step {iter_num}: {tr_log} | {val_log}")
            
            if metrics['loss']["val"] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = metrics['loss']["val"]
                if iter_num > 0:
                    ckpt = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        **metrics,
                    }
                    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
                    print(f"Saving checkpoint to {ckpt_path}")
                    torch.save(ckpt, ckpt_path)
            if iter_num == 0 and args.eval_only:
                break

        # Forward and backward pass with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(args.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps  # scale loss for accumulation
            # prefetch next batch asynchronously
            X, Y = get_batch("train", args, device)
            scaler.scale(loss).backward() # accumulate gradients

        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        lossf = loss.item() * args.gradient_accumulation_steps
        if local_iter_num >= 5:
            # Here you can add your own metric computation (e.g., mfu estimation)
            pass
        if iter_num % args.log_interval == 0:
            print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            if not args.nowandb:
                wandb.log({"loss": lossf, "time_ms": dt*1000})

        iter_num += 1
        local_iter_num += 1

    print("Training complete.")

if __name__ == "__main__":
    main()
