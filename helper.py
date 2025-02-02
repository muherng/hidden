# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx   

import data
import model

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            output = model(data)
            criterion = nn.NLLLoss()
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def perplexity(model): 
    corpus = data.Corpus(args.data)

def kl_loss(pred_logits, target_logits):
    """
    Computes the KL divergence loss between two discrete distributions defined by their logits.
    
    Assumes:
      pred_logits and target_logits have shape (batch_size, num_masked, vocab_size)
      where num_masked is the number of masked positions and vocab_size is 
      the number of tokens.
      
    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (B, M, V).
        target_logits (torch.Tensor): Target logits of shape (B, M, V).

    Returns:
        torch.Tensor: Scalar KL divergence loss averaged over the batch.
    """
    # Convert target logits into probabilities.
    target_probs = F.softmax(target_logits, dim=-1)
    # Convert predicted logits into log probabilities.
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
    
    # Compute KL divergence loss using batchmean reduction.
    loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
    return loss


