import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from nanogpt import GPT

# (Keep the definitions of LayerNorm, CausalSelfAttention, MLP, Block, GPTConfig, and GPT from your original code)
# For brevity, we assume these classes are defined above as in your full GPT implementation.

# -----------------------------------------------------------------------------
# New Model: CompressedGPT
# -----------------------------------------------------------------------------

@dataclass
class StateConfig:
    name: str = ''
    n_layers: int = 1
    n_embd: int = 256
    n_head: int = 2
    length: int = 1 # for transformer/LSTM/GRU, number of recurrences/autoregression

    def __repr__(self):
        return f'{self.name}(layers={self.n_layers}, embd={self.n_embd}, head={self.n_head}, length={self.length})'

class CompressedGPT(GPT):
    """
    A GPT-based language model that supports a two-phase forward pass to learn
    a compressed representation of the first part of the input sequence.
    
    In addition to the normal tokens, we learn a scratch token (not in the original vocabulary)
    which is inserted at a specified offset. During the first forward pass, the model processes
    the sequence with the inserted scratch token. The hidden state at that token (position)
    is then used as a compressed representation of all tokens before the offset.
    
    In the second pass, this compressed state is prepended to the embeddings of the tokens
    after the offset, and the model is run again to produce logits for predicting the remaining tokens.
    """
    def __init__(self, config, state_config):
        super().__init__(config)
        print('=> Initializing Compressed GPT model...')
        # Initialize a learned scratch token embedding.
        # This token is not in the vocabulary and is meant to capture a compressed representation.
        self.scratch_token = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)
        assert self.config.offset > 0, "Offset must be greater than 0."

        self.state_config = state_config
        self.init_state_model()

    def init_state_model(self):
        """
        Initialize a separate model to process the compressed state.
        """
        if self.state_config.name == '':
            return
        
        print(f'=> Using state model: {self.state_config}')
        if self.state_config.name == 'mlp':
            self.state_model = nn.Sequential(
                nn.Linear(self.config.n_embd, self.state_config.n_embd),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(self.state_config.n_embd, self.state_config.n_embd), nn.ReLU()) for _ in range(self.state_config.n_layers-1)]
            )    
        elif self.state_config.name == 'transformer':
            self.state_model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.config.n_embd, nhead=self.state_config.n_head),
                num_layers=self.state_config.n_layers
            )
        elif self.state_config.name == 'lstm':
            self.state_model = nn.LSTM(self.config.n_embd, self.state_config.n_embd, num_layers=self.state_config.n_layers, batch_first=True)
        elif self.state_config.name == 'gru':
            self.state_model = nn.GRU(self.config.n_embd, self.state_config.n_embd, num_layers=self.state_config.n_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown state model: {self.state_config.name}")

    def _process_state(self, x):
        """
        Process the hidden state at the scratch token's position.
        """
        if self.state_config.name == '':
            return x.unsqueeze(1)
        
        if self.state_config.name == 'mlp':
            x = self.state_model(x)
            return x.unsqueeze(1)
    
        bs = x.size(0)
        outs = []
        curr = x.unsqueeze(1) # shape: (bs, 1, n_embd)

        if self.state_config.name == 'gru':
            hid = (torch.randn(self.state_config.n_layers, bs, self.state_config.n_embd) * 0.02).to(x.device)
        elif self.state_config.name == 'lstm':
            h0 = (torch.randn(self.state_config.n_layers, bs, self.state_config.n_embd) * 0.02).to(x.device)
            c0 = (torch.randn(self.state_config.n_layers, bs, self.state_config.n_embd) * 0.02).to(x.device)
            hid = (h0, c0)

        for _ in range(self.state_config.length):
            if self.state_config.name == 'transformer':
                out = self.state_model(curr)
            elif self.state_config.name in ['lstm', 'gru']:
                out, hid = self.state_model(curr, hid)
            else:
                raise ValueError(f"Unknown state model: {self.state_config.name}")
            outs.append(out)
            curr = out

        return torch.cat(outs, dim=1)

    def _forward(self, x_in):
        pos = torch.arange(0, x_in.size(1), dtype=torch.long, device=x_in.device)
        pos_emb = self.transformer.wpe(pos)
        x = x_in + pos_emb

        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return x, logits
    
    def forward(self, idx, targets=None):
        """
        Performs a two-phase forward pass.
        
        Parameters:
          idx: LongTensor of shape (bs, seq) containing input token indices.
          offset: Integer. The index at which to insert the scratch token.
          targets: (Optional) LongTensor for computing loss in the second pass. 
                   Expected to be aligned with the decoder output (i.e. for tokens after the scratch token).
                   
        Returns:
          logits: The output logits from the decoder pass.
          loss: (Optional) Cross entropy loss computed on the decoder outputs if targets is provided.
          
        Procedure:
          1. Insert the scratch token at position 'offset', forming a new sequence of length seq+1.
          2. Compute token embeddings for the input and combine with positional embeddings.
          3. Run the transformer forward pass to get hidden states.
          4. Extract the hidden state at the scratch token's position; call this 'scratch_state'.
          5. Form a decoder input by concatenating scratch_state with the embeddings of the tokens after offset.
          6. Run a second transformer pass (reusing the same blocks) on the decoder input to produce logits.
        """
        bs, seq = idx.size()
        offset = self.config.offset

        assert offset < seq, "Offset must be less than the sequence length."
        assert seq <= self.config.block_size, f"Cannot forward sequence of length {seq}, block size is only {self.config.block_size}"
        
        token_emb = self.transformer.wte(idx)  # shape: (bs, seq, n_embd)
        scratch = self.scratch_token.expand(bs, 1, -1)  # shape: (bs, 1, n_embd)
        emb_first = token_emb[:, :offset, :]   # tokens before offset
        emb_rest  = token_emb[:, offset:, :]    # tokens after offset
        
        # First pass: Process the sequence with the scratch token.
        x_in = torch.cat([emb_first, scratch], dim=1)  # shape: (bs, offset+1, n_embd)
        out_1, logits_1 = self._forward(x_in)
        logits_1 = logits_1[:, :-1, :]  # remove the last token
        scratch_embd = out_1[:, offset, :]   # Extract the hidden state of the scratch token at position 'offset'

        
        # Process state embedding
        scratch_embds = self._process_state(scratch_embd)
        s_len = scratch_embds.size(1)
        
        # Second pass: Concatenate the scratch token's hidden state with the rest of the sequence.
        x_in = torch.cat([scratch_embds, emb_rest], dim=1)  # shape: (bs, (seq-offset)+length, n_embd)
        out_2, logits_2 = self._forward(x_in)
        logits_2 = logits_2[:, s_len:, :]   # remove the first s_len tokens

        logits = torch.cat([logits_1, logits_2], dim=1)
        if targets is not None:
            targets_1 = targets[:, :offset]  # targets for the first pass
            targets_2 = targets[:, offset:]  # targets for the second pass
            loss_1 = F.cross_entropy(logits_1.reshape(-1, logits_1.size(-1)), targets_1.reshape(-1), ignore_index=-1)
            loss_2 = F.cross_entropy(logits_2.reshape(-1, logits_2.size(-1)), targets_2.reshape(-1), ignore_index=-1)
            loss = loss_1 + loss_2
        else:
            loss = None
        
        return logits, loss
