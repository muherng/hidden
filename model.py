import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
    
    def collect_hidden_states_RNN(self,input_tensor,hidden): 
        # input: (seq_len, batch_size, input_dim)
        # Collect hidden states from all layers of RNN
        # This is useful for tasks like language modeling where we need to pass hidden states
        # from all layers to the decoder.
        #for now assume input_dim and hidden_dim are equal
        #TODO: handle different input and hidden dimensions
        seq_len = input_tensor.size(0)
        batch_size = input_tensor.size(1)
        data = []
        mask = []
        for t in range(seq_len):
            #although out[0].unsqueeze(0) does nothing, 
            #it is done to maintain notion of sequence length (dropped) 
            #then we add a dimension representing the sequence length
            #this maintains consistency with how we handle hidden_dim. 
            for layer in range(self.nlayers): 
                data.append(hidden[layer,:,:].unsqueeze(0))
            #data.append(out[0].unsqueeze(0))
            data.append(input_tensor[t].unsqueeze(0))
            #mask is incremented by number of layers + number of input vectors per timestep
            if t == 0: 
                mask.extend([0]*(self.nlayers + 1))
            if t > 0: 
                mask.extend([1]*self.nlayers + [0])
            
            out,hidden = self.rnn(input_tensor[t].unsqueeze(0),hidden) 
            #out shape: (1, batch_size, input_dim) because sequence length is 1
            #hidden shape: (nlayers, batch_size, hidden_dim) 
        data = torch.cat(data, dim=0)
        #mask is applied only to every token after the first 
        mask = torch.tensor(mask[1:])
        if data.shape != (seq_len*(self.nlayers + 1), batch_size, self.nhid):
            print("data shape: ", data.shape)
            print("expected shape: ", (seq_len*(self.nlayers + 1), batch_size, self.nhid))
            raise ValueError("Shape mismatch in data tensor")
        if mask.shape[0] != seq_len*(self.nlayers + 1) - 1:
            print("mask shape: ", mask.shape)
            print("expected shape: ", (seq_len*(2*self.nlayers + 1) - 1))
            raise ValueError("Shape mismatch in mask tensor")
        return data, mask
    
    def collect_hidden_from_tokens(self, hidden, out, input_tokens):
        # input_tokens: (seq_len, batch_size) of token indices
        # Collect hidden states from all layers of LSTM
        # This is useful for tasks like language modeling where we need to pass hidden states
        # from all layers to the decoder.
        #for now assume input_dim and hidden_dim are equal
        #TODO: handle different input and hidden dimensions
        #print('input_tokens: ', input_tokens.shape)
        input_tensor = self.encoder(input_tokens)
        #print('input_tensor: ', input_tensor.shape)
        
        #seq_len = input_tokens.size(0)
        #batch_size = input_tokens.size(1)
        #input_dim = self.encoder.embedding_dim
        #input_tensor = torch.randn(batch_size, seq_len, input_dim, device='cuda')
        #input_tensor = input_tensor.permute(1, 0, 2).contiguous()
        # Unpack the initial hidden states (h0, c0)
        h, c = hidden  # Each has shape (nlayers, batch_size, hidden_dim)

        # We’ll collect everything in lists first.
        data = []
        mask = []
        mask_out = []

        seq_len = input_tensor.size(0)
        batch_size = input_tensor.size(1)

        # Manually unroll LSTM over each time step
        for t in range(seq_len):
            for layer in range(self.nlayers):
                data.append(h[layer, :, :].unsqueeze(0))  # shape (1, batch_size, hidden_dim)
                data.append(c[layer, :, :].unsqueeze(0))
            #appending out to data, which is (1,batch_size, hidden_dim) 
            data.append(out)
            data.append(input_tensor[t].unsqueeze(0))  # shape (batch_size, hidden_dim)
            # LSTM expects (1, batch_size, input_dim) for a single time step
            # out: (1, batch_size, hidden_dim)
            # h, c: (nlayers, batch_size, hidden_dim)
            #print('out shape: ', out.shape)

            x_t = input_tensor[t].unsqueeze(0)  
            # One-step forward
            out, hidden = self.rnn(x_t, (h, c))
            (h,c) = hidden
            if t == 0:
                #start with hidden0, input0, out0, hidden1, input1, out1, hidden2, input2 ...
                # We have 2 * nlayers + 1 items per time step in data 
                # (h + c for each layer, input, and out) 
                #we predict the first out based on the initial hidden states and initial input
                mask.extend([0] * 2 * self.nlayers + [1] + [0])
                mask_out.extend([0] * 2 * self.nlayers + [1] + [0])
            else:
                mask.extend([1] * (2 * self.nlayers) + [1] + [0])
                mask_out.extend([0] * 2 * self.nlayers + [1] + [0])

        # data: for each time step, we appended (nlayers h) + (nlayers c) + (1 input) + (1 out) 
        # which is (2 * nlayers + 2) tokens per timestep
        # each token is (batch_size, hidden_dim), so stack them all
        data = torch.cat(data, dim=0)                # shape => (seq_len*(2*nlayers + 2), batch_size, hidden_dim)
        #print('data shape: ', data.shape)
        # mask (optional if you’re not using it)
        mask = torch.tensor(mask[1:])  # skip the very first element
        mask_out = torch.tensor(mask_out[1:]) 

        if data.shape != (seq_len*(2*self.nlayers + 2), batch_size, self.nhid):
            print("data shape: ", data.shape)
            print("expected shape: ", (seq_len*(2*self.nlayers + 2), batch_size, self.nhid))
            raise ValueError("Shape mismatch in data tensor")
        if mask.shape[0] != (seq_len*(2*self.nlayers + 2) - 1):
            print("mask shape: ", mask.shape)
            print("expected shape: ", (seq_len*(2*self.nlayers + 2) - 1))
            raise ValueError("Shape mismatch in mask tensor")

        # Return everything
        # all_hidden_states can be kept as a tuple for convenience: (all_h, all_c)
        return data, mask, mask_out, hidden, out,      

    def collect_hidden_states_LSTM(self, input_tensor, hidden):
        """
        This function manually unrolls the LSTM one timestep at a time, 
        collecting the outputs and hidden states at each step.
        
        Args:
            input_tensor: (seq_len, batch_size, input_dim)
            hidden: (h, c), each (nlayers, batch_size, hidden_dim)
        
        Returns:
            outputs: stacked LSTM outputs at each time step
            all_hidden_states: a tuple (all_h, all_c), 
                               each (seq_len+1, nlayers, batch_size, hidden_dim)
                               where seq_len+1 includes the initial hidden state (index 0).
            data: a tensor collecting (for each timestep t): 
                  [h[layer] for all layers, c[layer] for all layers, out(t)] 
            mask: a tensor analogous to your RNN code (may be ignored if not needed)
        """

        # Unpack the initial hidden states (h0, c0)
        h, c = hidden  # Each has shape (nlayers, batch_size, hidden_dim)

        # We’ll collect everything in lists first.
        data = []
        mask = []  # Initial mask
        #length of mask is 2*self.nlayers + seq_len * (1+2*self.nlayers) 

        seq_len = input_tensor.size(0)
        batch_size = input_tensor.size(1)

        # Manually unroll LSTM over each time step
        for t in range(seq_len):
            #first append the initial hidden/cell states for each layer
            for layer in range(self.nlayers):
                data.append(h[layer, :, :].unsqueeze(0))  # shape (1, batch_size, hidden_dim)
                data.append(c[layer, :, :].unsqueeze(0))
                print("c shape: ", data[-1].shape)
                print("h shape: ", data[-2].shape)
            #second, append the first input tensor
            data.append(input_tensor[t].unsqueeze(0))  # shape (batch_size, hidden_dim), will unsqueeze later
            print('input_tensor shape: ', input_tensor[t].unsqueeze(0).shape) 
            if t == 0: 
                mask.extend([0] * (2 * self.nlayers + 1))
            else: 
                mask.extend([1] * 2 * self.nlayers + [0])
            # LSTM expects (1, batch_size, input_dim) for a single time step
            x_t = input_tensor[t].unsqueeze(0)  
            # One-step forward
            out, (h, c) = self.rnn(x_t, (h, c))

            # out: (1, batch_size, hidden_dim)
            # h, c: (nlayers, batch_size, hidden_dim)

        data = torch.cat(data, dim=0)
        print('data shape: ', data.shape)
        mask = torch.tensor(mask[1:])  # skip the very first element
        print('mask shape: ', mask)
        print('data shape: ', data.shape)
        if data.shape != (seq_len*(2*self.nlayers + 1), batch_size, self.nhid):
            print("data shape: ", data.shape)
            print("expected shape: ", (seq_len*(2*self.nlayers + 1), batch_size, self.nhid))
            raise ValueError("Shape mismatch in data tensor")
        if mask.shape[0] != (seq_len*(2*self.nlayers + 1) - 1):
            print("mask shape: ", mask.shape)
            print("expected shape: ", (seq_len*(2*self.nlayers + 1) - 1))
            raise ValueError("Shape mismatch in mask tensor")
        # Return everything
        # all_hidden_states can be kept as a tuple for convenience: (all_h, all_c)
        return data, mask

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


class MultiLayerElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MultiLayerElmanRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Create a list of RNNCells for each layer
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(input_size if layer_idx == 0 else hidden_size,
                       hidden_size)
            for layer_idx in range(num_layers)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch, input_size)
        
        Returns:
            all_states: list of length seq_len, 
                        where each element is a list of length num_layers 
                        containing the hidden states for each layer at that timestep.
        """
        seq_len, batch_size, _ = x.size()

        # Initialize hidden states for each layer: list of tensors
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
             for _ in range(self.num_layers)]

        # To store hidden states at each timestep
        all_states = []

        # Manually unroll over time
        for t in range(seq_len):
            # Get input at time step t
            current_input = x[t]
            # Pass through each layer sequentially
            for layer_idx, cell in enumerate(self.rnn_cells):
                # Compute next hidden state for layer 'layer_idx'
                h[layer_idx] = cell(current_input, h[layer_idx])
                # The output of current layer becomes input to next layer
                current_input = h[layer_idx]
            # Store a copy of all layer hidden states at this timestep
            all_states.append([layer_state.clone() for layer_state in h])

        return all_states

if __name__ == '__main__': 
    # number of tokens, dim input, dim hidden, number of layers, dropout
    seq_len = 7
    batch_size = 32
    input_dim = 10
    hidden_dim = 10
    num_layers = 2
    num_tokens = 100
    
    rnn_type = 'LSTM'
    #rnn_type = 'RNN_TANH'
    #model = RNNModel('RNN_TANH', num_tokens, input_dim, hidden_dim, num_layers, dropout=0.5, tie_weights=False)
    model = RNNModel(rnn_type, num_tokens, input_dim, hidden_dim, num_layers, dropout=0.5, tie_weights=False)
    model.init_weights()
    model.eval()

    #input is of dimension (seq_len, batch, input_dim)
    input_tensor = torch.randn(seq_len, batch_size, input_dim)
    #outputs, all_hidden_states, data, mask = model.collect_hidden_states_RNN(input_tensor, model.init_hidden(batch_size))
    outputs, all_hidden_states, data, mask = model.collect_hidden_states_LSTM(input_tensor, model.init_hidden(batch_size))

    #print("all_hidden_states shape: ", all_hidden_states.shape)
    #print('all_hidden_states: ', all_hidden_states)
    #print('data: ', data)
    print('data shape: ', data.shape)
    print('mask: ', mask)

    #f = 'saved_models/model.pth'
    #torch.save(model, f)
    #model = torch.load(f, weights_only=False)
    #model.eval()
    #outputs, all_hidden_states, data, mask = model.collect_hidden_states_RNN(input_tensor, model.init_hidden(batch_size))
    
    #print('data shape: ', data.shape)
    #print('saved then output mask: ', mask)


