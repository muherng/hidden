import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModelWithoutEmbedding(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        super().__init__()
        #self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        # batch_first=False here matches (seq_len, batch_size, input_dim)
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

    def forward(self, inputs, hidden):
        """
        inputs shape:  (seq_len, batch_size, ninp)
        hidden shape:  depends on RNN type:
            - LSTM: (h_0, c_0) with each of shape (nlayers, batch_size, nhid)
            - GRU:  (nlayers, batch_size, nhid)
        """
        # output shape: (seq_len, batch_size, nhid)
        # new_hidden is the final hidden state (plus cell state for LSTM)
        output, new_hidden = self.rnn(inputs, hidden)

        # output has all time-step hidden states for the last layer
        # If needed, you can drop them:
        #output = self.drop(output)

        # Final projection of outputs (but we still keep them here for demonstration)
        decoded = self.decoder(output)  # shape: (seq_len, batch_size, ntoken)

        # Return all time-step hidden states (i.e., output) as well as final hidden state
        return decoded, new_hidden, output

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid)
            )
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

import torch

def collect_hidden_states_RNN(model, seq_len, batch_size, num_batches, input_dim):
    """
    Collects per-time-step hidden states for an RNN in one forward pass.
    Minimizes Python looping for better GPU utilization.
    """
    device = next(model.parameters()).device
    model.eval()

    cot_data_list = []
    hidden_data_list = []
    input_list = []

    with torch.no_grad():
        for _ in range(num_batches):
            print("Batch: ", _)
            # Initialize hidden state
            hidden = model.init_hidden(batch_size)

            # Create random inputs of shape (seq_len, batch_size, input_dim)
            # Putting them on the same device as the model
            inputs = torch.randn(seq_len, batch_size, input_dim, device=device)

            # Single forward pass to get:
            #   decoded     -> (seq_len, batch_size, ntoken), not used here
            #   new_hidden  -> final hidden state
            #   all_outputs -> (seq_len, batch_size, hidden_size)
            decoded, new_hidden, all_outputs = model(inputs, hidden)

            # "all_outputs[t]" is the hidden state at time t (for the last layer).
            # Interleave hidden states and inputs in 'cot_data' shape:
            #   (2 * seq_len, batch_size, input_dim)
            # so that 0,2,4,... are hidden states, and 1,3,5,... are inputs
            cot_data = torch.zeros(2 * seq_len, batch_size, input_dim, device=device)
            for t in range(seq_len):
                # If LSTM, 'new_hidden' is (h, c), but 'all_outputs' is for the last layer
                cot_data[2 * t]     = all_outputs[t]
                cot_data[2 * t + 1] = inputs[t]

            # Keep the full array of hidden states in hidden_data_list
            all_outputs = all_outputs.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
            cot_data = cot_data.permute(1, 0, 2)        # (batch_size, 2*seq_len, input_dim)
            hidden_data_list.append(all_outputs)  # shape: (batch_size, seq_len, hidden_size)
            cot_data_list.append(cot_data)        # shape: (batch_size, 2*seq_len, input_dim)
            input_list.append(inputs)

    # Concatenate along batch axis = dimension 0
    hidden_data = torch.cat(hidden_data_list, dim=0)  # (num_batches, seq_len, batch_size, hidden_size)
    cot_data = torch.cat(cot_data_list, dim=0)        # (num_batches, 2*seq_len, batch_size, input_dim)

    # If desired, you can permute or reshape them further
    # For example, to flatten the first two batch dimensions:
    #   hidden_data = hidden_data.view(-1, seq_len, batch_size, hidden_size)
    # or reorder the dimensions as needed.

    print("hidden_data shape:", hidden_data.shape)
    print("cot_data shape:", cot_data.shape)

    data = {
        'cot_data': cot_data,         # shape (num_batches, 2*seq_len, batch_size, input_dim)
        'hidden_data': hidden_data,   # shape (num_batches, seq_len, batch_size, hidden_size)
        'inputs': input_list
    }
    return data       