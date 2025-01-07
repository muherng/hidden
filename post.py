import torch
import torch.nn as nn
import torch.nn.functional as F

from train import batchify, repackage_hidden, get_batch, train, export_onnx

class RNNModelWithoutEmbedding(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModelWithoutEmbedding, self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
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

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        #output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# filepath: /raid/lingo/morrisyau/examples/word_language_model/load_model.py
#import torch
#from model import RNNModelWithoutEmbedding

#TODO: Handle deep RNNs
def collect_hidden_states_RNN(model, seq_len, batch_size, num_batches): 
    # Turn on evaluation mode which disables dropout.
    cot_data = []
    hidden_data = []
    input_data = []
    with torch.no_grad():
        for i in range(num_batches):
            hidden = model.init_hidden(batch_size)
            input_batch = torch.randn(seq_len, batch_size, input_dim)
            hidden_batch = []
            # Initialize cot_data, which interleaves hidden states and inputs
            cot_batch = torch.zeros(2 * seq_len, batch_size, input_dim)
            input_data.append(input_batch)
            for t in range(seq_len): 
                curr_timestep = input_batch[t:t+1,:,:]
                output, hidden = model(curr_timestep, hidden)
                #detach gradients 
                hidden = repackage_hidden(hidden)
                print('hidden size: ', hidden.size())
                #access only layer 1 of hidden states
                cot_batch[2 * t,:,:] = hidden[0,:,:]
                cot_batch[2 * t + 1,:,:] = input_batch[t,:,:]
                hidden_batch.append(hidden.unsqueeze(0))  # shape => (1, batch_size, hidden_size)
            # Concatenate along the first dimension (time)
            hidden_batch = torch.cat(hidden_batch, dim=0)   # (seq_len, num_layers, batch_size, hidden_size)
            print("hidden_batch shape:", hidden_batch.shape)  # (seq_len, num_layers, batch_size, hidden_size)
            hidden_data.append(hidden_batch)
            cot_data.append(cot_batch)
    hidden_data = torch.cat(hidden_data,dim=2) 
    cot_data = torch.cat(cot_data,dim=1)
    print("hidden_data shape:", hidden_data.shape)
    print('cot_data shape: ', cot_data.size())
    data = {'cot_data': cot_data, 'hidden_data': hidden_data, 'inputs': input_data}
    return data


if __name__ == '__main__': 
    # Load the original model
    original_model = torch.load('saved_models/RNN_TANH/model.pt')
    original_model.eval()  # Set the model to evaluation mode

    # Define the new model
    rnn_type = 'RNN_TANH'  # or 'GRU', depending on your original model
    ntoken = original_model.decoder.out_features
    ninp = original_model.encoder.embedding_dim  # Use the same input dimension as the embedding output
    nhid = original_model.rnn.hidden_size
    nlayers = original_model.rnn.num_layers
    dropout = original_model.drop.p

    new_model = RNNModelWithoutEmbedding(rnn_type, ntoken, ninp, nhid, nlayers, dropout)

    # Load the weights from the original model, skipping the embedding layer
    new_model.rnn.load_state_dict(original_model.rnn.state_dict())
    new_model.decoder.load_state_dict(original_model.decoder.state_dict())
    new_model.eval()

        # Parameters for dataset generation
    num_batches = 10**4  # Number of batches to generate
    batch_size = 32
    seq_len = 10
    input_dim = ninp

    data = collect_hidden_states_RNN(new_model, seq_len, batch_size, num_batches)
    torch.save(data, f'hidden_states/RNN_TANH_data.pt') 
    # Initialize lists to store inputs and hidden states