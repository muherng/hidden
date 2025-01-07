import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModelWithoutEmbedding(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModelWithoutEmbedding, self).__init__()
        self.drop = nn.Dropout(dropout)
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
        output = self.drop(output)
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

    # Now you can use new_model with vector-valued inputs directly

    # Initialize the hidden state
    batch_size = 32
    hidden = new_model.init_hidden(batch_size)

    # Create some dummy vector-valued inputs
    seq_len = 10
    input_dim = ninp
    inputs = torch.randn(seq_len, batch_size, input_dim)

    # Forward pass through the new model
    output, hidden = new_model(inputs, hidden)
    print(output)