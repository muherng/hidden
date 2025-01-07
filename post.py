import torch
import torch.nn as nn
import torch.nn.functional as F

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
    with torch.no_grad():
        for i in range(num_batches):
            hidden = model.init_hidden(batch_size)
            inputs = torch.randn(seq_len, batch_size, input_dim)
            h_list = []
            for t in range(seq_len): 
                curr_timestep = inputs[t:t+1,:,:]
                output, hidden = model(curr_timestep, hidden)
                #detach gradients 
                hidden = repackage_hidden(hidden)
                # Initialize cot_data, which interleaves hidden states and inputs
                cot_data = torch.zeros(2 * seq_len, batch_size, input_dim)
                cot_data[2 * t,:,:] = hidden[0,:,:]
                cot_data[2 * t + 1,:,:] = inputs[t,:,:]

                h_list.append(hidden.unsqueeze(0))  # shape => (1, batch_size, hidden_size)
            # Concatenate along the first dimension (time)
            h_all = torch.cat(h_list, dim=0)   # (seq_len, batch_size, hidden_size)
            print("h_all shape:", h_all.shape)  # (seq_len, num_layers, batch_size, hidden_size)
            h_data.append(h_all)
    h_data = torch.cat(h_data,dim=2) 
    print("h_data shape:", h_data.shape)
    # Save hidden states to file
    torch.save(h_data, f'hidden_states/{args.model}_h_data.pt') 
    return 


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
    num_batches = 100  # Number of batches to generate
    batch_size = 32
    seq_len = 10
    input_dim = ninp

    # Initialize lists to store inputs and hidden states
    inputs_list = []
    hidden_list = []
    cot_data_list = []

    for _ in range(num_batches):
        # Initialize the hidden state
        hidden = new_model.init_hidden(batch_size)

        # Create random vector-valued inputs
        inputs = torch.randn(seq_len, batch_size, input_dim)

        # Forward pass through the new model
        output, hidden = new_model(inputs, hidden)

        # Initialize cot_data
        cot_data = torch.zeros(2 * seq_len, batch_size, input_dim)

        # Append inputs and hidden states to the lists
        inputs_list.append(inputs)
        hidden_list.append(hidden)
        print('inputs size', inputs.size())
        print('hidden size', hidden.size())
        # Interleave inputs and hidden states
        for i in range(seq_len):
            cot_data[2 * i,:,:] = hidden[i,:,:]
            cot_data[2 * i + 1,:,:] = inputs[i,:,:]

        # Append cot_data to the list
        cot_data_list.append(cot_data)

    # Save the dataset
    dataset = {'cot_data': cot_data_list}
    torch.save(dataset, 'inputs_hidden_dataset.pt')

    print("Dataset saved successfully.")

    # Now you can use new_model with vector-valued inputs directly

"""     # Initialize the hidden state
    batch_size = 32
    hidden = new_model.init_hidden(batch_size)

    # Create some dummy vector-valued inputs
    seq_len = 10
    input_dim = ninp
    inputs = torch.randn(seq_len, batch_size, input_dim)

    # Forward pass through the new model
    output, hidden = new_model(inputs, hidden)
    #for now we will save input and hidden but not output 


    torch.save() """