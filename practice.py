import torch
import torch.nn as nn

# Hyperparameters
batch_size = 5
seq_len = 10
input_size = 7
hidden_size = 16
num_layers = 2

# Define an LSTM
lstm = nn.LSTM(input_size, hidden_size, num_layers)

# Create some input data: (seq_len, batch_size, input_size)
x = torch.randn(seq_len, batch_size, input_size)

# Initialize hidden and cell states for all layers
h = torch.zeros(num_layers, batch_size, hidden_size)
c = torch.zeros(num_layers, batch_size, hidden_size)

# Lists to store h and c at each timestep
h_list = []
c_list = []

# Process one timestep at a time
for t in range(seq_len):
    # Current timestep input: shape (1, batch_size, input_size)
    x_t = x[t : t+1]          
    
    # Feed it through the LSTM, along with the current (h, c)
    out, (h, c) = lstm(x_t, (h, c))
    
    # out: (1, batch_size, hidden_size)
    # h, c: (num_layers, batch_size, hidden_size)
    
    # Save hidden/cell states for this timestep
    h_list.append(h.unsqueeze(0))  # shape => (1, num_layers, batch_size, hidden_size)
    c_list.append(c.unsqueeze(0))  # shape => (1, num_layers, batch_size, hidden_size)

# Concatenate along the first dimension (time)
h_all = torch.cat(h_list, dim=0)   # (seq_len, num_layers, batch_size, hidden_size)
c_all = torch.cat(c_list, dim=0)   # (seq_len, num_layers, batch_size, hidden_size)

print("h_all shape:", h_all.shape)  # (seq_len, num_layers, batch_size, hidden_size)
print("c_all shape:", c_all.shape)  # (seq_len, num_layers, batch_size, hidden_size)
