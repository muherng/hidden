import torch
from torch.utils.data import Dataset
from data import Corpus

# ----------------------------------------------------
# 1. Synthetic Dataset of Random Vectors
# ----------------------------------------------------
class RandomVectorDataset(Dataset):
    """
    Each sample is a sequence of length seq_len (30), where each vector has dimension vector_dim (200).
    We'll generate random floats in [0,1).
    """
    def __init__(self, num_samples=1000, seq_len=30, vector_dim=200, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)  # Ensure reproducibility per split
        self.seq_len = seq_len
        self.vector_dim = vector_dim
        self.data = []
        for _ in range(num_samples):
            # shape: (seq_len, vector_dim)
            seq = torch.rand(seq_len, vector_dim)
            self.data.append(seq)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        We'll return a dictionary with 'inputs' and 'labels'.
        For autoregressive next-step prediction:
          inputs: shape (seq_len, vector_dim)
          labels: same shape (seq_len, vector_dim)
        """
        seq = self.data[idx]
        return {
            "inputs": seq,     # (30, 200)
            "labels": seq      # also (30, 200) - we'll do shift-by-1 in the model
        }

class FixedRotationDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=30, vector_dim=200, rotation_matrix=None, seed=None):
        """
        Synthetic dataset where each sequence is a fixed rotation of the first vector.

        Args:
            num_samples (int): Number of sequences in the dataset.
            seq_len (int): Length of each sequence.
            vector_dim (int): Dimensionality of each vector.
            rotation_matrix (torch.Tensor): Fixed rotation matrix of shape (vector_dim, vector_dim).
                                             If None, a random orthogonal matrix will be generated.
        """
        if seed is not None:
            torch.manual_seed(seed)  # Ensure reproducibility per split
        
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vector_dim = vector_dim

        # Generate a fixed rotation matrix if none is provided
        if rotation_matrix is None:
            self.rotation_matrix = self.generate_random_rotation_matrix(vector_dim)
        else:
            self.rotation_matrix = rotation_matrix

        # Generate the dataset
        self.data = self.generate_sequences()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "inputs": self.data[idx],  # Input sequence
            "labels": self.data[idx]  # Same as inputs for next-step prediction
        }

    def generate_random_rotation_matrix(self, dim):
        """
        Generates a random orthogonal rotation matrix using QR decomposition.

        Args:
            dim (int): Dimension of the rotation matrix.

        Returns:
            torch.Tensor: Orthogonal rotation matrix of shape (dim, dim).
        """
        q, _ = torch.linalg.qr(torch.randn(dim, dim))  # QR decomposition
        return q

    def generate_sequences(self):
        """
        Generate sequences where each vector is a fixed rotation of the previous vector.

        Returns:
            List[torch.Tensor]: List of sequences, each of shape (seq_len, vector_dim).
        """
        sequences = []
        for _ in range(self.num_samples):
            # Generate the first random vector
            first_vector = torch.randn(self.vector_dim)

            # Generate the sequence by applying the rotation matrix repeatedly
            sequence = [first_vector]
            for _ in range(1, self.seq_len):
                next_vector = sequence[-1] @ self.rotation_matrix  # Rotate the previous vector
                sequence.append(next_vector)

            # Stack the sequence into a tensor
            sequences.append(torch.stack(sequence))

        return sequences


class LinearDynamicsDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=30, vector_dim=10, A=None, B=None, seed=None):
        """
        Synthetic dataset based on the linear dynamics x_{t+1} = Ax_t + Bu_t.

        Args:
            num_samples (int): Number of sequences in the dataset.
            seq_len (int): Length of each sequence.
            vector_dim (int): Dimensionality of the state vector x_t.
            control_dim (int): Dimensionality of the control vector u_t.
            seed (int): Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vector_dim = vector_dim

        # Generate rotation matrices A and B
        if A is None or B is None: 
            self.A = self.generate_random_rotation_matrix(vector_dim)
            self.B = self.generate_random_rotation_matrix(vector_dim)
        else: 
            self.A = A
            self.B = B    
        # Generate the dataset
        self.data = self.generate_sequences()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "inputs": self.data[idx],  # Input sequence
            "labels": self.data[idx]  # Same as inputs for next-step prediction
        }

    def generate_random_rotation_matrix(self, dim):
        """
        Generates a random orthogonal rotation matrix using QR decomposition.

        Args:
            dim (int): Dimension of the rotation matrix.

        Returns:
            torch.Tensor: Orthogonal rotation matrix of shape (dim, dim).
        """
        q, _ = torch.linalg.qr(torch.randn(dim, dim))  # QR decomposition
        return q

    def generate_sequences(self):
        """
        Generate sequences of the form x_0, u_0, x_1, u_1, ..., x_t.

        Returns:
            List[torch.Tensor]: List of sequences, each of shape (seq_len, vector_dim + control_dim).
        """
        sequences = []
        for _ in range(self.num_samples):
            # Initialize x_0 (state vector) and generate sequence
            x_t = torch.randn(self.vector_dim)
            sequence = []

            # dividing seq_len by 2 to get the number of time steps (this is ad hoc)
            for _ in range(int(self.seq_len/2)):
                # Generate random control vector u_t
                u_t = torch.randn(self.vector_dim)

                # Append x_t and u_t to the sequence
                #sequence.append(torch.cat((x_t, u_t)))
                sequence.extend([x_t,u_t])

                # Compute x_{t+1} using linear dynamics
                x_t = torch.tanh(self.A @ x_t + self.B @ u_t)

            # Stack the sequence into a tensor
            sequences.append(torch.stack(sequence))
        return sequences

from post import RNNModelWithoutEmbedding, collect_hidden_states_RNN

class CustomRNN:
    def __init__(self, input_dim, hidden_dim, original_model, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_ih = original_model.rnn.weight_ih_l0
        self.W_hh = original_model.rnn.weight_hh_l0
        self.b_ih = original_model.rnn.bias_ih_l0
        self.b_hh = original_model.rnn.bias_hh_l0

    def forward(self, input_tensor, hidden):
        outputs = []
        all_hidden_states = [hidden.unsqueeze(0)]
        for t in range(input_tensor.size(0)):
            hidden = torch.tanh(
                input_tensor[t] @ self.W_ih.T + self.b_ih + 
                hidden @ self.W_hh.T + self.b_hh
            )
            outputs.append(hidden.unsqueeze(0))
            all_hidden_states.append(hidden.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        all_hidden_states = torch.cat(all_hidden_states, dim=0)
        return outputs, hidden, all_hidden_states

    def init_hidden(self, bsz):
        #return torch.zeros(bsz, self.hidden_dim, device=device)
        return torch.randn(bsz,self.hidden_dim, device=self.device)

    def to(self, device):
        self.W_ih = self.W_ih.to(device)
        self.W_hh = self.W_hh.to(device)
        self.b_ih = self.b_ih.to(device)
        self.b_hh = self.b_hh.to(device)


class RNN_TANH_Dataset(Dataset): 
    def __init__(self, num_samples=1000, seq_len=30, vector_dim=10, A=None, B=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vector_dim = vector_dim

        #TODO: generate the dataset 
        #load the dataset
        #self.data = torch.load('hidden_states/RNN_TANH_data.pt')['cot_data']
        self.data = self.generate_sequences()
        print('data size: ', self.data.size())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "inputs": self.data[idx],  # Input sequence
            "labels": self.data[idx]  # Same as inputs for next-step prediction
        }   
    
    def generate_sequences(self):   
        # Specify the device as CUDA
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the original model and move it to the device
        original_model = torch.load(f'saved_models/RNN_TANH/{self.vector_dim}_model.pt', map_location=device)
        original_model.to(device)

        # Define the new model
        input_dim = original_model.encoder.embedding_dim  # Use the same input dimension as the embedding output
        hidden_dim = original_model.rnn.hidden_size
        #nlayers = original_model.rnn.num_layers
        seq_len = self.seq_len
        num_samples = self.num_samples
        #input_dim = ninp
        print("Original RNN state_dict:")
        for param_tensor in original_model.rnn.state_dict():
            print(param_tensor, "\t", original_model.rnn.state_dict()[param_tensor].size())

        new_model = CustomRNN(input_dim, hidden_dim, original_model, device=device)
        new_model.to(device)

        all_inputs, all_hidden_states = generate_random_inputs_and_states(new_model, num_samples, int(seq_len/2), input_dim, device=device)
        data = interweave_inputs_and_hidden_states(all_inputs, all_hidden_states)
        torch.save(data, f'hidden_states/RNN_TANH_{input_dim}_data.pt') 
        return data.cpu()


#TODO: handle deep RNNs 
def interweave_inputs_and_hidden_states(input_tensor, all_hidden_states):
    """
    Interweaves input_tensor and all_hidden_states into a tensor T of shape (num_samples, 2*seq_len, input_dim).
    """
    num_samples, seq_len, input_dim = input_tensor.shape

    # Initialize the output tensor T with the desired shape
    T = torch.zeros(num_samples, 2 * seq_len, input_dim, device=input_tensor.device)

    # Interweave the tensors
    for t in range(seq_len):
        T[:, 2 * t, :] = all_hidden_states[:, t, :]
        T[:, 2 * t + 1, :] = input_tensor[:, t, :]

    return T

def generate_random_inputs_and_states(custom_rnn, num_samples, seq_len, input_dim, device='cpu'):
    """
    Generates random inputs of shape (num_samples, seq_len, input_dim),
    feeds them through the CustomRNN, and returns:
      - the randomly generated input tensor
      - the hidden states for all timesteps of shape (num_samples, seq_len, hidden_dim)
    """
    with torch.no_grad():
        # Create random inputs with desired shape
        input_tensor = torch.randn(num_samples, seq_len, input_dim, device=device)
        
        # Permute to match the CustomRNN input format (seq_len, batch_size, input_dim)
        input_for_rnn = input_tensor.permute(1, 0, 2).contiguous()
        
        # Initialize hidden state
        hidden = custom_rnn.init_hidden(num_samples)
        
        # Forward pass
        outputs, final_hidden, all_hidden_states = custom_rnn.forward(input_for_rnn, hidden)
        
        # all_hidden_states is shape (seq_len, batch_size, hidden_dim), permute to (batch_size, seq_len, hidden_dim)
        all_hidden_states = all_hidden_states.permute(1, 0, 2).contiguous()

        if debug == True: 
            # Check that all_hidden_states satisfies the RNN dynamics
            for a in range(num_samples):
                for t in range(seq_len - 1):
                    if not torch.allclose(all_hidden_states[a, t + 1, :], torch.tanh(input_tensor[a, t, :] + all_hidden_states[a, t, :]), atol=1e-6):
                        print(f"RNN dynamics check failed at sample {a}, timestep {t}")
                        return None, None

        #check that all_hidden_states satisfies the RNN dynamics.  
    
    # Return both the original input (num_samples, seq_len, input_dim) and the hidden states
    return input_tensor, all_hidden_states 


def check_rnn_equivalence(original_model, new_model, input_tensor):
    original_model.eval()
    new_model.eval()

    device = input_tensor.device
    original_model.to(device)
    new_model.to(device)

    with torch.no_grad():
        # Get the hidden state from the original model
        original_hidden = original_model.init_hidden(input_tensor.size(1))
        original_output, original_hidden = original_model.rnn(input_tensor, original_hidden)

        # Get the hidden state from the new model
        new_hidden = new_model.init_hidden(input_tensor.size(1))
        new_output, new_hidden = new_model.rnn(input_tensor, new_hidden)

    # Check if the outputs and hidden states are the same
    output_equal = torch.allclose(original_output, new_output, atol=1e-6)
    hidden_equal = all(torch.allclose(oh, nh, atol=1e-6) for oh, nh in zip(original_hidden, new_hidden))

    return output_equal and hidden_equal


class RNN_Dataset(Dataset): 
    def __init__(self, num_samples=1000, seq_len=30, vector_dim=10, num_layers = 1, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vector_dim = vector_dim
        self.num_layers = num_layers

        #TODO: generate the dataset 
        #load the dataset
        self.data, self.mask = self.generate_sequences()
        print('data size: ', self.data.size())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "inputs": self.data[idx],  # Input sequence
            "labels": self.data[idx],  # Same as inputs for next-step prediction
            "mask": self.mask
        }   
    
    def generate_sequences(self):   
        # Specify the device as CUDA
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the original model and move it to the device
        model = torch.load(f'saved_models/RNN_TANH/{self.vector_dim}_{self.num_layers}model.pt', map_location=device)
        model.to(device)

        # Define the new model
        input_dim = model.encoder.embedding_dim  # Use the same input dimension as the embedding output
        hidden_dim = model.rnn.hidden_size
        nlayers = model.rnn.num_layers
        seq_len = self.seq_len
        num_samples = self.num_samples
        #input_dim = ninp
        print("Original RNN state_dict:")
        for param_tensor in model.rnn.state_dict():
            print(param_tensor, "\t", model.rnn.state_dict()[param_tensor].size())

        #all_inputs, all_hidden_states = generate_random_inputs_and_states(new_model, num_samples, int(seq_len/2), input_dim, device=device)
        #data = interweave_inputs_and_hidden_states(all_inputs, all_hidden_states)
        
        with torch.no_grad():
            # Create random inputs with desired shape
            input_tensor = torch.randn(num_samples, seq_len, input_dim, device=device)
            
            # Permute to match the torch RNN input format (seq_len, batch_size, input_dim)
            input_rnn = input_tensor.permute(1, 0, 2).contiguous()

            data, mask = model.collect_hidden_states_RNN(input_rnn, model.init_hidden(num_samples))
            data = data.permute(1, 0, 2).contiguous()
            
            # all_hidden_states is shape (seq_len, batch_size, hidden_dim), permute to (batch_size, seq_len, hidden_dim)
            #all_hidden_states = all_hidden_states.permute(1, 0, 2).contiguous()
        
        #torch.save(data, f'hidden_states/RNN_TANH_{input_dim}_data.pt') 
        return data.cpu(), mask.cpu()   

from train import batchify, repackage_hidden, train, export_onnx

class LSTM_Dataset(Dataset): 
    def __init__(self, num_samples=1000, seq_len=30, vector_dim=10, num_layers = 1, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vector_dim = vector_dim
        self.num_layers = num_layers

        #TODO: generate the dataset 
        #load the dataset
        self.data, self.mask, self.mask_out, self.ntokens = self.generate_sequences()
        print('ntokens: ', self.ntokens)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "inputs": self.data[idx],  # Input sequence
            "labels": self.data[idx],  # Same as inputs for next-step prediction
        }   
    
    def generate_sequences(self,mode='dataset'):   
        # Specify the device as CUDA
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the original model and move it to the device
        model = torch.load(f'saved_models/LSTM/{self.vector_dim}_{self.num_layers}model.pt', map_location=device)
        model.to(device)

        # Define the new model
        input_dim = model.encoder.embedding_dim  # Use the same input dimension as the embedding output
        hidden_dim = model.rnn.hidden_size
        nlayers = model.rnn.num_layers
        seq_len = self.seq_len
        num_samples = self.num_samples
        #input_dim = ninp
        print("Original RNN state_dict:")
        for param_tensor in model.rnn.state_dict():
            print(param_tensor, "\t", model.rnn.state_dict()[param_tensor].size())

        #all_inputs, all_hidden_states = generate_random_inputs_and_states(new_model, num_samples, int(seq_len/2), input_dim, device=device)
        #data = interweave_inputs_and_hidden_states(all_inputs, all_hidden_states)
        
        mode = 'dataset'
        #mode = 'load'
        batch_size = 20

        def get_batch(source, i, seq_len):
            seq_len = min(seq_len, len(source) - 1 - i)
            data = source[i:i+seq_len]
            target = source[i+1:i+1+seq_len].view(-1)

            #print('data', data.size())
            #raise ValueError('stop here')
            return data, target

        with torch.no_grad():
            # Create random inputs with desired shape
            if mode == 'random': 
                input_tensor = torch.randn(num_samples, seq_len, input_dim, device=device)
                #hidden = (torch.randn(2,num_samples,input_dim, device='cuda'), torch.randn(2,num_samples,input_dim,device='cuda'))
                # Permute to match the torch RNN input format (seq_len, num_samples, input_dim)
                input_rnn = input_tensor.permute(1, 0, 2).contiguous()
                data, mask = model.collect_hidden_states_LSTM(input_rnn, model.init_hidden(num_samples))
                data_total = data.permute(1, 0, 2).contiguous()
            if mode == 'dataset':
                corpus = Corpus('./data/wikitext-2')
                ntokens = len(corpus.dictionary)
                print('ntokens: ', ntokens)
                train_data = batchify(corpus.train, batch_size, device=device)
                #input_tokens = batchify(corpus.train, batch_size, device) 
                ntokens = len(corpus.dictionary)
                hidden = model.init_hidden(batch_size)
                out = torch.zeros(1, batch_size, input_dim, device=device)
                data_total = []
                token_data = []
                mask = []
                mask_out = []
                max_data = train_data.size(0)
                print('train_data.size(): ', train_data.size()) 
                print('max_data: ', max_data)
                raise ValueError('stop here')
                for batch, i in enumerate(range(0, max_data, seq_len)):
                    input_batch, targets = get_batch(train_data, i, seq_len)
                    data_batch, mask_batch, mask_out, hidden, out = model.collect_hidden_from_tokens(hidden,out,input_batch)
                    if data_batch.size(0) != seq_len*(2*self.num_layers + 2):
                        raise ValueError('skipping batch: ', batch)
                    if batch == 0: 
                        mask = mask_batch 
                        mask_out = mask_out
                    data_total.append(data_batch)
                    token_data.append(input_batch)
                    print('data_batch.size():', data_batch.size())
                    #raise ValueError('Not implemented yet')
                    #output, hidden = model(data, hidden)
                data_total = torch.cat(data_total, dim=1)
                data_total = data_total.permute(1, 0, 2).contiguous()
                if data_total.size(0) > num_samples:
                    print('data_total.size() before', data_total.size())
                    data_total = data_total[:num_samples]
                else: 
                    self.num_samples = data_total.size(0)
                token_data = torch.cat(token_data, dim=1)
                token_data = token_data.permute(1, 0).contiguous()
                print('data_total.size()', data_total.size())
                print('train_data.size()', train_data.size())
                print("token_data.size()", token_data.size())
                raise ValueError('Not implemented yet')
                #print('data_total.size(): ', data_total.size())
                torch.save(data_total, f'hidden_states/LSTM_{input_dim}_{self.num_layers}_{self.seq_len}data.pt') 
                torch.save(mask, f'hidden_states/LSTM_{input_dim}_{self.num_layers}_{self.seq_len}mask.pt')
            if mode == 'load': 
                raise ValueError('Not implemented yet')
                data_total = torch.load(f'hidden_states/LSTM_{input_dim}_{self.num_layers}_{self.seq_len}data.pt')
                mask = torch.load(f'hidden_states/LSTM_{input_dim}_{self.num_layers}_{self.seq_len}mask.pt')
                print('data_total.size(): ', data_total.size())
                print('mask.size(): ', mask.size())
    
        return data_total.cpu(), mask.cpu(), mask_out.cpu(), ntokens  