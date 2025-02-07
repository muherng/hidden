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
        self.data, self.mask, self.mask_out, self.token_data, self.ntoken = self.generate_sequences()


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "inputs": self.data[idx],  # Input sequence
            "labels": self.data[idx],  # Same as inputs for next-step prediction
            "tokens": self.token_data[idx]
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
        print("Original RNN state_dict:")
        for param_tensor in model.rnn.state_dict():
            print(param_tensor, "\t", model.rnn.state_dict()[param_tensor].size())
        
        mode = 'dataset'
        #mode = 'load'
        #This batch size is just used to speed up the process of collecting the hidden states
        #The LSTM is run on the entire dataset but its hidden state is reset to zero batch_size times.  
        batch_size = 20

        def get_batch(source, i, seq_len):
            seq_len = min(seq_len, len(source) - 1 - i)
            data = source[i:i+seq_len]
            target = source[i+1:i+1+seq_len].view(-1)
            return data, target

        with torch.no_grad():
            if mode == 'dataset':
                corpus = Corpus('./data/wikitext-2')
                ntoken = len(corpus.dictionary)
                #print('ntoken: ', ntoken)
                train_data = batchify(corpus.train, batch_size, device=device)
                total_tokens = train_data.size(0) * train_data.size(1)
                hidden = model.init_hidden(batch_size)
                out = torch.zeros(1, batch_size, input_dim, device=device)
                data_total = []
                token_data = []
                mask = []
                mask_out = []
                max_data = min(int(self.num_samples*seq_len/batch_size), train_data.size(0))
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
                data_total = torch.cat(data_total, dim=1)
                data_total = data_total.permute(1, 0, 2).contiguous()
                if data_total.size(0) > num_samples:
                    data_total = data_total[:num_samples]
                else: 
                    self.num_samples = data_total.size(0)
                token_data = torch.cat(token_data, dim=1)
                token_data = token_data.permute(1, 0).contiguous()
                torch.save(data_total, f'hidden_states/LSTM_{input_dim}_{self.num_layers}_{self.seq_len}data.pt') 
                torch.save(mask, f'hidden_states/LSTM_{input_dim}_{self.num_layers}_{self.seq_len}mask.pt')
            if mode == 'load': 
                raise ValueError('Not implemented yet')
    
        return data_total.cpu(), mask.cpu(), mask_out.cpu(), token_data.cpu(), ntoken  