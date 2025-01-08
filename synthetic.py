import torch
from torch.utils.data import Dataset

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

from train import batchify, repackage_hidden, get_batch, train, export_onnx
from new_post import RNNModelWithoutEmbedding, collect_hidden_states_RNN

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
        #Load the original model
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
        batch_size = 100
        num_batches = int(self.num_samples/batch_size)  # Number of batches to generate
        seq_len = self.seq_len
        input_dim = ninp

        data = collect_hidden_states_RNN(new_model, seq_len, batch_size, num_batches, input_dim)
        torch.save(data, f'hidden_states/RNN_TANH_data.pt') 
        # Initialize lists to store inputs and hidden states
        return data['cot_data']

