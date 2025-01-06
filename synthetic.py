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
