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
    def __init__(self, num_samples=1000, seq_len=30, vector_dim=200, rotation_matrix=None):
        """
        Synthetic dataset where each sequence is a fixed rotation of the first vector.

        Args:
            num_samples (int): Number of sequences in the dataset.
            seq_len (int): Length of each sequence.
            vector_dim (int): Dimensionality of each vector.
            rotation_matrix (torch.Tensor): Fixed rotation matrix of shape (vector_dim, vector_dim).
                                             If None, a random orthogonal matrix will be generated.
        """
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
