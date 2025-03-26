#!/usr/bin/env python
import math
import torch
import torch.nn as nn

TOL = 1e-6  # tolerance for dummy identity

###############################################################################
#                           Minimal T0, T1 Modules
###############################################################################

class MiniT0(nn.Module):
    def __init__(self, chunk_size, hidden_dim):
        super().__init__()
        self.chunk_size = chunk_size
        self.lin = nn.Linear(chunk_size, hidden_dim)
    def forward(self, x):
        # x: (B, chunk_size)
        out = self.lin(x)  # (B, hidden_dim)
        return out.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, chunk_size, hidden_dim)

class MiniT1(nn.Module):
    def __init__(self, chunk_size, hidden_dim):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.lin = nn.Linear(2 * hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        # x: (B, 2*chunk_size, hidden_dim)
        B, L, H = x.shape
        assert L == 2*self.chunk_size, f"Expected L={2*self.chunk_size}, got {L}"
        left = x[:, :self.chunk_size, :]
        right = x[:, self.chunk_size:, :]
        merged = torch.cat([left, right], dim=-1)  # (B, chunk_size, 2*hidden_dim)
        out = self.lin(merged)
        out = self.ln(out)
        # Expand back to (B, 2*chunk_size, hidden_dim)
        return out.unsqueeze(1).expand(-1, 2, -1, -1).reshape(B, 2*self.chunk_size, H)

###############################################################################
#             Helper: Combine Two Blocks Using T1 (with identity check)
###############################################################################

def combine_chunks(x, y, model):
    """
    Combines two tensors x and y (each of shape (B, chunk_size, hidden_dim))
    by concatenating them along the token dimension, applying T1, and taking the
    rightmost chunk_size tokens.
    
    If either input is nearly zero (dummy), returns the other.
    """
    if torch.norm(x) < TOL:
        return y
    if torch.norm(y) < TOL:
        return x
    cat = torch.cat([x, y], dim=1)  # (B, 2*chunk_size, hidden_dim)
    out = model.T1(cat)
    if isinstance(out, tuple):
        out = out[0]
    return out[:, -model.chunk_size:, :]

###############################################################################
#      Phase 1: Inclusive Scan via Vectorized Blelloch Scan
###############################################################################

def inclusive_scan(chunks, model, debug=False):
    """
    Given a list of n chunks (each of shape (B, chunk_size, hidden_dim)), stack them into
    X of shape (B, n, chunk_size, hidden_dim). Then compute the inclusive scan Z such that:
        Z[0] = A0,
        Z[1] = T1(A0, A1),
        Z[2] = T1(T1(A0, A1), A2), etc.
    We pad X along the n-dimension to length M = 2^(ceil(log2(n))).
    The upsweep and downsweep phases are vectorized and run in O(log n) sequential steps.
    Returns Z of shape (B, n, chunk_size, hidden_dim).
    """
    # Fix: Unpack shape of a chunk (B, chunk_size, hidden_dim)
    B, csz, hdim = chunks[0].shape
    n = len(chunks)
    device = chunks[0].device
    # Stack chunks: X has shape (B, n, chunk_size, hidden_dim)
    X = torch.stack(chunks, dim=1)
    M = 2 ** math.ceil(math.log2(n))
    if M > n:
        pad_tensor = torch.zeros(B, M - n, csz, hdim, device=device)
        X = torch.cat([X, pad_tensor], dim=1)
    if debug:
        print(f"\n[Inclusive Scan] n={n}, padded M={M}")
        for i in range(M):
            print(f"  X[{i}] norm = {torch.norm(X[:, i]).item():.6f}")
    L_levels = int(math.log2(M))
    # Upsweep phase.
    for d in range(L_levels):
        step = 2 ** (d + 1)
        num_groups = M // step
        if debug:
            print(f"\n[Inclusive Scan Upsweep] Level d={d}, step={step}, num_groups={num_groups}")
        X = X.view(B, num_groups, step, csz, hdim)
        left_index = 2**d - 1
        right_index = step - 1
        if debug:
            print(f"  left_index={left_index}, right_index={right_index}")
        left = X[:, :, left_index, :, :]    # (B, num_groups, chunk_size, hidden_dim)
        right = X[:, :, right_index, :, :]    # (B, num_groups, chunk_size, hidden_dim)
        temp = torch.cat([left, right], dim=2)  # (B, num_groups, 2*chunk_size, hidden_dim)
        temp_flat = temp.view(B * num_groups, 2 * csz, hdim)
        if debug:
            print("  temp_flat shape before T1:", temp_flat.shape)
        temp_out = model.T1(temp_flat)
        if isinstance(temp_out, tuple):
            temp_out = temp_out[0]
        if debug:
            print("  temp_out shape after T1:", temp_out.shape)
        temp_out = temp_out.view(B, num_groups, 2 * csz, hdim)
        combined = temp_out[:, :, -csz:, :]
        X[:, :, right_index, :, :] = combined
        X = X.view(B, M, csz, hdim)
        if debug:
            for i in range(M):
                print(f"  After upsweep d={d}, X[{i}] norm = {torch.norm(X[:, i]).item():.6f}")
    # Downsweep phase.
    # For the inclusive scan, we want the exclusive scan later, so set X[M-1] = dummy (zeros)
    X[:, M-1, :, :] = torch.zeros_like(X[:, M-1, :, :])
    if debug:
        print("\n[Inclusive Scan Downsweep] After setting root to dummy:")
        for i in range(M):
            print(f"  X[{i}] norm = {torch.norm(X[:, i]).item():.6f}")
    for d in reversed(range(L_levels)):
        step = 2 ** (d + 1)
        num_groups = M // step
        if debug:
            print(f"\n[Inclusive Scan Downsweep] Level d={d}, step={step}, num_groups={num_groups}")
        X = X.view(B, num_groups, step, csz, hdim)
        left_index = 2**d - 1
        right_index = step - 1
        if debug:
            print(f"  left_index={left_index}, right_index={right_index}")
        old_left = X[:, :, left_index, :, :].clone()  # (B, num_groups, chunk_size, hidden_dim)
        X[:, :, left_index, :, :] = X[:, :, right_index, :, :]
        right = X[:, :, right_index, :, :]
        old_left_flat = old_left.view(B * num_groups, csz, hdim)
        right_flat = right.view(B * num_groups, csz, hdim)
        combined_flat = combine_chunks(old_left_flat, right_flat, model)
        combined = combined_flat.view(B, num_groups, csz, hdim)
        X[:, :, right_index, :, :] = combined
        X = X.view(B, M, csz, hdim)
        if debug:
            for i in range(M):
                print(f"  After downsweep d={d}, X[{i}] norm = {torch.norm(X[:, i]).item():.6f}")
    # For the inclusive scan, Z is the first n entries.
    Z = X[:, :n, :, :]
    if debug:
        print("\nAfter inclusive scan, final Z norms:")
        for i in range(n):
            print(f"  Z[{i}] norm = {torch.norm(Z[:, i]).item():.6f}")
    return Z

###############################################################################
#      Phase 2: Build Exclusive Scan from Inclusive Scan
###############################################################################

def parallel_prefix_scan(chunks, model, debug=False):
    """
    Computes the exclusive prefix scan P for input chunks using a two-phase approach:
      Phase 1: Compute the inclusive scan Z on the chunks (of shape (B, n, ...)).
      Phase 2: Convert Z to an exclusive scan by setting:
               P[0] = dummy, and for i>=1, P[i] = Z[i-1].
    Returns P of shape (B, n+1, chunk_size, hidden_dim).
    """
    n = len(chunks)
    B, csz, hdim = chunks[0].shape
    device = chunks[0].device
    Z = inclusive_scan(chunks, model, debug=debug)
    if debug:
        print("\n[Exclusive Conversion] Inclusive scan Z norms:")
        for i in range(n):
            print(f"  Z[{i}] norm = {torch.norm(Z[:, i]).item():.6f}")
    dummy = torch.zeros(B, csz, hdim, device=device)
    P_list = [dummy] + [Z[:, i-1, :, :] for i in range(1, n+1)]
    P = torch.stack(P_list, dim=1)
    if debug:
        print("\nFinal P (exclusive scan):")
        for i in range(n+1):
            print(f"  P[{i}] norm = {torch.norm(P[:, i]).item():.6f}")
    return P

###############################################################################
#                Sequential "Binary-Counter" (for Verification)
###############################################################################

def compute_sequential_prefix(model, input_data):
    B, seq_len = input_data.shape
    chunk_size = model.chunk_size
    n = seq_len // chunk_size
    states = []
    for i in range(n):
        x = input_data[:, i*chunk_size:(i+1)*chunk_size]
        out = model.T0(x)
        states.append(out)
    dummy = torch.zeros_like(states[0])
    L = [None]*(n.bit_length()+1)
    prefix_list = []
    for i in range(n):
        cur = states[i]
        j = 0
        while L[j] is not None:
            cat = torch.cat([L[j], cur], dim=1)
            out = model.T1(cat)
            out = out[:, -chunk_size:, :]
            cur = out
            L[j] = None
            j += 1
        L[j] = cur
        agg = None
        for k in reversed(range(len(L))):
            if L[k] is not None:
                if agg is None:
                    agg = L[k]
                else:
                    cat = torch.cat([agg, L[k]], dim=1)
                    out = model.T1(cat)
                    out = out[:, -chunk_size:, :]
                    agg = out
        prefix_list.append(agg)
    P_seq = [dummy] + prefix_list
    return torch.stack(P_seq, dim=1)

###############################################################################
#                                  MAIN
###############################################################################

def main():
    torch.manual_seed(0)
    chunk_size = 32
    hidden_dim = 256
    n_chunks = 3
    batch = 1

    class MyModel(nn.Module):
        def __init__(self, chunk_size, hidden_dim):
            super().__init__()
            self.chunk_size = chunk_size
            self.hidden_dim = hidden_dim
            self.T0 = MiniT0(chunk_size, hidden_dim)
            self.T1 = MiniT1(chunk_size, hidden_dim)
        def vectorized_prefix_scan(self, states, dummy, debug=False):
            # Call the exclusive scan: compute inclusive scan and then convert.
            return parallel_prefix_scan(states, self, debug=debug)
    model = MyModel(chunk_size, hidden_dim)
    input_data = torch.randn(batch, n_chunks * chunk_size)
    print("Input shape:", input_data.shape)

    # (A) Sequential prefix (binary-counter) for verification.
    P_seq = compute_sequential_prefix(model, input_data)
    print(f"\nP_seq shape => {P_seq.shape}")
    for i in range(P_seq.size(1)):
        print(f"  P_seq[{i}] norm = {torch.norm(P_seq[:, i]).item():.6f}")

    # (B) Build chunk states using T0.
    chunk_list = []
    for i in range(n_chunks):
        x = input_data[:, i*chunk_size:(i+1)*chunk_size]
        out = model.T0(x)
        chunk_list.append(out)

    # (C) Parallel prefix scan using the vectorized Blelloch approach.
    P_vec = model.vectorized_prefix_scan(chunk_list, torch.zeros_like(chunk_list[0]), debug=True)
    print(f"\nP_vec shape => {P_vec.shape}")
    for i in range(P_vec.size(1)):
        print(f"  P_vec[{i}] norm = {torch.norm(P_vec[:, i]).item():.6f}")

    diff = torch.abs(P_seq - P_vec)
    print(f"\nOverall prefix diff => max = {diff.max().item():.6f}, mean = {diff.mean().item():.6f}")
    for i in range(P_seq.size(1)):
        seq_flat = P_seq[:, i].reshape(-1)
        vec_flat = P_vec[:, i].reshape(-1)
        diff_flat = torch.abs(seq_flat - vec_flat)
        print(f"Prefix index {i}: max diff = {diff_flat.max().item():.6f}, mean diff = {diff_flat.mean().item():.6f}")
        print(f"  P_seq[{i}][:10] = {seq_flat[:10].tolist()}")
        print(f"  P_vec[{i}][:10] = {vec_flat[:10].tolist()}")

if __name__=="__main__":
    main()
