import math
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn

class BlellochScan(nn.Module):
    """
    A parallel Blelloch scan calls `combine_fn(left, right)` on every pair.
    Works on inputs X_in of shape (B, L, D, N), with L a power of two.
    """

    def __init__(self, combine_fn, identity=None, inclusive=True):
        super().__init__()
        self.combine_fn = combine_fn
        self.identity   = identity
        self.inclusive = inclusive

    def forward(self, X_in):
        # X_in: (B, L, D, N)
        B, L, D, N = X_in.shape
        P = 1 << (L - 1).bit_length() # padded length P = next power of two ≥ L
        # assert (L & (L - 1)) == 0, "L must be a power of two"

        # prepare identity (e.g. zero for additive scan)
        if self.identity is None:
            id_val = torch.zeros(B, D, N, device=X_in.device, dtype=X_in.dtype) # requires_grad_(False)
        else:
            id_val = self.identity.to(X_in.device).expand(B, D, N)

        # reformat to (B, D, L, N)
        X = X_in.transpose(2, 1).contiguous()
        if P != L:
            pad = id_val.unsqueeze(2).expand(B, D, P - L, N)
            X = torch.cat([X, pad], dim=2)               # → (B, D, P, N)
        X_orig = X.clone()


        levels = int(math.log2(P))

        # --- UPSWEEP (reduction) ---
        for lvl in range(levels):
            step = 2 ** lvl
            idx_l = torch.arange(step - 1, P, 2 * step, device=X.device)
            idx_r = idx_l + step

            left  = X[:, :, idx_l, :]
            right = X[:, :, idx_r, :]

            merged = self.combine_fn(left, right)
            X_next = X.clone()
            X_next[:, :, idx_r, :] = merged  # in-place on Xn but Xn is non-leaf, so PyTorch records gradients
            X = X_next

        # --- DOWNSWEEP (distribution) ---
        X_next = X.clone()
        X_next[:, :, -1, :] = id_val
        X = X_next

        for lvl in reversed(range(levels)):
            step = 2 ** lvl
            idx_l = torch.arange(step - 1, P, 2 * step, device=X.device)
            idx_r = idx_l + step

            old_l = X[:, :, idx_l, :].clone()
            old_r = X[:, :, idx_r, :]

            new_l = old_r                                   # ← set left to old right
            new_r = self.combine_fn(old_l, old_r)           # ← combine old left, old right

            X_next = X.clone()
            X_next[:, :, idx_l, :] = new_l
            X_next[:, :, idx_r, :] = new_r
            X = X_next

        # X is now the *exclusive* scan in (B, D, L, N).
        # To get *inclusive*, do combine(prefix_excl, original):
        if self.inclusive:
            X_incl = self.combine_fn(X, X_orig)
            return X_incl[:, :, :L, :].transpose(2, 1)   # → (B, L, D, N)

        return X[:, :, :L, :].transpose(2, 1)   # → (B, L, D, N)



import torch
def test():
    seq = torch.arange(1, 9, dtype=torch.float32)
    B, L = 1, seq.size(0)
    D, N = 1, 1

    # build input: (B, L, D, N)
    X = seq.view(B, L, D, N).requires_grad_(True)

    # our combiner: just add
    comb = lambda u, v: u + v

    # run generic scan
    scan = BlellochScan(comb, inclusive=True)
    out  = scan(X)   # (B, L, D, N)

    # should match cumsum along L
    expected = torch.cumsum(X, dim=1)
    print('-------- Inclusive Scan --------')
    print('Input:', X.flatten())
    print('Output:', out.flatten())
    print('Expected:', expected.flatten())
    assert torch.allclose(out, expected), f"forward mismatch {out.flatten()} vs {expected.flatten()}"

    # now test backward: sum(out) => dL/dX[i] = #of times x[i] contributes = L-i
    loss = out.sum()
    loss.backward()

    exp_grad = torch.tensor([L - i for i in range(L)], dtype=torch.float32)
    got_grad = X.grad.view(-1)
    assert torch.allclose(got_grad, exp_grad), f"backward mismatch {got_grad} vs {exp_grad}"

    print("BlellochScan (inclusive add) forward + backward working correctly!")

    #  ---- Exclusive test ----
    print('\n-------- Exclusive Scan --------')
    X.grad.zero_()
    
    scan = BlellochScan(comb, inclusive=False)
    out  = scan(X)

    # expected exclusive: [0, x0, x0+x1, ..., sum up to x[L-2]]
    zeros = torch.zeros(B, 1, D, N, dtype=X.dtype, device=X.device)
    shifted = torch.cumsum(X, dim=1)[:, :-1, ...]
    expected = torch.cat([zeros, shifted], dim=1)

    print("Output: ", out.flatten())
    print("Expected: ", expected.flatten())
    assert torch.allclose(out, expected), "Exclusive forward mismatch"

    # backward for exclusive: grad[i] = number of outputs j>i
    # since out_excl[j] = sum_{k<j} x[k],  sum_j out_excl[j] includes x[i] in positions j=i+1..L-1,
    # so grad[i] = (L-1 - i)
    loss_excl = out.sum()
    loss_excl.backward()
    exp_grad_excl = torch.tensor([L - 1 - i for i in range(L)], dtype=torch.float32)
    got_grad_excl = X.grad.view(-1)
    assert torch.allclose(got_grad_excl, exp_grad_excl), f"Exclusive backward mismatch {got_grad_excl} vs {exp_grad_excl}"
    print("BlellochScan (exclusive add) forward + backward working correctly!")



if __name__ == "__main__":
    test()

