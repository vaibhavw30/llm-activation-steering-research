import sys, os
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_hop as rh


class _ToySlice(nn.Module):
    """Stand-in for dct.SlicedModel: nonlinear per-position map + causal cross-position
    mixing, (B,T,d) -> (B,T,d). Frozen weights, like the real model."""
    def __init__(self, d=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(d, d, generator=g, dtype=torch.float64), requires_grad=False)
        self.W2 = nn.Parameter(torch.randn(d, d, generator=g, dtype=torch.float64), requires_grad=False)

    def forward(self, h):
        z = torch.tanh(h @ self.W1)
        return z.cumsum(dim=1) @ self.W2      # position t sees positions <= t


def _setup(B=2, T=4, d=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    sl = _ToySlice(d=d)
    h = torch.randn(B, T, d, generator=g, dtype=torch.float64)
    attn = torch.ones(B, T, dtype=torch.long)
    attn[1, -1] = 0                            # row 1 has one right-pad
    return sl, h, attn, B, T, d


def test_last_nonpad_index():
    _, _, attn, _, T, _ = _setup()
    last = rh.last_nonpad_index(attn)
    assert last.tolist() == [T - 1, T - 2]


def test_f0_equals_clean_forward():
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    out = f(torch.zeros(B, d))
    ref = sl(h)
    last = rh.last_nonpad_index(attn)
    for b in range(B):
        assert torch.allclose(out[b], ref[b, last[b]], atol=1e-6)


def test_broadcast_convention():
    """f(delta) must equal slicing h with delta added at EVERY position of that row."""
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    delta = torch.randn(B, d)
    out = f(delta)
    ref = sl(h + delta[:, None, :])
    last = rh.last_nonpad_index(attn)
    for b in range(B):
        assert torch.allclose(out[b], ref[b, last[b]], atol=1e-6)


def _fd_jacobian(f, B, d, h=1e-4):
    """Full J_b per row by central finite differences. Returns (B, d, d)."""
    J = torch.zeros(B, d, d, dtype=torch.float64)
    for j in range(d):
        e = torch.zeros(B, d, dtype=torch.float64); e[:, j] = h
        J[:, :, j] = (f(e) - f(-e)) / (2 * h)
    return J


def test_vjp_matches_finite_differences():
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    J = _fd_jacobian(f, B, d)
    W = torch.randn(3, d, dtype=torch.float64)
    _, G = rh.vjp_rows(f, torch.zeros(B, d, dtype=torch.float64), W)     # (K,B,d)
    # Toy path runs in float64, so this finite-difference check holds tightly
    # (no float32 roundoff floor to clear).
    for k in range(3):
        for b in range(B):
            ref = J[b].T @ W[k]
            assert torch.allclose(G[k, b], ref, atol=1e-5), (k, b)


def test_transposition_identity():
    """w·(Jv) via jvp == (Jᵀw)·v via vjp — guards Phases 1 and 2 simultaneously."""
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    torch.manual_seed(7)
    w = torch.randn(d, dtype=torch.float64); v = torch.randn(d, dtype=torch.float64)
    _, G = rh.vjp_rows(f, torch.zeros(B, d, dtype=torch.float64), w[None, :])
    _, JT = rh.jvp_cols(f, torch.zeros(B, d, dtype=torch.float64), v.expand(B, d).contiguous())
    for b in range(B):
        lhs = float(JT[b] @ w)
        rhs = float(G[0, b] @ v)
        assert abs(lhs - rhs) < 1e-5


def test_vjp_rows_are_per_row_independent():
    """Row b's gradient must not mix in row b'. Compare batch-of-2 vs single-row runs."""
    sl, h, attn, B, T, d = _setup()
    w = torch.randn(1, d)
    f2 = rh.make_hop(sl, h, attn)
    _, G2 = rh.vjp_rows(f2, torch.zeros(B, d), w)
    for b in range(B):
        f1 = rh.make_hop(sl, h[b:b + 1], attn[b:b + 1])
        _, G1 = rh.vjp_rows(f1, torch.zeros(1, d), w)
        assert torch.allclose(G2[0, b], G1[0, 0], atol=1e-5)
