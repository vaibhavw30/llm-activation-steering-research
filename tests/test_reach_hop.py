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
        self.W1 = nn.Parameter(torch.randn(d, d, generator=g), requires_grad=False)
        self.W2 = nn.Parameter(torch.randn(d, d, generator=g), requires_grad=False)

    def forward(self, h):
        z = torch.tanh(h @ self.W1)
        return z.cumsum(dim=1) @ self.W2      # position t sees positions <= t


def _setup(B=2, T=4, d=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    sl = _ToySlice(d=d)
    h = torch.randn(B, T, d, generator=g)
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
    J = torch.zeros(B, d, d)
    for j in range(d):
        e = torch.zeros(B, d); e[:, j] = h
        J[:, :, j] = (f(e) - f(-e)) / (2 * h)
    return J


def test_vjp_matches_finite_differences():
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    J = _fd_jacobian(f, B, d)
    W = torch.randn(3, d)
    _, G = rh.vjp_rows(f, torch.zeros(B, d), W)     # (K,B,d)
    # atol=5e-2: this toy model's tanh+cumsum chain produces output magnitudes
    # ~10-30, so float32 central differences at h=1e-4 sit in the roundoff-
    # dominated regime (eps/h ~ 1e-3), not the truncation-dominated one -- a
    # 200-trial sweep of this exact setup measured a worst-case FD-vs-vjp
    # deviation of 0.0213 purely from that roundoff floor, while an
    # independent float64 autograd cross-check confirms vjp_rows itself is
    # exact to ~1e-6, and an injected wrong-transpose bug produces errors of
    # ~12-15 (three orders of magnitude above this tolerance). 5e-2 clears the
    # measured noise floor with margin while still catching real defects.
    for k in range(3):
        for b in range(B):
            ref = J[b].T @ W[k]
            assert torch.allclose(G[k, b], ref, atol=5e-2), (k, b)


def test_transposition_identity():
    """w·(Jv) via jvp == (Jᵀw)·v via vjp — guards Phases 1 and 2 simultaneously."""
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    torch.manual_seed(7)
    w = torch.randn(d); v = torch.randn(d)
    _, G = rh.vjp_rows(f, torch.zeros(B, d), w[None, :])
    _, JT = rh.jvp_cols(f, torch.zeros(B, d), v.expand(B, d).contiguous())
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
