import sys, os
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dct


def test_warm_init_column_sets_unit_seed():
    V = torch.randn(5, 3)
    seed = torch.tensor([0.0, 3.0, 0.0, 4.0, 0.0])   # norm 5
    out = dct.warm_init_column(V.clone(), seed)
    assert torch.allclose(out[:, 0].norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(out[:, 0], seed / 5.0, atol=1e-6)
    assert torch.allclose(out[:, 1:], V[:, 1:])          # other columns untouched


def test_anchor_step_adds_only_to_col0():
    U = torch.zeros(4, 2)
    anchor = torch.tensor([1.0, 1.0, 1.0, 1.0])
    out = dct.anchor_step(U.clone(), anchor, 0.5)
    assert torch.allclose(out[:, 0], anchor * 0.5)
    assert torch.allclose(out[:, 1], torch.zeros(4))
    # lam=0 is a no-op
    assert torch.allclose(dct.anchor_step(U.clone(), anchor, 0.0), U)


class _LinearDelta(nn.Module):
    """Model-free DeltaActivations stub: delta(theta) = (x+theta)Wᵀ - y, averaged over seq."""
    def __init__(self, W):
        super().__init__()
        self.W = W
        self.device = W.device
        self.attention_mask = None

    def forward(self, theta, x, y, attention_mask=None):
        steered = (x + theta) @ self.W.T
        return (steered - y).mean(dim=1)


def test_fit_warm_keeps_factor0_near_seed_when_anchored():
    torch.manual_seed(0)
    d_src, d_tgt, n, seq = 6, 6, 4, 3
    W = torch.eye(d_tgt, d_src)
    da = _LinearDelta(W)
    X = torch.randn(n, seq, d_src)
    Y = torch.randn(n, seq, d_tgt)
    seed = torch.zeros(d_src); seed[0] = 1.0
    m = dct.ExponentialDCT(num_factors=4)
    U, V = m.fit(da, X, Y, batch_size=1, factor_batch_size=4, init="warm",
                 warm_seed=seed, anchor_lambda=5.0, input_scale=1.0, max_iters=5,
                 orthogonalize=True)
    cos = abs(float(torch.dot(V[:, 0] / V[:, 0].norm(), seed)))
    assert cos > 0.9         # strong anchor pins factor 0 to the seed direction
