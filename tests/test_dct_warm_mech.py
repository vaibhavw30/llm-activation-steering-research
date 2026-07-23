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


def test_anchor_step_scales_col0_by_its_norm():
    V_update = torch.zeros(4, 2)
    V_update[:, 0] = torch.tensor([3.0, 4.0, 0.0, 0.0])   # col0 norm 5
    anchor = torch.tensor([0.0, 0.0, 1.0, 0.0])           # unit, orthogonal to col0
    out = dct.anchor_step(V_update.clone(), anchor, 0.5)
    # col0 += 0.5 * ||col0||(=5) * anchor  =>  +2.5 in dim 2
    assert torch.allclose(out[:, 0], torch.tensor([3.0, 4.0, 2.5, 0.0]))
    assert torch.allclose(out[:, 1], torch.zeros(4))      # other columns untouched
    # lam=0 is a no-op
    assert torch.allclose(dct.anchor_step(V_update.clone(), anchor, 0.0), V_update)


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


def test_fit_u_anchor_pins_u0_near_target_seed():
    torch.manual_seed(0)
    d_src, d_tgt, n, seq = 6, 6, 4, 3
    W = torch.eye(d_tgt, d_src)
    da = _LinearDelta(W)
    X = torch.randn(n, seq, d_src)
    Y = torch.randn(n, seq, d_tgt)
    u_seed = torch.zeros(d_tgt); u_seed[0] = 1.0
    m = dct.ExponentialDCT(num_factors=4)
    U, V = m.fit(da, X, Y, batch_size=1, factor_batch_size=4, init="random",
                 u_anchor=u_seed, u_anchor_lambda=5.0, input_scale=1.0, max_iters=5,
                 orthogonalize=True)
    cos = abs(float(torch.dot(U[:, 0] / U[:, 0].norm(), u_seed)))
    assert cos > 0.9                       # strong U-anchor pins factor 0's effect direction
    assert torch.isfinite(V).all() and torch.isfinite(U).all()
    for j in range(4):                     # V columns stay unit (input side fully free)
        assert torch.allclose(V[:, j].norm(), torch.tensor(1.0), atol=1e-4)


def test_fit_u_anchor_zero_lambda_is_inert():
    torch.manual_seed(0)
    d, n, seq = 6, 4, 3
    da = _LinearDelta(torch.eye(d))
    X = torch.randn(n, seq, d); Y = torch.randn(n, seq, d)
    u_seed = torch.zeros(d); u_seed[0] = 1.0
    m = dct.ExponentialDCT(num_factors=4)
    torch.manual_seed(1)                      # reset RNG immediately before each fit so the
    U, V = m.fit(da, X, Y, batch_size=1,      # random inits are identical across the two runs
                 factor_batch_size=4, init="random",
                 u_anchor=u_seed, u_anchor_lambda=0.0, input_scale=1.0, max_iters=3,
                 orthogonalize=True)
    m2 = dct.ExponentialDCT(num_factors=4)
    torch.manual_seed(1)
    U2, V2 = m2.fit(da, X, Y, batch_size=1, factor_batch_size=4, init="random",
                    input_scale=1.0, max_iters=3, orthogonalize=True)
    assert torch.allclose(U, U2) and torch.allclose(V, V2)   # lam=0 == no anchor


def test_fit_u_anchor_rejects_separate_u():
    da = _LinearDelta(torch.eye(4))
    X = torch.randn(2, 3, 4); Y = torch.randn(2, 3, 4)
    m = dct.ExponentialDCT(num_factors=2)
    try:
        m.fit(da, X, Y, batch_size=1, factor_batch_size=2, init="random",
              u_anchor=torch.ones(4), u_anchor_lambda=1.0, input_scale=1.0,
              max_iters=1, separate_u=True)
        assert False, "expected AssertionError"
    except AssertionError:
        pass
