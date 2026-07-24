import sys, os
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_linerr as rl


class _LinearSlice(nn.Module):
    """Purely linear stand-in: rel error must be ~0 at every eps."""
    def __init__(self, d=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(d, d, generator=g), requires_grad=False)

    def forward(self, h):
        return h @ self.W


class _QuadSlice(nn.Module):
    """Quadratic term: rel error must grow with eps."""
    def __init__(self, d=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(d, d, generator=g), requires_grad=False)

    def forward(self, h):
        return h @ self.W + 0.05 * (h * h)


def _f_for(slice_mod, d=6, T=3, seed=1):
    import reach_hop as rh
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, T, d, generator=g)
    attn = torch.ones(1, T, dtype=torch.long)
    return rh.make_hop(slice_mod, h, attn)


def test_rel_error_zero_for_linear_map():
    f = _f_for(_LinearSlice())
    F0 = f(torch.zeros(1, 6))
    delta = torch.randn(6); delta /= delta.norm()
    errs = rl.rel_error_curve(f, F0, delta, [0.1, 1.0, 10.0, 100.0])
    assert all(e < 1e-4 for e in errs), errs


def test_rel_error_grows_for_nonlinear_map():
    f = _f_for(_QuadSlice())
    F0 = f(torch.zeros(1, 6))
    delta = torch.randn(6); delta /= delta.norm()
    errs = rl.rel_error_curve(f, F0, delta, [0.01, 1.0, 100.0])
    assert errs[0] < errs[-1]
    assert errs[0] < 0.05


def test_validity_radius_interpolation():
    eps = np.array([1.0, 2.0, 4.0, 8.0])
    med = np.array([0.05, 0.1, 0.3, 0.9])
    r = rl.validity_radius(eps, med, thresh=0.2)
    assert 2.0 < r < 4.0                       # crosses 0.2 between eps=2 and eps=4
    assert rl.validity_radius(eps, np.array([0.01, 0.02, 0.03, 0.04])) == 8.0
    assert rl.validity_radius(eps, np.array([0.5, 0.6, 0.7, 0.8])) == 0.0
