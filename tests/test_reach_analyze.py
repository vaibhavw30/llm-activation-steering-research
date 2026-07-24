import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_analyze as ra


def test_required_eps_hand_case():
    # 2-D hand case: J = diag(2, 1), w = e0  =>  J^T w = (2,0), m = 2.
    # h_tgt.w = 1.0, threshold t02 = 0  =>  g = 1.0, eps* = 0.5.
    g = np.array([1.0]); m = np.array([2.0])
    assert np.allclose(ra.required_eps(g, m), [0.5])


def test_required_eps_already_in_target_and_zero_margin():
    g = np.array([-0.3, 1.0]); m = np.array([1.0, 0.0])
    out = ra.required_eps(g, m)
    assert out[0] == 0.0               # already past the threshold: reachable at eps=0
    assert out[1] > 1e10               # zero margin: unreachable at any finite eps


def test_frac_reachable_monotone_in_eps():
    rng = np.random.default_rng(0)
    eps_star = rng.exponential(1.0, 500)
    grid = np.linspace(0, 5, 21)
    curve = ra.frac_reachable_curve(eps_star, grid)
    assert np.all(np.diff(curve) >= 0)
    assert curve[0] == (eps_star <= 0).mean()
    assert curve[-1] <= 1.0


def test_frac_reachable_exact_boundary():
    curve = ra.frac_reachable_curve(np.array([0.5, 1.0, 2.0]), np.array([0.4, 0.5, 1.0]))
    assert np.allclose(curve, [0.0, 1 / 3, 2 / 3])   # reachable iff eps* <= eps


def test_subspace_best_margin_is_top_singular_value():
    # rows J^T b_k for orthonormal {b_k}: best-case ||J^T w|| over unit w in span
    G = np.array([[3.0, 0.0, 0.0],
                  [0.0, 4.0, 0.0]])
    assert np.isclose(ra.subspace_best_margin(G), 4.0)
    # rotation of the basis must not change the answer
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert np.isclose(ra.subspace_best_margin(R @ G), 4.0)


def test_eps_grid_spans_1p5_input_scale():
    g = ra.eps_grid(40.0)
    assert len(g) == 61 and g[0] == 0.0 and np.isclose(g[-1], 60.0)
