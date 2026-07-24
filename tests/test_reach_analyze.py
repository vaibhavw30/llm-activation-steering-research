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


def test_truth_subspace_best_margins_excludes_redundant_truth_rows():
    # Orthonormal truth_sub basis {e0-image, e1-image} with ||J^T e0||=3, ||J^T e1||=4
    # => true best-case margin = sigma_max = 4. A redundant 'truth' row duplicating
    # e0 must be IGNORED; if it were stacked, sigma_max would inflate to sqrt(18)~4.24.
    names = ["mean_diff_tgt", "truth_sub_0", "truth_sub_1"]
    groups = ["truth", "truth_sub", "truth_sub"]
    store_names = names[:]                      # all three stored
    jtw = np.zeros((1, 3, 3))
    jtw[0, 0] = [1, 0, 0]                        # mean_diff image dir (redundant)
    jtw[0, 1] = [1, 0, 0]                        # truth_sub_0 image dir
    jtw[0, 2] = [0, 1, 0]                        # truth_sub_1 image dir
    margins = np.array([[3.0, 3.0, 4.0]])
    out = ra.truth_subspace_best_margins(jtw, margins, names, groups, store_names)
    assert np.isclose(out[0], 4.0)              # orthonormal-only sigma_max, NOT ~4.24


def test_truth_subspace_best_margins_empty_is_nan():
    out = ra.truth_subspace_best_margins(np.zeros((2, 1, 3)), np.ones((2, 1)),
                                         ["rand_0"], ["rand"], ["rand_0"])
    assert out.shape == (2,) and np.all(np.isnan(out))


def test_scale_grid_caps_and_brackets():
    import reach_steer as rs
    g = rs.scale_grid(eps_star=10.0, input_scale=40.0, fracs=[0.5, 1.0, 1.5, 2.0])
    assert 0.0 in g
    mags = sorted({abs(s) for s in g if s != 0})
    assert mags == [5.0, 10.0, 15.0, 20.0]
    g2 = rs.scale_grid(eps_star=100.0, input_scale=40.0, fracs=[1.0, 2.0])
    assert max(abs(s) for s in g2) == 60.0        # capped at 1.5 x input_scale
    assert rs.scale_grid(0.0, 40.0, [1.0]) == [0.0]


def test_stem_of():
    import reach_steer as rs
    assert rs.stem_of("The city of Paris is in France.") == "The city of Paris is in"
    assert rs.stem_of("Too short.") is None
