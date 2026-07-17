import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.directions import (
    v_Q, class_mean_diff, a_prefix_norm,
    build_lead_directions, residual_pc1, DIVERGENT_OPERATORS,
)
from funnel_utils import unit


def _lead_cache(L=3, n=80, d=6, seed=2):
    """Cache with a truth axis in the InputDelta SHIFT (dim 0) plus genuine off-axis residual
    variance, and a divergent Prefixed axis: the raw activation carries class signal in dim 1
    that the shift does not, so class_mean_diff(A_Qp) points off the shift's truth axis."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    cls = (y * 2.0 - 1.0)                             # +/-1 per class
    A_p = rng.standard_normal((L, n, d))
    A_p[:, :, 1] += cls * 2.0                         # raw/prefixed-only class signal in dim 1
    shift = rng.standard_normal((L, n, d)) * 0.4      # off-axis, class-independent residual
    shift[:, :, 0] += cls * 2.0                       # the SHIFT's truth axis in dim 0
    A_Qp = A_p + shift                                # InputDelta = shift
    A_Qpv = A_p + rng.standard_normal((L, n, d)) * 0.3
    A_Qpv[:, :, 4] += cls * 1.5
    A_EQp = A_p + rng.standard_normal((L, n, d)) * 0.3
    A_EQp[:, :, 5] += cls * 1.5
    per = lambda: rng.standard_normal((L, n, d))
    const = lambda: rng.standard_normal((L, d))
    c = {"A_p": A_p, "A_Qp": A_Qp, "A_Qpv": A_Qpv, "A_verdict": per(),
         "A_EQp": A_EQp, "A_Q": const(), "A_empty": const()}
    return c, y


def test_lead_directions_have_all_keys():
    c, y = _lead_cache()
    out = build_lead_directions(c, layer=1, labels_gold=y)
    for op in DIVERGENT_OPERATORS:
        assert f"u_{op}_unit" in out and out[f"u_{op}_unit"].shape == (6,)
        assert np.isfinite(out[f"acc_{op}"])
        assert abs(np.linalg.norm(out[f"u_{op}_unit"]) - 1.0) < 1e-5
    assert "resid_pc1_unit" in out and "resid_meandiff_norm" in out


def test_rank1_residual_meandiff_is_zero():
    # The core lead-#3 finding, proved in code: removing the primary truth axis kills ALL
    # linear class-mean signal, so the residual class-mean-diff is ~0 relative to the primary.
    c, y = _lead_cache()
    out = build_lead_directions(c, layer=1, labels_gold=y)
    assert out["resid_meandiff_norm"] < 1e-6 * out["u_primary_norm"]


def test_residual_pc1_orthogonal_to_primary():
    # residual-PC1 lives in the subspace with the truth axis removed → orthogonal to it.
    c, y = _lead_cache()
    out = build_lead_directions(c, layer=1, labels_gold=y)
    assert abs(out["cos_resid_pc1_primary"]) < 1e-6


def test_divergent_direction_reads_truth_but_diverges():
    # A planted divergent operator (prefixed carries class signal in dim 3, off the shift's
    # dim-0 truth axis) should read truth above chance yet not be collinear with the primary.
    c, y = _lead_cache()
    out = build_lead_directions(c, layer=1, labels_gold=y)
    assert out["acc_Prefixed"] > 0.6
    assert abs(out["cos_Prefixed_primary"]) < 0.99


def test_residual_pc1_is_unit_and_finite():
    c, y = _lead_cache()
    pc1 = residual_pc1(c, layer=1, primary_unit=unit(class_mean_diff(
        c["A_Qp"][1] - c["A_p"][1], y)))
    assert abs(np.linalg.norm(pc1) - 1.0) < 1e-6 and np.all(np.isfinite(pc1))


def test_vq_is_mean_of_shifts():
    A_p = np.array([[0.0, 0.0], [1.0, 1.0]])
    A_Qp = np.array([[1.0, 0.0], [3.0, 1.0]])
    # shifts: [1,0] and [2,0] -> mean [1.5, 0]
    assert np.allclose(v_Q(A_Qp, A_p), [1.5, 0.0])


def test_class_mean_diff_is_neg_minus_pos():
    feats = np.array([[2.0, 0.0],    # label 0
                      [4.0, 0.0],    # label 0
                      [0.0, 0.0]])   # label 1
    labels = np.array([0, 0, 1])
    # mean(neg)=[3,0], mean(pos)=[0,0] -> [3,0]
    assert np.allclose(class_mean_diff(feats, labels), [3.0, 0.0])


def test_a_prefix_norm_positive_and_is_mean_row_norm():
    A_Qp = np.array([[3.0, 4.0], [0.0, 0.0]])   # norms 5 and 0 -> mean 2.5
    assert abs(a_prefix_norm(A_Qp) - 2.5) < 1e-9


def test_class_mean_diff_handles_missing_class():
    feats = np.array([[1.0, 1.0], [2.0, 2.0]])
    labels = np.array([0, 0])                    # no positives
    out = class_mean_diff(feats, labels)
    assert out.shape == (2,) and np.all(np.isfinite(out))
