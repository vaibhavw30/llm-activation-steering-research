import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.operators import operator_features, all_operator_features


def _cache(L=3, n=5, d=4, seed=0):
    rng = np.random.default_rng(seed)
    per = lambda: rng.standard_normal((L, n, d))
    const = lambda: rng.standard_normal((L, d))
    return {"A_p": per(), "A_Qp": per(), "A_Qpv": per(), "A_verdict": per(),
            "A_EQp": per(), "A_Q": const(), "A_empty": const()}


def test_input_delta_is_prefixed_minus_direct():
    c = _cache(); L = 1
    got = operator_features("InputDelta", c, L)
    assert np.allclose(got, c["A_Qp"][L] - c["A_p"][L])


def test_interaction_adds_empty_constant():
    c = _cache(); L = 2
    got = operator_features("Interaction", c, L)
    assert np.allclose(got, c["A_Qp"][L] - c["A_p"][L] + c["A_empty"][L])


def test_question_delta_broadcasts_constant():
    c = _cache(); L = 0
    got = operator_features("QuestionDelta", c, L)
    assert np.allclose(got, c["A_Qp"][L] - c["A_Q"][L])   # (n,d) - (d,)


def test_direct_and_prefixed_passthrough():
    c = _cache(); L = 1
    assert np.allclose(operator_features("Direct", c, L), c["A_p"][L])
    assert np.allclose(operator_features("Prefixed", c, L), c["A_Qp"][L])


def test_operators_do_not_mutate_inputs():
    c = _cache(); L = 1
    before = c["A_Qp"][L].copy()
    operator_features("InputDelta", c, L)
    assert np.allclose(c["A_Qp"][L], before)


def test_all_operator_features_returns_eight_2d():
    c = _cache(); feats = all_operator_features(c, 1)
    assert len(feats) == 8
    assert all(v.ndim == 2 and v.shape == (5, 4) for v in feats.values())
