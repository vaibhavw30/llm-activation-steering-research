import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tqa_form import baseline_edges, cmh, length_strata


def _frame(rows):
    """rows: (frac, words, truthful) triples for one direction."""
    return pd.DataFrame([{"direction": "jtw_x", "frac": f, "words": w, "truthful": t,
                          "prompt": f"q{i}"} for i, (f, w, t) in enumerate(rows)])


def test_edges_come_from_the_baseline_only():
    d = _frame([(0.0, w, 0) for w in (1, 2, 3, 4, 5, 6, 7, 8)]
               + [(-2.0, 90, 1)] * 5)                 # steered lengths must not move edges
    edges = baseline_edges(d, "jtw_x")
    assert edges[-1] == 8                             # the baseline max closes the support
    assert all(e <= 8 for e in edges)


def test_answers_past_the_baseline_max_go_to_the_outside_bin():
    d = _frame([(0.0, 2, 0), (0.0, 4, 1), (0.0, 6, 0), (-2.0, 50, 1), (-2.0, 3, 1)])
    s = length_strata(d, "jtw_x", -2.0, score_col="truthful")
    out = s[s["stratum"] == "outside"].iloc[0]
    assert out["n_steer"] == 1 and out["n_base"] == 0 and out["k_steer"] == 1


def test_cmh_is_null_when_rates_match_in_every_stratum():
    strata = pd.DataFrame([{"k_steer": 5, "n_steer": 10, "k_base": 5, "n_base": 10},
                           {"k_steer": 2, "n_steer": 8, "k_base": 1, "n_base": 4}])
    r = cmh(strata)
    assert r["or_mh"] == pytest.approx(1.0)
    assert r["p"] > 0.9


def test_cmh_detects_a_consistent_within_stratum_gain():
    strata = pd.DataFrame([{"k_steer": 18, "n_steer": 20, "k_base": 4, "n_base": 20},
                           {"k_steer": 15, "n_steer": 20, "k_base": 3, "n_base": 20}])
    r = cmh(strata)
    assert r["or_mh"] > 10 and r["p"] < 1e-4


def test_cmh_ignores_strata_one_side_never_reaches():
    strata = pd.DataFrame([{"k_steer": 5, "n_steer": 10, "k_base": 5, "n_base": 10},
                           {"k_steer": 9, "n_steer": 9, "k_base": 0, "n_base": 0}])
    assert cmh(strata)["or_mh"] == pytest.approx(1.0)  # the one-sided stratum adds nothing
