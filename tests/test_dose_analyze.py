"""D1 analysis: the Holm adjustment, and whether the clean-window rule actually fires.

The decision rule is pre-registered, so it has to be shown to detect a planted effect and to
stay silent otherwise. These tests build both cases synthetically.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dose_analyze as A  # noqa: E402

SCALES = [-1.0, -0.52, -0.30, 0.30, 0.52, 1.0]
DIRS = ["sup_grad", "sup_mean_diff", "mag_resid_pc1", "rand_ctrl"]


def test_holm_is_step_down_and_monotone():
    # sorted p = .01 .03 .04, multipliers 3 2 1 -> .03 .06 .04, running max -> .03 .06 .06
    p = [0.01, 0.04, 0.03]
    adj = A.holm(p)
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    # the step-down enforcement must never let a larger raw p adjust to a smaller value
    order = np.argsort(p)
    assert np.all(np.diff(np.asarray(adj)[order]) >= -1e-12)


def test_holm_never_exceeds_one():
    assert np.all(A.holm([0.4, 0.5, 0.9]) <= 1.0)


def _summary(flip_by_dir, k_incoh=5, n=24, n_j=32):
    """One summary row per (direction, dose), plus the unsteered baseline row."""
    rows = [{"direction": "baseline", "scale": 0.0, "n_yesno": n, "k_flip": 0,
             "flip_rate": 0.0, "mean_d_margin": 0.0, "sd_d_margin": 0.0,
             "wilcoxon_p_raw": 1.0, "n_judged": n_j, "k_false": 1, "k_incoh": 5,
             "false_rate": 1 / n_j, "incoh_rate": 5 / n_j}]
    for d in DIRS:
        for s in SCALES:
            k = flip_by_dir(d, s)
            ki = k_incoh(d, s) if callable(k_incoh) else k_incoh
            rows.append({"direction": d, "scale": s, "n_yesno": n, "k_flip": k,
                         "flip_rate": k / n, "mean_d_margin": 0.0, "sd_d_margin": 0.1,
                         "wilcoxon_p_raw": 1.0, "n_judged": n_j, "k_false": k,
                         "k_incoh": ki, "false_rate": k / n_j, "incoh_rate": ki / n_j})
    return pd.DataFrame(rows)


def _windows(sm):
    return A.classify_windows(A.window_tests(sm, True), True)


def test_planted_clean_window_is_found_and_only_there():
    """sup_grad flips hard at |0.30| while coherence stays at baseline; nothing else moves."""
    def flips(d, s):
        return 18 if (d == "sup_grad" and abs(s) == 0.30) else 1
    w = _windows(_summary(flips))
    clean = w[w["clean_window"]]
    assert set(clean["direction"]) == {"sup_grad"}
    assert sorted(clean["scale"].abs().unique()) == [0.30]
    assert len(clean) == 2                      # both signs


def test_flips_that_come_with_incoherence_are_not_a_clean_window():
    """The whole point of the rule: a direction that only flips things by breaking the model
    must not be scored as behavioural control."""
    def flips(d, s):
        return 18 if (d == "sup_grad" and abs(s) == 1.0) else 1

    def incoh(d, s):
        return 28 if (d == "sup_grad" and abs(s) == 1.0) else 5
    w = _windows(_summary(flips, k_incoh=incoh))
    moved = w[w["moved"]]
    assert len(moved) == 2                      # it did move
    assert not w["clean_window"].any()          # but it broke the model doing it


def test_matching_the_random_control_is_never_a_window():
    """If a truth direction flips exactly as often as a same-norm random vector, the effect is
    norm-driven and must not be reported as a finding."""
    w = _windows(_summary(lambda d, s: 12))
    assert not w["clean_window"].any()


def test_control_and_baseline_rows_are_not_tested_against_themselves():
    w = _windows(_summary(lambda d, s: 1))
    assert A.CONTROL not in set(w["direction"])
    assert A.BASELINE not in set(w["direction"])


def test_missing_baseline_is_an_error_not_a_silent_pass():
    sm = _summary(lambda d, s: 1)
    with pytest.raises(SystemExit, match="no baseline"):
        A.window_tests(sm[sm["direction"] != "baseline"], True)


def test_without_the_judged_arm_coherence_is_unevaluable_not_satisfied():
    """No judge file means condition 2 cannot be checked. It must come back NaN, and a NaN
    must never count as a clean window."""
    sm = _summary(lambda d, s: 18 if d == "sup_grad" else 1).drop(
        columns=["n_judged", "k_false", "k_incoh", "false_rate", "incoh_rate"])
    w = A.classify_windows(A.window_tests(sm, False), False)
    assert w["moved"].any()
    assert w["coherent"].isna().all()
    assert not w["clean_window"].any()
