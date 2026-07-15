import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from investigate_steer import (proportion_diff_ci, tost_equivalent, benjamini_hochberg,
                               binom_upper, cochran_armitage, n_per_group)


def test_proportion_diff_ci_centered_and_ordered():
    diff, lo, hi = proportion_diff_ci(0.5, 100, 0.4, 100)
    assert abs(diff - 0.1) < 1e-9
    assert lo < diff < hi


def test_tost_equivalent_when_diff_tiny_and_n_large():
    # nearly-identical proportions with big n -> should be judged equivalent within a 0.10 margin
    eq, _ = tost_equivalent(0.201, 5000, 0.199, 5000, margin=0.10)
    assert eq is True


def test_tost_not_equivalent_when_underpowered():
    # a real 0.2 gap but tiny n -> CI too wide to declare equivalence within 0.10
    eq, _ = tost_equivalent(0.4, 8, 0.2, 8, margin=0.10)
    assert eq is False


def test_benjamini_hochberg_rejections():
    # one clearly-significant p among nulls: BH should reject exactly it at q=0.05
    rej = benjamini_hochberg([0.001, 0.5, 0.6, 0.9], q=0.05)
    assert rej == [True, False, False, False]


def test_benjamini_hochberg_all_null():
    assert benjamini_hochberg([0.4, 0.5, 0.9], q=0.05) == [False, False, False]


def test_binom_upper_rule_of_three():
    # 0/10 successes -> 95% upper bound in the classic ~0.26 ballpark
    ub = binom_upper(0, 10)
    assert 0.24 < ub < 0.28
    # 0/20 tighter
    assert binom_upper(0, 20) < ub


def test_cochran_armitage_detects_monotone_trend():
    # outcome rate climbs 0 -> 1 across levels -> large positive z, small p
    lc = {0.0: [0, 20], 1.0: [5, 20], 2.0: [15, 20], 3.0: [20, 20]}
    z, p = cochran_armitage(lc)
    assert z > 0 and p < 0.01


def test_cochran_armitage_flat_is_ns():
    lc = {0.0: [10, 20], 1.0: [10, 20], 2.0: [10, 20]}
    z, p = cochran_armitage(lc)
    assert abs(z) < 1e-6 and p > 0.99


def test_n_per_group_shrinks_with_bigger_effect():
    assert n_per_group(0.30, 0.20) > n_per_group(0.40, 0.20)
