"""Tests for src/token_conclusions.py.

Every test builds its own small DataFrame. None of them read a real artifact, so the
suite runs in well under a second and does not break if an artifact is ever
regenerated with different row counts.
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import token_conclusions as tc                                      # noqa: E402


def _arm(rows):
    """rows: (direction, stmt, frac, scale, hit_target, frac_margin, completion)."""
    return pd.DataFrame(rows, columns=["direction", "stmt", "frac", "scale",
                                       "hit_target", "frac_margin", "completion"])


def test_the_shared_set_is_the_intersection_not_the_smallest_direction():
    """Two directions each short in a different place must yield the intersection.

    Taking the smaller direction's set would return {1, 2} here, which contains a
    statement direction B never ran.
    """
    df = _arm([("A", 1, 1.0, 1.0, 0, 0.1, "x"),
               ("A", 2, 1.0, 1.0, 0, 0.1, "x"),
               ("A", 3, 1.0, 1.0, 0, 0.1, "x"),
               ("B", 2, 1.0, 1.0, 0, 0.1, "x"),
               ("B", 3, 1.0, 1.0, 0, 0.1, "x"),
               ("B", 4, 1.0, 1.0, 0, 0.1, "x")])
    assert tc.shared_statements(df) == [2, 3]


def test_restriction_changes_the_rate_when_coverage_differs():
    """The defect this guards against is silent: an unrestricted mean is still a
    number, just a number about a different population."""
    df = _arm([("A", 1, 1.0, 1.0, 1, 0.1, "x"),
               ("A", 2, 1.0, 1.0, 0, 0.1, "x"),
               ("B", 1, 1.0, 1.0, 0, 0.1, "x")])
    assert df.groupby("direction").hit_target.mean()["A"] == pytest.approx(0.5)
    r = tc.restrict_to_shared(df)
    assert r.groupby("direction").hit_target.mean()["A"] == pytest.approx(1.0)


def test_load_arm_keeps_empty_completions_as_empty_strings(tmp_path):
    """pandas turns an empty CSV field into NaN by default. All 740 empty completions
    in the cities post-norm arm are md_full emitting a newline and stopping, which is
    the evidence, not missing data. Losing them to NaN would drop it."""
    p = tmp_path / "arm.csv"
    p.write_text("direction,stmt,frac,scale,hit_target,frac_margin,completion\n"
                 "md_full,1,-1.0,-500.0,0,-1.0,\n"
                 "oracle,1,1.0,6.0,1,1.0,in China\n")
    df = tc.load_arm(str(p))
    assert df.completion.iloc[0] == ""
    assert df.completion.isna().sum() == 0


def test_load_arm_normalises_the_delta_column_under_either_header(tmp_path):
    """src/token_steer.py now writes `tgt_minus_top_delta`; the twelve files on disk
    still carry `readout_delta`. Both must load to the same column name or every
    downstream lookup breaks on whichever generation it was not written for."""
    old = tmp_path / "old.csv"
    old.write_text("direction,stmt,frac,scale,hit_target,frac_margin,readout_delta\n"
                   "A,1,1.0,1.0,0,0.01,0.13\n")
    new = tmp_path / "new.csv"
    new.write_text("direction,stmt,frac,scale,hit_target,frac_margin,tgt_minus_top_delta\n"
                   "A,1,1.0,1.0,0,0.01,0.13\n")
    assert "tgt_minus_top_delta" in tc.load_arm(str(old)).columns
    assert "readout_delta" not in tc.load_arm(str(old)).columns
    assert "tgt_minus_top_delta" in tc.load_arm(str(new)).columns


def test_the_alias_map_matches_token_steer():
    """token_steer.LEGACY_COLUMN_ALIASES is the source of truth and is duplicated in
    token_conclusions to avoid a 4.2s torch import for a one-entry dict. Read the
    literal out of the source text rather than importing, so this test stays fast and
    still fails loudly if the two ever drift."""
    src = os.path.join(os.path.dirname(__file__), "..", "src", "token_steer.py")
    with open(src) as f:
        line = next(l for l in f if l.startswith("LEGACY_COLUMN_ALIASES"))
    assert ast.literal_eval(line.split("=", 1)[1].strip()) == tc.LEGACY_COLUMN_ALIASES


def test_crossed_uses_the_legacy_budget_not_the_token_budget():
    """The two budgets share the symbol eps and differ by orders of magnitude.

    Here eps*_legacy is 0.5 and the swept scale is 50, so the readout crossed its
    threshold 100x over. A helper that mistakenly compared against a token-space budget
    (delta_cone / alpha, ~500 for this statement) would report not-crossed, and the
    dissociation would vanish. The assertion is on the eps_legacy column itself so the
    error cannot hide inside a boolean.
    """
    df = _arm([("jtw_legacy", 7, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {7: 0.5}, {7: 1})
    assert out.eps_legacy.iloc[0] == pytest.approx(0.5)
    assert bool(out.crossed.iloc[0]) is True


def test_crossed_flags_drops_statements_outside_the_subsample():
    """jtw_legacy covers 90 of 200 on common_claim. A statement with no reach_margins
    row has no eps* and must be dropped, not treated as eps*=0 and therefore always
    crossed."""
    df = _arm([("jtw_legacy", 7, 1.0, 50.0, 0, 0.02, "x"),
               ("jtw_legacy", 9, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {7: 0.5}, {7: 1})
    assert list(out.stmt) == [7]


def test_crossed_flags_can_restrict_to_one_label():
    """g <= 0 means the statement already sits inside the target halfspace, so eps* is
    0 and `crossed` is trivially true. Measured on cities: 96 of the 99 label-0
    statements have g <= 0 and 0 of the 101 label-1 statements do. Pooling them would
    report a crossed rate inflated by statements that never had to move."""
    df = _arm([("jtw_legacy", 7, 1.0, 50.0, 0, 0.02, "x"),
               ("jtw_legacy", 8, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {7: 0.5, 8: 0.0}, {7: 1, 8: 0}, label=1)
    assert list(out.stmt) == [7]


def test_cross_tab_handles_an_empty_column_without_raising():
    """jtw_legacy is expected to produce an all-zero flipped column. phi's denominator
    is then 0 and must come back as NaN rather than a ZeroDivisionError or, worse, a
    silent 0.0 that reads as 'no association measured' instead of 'not measurable'."""
    df = _arm([("jtw_legacy", 1, 1.0, 50.0, 0, 0.02, "x"),
               ("jtw_legacy", 2, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {1: 0.5, 2: 100.0}, {1: 1, 2: 1})
    t = tc.cross_tab(out)
    assert t["crossed_not_flipped"] == 1
    assert t["not_crossed_not_flipped"] == 1
    assert t["crossed_flipped"] == 0
    assert np.isnan(t["phi"])


def test_the_join_key_is_the_dataset_row_index_not_a_position():
    """token_steer.stmt and token_geom.idx carry original dataset row indices running
    to 1495 on cities, not 0-based positions running to 199. A merge that assumes
    positions lands on the wrong statements while still producing a full-looking join,
    so the error is invisible downstream."""
    df = _arm([("jtw_legacy", 1246, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {1246: 0.5, 0: 999.0}, {1246: 1, 0: 1})
    assert out.eps_legacy.iloc[0] == pytest.approx(0.5)
