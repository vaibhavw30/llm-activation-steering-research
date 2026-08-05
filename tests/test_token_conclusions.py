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
