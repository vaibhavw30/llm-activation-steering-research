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


def test_crossed_is_sign_blind_and_marks_a_negative_scale_as_crossed():
    """Pins current behavior, which is NOT the physically correct one.

    `crossed` is `|scale| >= eps*_legacy`. A negative scale moves the readout AWAY from
    the FALSE threshold, so the first-order argument in the docstring predicts no
    crossing at any magnitude, yet the row below comes back True. Every caller in the
    published analysis sweeps positive fracs only, so no reported number is affected,
    but the arms on disk carry fracs down to -2.0. This test exists so that making the
    comparison sign-aware is a deliberate act with a failing test attached rather than a
    silent change to a published column.
    """
    df = _arm([("jtw_legacy", 7, -1.0, -50.0, 0, -0.02, "x")])
    out = tc.crossed_flags(df, {7: 0.5}, {7: 1})
    assert out.scale.iloc[0] < 0
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


def _write_reach_npz(tmp_path, ds="tiny"):
    """Three synthetic artifacts with the real key names, shapes and axis order.

    Read off the real files: `reach_dirs_<ds>.npz` carries `W` as (n_dirs, d), `names`
    as an object array and `thresh02` as (n_dirs,); `reach_margins_<ds>.npz` carries
    `margins` as (n_rows, n_dirs); `reach_acts_<ds>.npz` carries `h_tgt` as (n_rows, d)
    plus `labels` and `row_index` as (n_rows,).

    `mean_diff_tgt` is deliberately at index 1 with a decoy at index 0, and the decoy's
    `W` row, `thresh02` entry and `margins` column all differ, so a wrong `names.index`
    or a `[k, :]` for a `[:, k]` changes every returned value rather than none.
    """
    np.savez(tmp_path / f"reach_dirs_{ds}.npz",
             names=np.array(["decoy", tc.LEGACY_NAME], dtype=object),
             W=np.array([[9.0, 9.0, 9.0], [1.0, -2.0, 0.0]], dtype=np.float32),
             thresh02=np.array([99.0, 2.0], dtype=np.float32))
    np.savez(tmp_path / f"reach_margins_{ds}.npz",
             margins=np.array([[7.0, 1.5], [7.0, 6.0], [7.0, 4.0], [7.0, 0.0]],
                              dtype=np.float32))
    np.savez(tmp_path / f"reach_acts_{ds}.npz",
             h_tgt=np.array([[5.0, 0.0, 3.0], [7.0, 1.0, 0.0],
                             [1.0, 0.0, 9.0], [4.0, 0.0, 0.0]], dtype=np.float32),
             labels=np.array([1, 0, 1, 0]),
             row_index=np.array([1246, 7, 33, 99]))
    return ds


def test_legacy_eps_star_reconstructs_g_over_m_from_the_three_artifacts(tmp_path):
    """The producer behind §8 Result 4, tested end to end rather than through its
    consumer. eps* = g/m with g = w.h_tgt - t02 and m = ||J^T w|| read from `margins`.

    Statement 1246: w.h = 5*1 + 0*(-2) + 3*0 = 5, g = 5 - 2 = 3, m = 1.5, eps* = 2.0.
    Statement 7:    w.h = 7 - 2 = 5, the same g = 3 against m = 6.0, eps* = 0.5. The two
    together pin g and m separately: a dropped `- t02` moves the first, a wrong margins
    column moves the second, and no single arithmetic slip reproduces both.
    """
    ds = _write_reach_npz(tmp_path)
    eps, lab = tc.legacy_eps_star(ds, root=str(tmp_path))
    assert eps[1246] == pytest.approx(2.0)
    assert eps[7] == pytest.approx(0.5)
    assert lab == {1246: 1, 7: 0, 33: 1, 99: 0}


def test_legacy_eps_star_returns_zero_for_a_statement_already_inside_the_target(tmp_path):
    """g <= 0 means the statement already sits in the FALSE halfspace and needs no
    perturbation, so eps* is 0 and every scale trivially crosses. Statement 33 has
    w.h = 1 against t02 = 2, so g = -1. Returning the raw negative ratio here would make
    `crossed` depend on the sign of a quantity that is not a budget."""
    ds = _write_reach_npz(tmp_path)
    eps, _ = tc.legacy_eps_star(ds, root=str(tmp_path))
    assert eps[33] == 0.0


def test_legacy_eps_star_marks_a_vanished_margin_unreachable_rather_than_dividing(tmp_path):
    """m = 0 means no perturbation at this site moves the readout at all. Statement 99
    has g = 2 and m = 0. The sentinel keeps it in the table as unreachable instead of
    producing an inf, a NaN, or a silently huge finite number from an epsilon floor."""
    ds = _write_reach_npz(tmp_path)
    eps, _ = tc.legacy_eps_star(ds, root=str(tmp_path))
    assert eps[99] == tc.UNREACHABLE


def test_cross_tab_phi_matches_the_hand_computed_value_on_a_full_table():
    """The only other phi test asserts `isnan` on a degenerate table, so a sign flip or
    a transposed numerator would pass the whole suite while corrupting every phi cell in
    the dump. Table: 3 crossed-and-flipped, 2 crossed-not-flipped, 1 not-crossed-flipped,
    4 neither. phi = (3*4 - 2*1) / sqrt(5 * 5 * 4 * 6) = 10 / sqrt(600) = 0.4082483.
    Inverting the association must flip the sign and nothing else, which is what the
    second frame checks."""
    pos = pd.DataFrame({"crossed": [1] * 5 + [0] * 5,
                        "hit_target": [1, 1, 1, 0, 0] + [1, 0, 0, 0, 0]})
    assert tc.cross_tab(pos)["phi"] == pytest.approx(10.0 / np.sqrt(600.0))

    neg = pd.DataFrame({"crossed": [1] * 5 + [0] * 5,
                        "hit_target": [1, 0, 0, 0, 0] + [1, 1, 1, 0, 0]})
    assert tc.cross_tab(neg)["phi"] == pytest.approx(-10.0 / np.sqrt(600.0))


def test_the_join_key_is_the_dataset_row_index_not_a_position():
    """token_steer.stmt and token_geom.idx carry original dataset row indices running
    to 1495 on cities, not 0-based positions running to 199. A merge that assumes
    positions lands on the wrong statements while still producing a full-looking join,
    so the error is invisible downstream."""
    df = _arm([("jtw_legacy", 1246, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {1246: 0.5, 0: 999.0}, {1246: 1, 0: 1})
    assert out.eps_legacy.iloc[0] == pytest.approx(0.5)


def test_margin_consumption_reports_median_and_p90_per_direction_and_frac():
    df = _arm([("A", 1, 1.0, 1.0, 0, 0.01, "x"),
               ("A", 2, 1.0, 1.0, 0, -0.03, "x"),
               ("A", 3, 1.0, 1.0, 0, 0.05, "x"),
               ("A", 1, 2.0, 2.0, 0, 1.00, "x")])
    out = tc.margin_consumption(df).set_index(["direction", "frac"])
    assert out.loc[("A", 1.0), "median_abs"] == pytest.approx(0.03)
    assert out.loc[("A", 1.0), "n"] == 3
    assert out.loc[("A", 2.0), "median_abs"] == pytest.approx(1.00)
    # p90_abs fills three columns of §8's Result 2 table, so it needs an assertion of
    # its own. |frac_margin| sorted is [0.01, 0.03, 0.05] and pandas interpolates
    # linearly: 0.9 * (3 - 1) = 1.8, so 0.03 + 0.8 * (0.05 - 0.03) = 0.046. Taking the
    # signed quantile instead would give 0.046 here too but the wrong answer on a frame
    # with negatives, which is why median_abs above uses -0.03 as its middle value.
    assert out.loc[("A", 1.0), "p90_abs"] == pytest.approx(0.046)
    assert out.loc[("A", 2.0), "p90_abs"] == pytest.approx(1.00)


def test_proportional_detector_fires_on_a_per_statement_rescale():
    """readout_delta = frac_margin * |m0| with |m0| constant within a statement and
    different across statements. This is what token_steer actually writes, verified:
    the per-statement std of the ratio is 2.3e-5 against a mean of 13.8."""
    df = pd.DataFrame({
        "direction": ["A"] * 4,
        "stmt": [1, 1, 2, 2],
        "frac_margin": [0.01, 0.02, 0.05, 0.10],
        "tgt_minus_top_delta": [0.13, 0.26, 1.00, 2.00],  # m0 = 13 then 20
    })
    assert tc.proportional_per_statement(df) is True


def test_proportional_detector_stays_silent_on_independent_columns():
    """If token_steer is corrected to log the real truth readout, the ratio stops being
    constant and this returns False, which is the notification that A1 and A2 must be
    revisited."""
    df = pd.DataFrame({
        "direction": ["A"] * 4,
        "stmt": [1, 1, 2, 2],
        "frac_margin": [0.01, 0.02, 0.05, 0.10],
        "tgt_minus_top_delta": [0.13, 0.90, 1.00, 0.20],
    })
    assert tc.proportional_per_statement(df) is False


def test_proportional_detector_ignores_the_frac_zero_rows():
    """frac == 0 makes frac_margin exactly 0 by construction, so the ratio is undefined
    there. Including those rows would make the detector return False on data that is in
    fact proportional."""
    df = pd.DataFrame({
        "direction": ["A"] * 3,
        "stmt": [1, 1, 1],
        "frac_margin": [0.0, 0.01, 0.02],
        "readout_delta": [0.0, 0.13, 0.26],
    })
    assert tc.proportional_per_statement(df, a="readout_delta") is True


def test_proportional_detector_returns_false_when_every_group_is_a_singleton():
    """A single row carries no evidence about within-statement proportionality: there is
    no second ratio to compare it against. Scoring a lone row as a perfect match would
    let the tripwire pass silently on data it cannot actually assess, so a frame where
    every (direction, stmt) group has exactly one surviving row must return False even
    though the two columns here are manifestly unrelated."""
    df = pd.DataFrame({
        "direction": ["A"] * 5,
        "stmt": [1, 2, 3, 4, 5],
        "frac_margin": [0.01, 0.02, 0.03, 0.04, 0.05],
        "tgt_minus_top_delta": [7.0, -3.5, 0.02, 100.0, 1.0],
    })
    assert tc.proportional_per_statement(df) is False


def test_proportional_detector_does_not_count_a_nan_partner_as_agreement():
    """A group of two rows where one ratio is NaN used to pass: pandas skips the NaN, so
    `std(ddof=0)` of the single surviving value is 0.0 and the group scores a perfect
    match on no comparison at all. That is exactly the leniency the two-row gate exists
    to prevent, so the NaN filter has to run before the gate, not after.

    The two frames below are the same data twice. With the partners present the columns
    are genuinely proportional and the detector says True; with them NaN there is no
    within-statement comparison left anywhere and it must say False.
    """
    clean = pd.DataFrame({
        "direction": ["A"] * 4,
        "stmt": [1, 1, 2, 2],
        "frac_margin": [0.01, 0.02, 0.05, 0.10],
        "tgt_minus_top_delta": [0.13, 0.26, 1.00, 2.00],
    })
    assert tc.proportional_per_statement(clean) is True

    partial = clean.copy()
    partial.loc[[1, 3], "tgt_minus_top_delta"] = np.nan
    assert tc.proportional_per_statement(partial) is False


def test_an_empty_completion_is_degenerate_not_missing():
    """All 740 empty completions in the cities post-norm arm belong to md_full, 739 of
    them at negative fracs where |scale| runs to 1.6e3, and their argmax_tok is a
    newline or <eos>. The model emitted a newline and stopped. 4,660 non-empty plus 740
    empty is exactly the 5,400 rows, so nothing was left ungenerated. Dropping them as
    missing would delete the loudest evidence of md_full's failure mode."""
    assert tc.is_degenerate("") is True
    assert tc.is_degenerate("   ") is True


def test_degeneracy_catches_repetition_and_non_alphabetic_output():
    assert tc.is_degenerate("the the the city") is True
    assert tc.is_degenerate("... , . ; -") is True
    assert tc.is_degenerate("the northern part of Hebei Province,") is False


def test_edit_ratio_is_zero_for_identical_and_one_for_disjoint():
    assert tc.edit_ratio("in China", "in China") == pytest.approx(0.0)
    assert tc.edit_ratio("aaaa", "bbbb") == pytest.approx(1.0)
    assert 0.0 < tc.edit_ratio("in China. It is", "in Japan. It is") < 1.0


def test_country_outcome_is_three_way_not_binary():
    """The third case, naming some country that is neither the correct one nor the
    steering target, is the interesting one. A plain flipped/not-flipped binary would
    hide it."""
    assert tc.country_outcome("in China. It is", "China", ("North Korea",)) == "correct"
    assert tc.country_outcome("in North Korea", "China", ("North Korea",)) == "target"
    assert tc.country_outcome("in Peru, a nation", "China", ("North Korea",)) == "none"
    assert tc.country_outcome("China and North Korea", "China", ("North Korea",)) == "both"


def test_target_country_set_is_every_country_sharing_the_target_token():
    """tok_tgt is the FIRST SUBWORD of a country name, so it does not identify one
    country. 'South' is shared by South Africa and South Korea. Treating it as a single
    country would either miss a hit or claim a specific one the data cannot support."""
    countries = ["China", "North Korea", "South Africa", "South Korea"]
    assert tc.target_country_set("South", countries) == ("South Africa", "South Korea")
    assert tc.target_country_set("North", countries) == ("North Korea",)
