# tests/test_refusal_analyze.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import refusal_analyze as ra
from refusal_analyze import (paired_frame, dose_table, mcnemar, two_by_two,
                             cochran_armitage, crossing_vs_magnitude)


def _row(stmt, frac, g, refused):
    return {"stmt_index": stmt, "frac": frac, "g_read": g, "refused": float(refused)}


def _complete(stmt, refused_map, g_map=None):
    """One statement observed at every core frac."""
    g_map = g_map or {f: 10.0 for f in ra.CORE_FRACS}
    return [_row(stmt, f, g_map[f], refused_map[f]) for f in ra.CORE_FRACS]


# ---------------------------------------------------------------- paired_frame

def test_paired_frame_keeps_only_complete_statements():
    rows = _complete(1, {f: 0 for f in ra.CORE_FRACS})
    rows += [_row(2, -1.0, 5.0, 0), _row(2, 0.0, 5.0, 0)]   # missing +-2
    frame = paired_frame(rows)
    assert set(frame) == {1}


def test_paired_frame_is_what_makes_the_comparison_within_statement():
    """The clamp gives the |frac|=2 buckets a different (near-boundary) statement set
    than frac 0. If incomplete statements leaked in, the -2 and 0 columns would hold
    different subjects and the sign control would stop being a control."""
    rows = _complete(1, {f: 0 for f in ra.CORE_FRACS})
    rows += [_row(9, 0.0, 5.0, 1)]           # baseline-only, would inflate frac 0
    frame = paired_frame(rows)
    assert 9 not in frame
    counts = {r["frac"]: r["n"] for r in dose_table(frame)}
    assert set(counts.values()) == {1}


def test_paired_frame_rounds_frac_to_two_places():
    rows = [_row(1, f + 0.001, 1.0, 0) for f in ra.CORE_FRACS]
    assert set(paired_frame(rows)) == {1}


def test_paired_frame_empty_when_nothing_is_complete():
    assert paired_frame([_row(1, 0.0, 1.0, 0)]) == {}


# ------------------------------------------------------------------ dose_table

def test_dose_table_counts_individual_crossings_not_the_mean():
    """Two statements: one deeply negative, one positive. The bucket MEAN at -2 is
    negative, but only ONE statement actually crossed. reach_control adjudicates on
    the mean; this table must not repeat that conflation."""
    g_a = {f: (-50.0 if f == -2.0 else 10.0) for f in ra.CORE_FRACS}
    g_b = {f: 10.0 for f in ra.CORE_FRACS}
    frame = paired_frame(_complete(1, {f: 0 for f in ra.CORE_FRACS}, g_a)
                         + _complete(2, {f: 0 for f in ra.CORE_FRACS}, g_b))
    row = next(r for r in dose_table(frame) if r["frac"] == -2.0)
    assert row["mean_g_read"] < 0        # the bucket mean says "crossed"
    assert row["n_crossed"] == 1         # but only one statement did
    assert row["n"] == 2


def test_dose_table_treats_g_exactly_zero_as_crossed():
    g = {f: (0.0 if f == -2.0 else 10.0) for f in ra.CORE_FRACS}
    frame = paired_frame(_complete(1, {f: 0 for f in ra.CORE_FRACS}, g))
    assert next(r for r in dose_table(frame) if r["frac"] == -2.0)["n_crossed"] == 1


# --------------------------------------------------------------------- mcnemar

def test_mcnemar_counts_gains_and_losses_separately():
    gain = _complete(1, {-2.0: 1, -1.0: 0, 0.0: 0, 1.0: 0, 2.0: 0})
    loss = _complete(2, {-2.0: 0, -1.0: 1, 0.0: 1, 1.0: 1, 2.0: 1})
    same = _complete(3, {f: 0 for f in ra.CORE_FRACS})
    m = mcnemar(paired_frame(gain + loss + same), 0.0, -2.0)
    assert (m["gained"], m["lost"], m["n_discordant"]) == (1, 1, 2)
    assert m["n_eligible_to_gain"] == 2 and m["n_eligible_to_lose"] == 1


def test_mcnemar_ignores_concordant_pairs():
    """A statement refusing at both fracs carries no evidence of change; adding a
    hundred of them must not move the p-value."""
    gain = _complete(1, {-2.0: 1, -1.0: 0, 0.0: 0, 1.0: 0, 2.0: 0})
    p_alone = mcnemar(paired_frame(gain), 0.0, -2.0)["p_exact"]
    both = []
    for i in range(100):
        both += _complete(10 + i, {f: 1 for f in ra.CORE_FRACS})
    assert mcnemar(paired_frame(gain + both), 0.0, -2.0)["p_exact"] == p_alone


def test_mcnemar_p_is_one_when_nothing_is_discordant():
    frame = paired_frame(_complete(1, {f: 0 for f in ra.CORE_FRACS}))
    m = mcnemar(frame, 0.0, -2.0)
    assert m["n_discordant"] == 0 and m["p_exact"] == 1.0


def test_mcnemar_direction_is_not_symmetric():
    gain = _complete(1, {-2.0: 1, -1.0: 0, 0.0: 0, 1.0: 0, 2.0: 0})
    frame = paired_frame(gain)
    assert mcnemar(frame, 0.0, -2.0)["gained"] == 1
    assert mcnemar(frame, -2.0, 0.0)["gained"] == 0
    assert mcnemar(frame, -2.0, 0.0)["lost"] == 1


# ------------------------------------------------------------------ two_by_two

def test_two_by_two_uses_the_same_denominator_for_both_fracs():
    rows = []
    for i in range(4):
        rows += _complete(i, {-2.0: 1, -1.0: 0, 0.0: 0, 1.0: 0, 2.0: 0})
    t = two_by_two(paired_frame(rows), -2.0, 2.0)
    assert t["n"] == 4 and t["n_refused_a"] == 4 and t["n_refused_b"] == 0


# ------------------------------------------------------------- cochran_armitage

def test_trend_slope_is_negative_when_refusal_rises_as_frac_falls():
    rows = []
    for i in range(20):
        rows += _complete(i, {-2.0: 1, -1.0: 1, 0.0: 0, 1.0: 0, 2.0: 0})
    t = cochran_armitage(rows)
    assert t["slope_per_eps_star"] < 0 and t["z"] < 0 and t["p"] < 0.01


def test_trend_is_null_when_refusal_does_not_depend_on_dose():
    rows = []
    for i in range(20):
        rows += _complete(i, {f: (i % 2) for f in ra.CORE_FRACS})
    t = cochran_armitage(rows)
    assert t["slope_per_eps_star"] == pytest.approx(0.0, abs=1e-9)
    assert t["p"] == pytest.approx(1.0)


def test_trend_handles_a_degenerate_outcome_without_dividing_by_zero():
    rows = []
    for i in range(5):
        rows += _complete(i, {f: 1 for f in ra.CORE_FRACS})
    assert cochran_armitage(rows) == {"n": 25, "z": 0.0, "p": 1.0,
                                      "slope_per_eps_star": 0.0}


def test_trend_ignores_fracs_outside_the_core_grid():
    rows = _complete(1, {f: 0 for f in ra.CORE_FRACS})
    rows += [_row(1, -1.86, 5.0, 1)] * 50      # clamp artefacts, not a dose level
    assert cochran_armitage(rows)["n"] == 5


# ------------------------------------------------------ crossing_vs_magnitude

def test_crossing_beats_magnitude_when_only_crossings_refuse():
    """Same |frac| stratum, same perturbation size: the rows that crossed refuse and
    the rows that did not do not. The OR must be driven by crossing, not dose."""
    rows = [_row(i, -2.0, -1.0, 1) for i in range(10)]          # crossed, refused
    rows += [_row(100 + i, -2.0, 40.0, 0) for i in range(10)]   # same dose, no cross
    x = crossing_vs_magnitude(rows)
    st = x["strata"][0]
    assert st["crossed_refused"] == 10 and st["not_crossed_refused"] == 0
    assert x["mh_odds_ratio"] > 1


def test_crossing_excludes_baseline_rows():
    """Baseline rows have g > 0 for every statement that has a boundary at all.
    Counting them would pack the not-crossed cell with rows never perturbed and
    understate the not-crossed refusal rate."""
    rows = [_row(i, 0.0, 40.0, 1) for i in range(50)]
    rows += [_row(100 + i, -2.0, -1.0, 1) for i in range(2)]
    st = crossing_vs_magnitude(rows)["strata"][0]
    assert st["n"] == 2


def test_crossing_stratifies_so_dose_cannot_masquerade_as_crossing():
    """Confounded input: every low-dose row is uncrossed and compliant, every
    high-dose row is crossed and refusing. Stratification must split them into two
    strata rather than pool them into one large spurious OR."""
    rows = [_row(i, -1.0, 40.0, 0) for i in range(10)]
    rows += [_row(100 + i, -2.0, -1.0, 1) for i in range(10)]
    strata = crossing_vs_magnitude(rows)["strata"]
    assert len(strata) == 2
    assert strata[0]["crossed_total"] == 0 and strata[1]["not_crossed_total"] == 0


def test_crossing_skips_strata_with_no_rows():
    rows = [_row(i, -2.0, -1.0, 1) for i in range(3)]
    assert len(crossing_vs_magnitude(rows)["strata"]) == 1
