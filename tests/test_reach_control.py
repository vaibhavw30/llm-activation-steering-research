# tests/test_reach_control.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import reach_control
from reach_control import (frac_of, align_stmt_rows, aggregate, readout_crossed,
                           behavior_delta, verdict, require_signal, mean_arm_rows)


def _j(scale, refused):
    return {"direction": "jtw_stmt", "scale": str(scale), "prompt": "p",
            "completion": "c", "refused": str(refused)}


def _m(scale, eps, g):
    return {"stmt_index": "0", "label": "1", "eps_star": str(eps),
            "scale": str(scale), "g_read": str(g)}


def test_frac_of_divides_by_eps_star():
    assert frac_of(4.0, 2.0) == 2.0
    assert frac_of(-2.0, 2.0) == -1.0


def test_frac_of_is_zero_when_eps_star_is_zero():
    assert frac_of(5.0, 0.0) == 0.0


def test_align_stmt_rows_pairs_by_row_order():
    out = align_stmt_rows([_j(0.0, 0), _j(3.0, 1)],
                          [_m(0.0, 3.0, 5.0), _m(3.0, 3.0, -1.0)])
    assert [r["frac"] for r in out] == [0.0, 1.0]
    assert [r["refused"] for r in out] == [0.0, 1.0]
    assert [r["g_read"] for r in out] == [5.0, -1.0]


def test_align_stmt_rows_rejects_length_mismatch():
    with pytest.raises(SystemExit, match="row-count"):
        align_stmt_rows([_j(0.0, 0)], [_m(0.0, 3.0, 1.0), _m(3.0, 3.0, -1.0)])


def test_align_stmt_rows_rejects_scale_mismatch():
    with pytest.raises(SystemExit, match="scale mismatch"):
        align_stmt_rows([_j(0.0, 0), _j(3.0, 1)],
                        [_m(0.0, 3.0, 5.0), _m(9.0, 3.0, -1.0)])


def test_aggregate_buckets_by_rounded_frac():
    rows = [{"frac": 1.0001, "g_read": -1.0, "refused": 1.0},
            {"frac": 0.9999, "g_read": -3.0, "refused": 0.0},
            {"frac": 0.0, "g_read": 5.0, "refused": 0.0}]
    t = aggregate(rows)
    assert set(t) == {0.0, 1.0}
    assert t[1.0]["n"] == 2
    assert t[1.0]["g_read"] == -2.0
    assert t[1.0]["frac_refused"] == 0.5


def test_readout_crossed_flags_nonpositive_g():
    t = {0.0: {"g_read": 5.0}, 1.0: {"g_read": -2.0}, -1.0: {"g_read": 9.0}}
    assert readout_crossed(t) == {0.0: False, 1.0: True, -1.0: False}


def test_behavior_delta_is_relative_to_the_zero_baseline():
    t = {0.0: {"frac_refused": 0.05}, 1.0: {"frac_refused": 0.55},
         -1.0: {"frac_refused": 0.02}}
    d = behavior_delta(t)
    assert abs(d[1.0] - 0.50) < 1e-12
    assert abs(d[-1.0] + 0.03) < 1e-12
    assert d[0.0] == 0.0


def test_behavior_delta_requires_a_baseline():
    with pytest.raises(SystemExit, match="baseline"):
        behavior_delta({1.0: {"frac_refused": 0.5}})


def test_verdict_actuatable_when_crossing_moves_behavior():
    assert verdict({0.0: False, 1.0: True}, {0.0: 0.0, 1.0: 0.5}) == "actuatable"


def test_verdict_readout_only_when_crossing_moves_nothing():
    assert verdict({0.0: False, 1.0: True}, {0.0: 0.0, 1.0: 0.01}) == "readout-only"


def test_verdict_inert_when_behavior_moves_without_a_crossing():
    assert verdict({0.0: False, 1.0: False}, {0.0: 0.0, 1.0: 0.5}) == "inert"


def test_verdict_no_crossing_when_nothing_crosses_and_nothing_moves():
    assert verdict({0.0: False, 1.0: False}, {0.0: 0.0, 1.0: 0.0}) == "no-crossing"


def test_verdict_requires_crossing_and_movement_at_the_same_frac():
    # Crossing only at frac=-1; movement only at frac=+2. These are DIFFERENT fracs,
    # so this must NOT be "actuatable" -- it is the off-target "inert"-shaped result
    # the module's own docstring calls out (crossings can only occur at negative
    # fracs per reach_steer.py:167; a large +2*eps* shift can degrade generations
    # with no crossing at all). Independent "any_cross"/"any_move" quantifiers over
    # different fracs would wrongly report "actuatable" here.
    crossed = {0.0: False, -1.0: True, 2.0: False}
    delta = {0.0: 0.0, -1.0: 0.0, 2.0: 0.5}
    assert verdict(crossed, delta) != "actuatable"
    assert verdict(crossed, delta) == "readout-only"


def test_verdict_min_delta_boundary_is_inclusive():
    # Exactly MIN_DELTA (0.10) at the SAME frac as the crossing must count as moved.
    crossed = {0.0: False, 1.0: True}
    delta = {0.0: 0.0, 1.0: 0.10}
    assert verdict(crossed, delta) == "actuatable"


def test_verdict_negative_delta_counts_as_moved():
    # A steered DROP in refusal rate is evidence of actuation, not of nothing --
    # abs() must be applied, not a signed comparison.
    crossed = {0.0: False, 1.0: True}
    delta = {0.0: 0.0, 1.0: -0.5}
    assert verdict(crossed, delta) == "actuatable"


def test_require_signal_raises_when_grid_fully_clamped_into_baseline():
    # A fully-clamped scale grid (eps* far exceeds 1.5*input_scale) rounds every
    # steered row's frac into the frac=0 baseline bucket. Reporting "no-crossing"
    # off a table with no non-baseline bucket would be a contaminated baseline
    # silently read as "underpowered" rather than "broken sweep".
    table = {0.0: {"n": 40, "g_read": 1.0, "frac_refused": 0.1}}
    with pytest.raises(SystemExit, match="clamp"):
        require_signal(table)


def test_mean_arm_rows_filters_direction_and_drops_unmatched_scale(capsys):
    judged = [
        {"direction": "jtw_mean_diff_tgt", "scale": "0.0", "refused": "0"},
        {"direction": "jtw_mean_diff_tgt", "scale": "5.0", "refused": "1"},
        {"direction": "jtw_other", "scale": "5.0", "refused": "1"},
    ]
    readout = [
        {"direction": "jtw_mean_diff_tgt", "scale": "0.0", "g_read": "3.0"},
        # no readout entry at scale=5.0 -- that judged row must be dropped, not
        # silently coerced into some other bucket.
    ]
    summ = {"directions": {"mean_diff_tgt": {"median_eps_star": 2.5}}}
    out, dropped = mean_arm_rows(judged, readout, summ, direction="jtw_mean_diff_tgt")
    assert out == [{"frac": 0.0, "g_read": 3.0, "refused": 0.0}]
    assert dropped == 1
    captured = capsys.readouterr()
    assert "dropped 1" in captured.out


def test_readout_crossed_requires_baseline_sign_change_not_membership():
    # Baseline g_read is ALREADY inside the halfspace (-1.5 <= 0): a "crossing" here
    # is vacuous -- the readout never had to move. Reachable by construction: eps*
    # is the dataset-wide median_eps_star (reach_steer.py:123) while the baseline
    # g_read is measured on a DIFFERENT population (FACTUAL_PROMPTS / the refusal
    # holdout, reach_steer.py:111,133) that t02 was never fit on, so nothing
    # guarantees that population's unsteered readout sits outside the boundary.
    t = {0.0: {"g_read": -1.5, "frac_refused": 0.05},
         -1.0: {"g_read": -4.0, "frac_refused": 0.55}}
    c = readout_crossed(t)
    assert c == {0.0: False, -1.0: False}
    assert verdict(c, behavior_delta(t)) != "actuatable"


def test_readout_crossed_detects_genuine_sign_change_baseline_never_flagged():
    # A real crossing: baseline starts outside (g>0), steered lands inside (g<=0).
    # The baseline frac itself must never be reported as crossed.
    t = {0.0: {"g_read": 3.0, "frac_refused": 0.0},
         1.0: {"g_read": -1.0, "frac_refused": 0.4}}
    c = readout_crossed(t)
    assert c[0.0] is False
    assert c[1.0] is True


def test_verdict_refuses_a_baseline_only_table():
    with pytest.raises(SystemExit, match="baseline"):
        verdict({0.0: False}, {0.0: 0.0})


def test_named_errors_instead_of_bare_exceptions():
    # frac_of on an unparseable eps_star must not leak a bare ValueError.
    with pytest.raises(SystemExit, match="frac_of"):
        frac_of(5.0, "")
    # _read_json on a missing file must not leak a bare FileNotFoundError (or a
    # leaked file handle from a raw json.load(open(...))).
    with pytest.raises(SystemExit, match="missing"):
        reach_control._read_json("/nonexistent/path/reach_summary_doesnotexist.json")
