# tests/test_reach_control.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from reach_control import (frac_of, align_stmt_rows, aggregate, readout_crossed,
                           behavior_delta, verdict)


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
