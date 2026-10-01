"""Tests for src/tqa_q2x.py (round 4, Q2x): the pure core and the CPU stages on fakes."""
import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import tqa_q2x as qx  # noqa: E402


# ------------------------------------------------------------------ doses, directions
def test_doses_point_past_the_predicted_crossing_and_start_at_the_anchor():
    assert qx.DOSES[0] == qx.ANCHOR == 2.0
    assert min(d for d in qx.DOSES if d > qx.PREDICTED_CROSSING) <= 4.0
    assert max(qx.DOSES) == 5.0 and 0.0 not in qx.DOSES


def test_check_doses_allows_q2s_grid_and_refuses_past_the_cap():
    # Q2's numbers: eps* 14.7065, input_scale 60.94 -> cap 91.4, top 73.5
    assert qx.check_doses(14.7065, 60.94) == pytest.approx(73.53, abs=0.01)
    with pytest.raises(SystemExit, match="cap"):
        qx.check_doses(14.7065, 40.0)


def test_q2x_directions_unit_seeded_and_not_jcs_null():
    q2 = np.array([3.0, 4.0, 0.0, 0.0])
    dirs = qx.q2x_directions(q2, n_rand=3)
    assert [n for n, _ in dirs] == ["q2_mean_diff", "rand_0", "rand_1", "rand_2"]
    assert np.allclose(dirs[0][1], q2 / 5)            # sign kept: q2_vector is truthful-signed
    assert all(np.linalg.norm(v) == pytest.approx(1.0) for _, v in dirs)
    again = qx.q2x_directions(q2, n_rand=3)
    assert all(np.allclose(a[1], b[1]) for a, b in zip(dirs, again))
    jc = np.random.default_rng(13).standard_normal(4)
    assert not np.allclose(dirs[1][1], jc / np.linalg.norm(jc))


# ------------------------------------------------------------------ form columns
def test_budget_hit_is_no_newline_in_the_raw_completion():
    assert qx.budget_hit(" a long answer that never stopped") is True
    assert qx.budget_hit(" Short answer.\nQ: next?") is False
    assert qx.budget_hit("  trailing newline only\n") is True     # stripped first


def test_distinct_ratio():
    assert qx.distinct_ratio("the the the cat") == pytest.approx(0.5)
    assert qx.distinct_ratio("") == 0.0


def test_truncated_row_cuts_to_the_baseline_length_and_keeps_the_full_text():
    r = qx.truncated_row({"answer": "one two three four", "question": "q"}, 2)
    assert (r["answer"], r["answer_full"], r["cut"], r["words_base"]) == \
        ("one two", "one two three four", 1, 2)
    short = qx.truncated_row({"answer": "one", "question": "q"}, 5)
    assert (short["answer"], short["cut"]) == ("one", 0)


def test_truncated_row_keeps_one_word_when_the_baseline_was_empty():
    r = qx.truncated_row({"answer": "one two", "question": "q"}, 0)
    assert r["answer"] == "one" and r["cut"] == 1


# ------------------------------------------------------------------ statistics
def test_holm():
    assert qx.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert qx.holm([0.5]) == [0.5]


def test_crossing_dose():
    assert qx.crossing_dose({2.0: 15.0, 3.0: 4.0, 3.5: -1.0, 4.0: -6.0}) == 3.5
    assert qx.crossing_dose({2.0: 15.0, 5.0: 0.5}) is None


def _sr(rows):
    return [dict(r) for r in rows]


def test_form_stats_per_direction_and_dose():
    rows = [{"direction": "q2_mean_diff", "frac": "3.5", "g": g, "crossed": int(g < 0),
             "words": w, "budget_hit": b, "distinct_ratio": d}
            for g, w, b, d in ((-1.0, 10, 1, 0.9), (2.0, 20, 0, 0.7), (-3.0, 30, 1, 0.4))]
    s = qx.form_stats(rows)[("q2_mean_diff", 3.5)]
    assert s["median_g"] == -1.0 and s["crossed_share"] == pytest.approx(2 / 3)
    assert s["mean_words"] == 20 and s["budget_hit_share"] == pytest.approx(2 / 3)
    assert s["median_distinct_ratio"] == 0.7


def _q2row(frac, *, full_rate=0.5, full_gain=0.2, full_p=0.001, full_perm=1 / 9,
           cut_gain=0.0, cut_p=0.9, cut_perm=0.5, median_g=-1.0, budget=0.1, distinct=0.9):
    return {"direction": "q2_mean_diff", "frac": frac, "full_rate": full_rate,
            "full_gain": full_gain, "full_mcnemar_p": full_p, "full_perm_p": full_perm,
            "cut_gain": cut_gain, "cut_mcnemar_p": cut_p, "cut_perm_p": cut_perm,
            "median_g": median_g, "budget_hit_share": budget,
            "median_distinct_ratio": distinct}


def test_classify_content_form_degeneration():
    r = _q2row(4.0, cut_gain=0.2, cut_p=0.001, cut_perm=1 / 9)
    assert qx.classify(r, 0.003, 0.003, 0.5) == ["a_content"]
    r = _q2row(4.0)                                         # full moves, cut flat
    assert qx.classify(r, 0.9, 0.003, 0.5) == ["b_form"]
    r = _q2row(5.0, full_rate=0.3, full_gain=0.0, full_p=1.0, full_perm=0.6,
               budget=0.7)
    assert qx.classify(r, 0.9, 1.0, 0.5) == ["c_degeneration"]
    r = _q2row(4.0, full_gain=0.05, full_p=0.4, full_perm=0.5)
    assert qx.classify(r, 0.9, 0.4, 0.5) == ["none"]


def test_classify_a_tie_with_a_random_direction_is_not_content():
    r = _q2row(4.0, cut_gain=0.2, cut_p=0.001, cut_perm=2 / 9)
    assert "a_content" not in qx.classify(r, 0.003, 0.003, 0.5)


def test_classify_one_random_direction_sets_the_bar_at_one_half():
    r = _q2row(4.0, cut_gain=0.2, cut_p=0.001, cut_perm=1 / 2)
    assert qx.classify(r, 0.003, 0.003, 0.5, n_rand=1) == ["a_content"]


def test_read_outcome_reads_from_the_crossing_and_holm_corrects():
    rows = [_q2row(2.0, median_g=15.0), _q2row(3.0, median_g=4.0),
            _q2row(3.5, median_g=-1.0), _q2row(4.0, median_g=-5.0)]
    out = qx.read_outcome(rows, 0.3958, 0.6337)
    assert out["gate"]["passed"] is True
    assert out["crossing"]["realized"] == 3.5 and out["uncrossed"] is False
    assert out["read_at"] == [3.5, 4.0]
    assert out["per_dose"]["3.5"]["full_p_holm"] == pytest.approx(0.002)
    assert out["headline"] == ["b_form"]


def test_read_outcome_uncrossed_reads_at_the_top_dose():
    rows = [_q2row(2.0, median_g=15.0), _q2row(5.0, median_g=0.5)]
    out = qx.read_outcome(rows, 0.3958, 0.6337)
    assert out["uncrossed"] is True and out["read_at"] == [5.0]


def test_read_outcome_gate_failure_withholds_the_headline():
    rows = [_q2row(2.0, full_rate=0.20, median_g=15.0), _q2row(4.0, median_g=-5.0)]
    out = qx.read_outcome(rows, 0.3958, 0.6337)
    assert out["gate"]["passed"] is False and out["headline"] == ["GATE FAILED"]


def test_summary_rows_merge_full_cut_and_form():
    t = {"direction": "q2_mean_diff", "frac": 4.0, "rate": 0.6, "lo": 0.5, "hi": 0.7,
         "gain": 0.3, "gained": 20, "lost": 1, "mcnemar_p": 0.001, "perm_p": 1 / 9}
    full = {"truthful": [t], "truthful_and_informative": [dict(t, rate=0.55)]}
    cut = {"truthful": [dict(t, rate=0.3, gain=0.02)],
           "truthful_and_informative": [dict(t, rate=0.28)]}
    form = {("q2_mean_diff", 4.0): {"median_g": -2.0}}
    (row,) = qx.summary_rows(full, cut, form)
    assert (row["full_rate"], row["cut_rate"], row["full_ti_rate"], row["cut_ti_rate"]) == \
        (0.6, 0.3, 0.55, 0.28)
    assert row["cut_gain"] == 0.02 and row["median_g"] == -2.0


# ------------------------------------------------------------------ the gate's interval
def test_q2_interval_reads_the_frac_minus_2_row(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "direction", "frac", "wilson_lo", "wilson_hi"])
        w.writeheader()
        w.writerow({"arm": "mean", "direction": "jtw_mean_diff_tgt", "frac": "-1.5",
                    "wilson_lo": 0.1, "wilson_hi": 0.2})
        w.writerow({"arm": "mean", "direction": "jtw_mean_diff_tgt", "frac": "-2.0",
                    "wilson_lo": 0.3958, "wilson_hi": 0.6337})
    assert qx.q2_interval(str(p)) == (0.3958, 0.6337)


def test_q2_interval_missing_row_stops_cleanly(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("arm,direction,frac,wilson_lo,wilson_hi\nmean,other,-2.0,0.1,0.2\n")
    with pytest.raises(SystemExit, match="frac -2"):
        qx.q2_interval(str(p))
