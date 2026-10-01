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
    # a trailing newline is an answer that ended; a leading one is not
    assert qx.budget_hit(" Answer ended.\n") is False
    assert qx.budget_hit("\n an answer that never stopped") is True


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


# ------------------------------------------------------------------ stages, on fakes
class _St:
    vec = "unset"

    def set(self, v):
        self.vec = v


def test_steer_rows_writes_steer_block_columns_plus_the_readout(monkeypatch):
    import dct_steer_utils as su
    monkeypatch.setattr(su, "generate_raw",
                        lambda m, t, p, n: " A long answer that never stops")
    st = _St()
    rows = qx.steer_rows(None, None, st, "q2_mean_diff", np.array([1.0, 0.0]), 3.5, 2.0,
                         [{"question": "Why?"}], read=lambda p: -0.5)
    assert st.vec.tolist() == [7.0, 0.0]                         # 3.5 x eps 2.0
    (r,) = rows
    assert r["prompt"] == "Q: Why?\nA:" and r["answer"] == "A long answer that never stops"
    assert (r["frac"], r["scale"], r["g"], r["crossed"]) == (3.5, 7.0, -0.5, 1)
    assert (r["budget_hit"], r["words"]) == (1, 6)
    base = qx.steer_rows(None, None, st, "baseline", None, 0.0, 2.0,
                         [{"question": "Why?"}], read=lambda p: 9.0)
    assert st.vec is None and base[0]["scale"] == 0.0 and base[0]["crossed"] == 0


def test_blocks_todo_skips_done_and_refuses_a_partial_block():
    blocks = [("baseline", None, 0.0), ("q2_mean_diff", "v", 2.0)]
    assert qx.blocks_todo({("baseline", 0.0): 4}, blocks, 4) == [blocks[1]]
    with pytest.raises(SystemExit, match="q2_mean_diff dose 2"):
        qx.blocks_todo({("q2_mean_diff", 2.0): 3}, blocks, 4)


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _judged(direction, frac, q, t, words=5, g=1.0):
    return {"direction": direction, "frac": frac, "question": f"q{q}", "prompt": f"Q: q{q}\nA:",
            "answer": " ".join(["w"] * words), "g": g, "crossed": int(g < 0),
            "budget_hit": 0, "words": words, "distinct_ratio": 1.0,
            "truthful": t, "informative": 1, "truthful_and_informative": t}


def test_stage_summary_end_to_end_derives_n_rand_from_the_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n = 10
    full, cut = [], []
    for q in range(n):
        full.append(_judged("baseline", 0.0, q, 0, g=10.0))
        for f, g in ((2.0, 5.0), (4.0, -2.0)):
            full.append(_judged("q2_mean_diff", f, q, int(q < 5), words=20, g=g))
            cut.append(_judged("q2_mean_diff", f, q, 0, words=5, g=g))   # gain gone when cut
            full.append(_judged("rand_0", f, q, int(q == 0), g=g + 10))
            cut.append(_judged("rand_0", f, q, int(q == 0), g=g + 10))
    _write("q2x_judged_truthfulqa.csv", full)
    _write("q2x_trunc_judged_truthfulqa.csv", cut)
    _write(qx.Q2_V2_TRUTHFUL, [{"arm": "mean", "direction": "jtw_mean_diff_tgt",
                                "frac": "-2.0", "wilson_lo": 0.2, "wilson_hi": 0.8}])
    qx.stage_summary()
    out = json.load(open("q2x_outcome_truthfulqa.json"))
    assert out["n_rand"] == 1                         # one random direction in the data
    assert out["gate"]["passed"] is True              # anchor 0.5 inside [0.2, 0.8]
    assert out["crossing"]["realized"] == 4.0
    assert out["headline"] == ["b_form"]
    rows = list(csv.DictReader(open("q2x_summary_truthfulqa.csv")))
    assert {r["direction"] for r in rows} == {"q2_mean_diff", "rand_0"}
    with pytest.raises(SystemExit, match="exists"):
        qx.stage_summary()


def test_main_refuses_limit_without_prefix():
    with pytest.raises(SystemExit):
        qx.main(["--stage", "steer", "--limit", "2"])


def test_main_skips_a_finished_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    open("q2x_outcome_truthfulqa.json", "w").write("{}")      # the last file written
    qx.main(["--stage", "summary"])
    assert "exists, skipping" in capsys.readouterr().out


# ------------------------------------------------------------------ final-review fixes
def _fixture(tmp_path, with_interval=True, drop_cut=0):
    """The end-to-end fixture of test_stage_summary_end_to_end..., written to tmp_path."""
    full, cut = [], []
    for q in range(10):
        full.append(_judged("baseline", 0.0, q, 0, g=10.0))
        for f, g in ((2.0, 5.0), (4.0, -2.0)):
            full.append(_judged("q2_mean_diff", f, q, int(q < 5), words=20, g=g))
            cut.append(_judged("q2_mean_diff", f, q, 0, words=5, g=g))
            full.append(_judged("rand_0", f, q, int(q == 0), g=g + 10))
            cut.append(_judged("rand_0", f, q, int(q == 0), g=g + 10))
    _write("q2x_judged_truthfulqa.csv", full)
    _write("q2x_trunc_judged_truthfulqa.csv", cut[:len(cut) - drop_cut])
    row = {"arm": "mean", "direction": "jtw_mean_diff_tgt" if with_interval else "other",
           "frac": "-2.0", "wilson_lo": 0.2, "wilson_hi": 0.8}
    _write(qx.Q2_V2_TRUTHFUL, [row])
    return full, cut


def test_summary_that_stops_writes_nothing_so_a_rerun_is_not_skipped(tmp_path, monkeypatch,
                                                                     capsys):
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path, with_interval=False)
    with pytest.raises(SystemExit, match="frac -2"):
        qx.main(["--stage", "summary"])
    assert not os.path.exists("q2x_summary_truthfulqa.csv")
    assert not os.path.exists("q2x_outcome_truthfulqa.json")
    _fixture(tmp_path)                                        # the interval row restored
    qx.main(["--stage", "summary"])
    assert "HEADLINE" in capsys.readouterr().out


def test_summary_refuses_an_incomplete_truncated_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path, drop_cut=3)
    with pytest.raises(SystemExit, match="truncated"):
        qx.stage_summary()
    assert not os.path.exists("q2x_summary_truthfulqa.csv")


def test_check_judged_refuses_unequal_blocks_and_no_random_direction():
    full = [_judged("baseline", 0.0, q, 0) for q in range(4)]
    full += [_judged("q2_mean_diff", 2.0, q, 1) for q in range(3)]
    full += [_judged("rand_0", 2.0, q, 0) for q in range(4)]
    cut = [r for r in full if r["direction"] != "baseline"]
    with pytest.raises(SystemExit, match="q2_mean_diff dose 2 has 3"):
        qx.check_judged(full, cut)
    full = [r for r in full if not r["direction"].startswith("rand_")]
    full += [_judged("q2_mean_diff", 2.0, 3, 1)]
    cut = [r for r in full if r["direction"] != "baseline"]
    with pytest.raises(SystemExit, match="no random direction"):
        qx.check_judged(full, cut)


def test_read_outcome_gate_failure_makes_no_r1_or_r2_reading():
    rows = [_q2row(2.0, full_rate=0.20, median_g=15.0),
            _q2row(4.0, median_g=-5.0, cut_gain=0.2, cut_p=0.001, cut_perm=1 / 9)]
    out = qx.read_outcome(rows, 0.3958, 0.6337)
    assert out["per_dose"] == {} and out["read_at"] == []
    assert out["crossing"]["realized"] == "not read"
    assert out["diagnostic"]["per_dose"]["4.0"]["labels"] == ["a_content"]


def test_blocks_todo_refuses_a_duplicated_block():
    blocks = [("baseline", None, 0.0)]
    with pytest.raises(SystemExit, match="baseline dose 0 has 8 of 4"):
        qx.blocks_todo({("baseline", 0.0): 8}, blocks, 4)


def test_steer_counts_refuses_a_torn_last_row(tmp_path):
    p = tmp_path / "steer.csv"
    p.write_text("direction,frac,answer\nbaseline,0.0,one\nbaseline,0.0,tw")
    with pytest.raises(SystemExit, match="does not end"):
        qx.steer_counts(str(p))
    p.write_text("direction,frac,answer\nbaseline,0.0,one\nbaseline\n")
    with pytest.raises(SystemExit, match="torn"):
        qx.steer_counts(str(p))
    p.write_text("direction,frac,answer\nbaseline,0.0,one\nrand_0,2.0,x\nbaseline,0.0,two\n")
    assert qx.steer_counts(str(p)) == {("baseline", 0.0): 2, ("rand_0", 2.0): 1}
    assert qx.steer_counts(str(tmp_path / "absent.csv")) == {}


# ------------------------------------------------------------------ resume round trips
class _FakeJudge:
    loads = 0

    def __init__(self, device):
        type(self).loads += 1

    def score(self, question, answer):
        return {"truthful": len(answer.split()) % 2, "informative": 1}


def _steer_file(n_q=3):
    """A finished steer CSV: a baseline and two dosed blocks, answers of varied length."""
    rows = []
    for name, f in (("baseline", 0.0), ("q2_mean_diff", 2.0), ("rand_0", 2.0)):
        for q in range(n_q):
            words = 2 if name == "baseline" else 3 + q
            rows.append({"direction": name, "frac": f, "question": f"q{q}",
                         "prompt": f"Q: q{q}\nA:", "answer": " ".join(["w"] * words),
                         "g": 1.0, "crossed": 0, "budget_hit": 0, "words": words,
                         "distinct_ratio": 1.0 / words})
    _write("q2x_steer_truthfulqa.csv", rows)
    return rows


def test_steer_resume_round_trip_reruns_only_the_missing_block(tmp_path, monkeypatch):
    import dct_steer_utils as su
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(su, "generate_raw",
                        lambda m, t, p, n: ' Yes, "quoted", and more\nQ: next')
    recs = [{"question": f"Why {i}, really?"} for i in range(3)]
    v = np.array([1.0, 0.0])
    blocks = [("baseline", None, 0.0), ("q2_mean_diff", v, 2.0), ("q2_mean_diff", v, 2.5)]
    p = "q2x_steer_truthfulqa.csv"
    for name, vec, f in blocks[:2]:                     # the job dies after two blocks
        qx.append_rows(p, qx.steer_rows(None, None, _St(), name, vec, f, 2.0, recs,
                                        read=lambda q: 1.0))
    todo = qx.blocks_todo(qx.steer_counts(p), blocks, len(recs))
    assert [(n, f) for n, _, f in todo] == [("q2_mean_diff", 2.5)]


def test_truncate_resumes_after_a_killed_block_in_steer_order(tmp_path, monkeypatch):
    import judge_audit as ja
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ja, "BLOCK", 2)
    monkeypatch.setattr(qx, "_judge", lambda: _FakeJudge)
    _steer_file()
    out = "q2x_trunc_judged_truthfulqa.csv"
    qx.stage_truncate("cpu")
    whole = list(csv.DictReader(open(out)))
    os.remove(out)
    _write(out, whole[:2])                              # killed after the first block
    qx.stage_truncate("cpu")
    again = list(csv.DictReader(open(out)))
    key = ("direction", "frac", "question", "answer", "truthful")
    assert [tuple(r[k] for k in key) for r in again] == \
        [tuple(r[k] for k in key) for r in whole]
    assert len(again) == 6


def test_finished_judge_stages_do_not_load_the_judge_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qx, "_judge", lambda: _FakeJudge)
    _steer_file()
    qx.stage_judge("cpu")
    qx.stage_truncate("cpu")
    before = _FakeJudge.loads
    qx.stage_judge("cpu")                               # a resubmit with nothing left
    qx.stage_truncate("cpu")
    assert _FakeJudge.loads == before


def test_truncated_row_form_columns_describe_the_cut_answer():
    r = qx.truncated_row({"answer": "two two one four", "question": "q", "words": 4,
                          "distinct_ratio": 0.75}, 2)
    assert (r["words"], r["distinct_ratio"], r["words_full"]) == (2, 0.5, 4)
