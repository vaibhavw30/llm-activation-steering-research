"""Tests for tqa_q2_analyze, the script the Q2 doc's numbers come from.

Everything here is synthetic. The real judged CSVs are cluster output and are not in
git, so a test that read them would pass on one machine and error on another.
"""
import json
import os

import pandas as pd
import pytest

import tqa_q2_analyze as q2

SUMM = {"directions": {"mean_diff_tgt": {"median_eps_star": 14.706505916088334},
                       "probe_grad_tgt": {"median_eps_star": 2.7029772528991374}}}


# --- eps_star_of: the frac axis every comparison in the doc is drawn against ---

def test_eps_star_strips_the_jtw_prefix():
    assert q2.eps_star_of("jtw_mean_diff_tgt", SUMM) == 14.706505916088334
    assert q2.eps_star_of("jtw_probe_grad_tgt", SUMM) == 2.7029772528991374


@pytest.mark.parametrize("d", ["rand_ctrl_0", "rand_ctrl_1", "rand_ctrl_17"])
def test_random_controls_borrow_the_mean_diff_eps_star(d):
    """arm_rand_ctrl reuses mean_diff_tgt's scale grid. If the control got its own
    eps* it would land on a different frac axis and would not control anything."""
    assert q2.eps_star_of(d, SUMM) == SUMM["directions"]["mean_diff_tgt"]["median_eps_star"]


def test_unknown_direction_names_itself_rather_than_KeyErroring():
    with pytest.raises(SystemExit) as e:
        q2.eps_star_of("jtw_not_a_direction", SUMM)
    assert "not_a_direction" in str(e.value)


def test_control_and_mean_arm_land_on_the_same_fracs():
    scales = [0.0, 14.706505916088334, -29.413011832176668]
    a = [round(s / q2.eps_star_of("jtw_mean_diff_tgt", SUMM), 2) for s in scales]
    b = [round(s / q2.eps_star_of("rand_ctrl_0", SUMM), 2) for s in scales]
    assert a == b == [0.0, 1.0, -2.0]


# --- mcnemar_exact: the test the paired design earns ---

def _s(bits):
    return pd.Series(list(bits), index=[f"q{i}" for i in range(len(bits))])


def test_mcnemar_counts_gains_and_losses_in_the_right_direction():
    gained, lost, _, n = q2.mcnemar_exact(_s([0, 0, 1, 1]), _s([1, 0, 0, 1]))
    assert (gained, lost, n) == (1, 1, 4)


def test_mcnemar_six_one_directional_flips_is_the_documented_p():
    """The power claim the Q2 design was sized on: 6 flips one way gives p=0.031."""
    gained, lost, p, _ = q2.mcnemar_exact(_s([0] * 6), _s([1] * 6))
    assert (gained, lost) == (6, 0)
    assert p == pytest.approx(0.03125)


def test_mcnemar_eight_one_directional_flips():
    _, _, p, _ = q2.mcnemar_exact(_s([0] * 8), _s([1] * 8))
    assert p == pytest.approx(0.0078125)


def test_mcnemar_no_discordant_pairs_is_p_one():
    gained, lost, p, _ = q2.mcnemar_exact(_s([1, 0, 1]), _s([1, 0, 1]))
    assert (gained, lost, p) == (0, 0, 1.0)


def test_mcnemar_is_symmetric_under_swapping_the_arms():
    a, b = _s([0, 0, 1, 0, 1]), _s([1, 1, 0, 1, 1])
    g1, l1, p1, _ = q2.mcnemar_exact(a, b)
    g2, l2, p2, _ = q2.mcnemar_exact(b, a)
    assert (g1, l1) == (l2, g2) and p1 == p2


def test_mcnemar_pairs_on_the_index_not_on_row_order():
    """Rows are keyed by prompt. A judged CSV whose arms are in different orders must
    still pair each question with itself."""
    base = pd.Series([0, 1], index=["qA", "qB"])
    arm = pd.Series([1, 1], index=["qB", "qA"])
    gained, lost, _, n = q2.mcnemar_exact(base, arm)
    assert (gained, lost, n) == (1, 0, 2)


def test_mcnemar_drops_questions_missing_from_one_arm():
    base = pd.Series([0, 0], index=["qA", "qB"])
    arm = pd.Series([1], index=["qA"])
    gained, lost, _, n = q2.mcnemar_exact(base, arm)
    assert (gained, lost, n) == (1, 0, 1)


# --- length_adjusted_p: must degrade, never crash ---

def test_length_adjustment_returns_none_when_the_outcome_is_constant():
    d = pd.DataFrame({q2.SCORE_COL: [1, 1, 1, 1], "words": [3, 9, 12, 30],
                      "frac": [0, -1, -2, -2]})
    assert q2.length_adjusted_p(d) is None


def test_length_adjustment_returns_a_probability():
    d = pd.DataFrame({q2.SCORE_COL: [0, 0, 1, 1, 0, 1] * 6,
                      "words": [2, 3, 20, 25, 4, 30] * 6,
                      "frac": [0, 0, -2, -2, 1, -2] * 6})
    got = q2.length_adjusted_p(d)
    if got is not None:            # sklearn absent is a legal outcome
        p, stat = got
        assert 0.0 <= p <= 1.0 and stat >= 0.0


# --- load_arm: a partial run must say which stage is missing ---

def test_load_arm_names_the_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        q2.load_arm("truthfulqa", "randctrl", SUMM)
    assert "judge_refusal_truthfulqa_randctrl.csv" in str(e.value)


def test_load_arm_names_a_missing_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"direction": ["jtw_mean_diff_tgt"], "scale": [0.0],
                  "prompt": ["Q: x\nA:"]}).to_csv(
        "judge_refusal_truthfulqa_mean.csv", index=False)
    with pytest.raises(SystemExit) as e:
        q2.load_arm("truthfulqa", "mean", SUMM)
    assert "answer" in str(e.value)


def test_load_arm_adds_frac_and_words(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"direction": ["jtw_mean_diff_tgt"] * 2,
                  "scale": [0.0, -29.413011832176668],
                  "prompt": ["Q: x\nA:"] * 2,
                  "answer": ["Paris", "The answer is probably Paris"],
                  q2.SCORE_COL: [0, 1]}).to_csv(
        "judge_refusal_truthfulqa_mean.csv", index=False)
    d = q2.load_arm("truthfulqa", "mean", SUMM)
    assert d["frac"].tolist() == [0.0, -2.0]
    assert d["words"].tolist() == [1, 5]
    assert d["arm"].unique().tolist() == ["mean"]


# --- end to end ---

def test_analyze_reports_every_frac_and_flags_the_paired_shift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    json.dump(SUMM, open("reach_summary_truthfulqa.json", "w"))
    eps = SUMM["directions"]["mean_diff_tgt"]["median_eps_star"]
    prompts = [f"Q: q{i}\nA:" for i in range(10)]
    rows = []
    for frac, scores in ((0.0, [0] * 10), (-2.0, [1] * 8 + [0, 0])):
        for p, s in zip(prompts, scores):
            rows.append({"direction": "jtw_mean_diff_tgt", "scale": frac * eps,
                         "prompt": p, "answer": "word " * (3 if s else 1),
                         "truthful": s, "informative": 1, q2.SCORE_COL: s})
    pd.DataFrame(rows).to_csv("judge_refusal_truthfulqa_mean.csv", index=False)

    out = q2.analyze("truthfulqa", ["mean"])
    by = {r["frac"]: r for r in out}
    assert set(by) == {0.0, -2.0}
    assert by[0.0]["mcnemar_gained"] == 0 and by[0.0]["mcnemar_p"] == 1.0
    assert by[-2.0]["mcnemar_gained"] == 8 and by[-2.0]["mcnemar_lost"] == 0
    assert by[-2.0]["mcnemar_p"] == pytest.approx(0.00781, rel=1e-3)
    assert by[-2.0]["rate"] == 0.8
    assert by[-2.0]["mean_words"] > by[0.0]["mean_words"]


def test_tiny_p_values_survive_the_rounding():
    """round(p, 6) would flatten a decisive p to 0.0 and read as a formatting bug."""
    _, _, p, _ = q2.mcnemar_exact(_s([0] * 25), _s([1] * 25))
    assert float(f"{p:.3g}") > 0.0


def test_report_writes_a_csv_whose_header_matches_the_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [{"arm": "mean", "direction": "jtw_mean_diff_tgt", "frac": 0.0, "n": 1,
             "n_truthful_informative": 0, "rate": 0.0, "wilson_lo": 0.0,
             "wilson_hi": 1.0, "rate_truthful": 0.0, "rate_informative": 1.0,
             "mean_words": 1.0, "mcnemar_gained": 0, "mcnemar_lost": 0,
             "mcnemar_p": 1.0, "length_adjusted_frac_p": ""}]
    q2.report(rows, "out.csv")
    assert os.path.exists("out.csv")
    back = pd.read_csv("out.csv")
    assert list(back.columns) == list(rows[0].keys())
    assert len(back) == 1


# --- control_contrast: the registered test, target against each random control ---

def _arm(direction, scores, prompts, eps, words=1):
    """One (direction, dose) block as load_arm would hand it to control_contrast."""
    return pd.DataFrame([{"direction": direction, "arm": "x", "frac": -2.0,
                          "scale": -2.0 * eps, "prompt": p,
                          "answer": "word " * words, "words": words,
                          "truthful": s, "informative": 1, q2.SCORE_COL: s}
                         for p, s in zip(prompts, scores)])


def test_control_contrast_pairs_the_target_against_each_control():
    """The registered bar: beat a norm-matched control at the same dose. Two controls
    here, so two rows, each paired on the same prompts as the target."""
    eps = SUMM["directions"]["mean_diff_tgt"]["median_eps_star"]
    prompts = [f"Q: q{i}\nA:" for i in range(10)]
    frames = [_arm(q2.TARGET, [1] * 8 + [0, 0], prompts, eps, words=9),
              _arm("rand_ctrl_0", [0] * 10, prompts, eps),
              _arm("rand_ctrl_1", [1] + [0] * 9, prompts, eps)]

    rows = q2.control_contrast(frames, frac=-2.0)
    by = {r["control"]: r for r in rows}
    assert set(by) == {"rand_ctrl_0", "rand_ctrl_1"}
    assert by["rand_ctrl_0"]["target_wins"] == 8
    assert by["rand_ctrl_0"]["control_wins"] == 0
    assert by["rand_ctrl_1"]["target_wins"] == 7
    assert by["rand_ctrl_1"]["control_wins"] == 0
    assert by["rand_ctrl_0"]["n_paired"] == 10
    assert by["rand_ctrl_0"]["target_rate"] == 0.8
    assert by["rand_ctrl_0"]["target_words"] > by["rand_ctrl_0"]["control_words"]


def test_control_contrast_is_paired_not_a_difference_of_rates():
    """Two arms with the SAME rate but disjoint successes. An unpaired comparison sees
    nothing; the paired one sees every question as discordant, which is the whole
    reason the registered test is McNemar and not two proportions."""
    eps = SUMM["directions"]["mean_diff_tgt"]["median_eps_star"]
    prompts = [f"Q: q{i}\nA:" for i in range(8)]
    frames = [_arm(q2.TARGET, [1, 1, 1, 1, 0, 0, 0, 0], prompts, eps),
              _arm("rand_ctrl_0", [0, 0, 0, 0, 1, 1, 1, 1], prompts, eps)]

    r, = q2.control_contrast(frames, frac=-2.0)
    assert r["target_rate"] == r["control_rate"] == 0.5
    assert r["target_wins"] == 4 and r["control_wins"] == 4
    assert r["mcnemar_p"] == 1.0


def test_control_contrast_is_empty_without_the_target_arm():
    """A one-arm run must still report rather than KeyError on the missing target."""
    eps = SUMM["directions"]["mean_diff_tgt"]["median_eps_star"]
    prompts = [f"Q: q{i}\nA:" for i in range(4)]
    assert q2.control_contrast([_arm("rand_ctrl_0", [1, 0, 1, 0], prompts, eps)]) == []


def test_control_contrast_selects_the_requested_dose():
    """frac is a knob, not a constant: asking for a dose no row carries returns
    nothing rather than silently contrasting a different dose."""
    eps = SUMM["directions"]["mean_diff_tgt"]["median_eps_star"]
    prompts = [f"Q: q{i}\nA:" for i in range(4)]
    frames = [_arm(q2.TARGET, [1, 1, 1, 0], prompts, eps),
              _arm("rand_ctrl_0", [0, 0, 0, 0], prompts, eps)]
    assert q2.control_contrast(frames, frac=-1.0) == []
    assert len(q2.control_contrast(frames, frac=-2.0)) == 1
