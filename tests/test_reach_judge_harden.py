import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_judge_harden as rjh


def _rows():
    # prompt "bad" fails at scale 0; "good" is TRUE everywhere
    rows = []
    for s in (0.0, 1.0, -1.0):
        rows.append({"direction": "d1", "scale": str(s), "prompt": "good",
                     "completion": "x", "verdict": "TRUE", "reason": ""})
        rows.append({"direction": "d1", "scale": str(s), "prompt": "bad",
                     "completion": "x",
                     "verdict": "FALSE" if s == 0.0 else "TRUE", "reason": ""})
    return rows


def test_problem_prompts_identifies_scale0_failures():
    assert rjh.problem_prompts(_rows()) == {"bad"}


def test_fractions_change_as_hand_computed():
    rows = _rows()
    raw = rjh.fractions(rows)
    hard = rjh.fractions(rows, exclude={"bad"})
    assert raw[("d1", 0.0)] == (2, 0.5, 0.5, 0.0)     # good TRUE + bad FALSE
    assert hard[("d1", 0.0)] == (1, 1.0, 0.0, 0.0)    # bad dropped
    assert hard[("d1", 1.0)] == (1, 1.0, 0.0, 0.0)


def test_spearman_monotone_and_constant():
    assert abs(rjh.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(rjh.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert np.isnan(rjh.spearman([1, 2, 3], [5, 5, 5]))


def test_harden_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with open("judge_reach_steer_toy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "scale", "prompt",
                                          "completion", "verdict", "reason"])
        w.writeheader()
        w.writerows(_rows())
    rjh.harden("toy")
    text = capsys.readouterr().out
    assert "1 problem prompts" in text
    with open("judge_hardened_fracs_toy.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    raw0 = [r for r in rows if r["hardened"] == "0" and r["scale"] == "0.0"][0]
    hard0 = [r for r in rows if r["hardened"] == "1" and r["scale"] == "0.0"][0]
    assert raw0["frac_false"] == "0.5000" and hard0["frac_false"] == "0.0000"
