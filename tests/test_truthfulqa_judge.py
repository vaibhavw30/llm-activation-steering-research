"""Tests for src/truthfulqa_judge.py. Torch-free: the scorer is always injected."""
import csv
import sys

import pytest

sys.path.insert(0, "src")

import truthfulqa_judge as tj


# --- question_of -----------------------------------------------------------------

def test_question_of_recovers_the_bare_question():
    assert tj.question_of("Q: What is 2+2?\nA:") == "What is 2+2?"


def test_question_of_ignores_the_answer_half():
    assert tj.question_of("Q: Why is the sky blue?\nA: Because of") == "Why is the sky blue?"


def test_question_of_handles_a_question_containing_a_colon():
    p = "Q: What does the saying mean: a stitch in time?\nA:"
    assert tj.question_of(p) == "What does the saying mean: a stitch in time?"


def test_question_of_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        tj.question_of("What is 2+2?")


def test_question_of_never_double_prefixes():
    """The bug this exists to prevent: adapters.truthfulqa_prompt adds its own 'Q: '."""
    assert not tj.question_of("Q: What is 2+2?\nA:").startswith("Q:")


@pytest.mark.parametrize("path", ["got_datasets/truthfulqa_holdout.csv",
                                  "got_datasets/truthfulqa.csv"])
def test_question_of_parses_every_real_statement(path):
    import pandas as pd
    col = pd.read_csv(path)["statement"].astype(str)
    for s in col:
        assert tj.question_of(s)


# --- score_rows ------------------------------------------------------------------

def _fake_score(truthful, informative):
    return lambda q, a: {"truthful": truthful, "informative": informative}


def test_score_rows_sets_all_four_columns():
    rows = [{"prompt": "Q: q?\nA:", "answer": "yes"}]
    out, n_empty = tj.score_rows(rows, _fake_score(True, True))
    assert out[0]["truthful"] == 1
    assert out[0]["informative"] == 1
    assert out[0]["truthful_and_informative"] == 1
    assert n_empty == 0


def test_refused_is_a_copy_of_truthful_and_informative():
    """reach_control.py keys on `refused`; the shim must track the real metric."""
    for t, i in [(True, True), (True, False), (False, True), (False, False)]:
        rows = [{"prompt": "Q: q?\nA:", "answer": "x"}]
        out, _ = tj.score_rows(rows, _fake_score(t, i))
        assert out[0]["refused"] == out[0]["truthful_and_informative"]
        assert out[0]["refused"] == int(t and i)


def test_truthful_but_uninformative_is_not_a_success():
    """'I have no comment' is 100% truthful and useless. Q1 found exactly one."""
    rows = [{"prompt": "Q: q?\nA:", "answer": "I have no comment"}]
    out, _ = tj.score_rows(rows, _fake_score(True, False))
    assert out[0]["truthful"] == 1
    assert out[0]["truthful_and_informative"] == 0


def test_empty_answer_scores_zero_without_calling_the_judge():
    def boom(q, a):
        raise AssertionError("judge must not be called on an empty answer")
    rows = [{"prompt": "Q: q?\nA:", "answer": "   "}]
    out, n_empty = tj.score_rows(rows, boom)
    assert n_empty == 1
    assert out[0]["truthful"] == 0 and out[0]["truthful_and_informative"] == 0


def test_score_rows_preserves_order_and_original_columns():
    rows = [{"prompt": "Q: a?\nA:", "answer": "1", "scale": "0.0", "direction": "d"},
            {"prompt": "Q: b?\nA:", "answer": "2", "scale": "1.5", "direction": "d"}]
    out, _ = tj.score_rows(rows, _fake_score(True, True))
    assert [r["scale"] for r in out] == ["0.0", "1.5"]
    assert [r["answer"] for r in out] == ["1", "2"]


def test_truth_only_scorer_returns_none_informative_and_counts_as_informative():
    """--truth-only leaves informativeness unmeasured; it must not silently zero it."""
    rows = [{"prompt": "Q: q?\nA:", "answer": "x"}]
    out, _ = tj.score_rows(rows, _fake_score(True, None))
    assert out[0]["informative"] == 1
    assert out[0]["truthful_and_informative"] == 1


def test_score_rows_judges_answer_not_completion():
    """The whole point: `completion` carries a fabricated next turn."""
    seen = {}

    def spy(q, a):
        seen["a"] = a
        return {"truthful": True, "informative": True}

    rows = [{"prompt": "Q: q?\nA:",
             "completion": "Paris. Q: And Spain? A: Madrid",
             "answer": "Paris."}]
    tj.score_rows(rows, spy)
    assert seen["a"] == "Paris."


# --- read_rows -------------------------------------------------------------------

def test_read_rows_refuses_a_pre_fix_csv(tmp_path):
    p = tmp_path / "old.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "scale", "prompt", "completion"])
        w.writerow(["d", "0.0", "Q: q?\nA:", "flattened text"])
    with pytest.raises(SystemExit, match="no `answer` column"):
        tj.read_rows(str(p))


def test_read_rows_refuses_an_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerow(["direction", "scale", "prompt", "completion", "answer"])
    with pytest.raises(SystemExit, match="no data rows"):
        tj.read_rows(str(p))


def test_read_rows_accepts_the_post_fix_schema(tmp_path):
    p = tmp_path / "new.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "scale", "prompt", "completion", "answer"])
        w.writerow(["d", "0.0", "Q: q?\nA:", "Paris. Q: x A: y", "Paris."])
    assert len(tj.read_rows(str(p))) == 1


# --- CLI -------------------------------------------------------------------------

def test_parser_defaults_to_truthfulqa_and_requires_an_arm():
    a = tj.build_parser().parse_args(["--arm", "mean"])
    assert a.dataset == "truthfulqa" and a.device == "cuda" and a.limit == 0
    with pytest.raises(SystemExit):
        tj.build_parser().parse_args([])


def test_arm_files_match_reach_steer_output_names():
    assert tj.ARM_FILES["mean"].format(ds="truthfulqa") == "reach_steer_truthfulqa.csv"
    assert tj.ARM_FILES["stmt"].format(ds="truthfulqa") == "reach_steer_stmt_truthfulqa.csv"


def test_module_imports_no_torch(monkeypatch):
    # monkeypatch, not a bare pop: the entries come back after the test. A popped torch
    # left a later `import torch` loading a SECOND copy, which dies re-registering its
    # TORCH_LIBRARY namespaces (the suite's test-order failure).
    import importlib
    import sys as _s
    for m in ("torch", "truthfulqa_judge"):
        monkeypatch.delitem(_s.modules, m, raising=False)
    importlib.import_module("truthfulqa_judge")
    assert "torch" not in _s.modules


def test_score_rows_carries_the_judge_probabilities_through():
    """J1: p(yes) and the yes+no mass reach the CSV, so strictness can be measured."""
    def score(q, a):
        return {"truthful": True, "informative": False, "p_truthful": 0.81,
                "p_informative": 0.3, "yesno_mass_truthful": 0.97,
                "yesno_mass_informative": 0.95}
    out, _ = tj.score_rows([{"prompt": "Q: q?\nA:", "answer": "yes"}], score)
    assert out[0]["p_truthful"] == 0.81 and out[0]["p_informative"] == 0.3
    assert out[0]["yesno_mass_truthful"] == 0.97


def test_score_rows_leaves_probability_columns_blank_for_an_empty_answer():
    out, n_empty = tj.score_rows([{"prompt": "Q: q?\nA:", "answer": "  "}],
                                 lambda q, a: pytest.fail("judge called on empty"))
    assert n_empty == 1 and out[0]["p_truthful"] == ""
