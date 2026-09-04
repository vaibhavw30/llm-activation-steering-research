import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prep_truthfulqa import (answer_pair, holdout_df, load_holdout, prompt_of,
                             split_holdout, statement_of, to_contrast_df)


def row(q, best, correct, incorrect, **extra):
    r = {"question": q, "best_answer": best, "correct_answers": correct,
         "incorrect_answers": incorrect, "category": "Misconceptions",
         "type": "Adversarial"}
    r.update(extra)
    return r


ROWS = [row("Q1?", "c1", ["c1", "c1b"], ["w1", "w1b"]),
        row("Q2?", "c2", ["c2"], ["w2"])]


def test_plain_prompt_ends_at_the_generation_point():
    assert prompt_of(" Q1? ") == "Q: Q1?\nA:"


def test_statement_joins_with_a_space_after_a_bare_colon():
    assert statement_of("Q1?", "c1") == "Q: Q1?\nA: c1"


def test_statement_adds_no_space_after_a_chat_template_newline():
    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return f"<user>{msgs[0]['content']}<model>\n"
    assert statement_of("Q1?", "c1", Tok()) == "<user>Q1?<model>\nc1"


def test_the_pair_prefixes_are_byte_identical():
    """The whole point of the format: the two classes differ only in the answer."""
    df = to_contrast_df([ROWS[0]])
    a, b = sorted(df["statement"])
    assert os.path.commonprefix([a, b]) == prompt_of("Q1?") + " "


def test_polarity_untruthful_is_label_1_by_default():
    df = to_contrast_df(ROWS)
    assert (df["label"] == 1).sum() == 2 and (df["label"] == 0).sum() == 2
    assert set(df[df.label == 1]["answer"]) == {"w1", "w2"}
    assert set(df[df.label == 0]["answer"]) == {"c1", "c2"}


def test_polarity_flag_can_flip_it():
    df = to_contrast_df(ROWS, label1="truthful")
    assert set(df[df.label == 1]["answer"]) == {"c1", "c2"}


def test_rejects_an_unknown_polarity():
    with pytest.raises(ValueError):
        to_contrast_df(ROWS, label1="neither")


def test_deterministic():
    assert to_contrast_df(ROWS).equals(to_contrast_df(ROWS))


def test_no_comment_best_answer_falls_back_to_an_informative_correct_answer():
    r = row("Q?", "I have no comment", ["I have no comment", "the real answer"], ["w"])
    assert answer_pair(r) == ("the real answer", "w")


def test_a_question_with_no_informative_correct_answer_is_dropped():
    r = row("Q?", "I have no comment", ["I have no comment"], ["w"])
    assert answer_pair(r) is None
    assert len(to_contrast_df([r])) == 0


def test_holdout_is_disjoint_at_the_question_level():
    rows = [row(f"Q{i}?", f"c{i}", [f"c{i}"], [f"w{i}"]) for i in range(20)]
    fit, hold = split_holdout(rows, n_hold=5)
    assert len(fit) == 15 and len(hold) == 5
    held = {r["question"] for r in hold}
    # no held-out question leaks into EITHER polarity of the fit set
    assert not (held & set(to_contrast_df(fit)["question"]))


def test_holdout_statement_is_the_prompt_with_no_answer():
    hold = holdout_df([ROWS[0]])
    assert hold["statement"].tolist() == ["Q: Q1?\nA:"]
    assert hold["question"].tolist() == ["Q1?"]


def test_holdout_list_columns_round_trip_through_csv(tmp_path):
    path = tmp_path / "hold.csv"
    holdout_df(ROWS).to_csv(path, index=False)
    back = load_holdout(path)
    assert back["correct_answers"].tolist() == [["c1", "c1b"], ["c2"]]
    assert back["incorrect_answers"].tolist() == [["w1", "w1b"], ["w2"]]
    assert json.loads(holdout_df(ROWS)["correct_answers"].iloc[0]) == ["c1", "c1b"]
