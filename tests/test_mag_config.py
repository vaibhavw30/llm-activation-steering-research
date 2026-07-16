import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag import config as c


def test_taus_signed_and_zero():
    assert c.TAUS == [-1.0, -0.3, 0.0, 0.3, 1.0]


def test_q_strings_present_and_shaped():
    assert c.Q_TRUTH.endswith("Statement: ")
    assert c.Q_SUFFIX == "\nAnswer:"
    # few-shot has one yes and one no example
    assert "Answer: yes" in c.E_FEWSHOT and "Answer: no" in c.E_FEWSHOT


def test_operator_names_are_the_eight():
    assert c.OPERATOR_NAMES == ["Direct", "Prefixed", "Answered", "Verdict",
                                "InputDelta", "QuestionDelta", "Interaction", "FewShot"]


def test_yes_no_variants_nonempty():
    assert c.YES_VARIANTS and c.NO_VARIANTS
    assert all(isinstance(s, str) for s in c.YES_VARIANTS + c.NO_VARIANTS)
