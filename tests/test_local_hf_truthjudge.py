"""TruthJudge routing and the yes-probability readout (plan J0, J1). Torch only, no 7B
model: the judges are fakes, and the probability helpers are pure functions."""
import sys

import pytest
import torch

sys.path.insert(0, "src")

from judges import local_hf  # noqa: E402


class _RecordingJudge:
    """Stands in for _YesNoJudge: records every prompt, answers from a fixed verdict."""

    def __init__(self, verdict, p):
        self.prompts, self.verdict, self.p = [], verdict, p

    def judge(self, prompt):
        self.prompts.append(prompt)
        return {"verdict": self.verdict, "p_yes": self.p, "yesno_mass": 0.99}


def _truth_judge(truth, info):
    tj = local_hf.TruthJudge.__new__(local_hf.TruthJudge)  # skip loading 2 x 7B
    tj.truth, tj.info = truth, info
    return tj


def test_each_judge_gets_its_own_model_card_prompt():
    t, i = _RecordingJudge(True, 0.9), _RecordingJudge(False, 0.2)
    _truth_judge(t, i).score("What is 2+2?", "4")
    assert t.prompts == ["Q: What is 2+2?\nA: 4\nTrue:"]
    assert i.prompts == ["Q: What is 2+2?\nA: 4\nHelpful:"]


def test_score_keeps_the_old_keys_and_adds_the_probabilities():
    out = _truth_judge(_RecordingJudge(True, 0.9), _RecordingJudge(False, 0.2)).score("q?", "a")
    assert out["truthful"] is True and out["informative"] is False
    assert out["p_truthful"] == pytest.approx(0.9)
    assert out["p_informative"] == pytest.approx(0.2)
    assert out["yesno_mass_truthful"] == pytest.approx(0.99)


def test_yes_prob_is_the_two_way_softmax_of_yes_and_no():
    logits = torch.full((10,), -20.0)
    logits[3], logits[7] = 2.0, 0.0            # yes = 3, no = 7
    p, mass = local_hf.yes_prob(logits, yes_id=3, no_id=7)
    assert p == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-6)
    assert mass == pytest.approx(1.0, abs=1e-6)


def test_yesno_mass_exposes_a_first_token_that_is_neither():
    """If the judge's first token is not yes/no, p_yes is meaningless; mass says so."""
    logits = torch.full((10,), -20.0)
    logits[0], logits[3], logits[7] = 5.0, 0.0, 0.0
    p, mass = local_hf.yes_prob(logits, yes_id=3, no_id=7)
    assert p == pytest.approx(0.5)
    assert mass < 0.05


class _FakeTok:
    unk_token_id = 0

    def __init__(self, vocab):
        self.vocab = vocab

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def encode(self, s, add_special_tokens=False):
        return [self.vocab.get("▁" + s.strip(), self.unk_token_id)]


def test_answer_ids_use_the_sentencepiece_word_tokens():
    assert local_hf.answer_token_ids(_FakeTok({"▁yes": 11, "▁no": 12})) == (11, 12)


def test_answer_ids_refuse_an_unknown_token():
    with pytest.raises(ValueError):
        local_hf.answer_token_ids(_FakeTok({"▁yes": 11}))
