"""Tests for src/tqa_mc.py and src/tqa_learned.py (J-E)."""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import tqa_mc as mc  # noqa: E402


class FakeTok:
    """Character tokenizer: ids 1..50 for characters, 51 is BOS, 0 is pad."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) % 50 + 1 for c in text]
        return {"input_ids": ([51] if add_special_tokens else []) + ids}


class FakeModel(torch.nn.Module):
    """Twelve residual layers, no attention: each position depends only on its own
    token, so batching and padding must not change any score."""

    def __init__(self, d=8, vocab=60, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(hidden_size=d)
        self.emb = torch.nn.Embedding(vocab, d)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(torch.nn.Linear(d, d) for _ in range(12))
        self.head = torch.nn.Linear(d, vocab)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        h = self.emb(input_ids)
        for layer in self.model.layers:
            h = h + torch.tanh(layer(h))
        return SimpleNamespace(logits=self.head(h))


# ------------------------------------------------------------------ core
def test_parse_answers_reads_the_holdout_list_format_and_drops_blanks():
    assert mc.parse_answers('["Yes, it is", "", " No "]') == ["Yes, it is", "No"]


def test_encode_pairs_left_pads_and_masks_only_the_answer():
    tok = FakeTok()
    ids, att, am = mc.encode_pairs(tok, ["ab", "abcd"], ["x", "y"])
    # row 0: BOS a b ' ' x = 5 tokens, row 1: BOS a b c d ' ' y = 7 tokens
    assert ids.shape == (2, 7)
    assert att[0].tolist() == [0, 0, 1, 1, 1, 1, 1]
    assert am[0].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert am[1].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert ids[1, :5].tolist() == tok("abcd")["input_ids"]      # prompt ids untouched


def test_position_ids_restart_at_the_first_real_token():
    att = torch.tensor([[0, 0, 1, 1, 1]])
    assert mc.position_ids(att).tolist() == [[0, 0, 0, 1, 2]]


def test_token_logprobs_scores_token_t_with_the_logits_at_t_minus_1():
    logits = torch.zeros(1, 3, 2)
    logits[0, 1, 1] = math.log(3.0)          # position 1 predicts token 1 with p = 3/4
    ids = torch.tensor([[0, 0, 1]])
    am = torch.tensor([[0, 1, 1]])
    s, m, n = mc.token_logprobs(logits, ids, am)
    want = math.log(0.5) + math.log(0.75)
    assert n.item() == 2
    assert s.item() == pytest.approx(want, abs=1e-6)
    assert m.item() == pytest.approx(want / 2, abs=1e-6)


def test_answer_logprob_is_the_same_batched_and_one_at_a_time():
    tok, model = FakeTok(), FakeModel()
    prompts, answers = ["short", "a much longer prompt"], ["yes", "no it is not"]
    s_b, m_b, n_b = mc.answer_logprob(model, tok, prompts, answers, batch=2)
    for i in range(2):
        s1, m1, n1 = mc.answer_logprob(model, tok, prompts[i:i + 1], answers[i:i + 1])
        assert s1.item() == pytest.approx(s_b[i].item(), abs=1e-5)
        assert n1.item() == n_b[i].item()


def test_mc_metrics_hand_computed():
    mean = np.array([-1.0, -3.0, -2.0, -4.0])
    summ = np.array([-2.0, -3.0, -6.0, -4.0])
    corr = np.array([True, True, False, False])
    r = mc.mc_metrics(mean, summ, corr)
    assert r["margin"] == pytest.approx(-2.0 - -3.0)
    assert r["mc1"] == 1                                  # argmax mean is answer 0
    w = np.exp(summ)
    assert r["mc2"] == pytest.approx(w[:2].sum() / w.sum())


def test_holdout_items_cover_every_reference_answer():
    items = mc.holdout_items(limit=2)
    h = pd.read_csv("got_datasets/truthfulqa_holdout.csv").head(2)
    n = sum(len(mc.parse_answers(h[c].iloc[i])) for i in range(2)
            for c in ("correct_answers", "incorrect_answers"))
    assert len(items) == n
    assert items[0]["prompt"].startswith("Q: ") and items[0]["prompt"].endswith("\nA:")
    assert {it["correct"] for it in items} == {0, 1}


# ------------------------------------------------------------------ learned
import tqa_learned as tl  # noqa: E402


def test_split_questions_is_disjoint_sized_and_deterministic():
    qs = [f"q{i}" for i in range(20)]
    tr, va = tl.split_questions(qs + qs[:3], n_val=5, seed=1)     # duplicates collapse
    assert len(tr) == 15 and len(va) == 5
    assert not set(tr) & set(va)
    assert (tr, va) == tl.split_questions(qs, n_val=5, seed=1)


def test_train_pairs_have_one_true_and_one_false_answer_per_question():
    pairs = tl.train_pairs()
    assert len(pairs) == 744
    assert len({p["question"] for p in pairs}) == 744
    assert all(p["true"] and p["false"] and p["true"] != p["false"] for p in pairs)


def test_train_questions_never_appear_in_the_holdout():
    hold = set(pd.read_csv("got_datasets/truthfulqa_holdout.csv")["question"]
               .astype(str).str.strip())
    assert not hold & {p["question"] for p in tl.train_pairs()}


def test_project_puts_the_vector_on_the_sphere():
    v = torch.tensor([3.0, 4.0])
    assert tl.project(v, 2.0).norm().item() == pytest.approx(2.0)
    assert torch.allclose(tl.project(v, 2.0), torch.tensor([1.2, 1.6]))


def test_train_one_beats_the_unsteered_objective_on_a_toy(monkeypatch):
    import dct_steer_utils as su
    monkeypatch.setattr(tl, "LR_PER_R", 0.3)
    monkeypatch.setattr(tl, "EVAL_EVERY", 5)
    tok, model = FakeTok(), FakeModel(seed=3)
    pairs = [{"question": f"q{i}", "prompt": f"Q{i}", "true": "aaa", "false": "zzz"}
             for i in range(8)]
    with su.Steerer(model, 11) as st:
        base, _ = tl.evaluate(model, tok, pairs)
        st.set(tl.project(torch.randn(8, generator=torch.Generator().manual_seed(0)), 2.0))
        start, _ = tl.evaluate(model, tok, pairs)
        st.set(None)
        u, obj, acc = tl.train_one(model, tok, st, pairs, pairs, 2.0, seed=0,
                                   max_steps=40)
        assert st.vec is None                     # training leaves no steering behind
    assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-5)
    assert obj > base
    assert obj > start + 1e-3                      # training must beat its own start, not
                                                    # just an already-lucky random start


def test_load_learned_names_each_vector_by_frac_and_seed(tmp_path):
    p = tmp_path / "mc_learned.npz"
    np.savez(p, vecs=np.eye(2), frac=np.array([0.25, 0.5]), seed=np.array([0, 2]),
             val_obj=np.zeros(2), val_acc=np.zeros(2), norm_med=100.0,
             base_val_obj=0.0, base_val_acc=0.0)
    got = tl.load_learned(str(p))
    assert [(n, f) for n, _, f in got] == [("learned_f0.25_s0", 0.25),
                                           ("learned_f0.5_s2", 0.5)]


def test_save_then_load_learned_round_trips_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "mc_learned.npz"
    rows = [(np.array([1.0, 0.0]), 0.25, 0, -0.5, 0.6),
            (np.array([0.0, 1.0]), 0.5, 1, -0.4, 0.7)]
    tl._save(str(p), rows, {"norm_med": 100.0, "base_val_obj": 0.0, "base_val_acc": 0.0})
    got = tl.load_learned(str(p))
    assert [(n, f) for n, _, f in got] == [("learned_f0.25_s0", 0.25),
                                           ("learned_f0.5_s1", 0.5)]
    assert list(p.parent.iterdir()) == [p]         # no leftover .tmp.npz
