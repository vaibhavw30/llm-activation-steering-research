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


# ------------------------------------------------------------------ stages
def test_blocks_put_baseline_first_and_learned_only_at_their_own_frac():
    grid = [("norm", 0.25, 25.0), ("norm", -0.25, -25.0)]
    dirs = [("a", np.ones(2))]
    learned = [("learned_f0.5_s0", np.ones(2), 0.5)]
    bl = mc.blocks(dirs, grid, learned, norm_med=100.0)
    assert bl[0][:4] == ("baseline", "none", 0.0, 0.0) and bl[0][4] is None
    assert [b[:4] for b in bl[1:]] == [("a", "norm", 0.25, 25.0), ("a", "norm", -0.25, -25.0),
                                      ("learned_f0.5_s0", "norm", 0.5, 50.0)]


def test_smoke_dirs_keeps_three_real_and_two_random():
    dirs = [(n, None) for n in ("x", "y", "z", "w", "rand_0", "rand_1", "rand_2")]
    assert [d[0] for d in mc.smoke_dirs(dirs)] == ["x", "y", "z", "rand_0", "rand_1"]


def test_score_blocks_writes_one_row_per_item_and_resumes(tmp_path):
    import dct_steer_utils as su
    tok, model = FakeTok(), FakeModel()
    items = [{"question": "q", "prompt": "Q: q\nA:", "answer": a, "correct": c}
             for a, c in (("yes", 1), ("no", 0))]
    bl = [("baseline", "none", 0.0, 0.0, None), ("a", "norm", 0.5, 1.0, np.ones(8))]
    out = str(tmp_path / "s.csv")
    with su.Steerer(model, 11) as st:
        mc.score_blocks(model, tok, st, bl, items, out)
        mc.score_blocks(model, tok, st, bl, items, out)          # resume: nothing added
    df = pd.read_csv(out)
    assert len(df) == 4
    assert set(df.columns) >= {"direction", "unit", "frac", "scale", "question",
                               "answer", "correct", "sum_lp", "mean_lp", "n_tok"}
    base = df[df.direction == "baseline"].sum_lp.values
    steer = df[df.direction == "a"].sum_lp.values
    assert not np.allclose(base, steer)                        # the vector was applied


def test_check_padding_uses_first_answer_of_first_four_distinct_questions(monkeypatch):
    """items[:4] would be four answers to ONE question; the check must instead span
    prompts of different lengths, which is the case left-padding can actually break."""
    calls = []

    def spy(model, tok, prompts, answers, grad=False, batch=mc.BATCH):
        calls.append(list(prompts))
        z = torch.zeros(len(prompts))
        return z, z, z

    monkeypatch.setattr(mc, "answer_logprob", spy)
    items = [{"question": f"q{q}", "prompt": f"P{q}", "answer": f"a{q}_{k}"}
             for q in range(5) for k in range(3)]
    mc.check_padding(None, None, items)
    batched = max(calls, key=len)
    assert len(batched) == 4
    assert len(set(batched)) == 4


def test_score_blocks_clears_steering_before_a_baseline_that_runs_second(tmp_path):
    """Blocks ordered steered-then-baseline: a missing `st.set(None)` before scoring the
    baseline block would leak the steering vector's effect into its sum_lp."""
    import dct_steer_utils as su
    tok, model = FakeTok(), FakeModel()
    items = [{"question": "q", "prompt": "Q: q\nA:", "answer": a, "correct": c}
             for a, c in (("yes", 1), ("no", 0))]
    bl = [("a", "norm", 0.5, 1.0, np.ones(8)), ("baseline", "none", 0.0, 0.0, None)]
    out = str(tmp_path / "s.csv")
    with su.Steerer(model, 11) as st:
        mc.score_blocks(model, tok, st, bl, items, out)
        st.set(None)
        want = mc.answer_logprob(model, tok, [i["prompt"] for i in items],
                                 [i["answer"] for i in items])[0]
    df = pd.read_csv(out)
    got = df[df.direction == "baseline"].sum_lp.values
    assert np.allclose(got, want.numpy(), atol=1e-5)


def test_main_rejects_limit_without_prefix():
    with pytest.raises(SystemExit):
        mc.main(["--stage", "mc", "--limit", "4"])


def test_running_as_a_script_sets_the_prefix_the_importers_see(monkeypatch):
    import runpy
    seen = []
    monkeypatch.setattr(mc, "PREFIX", "")
    monkeypatch.setattr(mc, "stage_judge", lambda device: seen.append(mc.PREFIX))
    monkeypatch.setattr(sys, "argv", ["tqa_mc.py", "--stage", "judge", "--prefix", "zz_"])
    runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "src", "tqa_mc.py"),
                   run_name="__main__")
    assert seen == ["zz_"]


# ------------------------------------------------------------------ summary
def _mc_frame(shift):
    """10 questions, one correct and one incorrect answer each; each direction adds
    shift[name] to every correct answer's mean log-prob."""
    rows = []
    for name, (u, f) in (("baseline", ("none", 0.0)), ("good", ("norm", 0.25)),
                         ("flat", ("norm", 0.25)), ("rand_0", ("norm", 0.25)),
                         ("rand_1", ("norm", 0.25)), ("rand_2", ("norm", 0.25))):
        for q in range(10):
            jitter = 0.01 * q
            for c in (1, 0):
                lp = -2.0 - c * 0.0 + (shift.get(name, 0.0) + jitter if c else 0.0)
                rows.append({"direction": name, "unit": u, "frac": f, "question": f"q{q}",
                             "correct": c, "mean_lp": lp, "sum_lp": 3 * lp})
    return pd.DataFrame(rows)


def test_summarize_mc_moves_only_the_direction_that_beats_null_and_baseline():
    df = _mc_frame({"good": 1.0, "rand_0": 0.1, "rand_1": -0.1, "rand_2": 0.05})
    rows = {r["direction"]: r for r in mc.summarize_mc(df)}
    assert rows["good"]["e_margin"] == pytest.approx(1.0)
    assert rows["good"]["n_null"] == 3
    assert rows["good"]["p_margin"] == pytest.approx(1 / 4)     # beats all three
    assert rows["good"]["moves"] is True
    assert rows["flat"]["moves"] is False
    assert rows["rand_0"]["moves"] is False and rows["rand_0"]["p_margin"] == ""


def test_summarize_mc_a_negative_dose_moves_only_toward_what_it_pushes():
    df = _mc_frame({"good": -1.0})
    df.loc[df.direction == "good", "frac"] = -0.25
    rows = {r["direction"]: r for r in mc.summarize_mc(df)}
    assert rows["good"]["e_margin"] == pytest.approx(1.0)       # sign(frac) x change


def test_summarize_long_uses_mcnemar_on_gen_correct():
    rows = []
    for name, f, hits in (("baseline", 0.0, 0), ("up", 0.25, 12), ("rand_0", 0.25, 0)):
        for i in range(12):
            rows.append({"direction": name, "unit": "none" if name == "baseline" else "norm",
                         "frac": f, "stmt": i, "gen_correct": int(i < hits),
                         "words": 5, "incoherent": 0})
    got = {r["direction"]: r for r in mc.summarize_long(pd.DataFrame(rows))}
    assert got["up"]["gained"] == 12 and got["up"]["mcnemar_p"] < 0.001
    assert got["up"]["e_gen"] == pytest.approx(1.0)
    assert got["up"]["moves"] is True


def test_summarize_train_flags_in_sample_and_splits_learned_by_validation():
    rows = []
    for name in ("baseline", "tqa:sup_jtw", "cities:mean_diff", "learned_f0.25_s0"):
        for q in ("a", "b"):
            for c in (1, 0):
                rows.append({"direction": name, "unit": "norm", "frac": 0.25,
                             "question": q, "correct": c,
                             "mean_lp": -1.0 if (c and name != "baseline") else -2.0})
    got = mc.summarize_train(pd.DataFrame(rows), val_qs=["b"])
    by = {(r["direction"], r["subset"]): r for r in got}
    assert by[("tqa:sup_jtw", "all")]["in_sample"] is True
    assert by[("cities:mean_diff", "all")]["in_sample"] is False
    assert by[("learned_f0.25_s0", "val")]["n"] == 1
    assert by[("learned_f0.25_s0", "val")]["in_sample"] is False
    assert by[("learned_f0.25_s0", "train")]["in_sample"] is True


def test_summarize_judged_rates_against_the_unsteered_answers():
    rows = [{"direction": d, "question": f"q{i}", "truthful": t,
             "truthful_and_informative": t}
            for d, ts in (("baseline", [0, 0, 1, 1]), ("learned_f0.25_s0", [1, 1, 1, 1]))
            for i, t in enumerate(ts)]
    got = {r["direction"]: r for r in mc.summarize_judged(pd.DataFrame(rows))}
    r = got["learned_f0.25_s0"]
    assert r["truthful"] == 1.0 and r["base_truthful"] == 0.5
    assert r["gained_truthful"] == 2 and r["lost_truthful"] == 0
