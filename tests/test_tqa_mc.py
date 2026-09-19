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

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
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


def _linear_objective_run(monkeypatch, d=512, r=100.0, steps=150, seed=0):
    """train_one on an objective linear in the steering vector, (v . w) / r, whose optimum
    on the sphere is exactly v = r w. Returns cos(learned u, w)."""
    import dct_steer_utils as su
    w = torch.randn(d, generator=torch.Generator().manual_seed(123))
    w = w / w.norm()
    model = FakeModel(d=d)
    with su.Steerer(model, 11) as st:
        # st.vec IS the trained tensor (Steerer keeps the reference), so the objective
        # carries its gradient; evaluate() calls this under no_grad, which is fine too.
        monkeypatch.setattr(tl, "pair_objective",
                            lambda m, t, pairs, grad: ((st.vec @ w) / r, torch.tensor(0.5)))
        pairs = [{"question": f"q{i}"} for i in range(32)]
        u, _, _ = tl.train_one(model, None, st, pairs, pairs, r, seed=seed, max_steps=steps)
    return float(u @ w.double().numpy())


def test_train_one_finds_a_known_optimum_in_the_real_step_budget(monkeypatch):
    """Adam's per-coordinate steps are sign-like, so a step's length scales with sqrt(d)
    and a radial gradient component is wasted on moving off the sphere. At a realistic d
    and ~150 steps (4 epochs x ceil(600/16)) the optimizer must still land on the optimum."""
    assert _linear_objective_run(monkeypatch) > 0.99


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


@pytest.fixture
def train_stage(monkeypatch, tmp_path):
    """stage_train on fakes: 150 toy pairs (6 train / 144 val), 2 fracs x 2 seeds, and a
    train_one that records which (r, seed) it was asked for."""
    import dct_steer_utils as su
    import xfer_common as xc
    calls = []
    monkeypatch.setattr(mc, "PREFIX", str(tmp_path) + os.sep)
    monkeypatch.setattr(mc, "load_model_left", lambda device: (su, FakeTok(), FakeModel()))
    monkeypatch.setattr(xc, "median_last_norm", lambda model, tok, prompts: 100.0)
    monkeypatch.setattr(xc, "build_directions", lambda *a: ([("a", np.ones(8))], []))
    monkeypatch.setattr(xc, "NORM_FRACS", (0.25, 0.5))
    monkeypatch.setattr(tl, "SEEDS", (0, 1))
    monkeypatch.setattr(tl, "train_pairs", lambda: [
        {"question": f"q{i:03d}", "prompt": f"Q{i}", "true": "a", "false": "z"}
        for i in range(150)])
    monkeypatch.setattr(tl, "evaluate", lambda model, tok, pairs: (-0.7, 0.5))

    def fake_train_one(model, tok, st, tr, va, r, seed, max_steps=None):
        calls.append((r, seed))
        return np.eye(8)[seed], -0.5, 0.6

    monkeypatch.setattr(tl, "train_one", fake_train_one)
    return SimpleNamespace(calls=calls, out=mc.path("mc_learned.npz"))


def test_stage_train_stores_its_settings_and_resumes_only_what_is_missing(train_stage):
    tl.stage_train("cpu")
    z = dict(np.load(train_stage.out))
    assert float(z["lr_per_r"]) == tl.LR_PER_R and int(z["max_epochs"]) == tl.MAX_EPOCHS
    assert int(z["eval_every"]) == tl.EVAL_EVERY and int(z["batch_pairs"]) == tl.BATCH_PAIRS
    assert (int(z["n_train"]), int(z["n_val"])) == (6, 144)
    assert float(z["norm_med"]) == 100.0
    # Drop the last (frac, seed) and resume: only it is trained again.
    rows = [(v, f, s, o, a) for v, f, s, o, a in
            zip(z["vecs"], z["frac"], z["seed"], z["val_obj"], z["val_acc"])][:-1]
    tl._save(train_stage.out, rows, {k: z[k] for k in tl.SETTINGS + ("norm_med",
                                                        "base_val_obj", "base_val_acc")})
    train_stage.calls.clear()
    tl.stage_train("cpu")
    assert train_stage.calls == [(50.0, 1)]
    assert len(tl.load_learned(train_stage.out)) == 4


def test_stage_train_refuses_to_resume_under_a_changed_setting(train_stage, monkeypatch):
    tl.stage_train("cpu")
    train_stage.calls.clear()
    monkeypatch.setattr(tl, "LR_PER_R", tl.LR_PER_R * 2)
    with pytest.raises(SystemExit, match="lr_per_r"):
        tl.stage_train("cpu")
    assert train_stage.calls == []


def test_stage_train_refuses_to_resume_under_a_changed_norm(train_stage, monkeypatch):
    import xfer_common as xc
    tl.stage_train("cpu")
    monkeypatch.setattr(xc, "median_last_norm", lambda model, tok, prompts: 101.0)
    with pytest.raises(SystemExit, match="norm_med"):
        tl.stage_train("cpu")


def test_stage_train_refuses_a_file_from_before_the_settings_were_stored(train_stage):
    np.savez(train_stage.out, vecs=np.eye(8)[:1], frac=np.array([0.25]), seed=np.array([0]),
             val_obj=np.array([-0.5]), val_acc=np.array([0.6]), norm_med=100.0,
             base_val_obj=-0.7, base_val_acc=0.5)
    with pytest.raises(SystemExit, match="(?i)move it aside"):
        tl.stage_train("cpu")
    assert train_stage.calls == []


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


def _cells_frame(cells, nq=10):
    """Baseline plus one block per (name, unit, frac, shift); shift is a scalar or one
    value per question, added to that question's correct answer's mean log-prob, so it
    is exactly the per-question change in margin."""
    rows = []
    for name, u, f, shift in [("baseline", "none", 0.0, 0.0)] + list(cells):
        sh = np.broadcast_to(np.asarray(shift, float), (nq,))
        for q in range(nq):
            for c in (1, 0):
                lp = -2.0 + (sh[q] + 0.01 * q if c else 0.0)
                rows.append({"direction": name, "unit": u, "frac": f, "question": f"q{q}",
                             "correct": c, "mean_lp": lp, "sum_lp": 3 * lp})
    return pd.DataFrame(rows)


def test_summarize_mc_each_clause_of_moves_is_needed():
    """Each row below fails exactly one clause of _moves; dropping that clause would call
    it a mover. At norm 0.5 every random lowers the margin MORE than `neg` does."""
    spiky = [5.0] + [-0.1] * 9                   # mean +0.41: beats every random...
    rands25 = [("rand_0", "norm", 0.25, 0.1), ("rand_1", "norm", 0.25, -0.1),
               ("rand_2", "norm", 0.25, 0.05)]
    rands50 = [("rand_0", "norm", 0.5, -1.0), ("rand_1", "norm", 0.5, -0.8),
               ("rand_2", "norm", 0.5, -0.9)]
    df = _cells_frame(rands25 + rands50 + [
        ("good", "norm", 0.25, 1.0), ("spiky", "norm", 0.25, spiky),
        ("inside", "norm", 0.25, 0.07), ("neg", "norm", 0.5, -0.2)])
    rows = {(r["direction"], r["frac"]): r for r in mc.summarize_mc(df)}
    good, spk = rows[("good", 0.25)], rows[("spiky", 0.25)]
    ins, neg = rows[("inside", 0.25)], rows[("neg", 0.5)]
    assert good["moves"] is True
    # ...but its Wilcoxon p is not significant
    assert spk["p_margin"] == pytest.approx(1 / 4) and spk["wilcoxon_p"] > 0.05
    assert spk["moves"] is False
    # significant per question, but inside the random spread
    assert ins["wilcoxon_p"] <= 0.05 and ins["p_margin"] == pytest.approx(2 / 4)
    assert ins["moves"] is False
    # significant and beats every random at its dose, but the margin went DOWN
    assert neg["wilcoxon_p"] <= 0.05 and neg["p_margin"] == pytest.approx(1 / 4)
    assert neg["e_margin"] == pytest.approx(-0.2)
    assert neg["moves"] is False


def test_add_null_keys_the_randoms_by_unit_and_frac():
    """Randoms at norm 0.5 (all far above `good`) must not enter norm 0.25's null."""
    df = _cells_frame([("rand_0", "norm", 0.25, 0.1), ("rand_1", "norm", 0.25, -0.1),
                       ("rand_2", "norm", 0.25, 0.05), ("rand_0", "norm", 0.5, 9.0),
                       ("rand_1", "norm", 0.5, 9.0), ("good", "norm", 0.25, 1.0)])
    good = {(r["direction"], r["frac"]): r for r in mc.summarize_mc(df)}[("good", 0.25)]
    assert good["n_null"] == 3 and good["p_margin"] == pytest.approx(1 / 4)


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
    assert by[("learned_f0.25_s0", "val_selection")]["n"] == 1     # it chose the checkpoint
    assert by[("learned_f0.25_s0", "val_selection")]["in_sample"] is False
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


# ------------------------------------------------------------------ readings
def _write_summary_inputs(pre, learned_shift, val_obj=-0.5, base_val_obj=-0.69,
                          outcomes=True, mean_diff_shift=1.0, long_hits=12,
                          control_ok=True, oracle_ok=True):
    """Small synthetic J-E outputs under the path prefix `pre`: three randoms at each of
    norm +0.25 and +0.5, a truth direction moving at both (mean_diff_shift), a
    potency-matched control moving at the read dose, and one learned vector at the read
    frac. `long_hits` sets how many of tqa:sup_jtw's 12 cities_long statements flip (12 =
    moves, 0 = baseline-flat = does not move). `control_ok`/`oracle_ok` set round 2's
    validity flags (positive_control_reproduced, oracle_ok)."""
    import json
    rands = [(f"rand_{j}", "norm", f, s) for f in (0.25, 0.5)
             for j, s in enumerate((0.1, -0.1, 0.05))]
    _cells_frame(rands + [
        ("cities:mean_diff", "norm", 0.25, mean_diff_shift),
        ("cities:mean_diff", "norm", 0.5, mean_diff_shift),
        ("tqa:sup_jtw", "norm", 0.25, 0.0), ("cities:dct_ctl_7", "norm", 0.25, 1.0),
        ("learned_f0.25_s0", "norm", 0.25, learned_shift)]).to_csv(
        pre + "mc_tqa_scores.csv", index=False)
    rows = []
    for name, f, hits in (("baseline", 0.0, 0), ("tqa:sup_jtw", 0.25, long_hits),
                          ("rand_0", 0.25, 0), ("rand_1", 0.25, 1)):
        for i in range(12):
            rows.append({"direction": name, "unit": "none" if name == "baseline" else "norm",
                         "frac": f, "stmt": i, "gen_correct": int(i < hits),
                         "words": 5, "incoherent": 0})
    pd.DataFrame(rows).to_csv(pre + "mc_cities_long.csv", index=False)
    np.savez(pre + "mc_learned.npz", vecs=np.eye(4)[:3], frac=np.array([0.25, 0.25, 0.5]),
             seed=np.array([0, 1, 0]), val_obj=np.array([val_obj, -0.6, -0.4]),
             val_acc=np.full(3, 0.6), norm_med=100.0, base_val_obj=base_val_obj,
             base_val_acc=0.5)
    if outcomes:
        for name, oc, flag_key, flag_val in (
                ("cities", {"cities:mean_diff": "c", "tqa:sup_jtw": "gen-only"},
                 "oracle_ok", oracle_ok),
                ("truthfulqa", {"cities:mean_diff": "none", "tqa:sup_jtw": "gain",
                                "cities:dct_ctl_7": "gain"},
                 "positive_control_reproduced", control_ok)):
            with open(pre + f"xfer_{name}_outcomes.json", "w") as f:
                json.dump({"outcomes": oc, flag_key: flag_val}, f)


def _summary_lines(monkeypatch, tmp_path, capsys, **kw):
    pre = str(tmp_path) + os.sep
    monkeypatch.setattr(mc, "PREFIX", pre)
    _write_summary_inputs(pre, **kw)
    mc.stage_summary()
    return capsys.readouterr().out.splitlines()


def _line(lines, start):
    got = [ln for ln in lines if ln.startswith(start)]
    assert len(got) == 1, (start, lines)
    return got[0]


def test_readings_content_lists_only_read_dose_truth_directions(monkeypatch, tmp_path,
                                                                 capsys):
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0)
    content = _line(lines, "[reading] CONTENT")
    assert "cities:mean_diff norm +0.25" in content
    assert "+0.5" not in content and "dct_ctl" not in content and "learned" not in content
    assert "cities:dct_ctl_7" in _line(lines, "[reading] controls")
    sec = _line(lines, "[secondary]")
    assert "not corrected for multiplicity" in sec and "cities:mean_diff norm +0.5" in sec
    assert "tqa:sup_jtw" in _line(lines, "[reading] cities LONG FORM")
    assert any("e > 0" in ln for ln in lines)                    # F6, said once
    assert sum("e > 0" in ln for ln in lines) == 1


def test_readings_method_limit_fires_when_a_trained_ceiling_stays_in_the_null(
        monkeypatch, tmp_path, capsys):
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0)
    assert any(ln.startswith("[reading] METHOD LIMIT") for ln in lines)
    assert not any("TRAINING FAILED" in ln for ln in lines)


def test_readings_no_method_limit_when_the_ceiling_moves(monkeypatch, tmp_path, capsys):
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=1.0)
    assert not any(ln.startswith("[reading] METHOD LIMIT") for ln in lines)


def test_readings_a_failed_training_run_is_flagged_and_not_read_as_a_limit(
        monkeypatch, tmp_path, capsys):
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0,
                           val_obj=-0.7, base_val_obj=-0.69)
    assert "learned_f0.25_s0" in _line(lines, "[reading] TRAINING FAILED")
    assert not any(ln.startswith("[reading] METHOD LIMIT") for ln in lines)


def test_readings_identifiability_is_per_frac_with_each_seeds_objective(
        monkeypatch, tmp_path, capsys):
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0)
    ident = [ln for ln in lines if ln.startswith("[reading] IDENTIFIABILITY")]
    assert len(ident) == 2                                      # fracs 0.25 and 0.5
    f25 = [ln for ln in ident if "norm 0.25" in ln][0]
    assert "median 0.000" in f25 and "s0 -0.5000" in f25 and "s1 -0.6000" in f25


def test_readings_form_vs_content_per_truth_direction(monkeypatch, tmp_path, capsys):
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0)
    form = [ln for ln in lines if ln.startswith("[reading] FORM vs CONTENT ")
            and "verdict" not in ln]
    assert {ln.split()[4].rstrip(":") for ln in form} == {"cities:mean_diff",
                                                          "tqa:sup_jtw"}
    jtw = [ln for ln in form if "tqa:sup_jtw" in ln][0]
    assert "J-D1 gen-only" in jtw and "J-D2 gain" in jtw
    # cities:mean_diff moves mc at the read dose, so the verdict is CONTENT
    assert _line(lines, "[reading] FORM vs CONTENT verdict").endswith("CONTENT")


def test_readings_form_not_content_when_only_long_form_moves(monkeypatch, tmp_path,
                                                             capsys):
    # cities:mean_diff is scored but does not move mc; tqa:sup_jtw's cities_long
    # gain (long_hits=12, the default) is the only mover, so the verdict is FORM NOT
    # CONTENT, not CONTENT or INCOMPLETE.
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0,
                           mean_diff_shift=0.0)
    assert _line(lines, "[reading] FORM vs CONTENT verdict") == \
        "[reading] FORM vs CONTENT verdict: FORM NOT CONTENT"


def test_readings_form_incomplete_when_mc_scores_file_is_absent(monkeypatch, tmp_path,
                                                                 capsys):
    """R-A: mc_tqa_scores.csv absent (J-D2 still says 'gain', control reproduced) must
    read INCOMPLETE, never FORM NOT CONTENT."""
    pre = str(tmp_path) + os.sep
    monkeypatch.setattr(mc, "PREFIX", pre)
    _write_summary_inputs(pre, learned_shift=0.0)
    os.remove(pre + "mc_tqa_scores.csv")
    mc.stage_summary()
    lines = capsys.readouterr().out.splitlines()
    verdict = _line(lines, "[reading] FORM vs CONTENT verdict")
    assert "INCOMPLETE" in verdict and "FORM NOT CONTENT" not in verdict
    assert "cities:mean_diff" in verdict and "tqa:sup_jtw" in verdict


def test_readings_form_incomplete_when_one_truth_direction_is_unscored_in_mc(
        monkeypatch, tmp_path, capsys):
    """R-A: one truth direction scored and not moving (tqa:sup_jtw), one entirely
    unscored (cities:mean_diff dropped from the mc CSV) must read INCOMPLETE."""
    pre = str(tmp_path) + os.sep
    monkeypatch.setattr(mc, "PREFIX", pre)
    _write_summary_inputs(pre, learned_shift=0.0, mean_diff_shift=0.0)
    df = pd.read_csv(pre + "mc_tqa_scores.csv")
    df = df[df.direction != "cities:mean_diff"]
    df.to_csv(pre + "mc_tqa_scores.csv", index=False)
    mc.stage_summary()
    lines = capsys.readouterr().out.splitlines()
    verdict = _line(lines, "[reading] FORM vs CONTENT verdict")
    assert "INCOMPLETE" in verdict and "FORM NOT CONTENT" not in verdict
    assert "cities:mean_diff" in verdict and "tqa:sup_jtw" not in verdict


def test_readings_form_content_when_a_scored_mover_outweighs_an_unscored_direction(
        monkeypatch, tmp_path, capsys):
    """R-A: a scored mover (cities:mean_diff) is valid evidence even while another truth
    direction (tqa:sup_jtw) is unscored; the verdict is CONTENT, not INCOMPLETE."""
    pre = str(tmp_path) + os.sep
    monkeypatch.setattr(mc, "PREFIX", pre)
    _write_summary_inputs(pre, learned_shift=0.0)                # mean_diff_shift=1.0
    df = pd.read_csv(pre + "mc_tqa_scores.csv")
    df = df[df.direction != "tqa:sup_jtw"]
    df.to_csv(pre + "mc_tqa_scores.csv", index=False)
    mc.stage_summary()
    lines = capsys.readouterr().out.splitlines()
    assert _line(lines, "[reading] FORM vs CONTENT verdict") == \
        "[reading] FORM vs CONTENT verdict: CONTENT"


def test_readings_form_neither_when_positive_control_not_reproduced(monkeypatch,
                                                                     tmp_path, capsys):
    """R-B: with the J-D2 positive control not reproduced, its 'gain' outcome must not
    count toward long_moved; with mc scored-and-flat and no cities_long mover the
    verdict is NEITHER, and the unusable line prints once."""
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0,
                           mean_diff_shift=0.0, long_hits=0, control_ok=False)
    assert _line(lines, "[reading] FORM vs CONTENT verdict") == \
        "[reading] FORM vs CONTENT verdict: NEITHER"
    unusable = [ln for ln in lines if "positive control NOT reproduced" in ln]
    assert unusable == ["[reading] FORM vs CONTENT: (J-D2 positive control NOT "
                        "reproduced: unusable)"]


def test_readings_form_flags_an_unreadable_oracle(monkeypatch, tmp_path, capsys):
    """R-B: oracle_ok False must print the UNREADABLE line; J-D1 never feeds the
    verdict either way."""
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0,
                           oracle_ok=False)
    assert "[reading] FORM vs CONTENT: J-D1 oracle did not flip: its outcomes are " \
        "UNREADABLE" in lines


def test_readings_form_needs_the_round2_outcomes(monkeypatch, tmp_path, capsys):
    lines = _summary_lines(monkeypatch, tmp_path, capsys, learned_shift=0.0,
                           outcomes=False)
    form = _line(lines, "[reading] FORM vs CONTENT")
    assert "xfer_cities_outcomes.json" in form and "xfer_truthfulqa_outcomes.json" in form


def test_summary_with_no_inputs_prints_every_reading_and_writes_nothing(
        monkeypatch, tmp_path, capsys):
    pre = str(tmp_path) + os.sep
    monkeypatch.setattr(mc, "PREFIX", pre)
    mc.stage_summary()
    out = capsys.readouterr().out
    for key in ("CONTENT", "LONG FORM", "CEILING", "FORM vs CONTENT"):
        assert key in out
    assert "[reading] METHOD LIMIT" not in out
    assert list(tmp_path.iterdir()) == []
