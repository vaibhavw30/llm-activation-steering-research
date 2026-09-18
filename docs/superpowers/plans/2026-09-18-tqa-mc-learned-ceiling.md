# Judge-free TruthfulQA and the Learned Ceiling (J-E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One cluster job (J-E) that scores TruthfulQA by answer log-probability under steering (no judge), completes the dataset x format 2x2 with a long-form cities arm, and trains the best single layer-11 steering vector as a ceiling.

**Architecture:** `src/tqa_mc.py` holds the scoring core (`answer_logprob`), the MC metrics, the scoring/generation stages, the summaries and `main`. `src/tqa_learned.py` holds the training data split, sphere projection and training loop (stage `train`). Both reuse `xfer_common` (directions, dose grid, statistics) and `xfer_cities` / `xfer_tqa` helpers unchanged. `deltaai/run_tqa_mc.slurm` runs a smoke then the full job.

**Tech Stack:** Python 3.13, torch, transformers (gemma-2-2b fp32 via `dct_steer_utils.load_model`), numpy, pandas, scipy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md`

## Global Constraints

- Steering layer 11 via `dct_steer_utils.Steerer` (adds at every position, as generation does); vectors built with `xfer_common.steer_vec`.
- Prompt format `prep_truthfulqa.prompt_of(question)` = `"Q: {q}\nA:"`; answer text is `" " + answer`.
- Primary MC metric: per-token MEAN log-prob of correct answers minus incorrect (`margin`). `mc1` uses mean log-prob; `mc2` uses summed log-prob.
- MC stage nulls: `N_RAND, RAND_SEED = 32, 41`. Cities long-form nulls: `8, 29`.
- Training: split seed 37, 144 validation questions, fracs `xc.NORM_FRACS` = (0.125, 0.25, 0.5), seeds (0, 1, 2), Adam lr `1e-2 * r`, 16 pairs per batch, at most 4 epochs, validation every 10 steps, keep the best step.
- Learned vectors are steered only at their own frac, positive sign, unit `norm`, scale = frac x the TARGET's median ||h_11||.
- Cities long form: prompt `"Q: Where is the city of {city}?\nA:"`, 48 new tokens, repetition penalty 1.0, doses norm +/-`xc.READ_FRAC` (0.25) only.
- "Moves" = paired test p <= 0.05 (Wilcoxon on margin; exact McNemar on 0/1) AND `xc.beyond_null(p_perm, n_null)` AND effect > 0, where effect = sign(frac) x change vs baseline.
- Nothing is overwritten except derived summary CSVs; every output file is named `mc_*` (with `--prefix smoke_` for the smoke). Every stage resumes.
- Commands are labelled LAPTOP or CLUSTER. Laptop tests: `./.venv/bin/python -m pytest`. Do not rsync to the cluster while jobs 3169838 / 3169899 are pending.
- Commit only the files each task names.

---

### Task 1: The scoring core and MC metrics

**Files:**
- Create: `src/tqa_mc.py`
- Create: `tests/test_tqa_mc.py`
- Modify: `docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md` (the "Shared core" bullet on the seam)

**Interfaces:**
- Produces:
  - `tqa_mc.parse_answers(cell: str) -> list[str]`
  - `tqa_mc.encode_pairs(tok, prompts: list[str], answers: list[str]) -> (ids LongTensor (B,T), att LongTensor (B,T), am LongTensor (B,T))`, left padded
  - `tqa_mc.position_ids(att) -> LongTensor`
  - `tqa_mc.token_logprobs(logits, ids, am) -> (sum (B,), mean (B,), n (B,))` torch tensors
  - `tqa_mc.answer_logprob(model, tok, prompts, answers, grad=False, batch=BATCH) -> (sum, mean, n)` torch tensors on the model's device
  - `tqa_mc.mc_metrics(mean_lp, sum_lp, correct) -> {"margin": float, "mc1": int, "mc2": float}`
  - `tqa_mc.holdout_items(limit=0) -> list[dict(question, prompt, answer, correct)]`
  - constants `DS, N_RAND, RAND_SEED, N_RAND_LONG, RAND_SEED_LONG, BATCH, LONG_PROMPT, LONG_MAX_NEW, LONG_REP, PAD_TOL, PREFIX`, and `path(name)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tqa_mc.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run (LAPTOP): `./.venv/bin/python -m pytest tests/test_tqa_mc.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'tqa_mc'`.

- [ ] **Step 3: Write the core**

Create `src/tqa_mc.py`:

```python
"""tqa_mc.py: judge-free TruthfulQA and the dataset x format grid (J-E, round 3).

    PYTHONPATH=src python src/tqa_mc.py --stage all --device cuda                  # CLUSTER
    PYTHONPATH=src python src/tqa_mc.py --stage all --device cuda \\
        --limit 4 --steps 1 --prefix smoke_                                       # its smoke
    PYTHONPATH=src python src/tqa_mc.py --stage summary                            # LAPTOP

Spec: docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md.

Every TruthfulQA number before this went through a judge, and the dataset card's
registered hypothesis is that TQA steers only because the judged score rewards form. Here
the score is the model's own log-probability of TruthfulQA's reference answers, so no
judge and no generated length is involved:

  margin  (PRIMARY) mean over correct answers of the per-token mean log-prob, minus the
          same over incorrect ones. Per token, so a direction cannot win by preferring
          shorter answers.
  mc1     1 if the answer with the highest per-token mean log-prob is a correct one.
  mc2     summed-log-prob probability mass on the correct answers (the standard MC2).

|            | short answer              | long answer               |
|------------|---------------------------|---------------------------|
| cities     | J-D1                      | stage cities_long (here)  |
| TruthfulQA | stage mc (here)           | J-D2                      |

The learned vectors (tqa_learned.py, stage train) are scored in every stage: they are
the ceiling every other direction is a fraction of.
"""
import argparse
import ast
import json
import os

import numpy as np

import tqa_discovery as td
import xfer_common as xc

DS = "truthfulqa"
N_RAND, RAND_SEED = 32, 41             # the MC null: a permutation p can reach 1/33
N_RAND_LONG, RAND_SEED_LONG = 8, 29    # the cities long-form null, J-D1's randoms
BATCH = 32
LONG_PROMPT = "Q: Where is the city of {city}?\nA:"
LONG_MAX_NEW, LONG_REP = 48, 1.0
PAD_TOL = 1e-3                         # batched vs single summed log-prob, fp32
PREFIX = ""


def path(name):
    return PREFIX + name


# ------------------------------------------------------------------ pure
def parse_answers(cell):
    """The holdout's answer lists are Python list literals."""
    return [str(a).strip() for a in ast.literal_eval(str(cell)) if str(a).strip()]


def encode_pairs(tok, prompts, answers):
    """Left-padded (ids, attention mask, answer mask).

    The prompt and the answer are tokenized SEPARATELY and concatenated. The prompt's
    ids are then exactly the ones generation starts from, and the answer's are the ones
    the model would have to emit after them; tokenizing the joined string could merge
    tokens across the seam and score a sequence generation never produces."""
    import torch
    seqs, masks = [], []
    for p, a in zip(prompts, answers):
        pi = list(tok(p)["input_ids"])
        ai = list(tok(" " + a, add_special_tokens=False)["input_ids"])
        seqs.append(pi + ai)
        masks.append([0] * len(pi) + [1] * len(ai))
    T, pad = max(map(len, seqs)), tok.pad_token_id
    ids = torch.tensor([[pad] * (T - len(s)) + s for s in seqs])
    att = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs])
    am = torch.tensor([[0] * (T - len(m)) + m for m in masks])
    return ids, att, am


def position_ids(att):
    """Positions counted from the first real token. A plain forward with left padding
    otherwise numbers the pads, shifting every real token's position."""
    return (att.cumsum(-1) - 1).clamp(min=0)


def token_logprobs(logits, ids, am):
    """(sum, mean, n) of log p(ids[t] | ids[<t]) over positions with am == 1.
    logits[:, t - 1] is the prediction for ids[:, t]."""
    import torch
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = am[:, 1:].to(tok_lp.dtype)
    s = (tok_lp * m).sum(-1)
    n = m.sum(-1)
    return s, s / n.clamp(min=1), n


def answer_logprob(model, tok, prompts, answers, grad=False, batch=BATCH):
    """Per-row (sum, mean, n) of the answer's log-prob given the prompt, under whatever
    steering the caller's Steerer holds. grad=True keeps the graph (training)."""
    import torch
    dev = next(model.parameters()).device
    outs = []
    for b0 in range(0, len(prompts), batch):
        ids, att, am = (t.to(dev) for t in encode_pairs(
            tok, prompts[b0:b0 + batch], answers[b0:b0 + batch]))
        with torch.set_grad_enabled(grad):
            logits = model(input_ids=ids, attention_mask=att,
                           position_ids=position_ids(att)).logits
            outs.append(token_logprobs(logits, ids, am))
    return tuple(torch.cat([o[i] for o in outs]) for i in range(3))


def mc_metrics(mean_lp, sum_lp, correct):
    mean_lp, sum_lp = np.asarray(mean_lp, float), np.asarray(sum_lp, float)
    c = np.asarray(correct).astype(bool)
    w = np.exp(sum_lp - sum_lp.max())
    return {"margin": float(mean_lp[c].mean() - mean_lp[~c].mean()),
            "mc1": int(c[int(np.argmax(mean_lp))]),
            "mc2": float(w[c].sum() / w.sum())}


def holdout_items(limit=0):
    """One item per (holdout question, reference answer), correct = 1 or 0."""
    import pandas as pd

    from prep_truthfulqa import prompt_of
    h = pd.read_csv(td.HOLDOUT)
    if limit:
        h = h.head(limit)
    out = []
    for _, r in h.iterrows():
        q = str(r["question"]).strip()
        for kind, col in ((1, "correct_answers"), (0, "incorrect_answers")):
            for a in parse_answers(r[col]):
                out.append({"question": q, "prompt": prompt_of(q), "answer": a,
                            "correct": kind})
    return out
```

- [ ] **Step 4: Run the tests**

Run (LAPTOP): `./.venv/bin/python -m pytest tests/test_tqa_mc.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Record the seam decision in the spec**

In the spec, replace the bullet beginning "Text is `prompt_of(question) + " " + answer`" with:

```markdown
- The prompt is `prompt_of(question)` and the answer `" " + answer`, tokenized SEPARATELY
  and concatenated, so the prompt's ids are exactly generation's and the answer's are what
  the model would emit after them (no merge across the seam). Stage `mc` also checks once
  that batched and one-at-a-time scores agree within 1e-3.
```

- [ ] **Step 6: Commit**

```bash
git add src/tqa_mc.py tests/test_tqa_mc.py docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md
git commit -m "feat(J-E): answer log-prob under steering and the TruthfulQA MC metrics"
```

---

### Task 2: The learned ceiling (training)

**Files:**
- Create: `src/tqa_learned.py`
- Modify: `tests/test_tqa_mc.py` (append)

**Interfaces:**
- Consumes: `tqa_mc.answer_logprob`, `tqa_mc.path`, `tqa_mc.load_model_left` (defined in Task 3; `stage_train` is only run after Task 3 exists, the tested functions here do not need it).
- Produces:
  - `tqa_learned.train_pairs() -> list[dict(question, prompt, true, false)]`
  - `tqa_learned.split_questions(qs, n_val=N_VAL, seed=SPLIT_SEED) -> (train_qs list, val_qs list)`
  - `tqa_learned.project(v: Tensor, r: float) -> Tensor`
  - `tqa_learned.pair_objective(model, tok, pairs, grad) -> (objective Tensor scalar, accuracy Tensor scalar)`
  - `tqa_learned.evaluate(model, tok, pairs) -> (objective float, accuracy float)`
  - `tqa_learned.train_one(model, tok, st, tr, va, r, seed, max_steps=None) -> (unit vector np.ndarray (d,), best val objective float, its val accuracy float)`
  - `tqa_learned.stage_train(device, limit=0, steps=None, jb_prefix="")` writing `mc_learned.npz` with arrays `vecs (n,d)`, `frac (n,)`, `seed (n,)`, `val_obj (n,)`, `val_acc (n,)`, scalars `norm_med`, `base_val_obj`, `base_val_acc`; and `mc_learned_cos.csv`
  - `tqa_learned.load_learned(p) -> list[(name, unit vec, frac)]`, names `learned_f{frac:g}_s{seed}`

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_tqa_mc.py`:

```python
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
        u, obj, acc = tl.train_one(model, tok, st, pairs, pairs, 2.0, seed=0,
                                   max_steps=40)
        assert st.vec is None                     # training leaves no steering behind
    assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-5)
    assert obj > base


def test_load_learned_names_each_vector_by_frac_and_seed(tmp_path):
    p = tmp_path / "mc_learned.npz"
    np.savez(p, vecs=np.eye(2), frac=np.array([0.25, 0.5]), seed=np.array([0, 2]),
             val_obj=np.zeros(2), val_acc=np.zeros(2), norm_med=100.0,
             base_val_obj=0.0, base_val_acc=0.0)
    got = tl.load_learned(str(p))
    assert [(n, f) for n, _, f in got] == [("learned_f0.25_s0", 0.25),
                                           ("learned_f0.5_s2", 0.5)]
```

- [ ] **Step 2: Run them to see them fail**

Run (LAPTOP): `./.venv/bin/python -m pytest tests/test_tqa_mc.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'tqa_learned'`.

- [ ] **Step 3: Write the training module**

Create `src/tqa_learned.py`:

```python
"""tqa_learned.py: the best single layer-11 steering vector, trained (J-E, stage train).

Spec: docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md.

The ceiling every steering number is a fraction of. For each norm r = frac x the median
||h_11|| of the holdout prompts, a vector v with ||v|| = r is trained by Adam to maximize

    mean over pairs of logsigmoid( mean_lp(true answer) - mean_lp(false answer) )

on 600 of the 744 got_datasets/truthfulqa.csv questions (one true and one false answer
each), projected back to the sphere after every step, and the step with the best
objective on the other 144 is kept. Per-token means, so it cannot win on length. None of
the 744 questions is in the 64-question holdout every other stage is scored on.

Three random starts per frac: cosines near 1 between them mean one optimal direction;
low cosines at equal objectives mean many (non-identifiability in miniature).
"""
import math
import os

import numpy as np

import xfer_common as xc

SPLIT_SEED, N_VAL = 37, 144
SEEDS = (0, 1, 2)
LR_PER_R, BATCH_PAIRS, MAX_EPOCHS, EVAL_EVERY = 1e-2, 16, 4, 10


# ------------------------------------------------------------------ pure
def train_pairs():
    import pandas as pd

    from prep_truthfulqa import prompt_of
    df = pd.read_csv("got_datasets/truthfulqa.csv")
    out = []
    for q, g in df.groupby("question", sort=True):
        t, f = g[g["label"] == 1]["answer"], g[g["label"] == 0]["answer"]
        if len(t) != 1 or len(f) != 1:          # every question has one of each (2026-09-18)
            continue
        q = str(q).strip()
        out.append({"question": q, "prompt": prompt_of(q),
                    "true": str(t.iloc[0]).strip(), "false": str(f.iloc[0]).strip()})
    return out


def split_questions(qs, n_val=N_VAL, seed=SPLIT_SEED):
    qs = sorted(set(qs))
    perm = np.random.default_rng(seed).permutation(len(qs))
    val = {qs[i] for i in perm[:n_val]}
    return [q for q in qs if q not in val], sorted(val)


def project(v, r):
    return v * (r / v.norm().clamp_min(1e-12))


# ------------------------------------------------------------------ training
def pair_objective(model, tok, pairs, grad):
    import torch

    import tqa_mc as mc
    k = len(pairs)
    _, mean, _ = mc.answer_logprob(
        model, tok, [p["prompt"] for p in pairs] * 2,
        [p["true"] for p in pairs] + [p["false"] for p in pairs], grad=grad, batch=2 * k)
    d = mean[:k] - mean[k:]
    return torch.nn.functional.logsigmoid(d).mean(), (d > 0).float().mean()


def evaluate(model, tok, pairs):
    obj = acc = 0.0
    for b0 in range(0, len(pairs), BATCH_PAIRS):
        chunk = pairs[b0:b0 + BATCH_PAIRS]
        o, a = pair_objective(model, tok, chunk, grad=False)
        obj += float(o) * len(chunk)
        acc += float(a) * len(chunk)
    return obj / len(pairs), acc / len(pairs)


def train_one(model, tok, st, tr, va, r, seed, max_steps=None):
    """(unit vector, best validation objective, its accuracy). `st` is a Steerer at the
    steering layer; it holds v during training and is cleared before returning."""
    import torch
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    v = project(torch.randn(model.config.hidden_size, generator=g), r).to(dev)
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=LR_PER_R * r)
    rng = np.random.default_rng(seed)
    per_epoch = math.ceil(len(tr) / BATCH_PAIRS)
    total = max_steps or MAX_EPOCHS * per_epoch
    best = (-np.inf, 0.0, v.detach().clone())
    step = 0
    st.set(v)
    try:
        while step < total:
            for idx in np.array_split(rng.permutation(len(tr)), per_epoch):
                loss = -pair_objective(model, tok, [tr[i] for i in idx], grad=True)[0]
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    v.copy_(project(v, r))
                step += 1
                if step % EVAL_EVERY == 0 or step == total:
                    with torch.no_grad():
                        o, a = evaluate(model, tok, va)
                    if o > best[0]:
                        best = (o, a, v.detach().clone())
                    print(f"  [train] r={r:.4g} seed={seed} step {step}/{total} "
                          f"val obj {o:.4f} acc {a:.3f}", flush=True)
                if step >= total:
                    break
    finally:
        st.set(None)
    u = (best[2] / r).cpu().double().numpy()
    return u, float(best[0]), float(best[1])


# ------------------------------------------------------------------ io
def load_learned(p):
    if not os.path.exists(p):
        return []
    z = np.load(p, allow_pickle=True)
    return [(f"learned_f{float(f):g}_s{int(s)}", np.asarray(v, np.float64), float(f))
            for v, f, s in zip(z["vecs"], z["frac"], z["seed"])]


def _save(p, rows, extra):
    np.savez(p, vecs=np.array([r[0] for r in rows]), frac=np.array([r[1] for r in rows]),
             seed=np.array([r[2] for r in rows]), val_obj=np.array([r[3] for r in rows]),
             val_acc=np.array([r[4] for r in rows]), **extra)


def learned_cos(learned, dirs):
    """Seed-vs-seed cosines per frac, and every learned vector against every direction."""
    rows = []
    for i, (n1, v1, f1) in enumerate(learned):
        for n2, v2, f2 in learned[i + 1:]:
            if f1 == f2:
                rows.append({"a": n1, "b": n2, "kind": "seed_vs_seed",
                             "cos": float(v1 @ v2)})
        for n2, v2 in dirs:
            rows.append({"a": n1, "b": n2, "kind": "vs_direction",
                         "cos": float(v1 @ xc.unit(v2))})
    return rows


def stage_train(device, limit=0, steps=None, jb_prefix=""):
    import pandas as pd

    import tqa_mc as mc
    import xfer_tqa as xtqa
    from prep_truthfulqa import prompt_of
    out = mc.path("mc_learned.npz")
    su, tok, model = mc.load_model_left(device)
    norm_med = xc.median_last_norm(model, tok, [prompt_of(q)
                                                for q in xtqa.questions(limit)])
    pairs = train_pairs()
    tr_q, va_q = split_questions([p["question"] for p in pairs])
    tr = [p for p in pairs if p["question"] in set(tr_q)]
    va = [p for p in pairs if p["question"] in set(va_q)]
    if limit:
        tr, va = tr[:4 * limit], va[:limit]
    rows = []
    if os.path.exists(out):
        z = np.load(out)
        rows = [(v, float(f), int(s), float(o), float(a)) for v, f, s, o, a in
                zip(z["vecs"], z["frac"], z["seed"], z["val_obj"], z["val_acc"])]
    done = {(r[1], r[2]) for r in rows}
    base_o, base_a = evaluate(model, tok, va)
    extra = {"norm_med": norm_med, "base_val_obj": base_o, "base_val_acc": base_a}
    print(f"[train] {len(tr)} train / {len(va)} val pairs; median ||h_11|| {norm_med:.4g}; "
          f"unsteered val obj {base_o:.4f} acc {base_a:.3f}", flush=True)
    with su.Steerer(model, xc.LAYER) as st:
        for f in xc.NORM_FRACS:
            for s in SEEDS:
                if (f, s) in done:
                    continue
                u, o, a = train_one(model, tok, st, tr, va, f * norm_med, s, steps)
                rows.append((u, f, s, o, a))
                _save(out, rows, extra)
                print(f"[train] frac {f:g} seed {s}: val obj {o:.4f} acc {a:.3f} "
                      f"(unsteered {base_o:.4f} / {base_a:.3f})", flush=True)
    dirs, missing = xc.build_directions(0, 0, jb_prefix)
    for m in missing:
        print(f"[train] cosines leave out: {m}", flush=True)
    pd.DataFrame(learned_cos(load_learned(out), dirs)).to_csv(
        mc.path("mc_learned_cos.csv"), index=False)
    del model
```

- [ ] **Step 4: Run the tests**

Run (LAPTOP): `./.venv/bin/python -m pytest tests/test_tqa_mc.py -q`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/tqa_learned.py tests/test_tqa_mc.py
git commit -m "feat(J-E): train the best single layer-11 vector per norm, three starts each"
```

---

### Task 3: The scoring and generation stages, and `main`

**Files:**
- Modify: `src/tqa_mc.py` (append)
- Modify: `tests/test_tqa_mc.py` (append)

**Interfaces:**
- Consumes: Task 1 core; `tqa_learned.stage_train`, `tqa_learned.load_learned`, `tqa_learned.train_pairs`; `xfer_common.build_directions / dose_grid / median_last_norm / steer_vec / gen_batch / done_blocks / append_rows / eps_star / READ_UNIT / READ_FRAC / LAYER`; `xfer_cities.countries_of / clean_pool / targets / score_gen`; `xfer_tqa.questions / generate / answer_rows`; `judge_audit.judge_resumable`; `judges.local_hf.TruthJudge`.
- Produces:
  - `tqa_mc.load_model_left(device) -> (dct_steer_utils module, tok, model)`
  - `tqa_mc.blocks(dirs, grid, learned, norm_med) -> list[(name, unit, frac, scale, vec)]`, baseline first
  - `tqa_mc.smoke_dirs(dirs) -> list` (3 non-random, 2 random)
  - `tqa_mc.check_padding(model, tok, items)`
  - `tqa_mc.score_blocks(model, tok, st, bl, items, out)` writing rows `direction, unit, frac, scale, question, answer, correct, sum_lp, mean_lp, n_tok`
  - stages `stage_mc`, `stage_cities_long`, `stage_gen`, `stage_judge`; `STAGES`; `main(argv=None)`
  - files `mc_tqa_meta.json`, `mc_tqa_scores.csv`, `mc_tqa_train_scores.csv`, `mc_cities_long.csv`, `mc_learned_gen.csv`, `mc_learned_judged.csv`

- [ ] **Step 1: Append the failing tests**

```python
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


def test_main_rejects_limit_without_prefix():
    with pytest.raises(SystemExit):
        mc.main(["--stage", "mc", "--limit", "4"])
```

- [ ] **Step 2: Run them to see them fail**

Run (LAPTOP): `./.venv/bin/python -m pytest tests/test_tqa_mc.py -q`
Expected: 4 failures, `AttributeError: module 'tqa_mc' has no attribute 'blocks'` (and similar).

- [ ] **Step 3: Append the stages**

Append to `src/tqa_mc.py`:

```python
# ------------------------------------------------------------------ blocks
def blocks(dirs, grid, learned, norm_med):
    """Baseline, every direction at every dose, then each learned vector at its own frac
    only (a vector trained at one norm is not claimed at another), toward truthful."""
    out = [("baseline", "none", 0.0, 0.0, None)]
    out += [(n, u, f, s, v) for n, v in dirs for u, f, s in grid]
    out += [(n, "norm", f, f * norm_med, v) for n, v, f in learned]
    return out


def smoke_dirs(dirs):
    return ([d for d in dirs if not d[0].startswith("rand_")][:3] +
            [d for d in dirs if d[0].startswith("rand_")][:2])


def load_model_left(device):
    import dct_steer_utils as su
    from reach_hop import load_meta
    src, _, _, model_name = load_meta(DS)
    assert src == xc.LAYER, f"TQA meta says source layer {src}, J-E assumes {xc.LAYER}"
    tok, model, _ = su.load_model(device, model_name=model_name)
    tok.padding_side = "left"
    return su, tok, model


def check_padding(model, tok, items):
    """Batched and one-at-a-time scores must agree: the left-padding and position-id
    handling is otherwise unverified on the real model."""
    it = items[:4]
    s_b = answer_logprob(model, tok, [i["prompt"] for i in it], [i["answer"] for i in it])[0]
    s_1 = [float(answer_logprob(model, tok, [i["prompt"]], [i["answer"]])[0][0]) for i in it]
    diff = float(np.max(np.abs(s_b.cpu().numpy() - np.array(s_1))))
    print(f"[mc] padding check: max |batched - single| summed log-prob {diff:.2e}",
          flush=True)
    if diff > PAD_TOL:
        raise SystemExit(f"[mc] !!!! padding changes scores by {diff:.2e} > {PAD_TOL}")


def score_blocks(model, tok, st, bl, items, out):
    done = xc.done_blocks(out)
    prompts, answers = [i["prompt"] for i in items], [i["answer"] for i in items]
    for name, u, f, s, v in bl:
        if done.get((name, u, float(f)), 0) >= len(items):
            continue
        st.set(None if v is None else xc.steer_vec(v, s))
        sm, mn, n = (t.cpu().numpy() for t in answer_logprob(model, tok, prompts, answers))
        st.set(None)
        xc.append_rows(out, [{"direction": name, "unit": u, "frac": f, "scale": s,
                              "question": i["question"], "answer": i["answer"],
                              "correct": i["correct"], "sum_lp": float(a),
                              "mean_lp": float(b), "n_tok": int(c)}
                             for i, a, b, c in zip(items, sm, mn, n)])
        print(f"[score] {os.path.basename(out)} {name} {u} {f:+g} done", flush=True)


# ------------------------------------------------------------------ stages
def stage_mc(device, limit=0, jb_prefix=""):
    import tqa_learned as tl
    items = holdout_items(limit)
    dirs, missing = xc.build_directions(N_RAND, RAND_SEED, jb_prefix)
    if limit:
        dirs = smoke_dirs(dirs)
    for m in missing:
        print(f"[mc] !!!! left out: {m}", flush=True)
    learned = tl.load_learned(path("mc_learned.npz"))
    if not learned:
        print("[mc] !!!! no learned vectors: stage train has not run", flush=True)
    su, tok, model = load_model_left(device)
    norm_med = xc.median_last_norm(model, tok, sorted({i["prompt"] for i in items}))
    grid = xc.dose_grid(norm_med, xc.eps_star(DS))
    json.dump({"n_questions": len({i["question"] for i in items}), "n_items": len(items),
               "norm_median_layer11": norm_med, "directions": [d[0] for d in dirs],
               "learned": [n for n, _, _ in learned], "left_out": missing,
               "grid": [list(g) for g in grid], "n_rand": N_RAND},
              open(path("mc_tqa_meta.json"), "w"), indent=2)
    check_padding(model, tok, items)
    with su.Steerer(model, xc.LAYER) as st:
        score_blocks(model, tok, st, blocks(dirs, grid, learned, norm_med), items,
                     path("mc_tqa_scores.csv"))
        # Secondary: the 744 training pairs at the read dose. In-sample for every
        # TQA-sourced direction and (outside the validation split) every learned one.
        pairs = tl.train_pairs()[:limit or None]
        titems = [{"question": p["question"], "prompt": p["prompt"], "answer": p[k],
                   "correct": c} for p in pairs for k, c in (("true", 1), ("false", 0))]
        rgrid = [(xc.READ_UNIT, xc.READ_FRAC, xc.READ_FRAC * norm_med)]
        rlearned = [x for x in learned if x[2] == xc.READ_FRAC]
        score_blocks(model, tok, st, blocks(dirs, rgrid, rlearned, norm_med), titems,
                     path("mc_tqa_train_scores.csv"))
    del model


def stage_cities_long(device, limit=0, jb_prefix=""):
    import tqa_learned as tl
    import xfer_cities as xcit
    dirs, missing = xc.build_directions(N_RAND_LONG, RAND_SEED_LONG, jb_prefix)
    if limit:
        dirs = smoke_dirs(dirs)
    for m in missing:
        print(f"[long] !!!! left out: {m}", flush=True)
    learned = [x for x in tl.load_learned(path("mc_learned.npz")) if x[2] == xc.READ_FRAC]
    su, tok, model = load_model_left(device)
    countries = xcit.countries_of()
    pool = xcit.clean_pool({c: int(tok(" " + c, add_special_tokens=False)["input_ids"][0])
                            for c in countries})
    recs = xcit.targets(pool, limit)
    prompts = [LONG_PROMPT.format(city=r["city"]) for r in recs]
    norm_med = xc.median_last_norm(model, tok, prompts)
    grid = [("norm", s * xc.READ_FRAC, s * xc.READ_FRAC * norm_med) for s in (1.0, -1.0)]
    out = path("mc_cities_long.csv")
    done = xc.done_blocks(out)
    print(f"[long] {len(recs)} cities; median ||h_11|| {norm_med:.4g}", flush=True)
    with su.Steerer(model, xc.LAYER) as st:
        for name, u, f, s, v in blocks(dirs, grid, learned, norm_med):
            if done.get((name, u, float(f)), 0) >= len(recs):
                continue
            st.set(None if v is None else xc.steer_vec(v, s))
            raws = []
            for b0 in range(0, len(prompts), BATCH):
                raws += xc.gen_batch(model, tok, prompts[b0:b0 + BATCH], LONG_MAX_NEW,
                                     LONG_REP)
            st.set(None)
            xc.append_rows(out, [dict({"direction": name, "unit": u, "frac": f,
                                       "scale": s, "stmt": r["stmt"], "city": r["city"],
                                       "correct": r["correct"],
                                       "completion": raw.replace("\n", " ").strip()},
                                      **xcit.score_gen(raw, r["correct"], countries))
                                 for r, raw in zip(recs, raws)])
            print(f"[long] {name} {u} {f:+g} done", flush=True)
    del model


def stage_gen(device, limit=0):
    """The learned vectors, generated and later judged: the ceiling on the judged scale."""
    import tqa_learned as tl
    import xfer_tqa as xtqa
    from prep_truthfulqa import prompt_of
    learned = tl.load_learned(path("mc_learned.npz"))
    if not learned:
        print("[gen] !!!! no learned vectors, nothing to generate", flush=True)
        return
    qs = xtqa.questions(limit)
    su, tok, model = load_model_left(device)
    norm_med = xc.median_last_norm(model, tok, [prompt_of(q) for q in qs])
    out = path("mc_learned_gen.csv")
    done = xc.done_blocks(out)
    with su.Steerer(model, xc.LAYER) as st:
        for name, u, f, s, v in blocks([], [], learned, norm_med):
            if done.get((name, u, float(f)), 0) >= len(qs):
                continue
            xc.append_rows(out, xtqa.answer_rows(name, u, f, s, qs,
                                                 xtqa.generate(model, tok, st, v, s, qs)))
            print(f"[gen] {name} {u} {f:+g} done", flush=True)
    del model


def stage_judge(device):
    from judge_audit import judge_resumable
    from judges.local_hf import TruthJudge
    p = path("mc_learned_gen.csv")
    if not os.path.exists(p):
        print("[judge] nothing generated, skipping", flush=True)
        return
    judge_resumable(td.read_csv(p), TruthJudge(device).score, path("mc_learned_judged.csv"))


STAGES = ("train", "mc", "cities_long", "gen", "judge", "summary")


def main(argv=None):
    global PREFIX
    import tqa_learned as tl
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="cap training steps (smoke)")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--jb-prefix", default="")
    a = ap.parse_args(argv)
    if a.limit and not a.prefix:
        ap.error("--limit writes partial files: give it a --prefix")
    PREFIX = a.prefix
    todo = STAGES if a.stage == "all" else (a.stage,)
    for s in todo:
        print(f"=== tqa_mc stage {s} ===", flush=True)
        if s == "train":
            tl.stage_train(a.device, a.limit, a.steps, a.jb_prefix)
        elif s == "mc":
            stage_mc(a.device, a.limit, a.jb_prefix)
        elif s == "cities_long":
            stage_cities_long(a.device, a.limit, a.jb_prefix)
        elif s == "gen":
            stage_gen(a.device, a.limit)
        elif s == "judge":
            stage_judge(a.device)
        else:
            stage_summary()


if __name__ == "__main__":
    main()
```

Note: `stage_summary` is added in Task 4; until then `--stage summary` raises `NameError`, which no test exercises.

- [ ] **Step 4: Run the tests**

Run (LAPTOP): `./.venv/bin/python -m pytest tests/test_tqa_mc.py -q`
Expected: `17 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/tqa_mc.py tests/test_tqa_mc.py
git commit -m "feat(J-E): the mc, cities_long, gen and judge stages, resumable"
```

---

### Task 4: The summaries and the registered readings

**Files:**
- Modify: `src/tqa_mc.py` (insert above `STAGES = ...`)
- Modify: `tests/test_tqa_mc.py` (append)

**Interfaces:**
- Consumes: `mc_metrics`; `xfer_common.perm_p / beyond_null / ALPHA / READ_FRAC`; `tqa_q2_analyze.mcnemar_exact(base, arm) -> (gained, lost, p, n)`; `tqa_baseline.wilson(k, n) -> (lo, hi)`; `tqa_learned.split_questions / train_pairs / load_learned`.
- Produces:
  - `tqa_mc.per_question(df) -> DataFrame(direction, unit, frac, question, margin, mc1, mc2)`
  - `tqa_mc.add_null(rows, effect_key, p_key) -> rows` (sets `n_null`, `p_key`)
  - `tqa_mc.summarize_mc(df) -> list[dict]` with `e_margin, wilcoxon_p, p_margin, n_null, moves`
  - `tqa_mc.summarize_long(df) -> list[dict]` with `e_gen, mcnemar_p, p_gen, n_null, moves`
  - `tqa_mc.summarize_train(df, val_qs) -> list[dict]`
  - `tqa_mc.summarize_judged(df) -> list[dict]`
  - `tqa_mc.stage_summary()` writing `mc_tqa_summary.csv`, `mc_tqa_train_summary.csv`, `mc_cities_long_summary.csv`, `mc_learned_judged_summary.csv`

- [ ] **Step 1: Append the failing tests**

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run (LAPTOP): `./.venv/bin/python -m pytest tests/test_tqa_mc.py -q`
Expected: 5 failures, `AttributeError: module 'tqa_mc' has no attribute 'summarize_mc'` (and similar).

- [ ] **Step 3: Insert the summaries above `STAGES = ...` in `src/tqa_mc.py`**

```python
# ------------------------------------------------------------------ summary
def per_question(df):
    import pandas as pd
    rows = []
    for k, g in df.groupby(["direction", "unit", "frac", "question"]):
        rows.append(dict(zip(("direction", "unit", "frac", "question"), k),
                         **mc_metrics(g["mean_lp"].values, g["sum_lp"].values,
                                      g["correct"].values)))
    return pd.DataFrame(rows)


def add_null(rows, effect_key, p_key):
    """Permutation p of each row's effect among the random directions at the same
    (unit, frac); blank for the randoms themselves and where there is no null."""
    rand = {}
    for r in rows:
        if r["direction"].startswith("rand_"):
            rand.setdefault((r["unit"], r["frac"]), []).append(r[effect_key])
    for r in rows:
        rs = rand.get((r["unit"], r["frac"]), [])
        r["n_null"] = len(rs)
        r[p_key] = ("" if r["direction"].startswith("rand_") or not rs
                    else xc.perm_p(r[effect_key], rs))
    return rows


def _moves(r, p_test, p_key, e_key, alpha=xc.ALPHA):
    return bool(r[p_key] != "" and r[p_test] <= alpha and r[e_key] > 0
                and xc.beyond_null(r[p_key], r["n_null"], alpha))


def summarize_mc(df):
    from scipy.stats import wilcoxon

    from tqa_q2_analyze import mcnemar_exact
    pq = per_question(df)
    base = pq[pq["direction"] == "baseline"].set_index("question")
    out = []
    for (name, u, f), g in pq[pq["direction"] != "baseline"].groupby(
            ["direction", "unit", "frac"]):
        g = g.set_index("question")
        b = base.loc[g.index]
        sgn = float(np.sign(f)) or 1.0
        dm = g["margin"] - b["margin"]
        try:
            p_w = float(wilcoxon(dm).pvalue) if (dm != 0).any() else 1.0
        except ValueError:
            p_w = 1.0
        gained, lost, p_mc, _ = mcnemar_exact(b["mc1"], g["mc1"])
        out.append({"direction": name, "unit": u, "frac": float(f), "n": len(g),
                    "margin": float(g["margin"].mean()),
                    "base_margin": float(b["margin"].mean()),
                    "e_margin": sgn * float(dm.mean()), "wilcoxon_p": p_w,
                    "mc1": float(g["mc1"].mean()), "base_mc1": float(b["mc1"].mean()),
                    "mc1_gained": gained, "mc1_lost": lost, "mc1_mcnemar_p": p_mc,
                    "mc2": float(g["mc2"].mean()), "base_mc2": float(b["mc2"].mean()),
                    "e_mc2": sgn * float((g["mc2"] - b["mc2"]).mean())})
    add_null(out, "e_margin", "p_margin")
    for r in out:
        r["moves"] = _moves(r, "wilcoxon_p", "p_margin", "e_margin")
    return sorted(out, key=lambda r: (r["direction"], r["unit"], r["frac"]))


def summarize_long(df):
    from tqa_q2_analyze import mcnemar_exact
    base = df[df["direction"] == "baseline"].set_index("stmt")
    out = []
    for (name, u, f), g in df[df["direction"] != "baseline"].groupby(
            ["direction", "unit", "frac"]):
        g = g.set_index("stmt")
        b = base.loc[g.index]
        sgn = float(np.sign(f)) or 1.0
        gained, lost, p_mc, _ = mcnemar_exact(b["gen_correct"], g["gen_correct"])
        out.append({"direction": name, "unit": u, "frac": float(f), "n": len(g),
                    "gen_correct": float(g["gen_correct"].mean()),
                    "base_gen_correct": float(b["gen_correct"].mean()),
                    "gained": gained, "lost": lost, "mcnemar_p": p_mc,
                    "e_gen": sgn * float(g["gen_correct"].mean() - b["gen_correct"].mean()),
                    "words": float(g["words"].mean()), "base_words": float(b["words"].mean()),
                    "incoherent": float(g["incoherent"].mean()),
                    "base_incoherent": float(b["incoherent"].mean())})
    add_null(out, "e_gen", "p_gen")
    for r in out:
        r["moves"] = _moves(r, "mcnemar_p", "p_gen", "e_gen")
    return sorted(out, key=lambda r: (r["direction"], r["unit"], r["frac"]))


def summarize_train(df, val_qs):
    """Per direction: mean over pairs of the change in mean_lp(true) - mean_lp(false).
    TQA-sourced directions were fitted on these pairs; learned vectors trained on all but
    the validation questions, so they get a separate `val` row."""
    from scipy.stats import wilcoxon
    gap = (df.pivot_table(index=["direction", "question"], columns="correct",
                          values="mean_lp").reset_index())
    gap["gap"] = gap[1] - gap[0]
    base = gap[gap["direction"] == "baseline"].set_index("question")["gap"]
    val = set(val_qs)
    out = []
    for name, g in gap[gap["direction"] != "baseline"].groupby("direction"):
        g = g.set_index("question")["gap"]
        learned = name.startswith("learned_")
        subsets = ([("train", [q for q in g.index if q not in val]),
                    ("val", [q for q in g.index if q in val])] if learned
                   else [("all", list(g.index))])
        for sub, qs in subsets:
            if not qs:
                continue
            d = g.loc[qs] - base.loc[qs]
            try:
                p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            except ValueError:
                p = 1.0
            out.append({"direction": name, "subset": sub, "n": len(qs),
                        "d_gap": float(d.mean()), "wilcoxon_p": p,
                        "in_sample": bool(name.startswith("tqa:") or
                                          (learned and sub == "train"))})
    return out


def summarize_judged(df):
    from tqa_baseline import wilson
    from tqa_q2_analyze import mcnemar_exact
    base = df[df["direction"] == "baseline"].set_index("question")
    out = []
    for name, g in df[df["direction"] != "baseline"].groupby("direction"):
        g = g.set_index("question")
        b = base.loc[g.index]
        r = {"direction": name, "n": len(g)}
        for col in ("truthful", "truthful_and_informative"):
            x, y = b[col].astype(int), g[col].astype(int)
            gained, lost, p, _ = mcnemar_exact(x, y)
            lo, hi = wilson(int(y.sum()), len(y))
            r.update({col: float(y.mean()), f"base_{col}": float(x.mean()),
                      f"lo_{col}": lo, f"hi_{col}": hi, f"gained_{col}": gained,
                      f"lost_{col}": lost, f"mcnemar_p_{col}": p})
        out.append(r)
    return out


def stage_summary():
    """LAPTOP, from pulled CSVs. Prints the registered readings of the spec."""
    import pandas as pd

    import tqa_learned as tl

    def write(rows, name):
        if rows:
            pd.DataFrame(rows).to_csv(path(name), index=False)
            print(f"[summary] wrote {path(name)}", flush=True)
        return rows

    have = lambda n: os.path.exists(path(n))       # noqa: E731
    mc_rows = long_rows = []
    if have("mc_tqa_scores.csv"):
        mc_rows = write(summarize_mc(pd.read_csv(path("mc_tqa_scores.csv"))),
                        "mc_tqa_summary.csv")
    if have("mc_tqa_train_scores.csv"):
        _, va = tl.split_questions([p["question"] for p in tl.train_pairs()])
        write(summarize_train(pd.read_csv(path("mc_tqa_train_scores.csv")), va),
              "mc_tqa_train_summary.csv")
    if have("mc_cities_long.csv"):
        long_rows = write(summarize_long(pd.read_csv(path("mc_cities_long.csv"))),
                          "mc_cities_long_summary.csv")
    if have("mc_learned_judged.csv"):
        write(summarize_judged(pd.read_csv(path("mc_learned_judged.csv"))),
              "mc_learned_judged_summary.csv")

    real = [r for r in mc_rows if not r["direction"].startswith(("rand_", "learned_"))]
    moved = [f"{r['direction']} {r['unit']} {r['frac']:+g}" for r in real if r["moves"]]
    print(f"\n[reading] CONTENT: truth directions moving the MC margin beyond null: "
          f"{moved or 'none'}")
    ceil = [r for r in mc_rows if r["direction"].startswith("learned_")
            and r["frac"] == xc.READ_FRAC]
    for r in ceil:
        print(f"[reading] CEILING {r['direction']}: e_margin {r['e_margin']:+.3f}, "
              f"wilcoxon p {r['wilcoxon_p']:.2g}, perm p {r['p_margin']}, "
              f"moves={r['moves']}")
    if ceil and not any(r["moves"] for r in ceil):
        print("[reading] METHOD LIMIT: no learned vector at norm "
              f"{xc.READ_FRAC} moves beyond null. Every steering null is about "
              "single-vector steering at layer 11, not about truth.")
    lmoved = [f"{r['direction']} {r['frac']:+g}" for r in long_rows
              if r["moves"] and not r["direction"].startswith("rand_")]
    print(f"[reading] cities LONG FORM, directions moving gen_correct beyond null: "
          f"{lmoved or 'none'}")
    if have("mc_learned_cos.csv"):
        c = pd.read_csv(path("mc_learned_cos.csv"))
        s = c[c["kind"] == "seed_vs_seed"]
        if len(s):
            print(f"[reading] IDENTIFIABILITY: seed-vs-seed cosine median "
                  f"{s['cos'].median():.3f} (min {s['cos'].min():.3f})")
```

- [ ] **Step 4: Run the tests**

Run (LAPTOP): `./.venv/bin/python -m pytest tests/test_tqa_mc.py -q`
Expected: `22 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/tqa_mc.py tests/test_tqa_mc.py
git commit -m "feat(J-E): summaries with the paired-plus-null rule and the registered readings"
```

---

### Task 5: The job, the submitter, and the docs

**Files:**
- Create: `deltaai/run_tqa_mc.slurm`
- Modify: `deltaai/submit_pi_feedback.sh` (the `case` block and usage line)
- Modify: `deltaai/README.md` (one row after the `tokgeom` row)
- Modify: `docs/PLAN_PI_FEEDBACK_2026-09-18.md` (a note at the end of section 8)

**Interfaces:**
- Consumes: `python3 src/tqa_mc.py --stage all --device cuda [--limit 4 --steps 1 --prefix smoke_]`.
- Produces: `bash deltaai/submit_pi_feedback.sh round3`.

- [ ] **Step 1: Write the job file**

Create `deltaai/run_tqa_mc.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=tqa_mc
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=tqa_mc_%j.out

# J-E (round 3): judge-free TruthfulQA and the learned single-vector ceiling.
# docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md; src/tqa_mc.py and
# src/tqa_learned.py.
#   train        the best ||v||-fixed layer-11 vector per norm frac, 3 starts (~15 min)
#   mc           every direction x dose on the 64 holdout questions' reference answers,
#                scored by log-prob, 32 randoms; plus the 744 training pairs at the read dose
#   cities_long  "Q: Where is the city of X?" 48 tokens, norm +/-0.25, 8 randoms
#   gen, judge   the learned vectors generated and judged by the v2 allenai judges
# Full direction list needs J-B (TQA DCT picks, G0 MAG); without it they are left out
# loudly. Writes only mc_*. Every stage resumes after a timeout.
#
# READ THIS FIRST WHEN IT FINISHES:
#   [mc] padding check              must be <= 1e-3 or the stage stopped
#   [train] frac 0.25 seed ...       learned val objective vs unsteered
#   then LAPTOP: python src/tqa_mc.py --stage summary, the [reading] lines

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

echo "=== stage 0: preflight ==="
for ds in cities truthfulqa; do
  for f in reach_summary_${ds}.json reach_margins_${ds}.npz reach_acts_${ds}.npz \
           reach_dirs_${ds}.npz truth_dir_${ds}.npz dct_meta_${ds}.json \
           dct_V_${ds}.pt dct_U_${ds}.pt; do
    test -f "$f" || { echo "!!!! $f missing."; exit 1; }
  done
done
for f in got_datasets/truthfulqa_holdout.csv got_datasets/truthfulqa.csv \
         got_datasets/cities.csv token_acts_cities.npz mag_dir_cities.npz; do
  test -f "$f" || { echo "!!!! $f missing."; exit 1; }
done
test -f dct_selection_truthfulqa.json \
  || echo "[preflight] !!!! no J-B selection: the TQA DCT rows will be absent"
test -f jb_truthfulqa/mag_dir_truthfulqa.npz \
  || echo "[preflight] no TQA MAG directions from J-B; those rows will be absent"
for m in models--google--gemma-2-2b models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done
python3 -c "
import json, sys
m = json.load(open('dct_meta_truthfulqa.json'))
sys.exit(0 if m.get('num_factors') is None and m['source_layer'] == 11 else
         '!!!! dct_meta_truthfulqa.json is not the Q2 meta')"

echo "=== stage 1: smoke, 4 questions, 1 training step, smoke_ outputs ==="
python3 src/tqa_mc.py --stage all --device cuda --limit 4 --steps 1 --prefix smoke_
rm -f smoke_mc_*

echo "=== stage 2: the full run ==="
python3 src/tqa_mc.py --stage all --device cuda

echo "=== job complete ==="
ls -la mc_*
```

- [ ] **Step 2: Add `round3` to the submitter**

In `deltaai/submit_pi_feedback.sh`, after the `tokgeom)` line add:

```bash
  round3) files=(deltaai/run_tqa_mc.slurm) ;;
```

and change the usage line to:

```bash
  *) echo "usage: $0 round1|round2|round3|tokgeom"; exit 2 ;;
```

and the header comment `# submit_pi_feedback.sh <round|tokgeom>:` stays as it is.

- [ ] **Step 3: Check the shell syntax**

Run (LAPTOP): `bash -n deltaai/run_tqa_mc.slurm && bash -n deltaai/submit_pi_feedback.sh && echo ok`
Expected: `ok`.

- [ ] **Step 4: README row**

In `deltaai/README.md`, after the row beginning `| Token geometry on the "country of" prompt |`, add:

```markdown
| PI feedback, round 3 (J-E) | [`../docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md`](../docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md) | `run_tqa_mc.slurm` (~1.5 h, 3 h wall), best after J-B, via `submit_pi_feedback.sh round3` |
```

- [ ] **Step 5: Plan note**

Append to the end of section 8 of `docs/PLAN_PI_FEEDBACK_2026-09-18.md` (immediately before the `## 9. Track C` heading):

```markdown
**Added 2026-09-18: J-E, round 3** (`deltaai/run_tqa_mc.slurm`, spec
`docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md`). Two measurements that
do not depend on which way X comes out. (1) TruthfulQA scored by the model's log-probability
of its reference answers, no judge and no generation, which with a long-form cities arm
completes the dataset x format 2x2 around J-D1 and J-D2. (2) The best single layer-11 vector
at each norm, trained on the 744 non-holdout questions: the ceiling every steering effect
here is a fraction of. Registered readings are in the spec.
```

- [ ] **Step 6: Full suite**

Run (LAPTOP): `./.venv/bin/python -m pytest -q`
Expected: all pass (643 before this plan, plus 22 new: `665 passed, 2 skipped`).

- [ ] **Step 7: Commit**

```bash
git add deltaai/run_tqa_mc.slurm deltaai/submit_pi_feedback.sh deltaai/README.md docs/PLAN_PI_FEEDBACK_2026-09-18.md
git commit -m "feat(J-E): the round-3 job, its submitter entry, and the plan note"
```
