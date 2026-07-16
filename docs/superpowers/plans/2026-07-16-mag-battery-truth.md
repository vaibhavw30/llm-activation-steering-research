# MAG (Mining via Activation Geometry) Battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a second, independent unsupervised feature-mining method (MAG) adapted to our truth question on `gemma-2-2b` base, and run it head-to-head against DCT for decoding (E1–E3), calibrated steering (E4), and transfer ranking (§4).

**Architecture:** A new `src/mag/` package mirroring the existing DCT/funnel arm. Pure math (operators, verdict, directions, probes, α-calibration) is separated from I/O (extraction, generation) so it is unit-testable without loading gemma. One `run_mag.py` CLI drives every probe from a cached `mag_acts_<ds>.npz`; `viz_mag.py` plots each probe; E4 reuses the existing `dct_steer_utils.Steerer`, `judge_results` OLMo path, and `investigate_steer.py` battery so MAG steering gets the same treatment as DCT steering.

**Tech Stack:** Python 3.13, PyTorch (fp32, MPS/CPU), transformers, numpy, scikit-learn, scipy, matplotlib, pytest. Model: `google/gemma-2-2b` (base).

## Global Constraints

- **Model:** `google/gemma-2-2b` base, fp32, eager attention. Load via `dct_steer_utils.load_model` — never re-implement loading.
- **Determinism:** seed 42 everywhere; greedy decode (`do_sample=False`), `repetition_penalty=1.3` (matches `dct_steer_utils.generate`).
- **Reuse, do not reinvent:** `funnel_utils` (`unit`, `resolve_layer`, `load_acts`, `mean_diff_dir`, `grad_dir`, `load_dct`, `top_k_by_potency`, `BEST_LAYER`); `dct_steer_utils` (`load_model`, `Steerer`, `generate`); `judge_results` (`run_steer`, `run_steer_local`, `_steer_summary_and_plot`); `investigate_steer` pure helpers.
- **Peak layers (from `funnel_utils.BEST_LAYER`):** cities 11, sp_en_trans 7, companies_true_false 14, common_claim_true_false 13. Final block = 26 (gemma-2-2b has 26 layers; `activations` axis 0 has length 27 incl. embeddings).
- **`y^M` (MAG Eq. 1):** `y^M(p) = 1[Pr(yes|Q‖p) > Pr(no|Q‖p)]`, read at the first generated token after `Q_TRUTH + p + Q_SUFFIX`.
- **`α(τ)` calibration (MAG):** `α(τ) = τ · A_prefix_norm[L_inject]` where `A_prefix_norm[L] = mean_p ‖A(Q‖p)[L]‖`, injected onto a **unit** direction. `TAUS = [-1.0, -0.3, 0, 0.3, 1.0]`.
- **Tests:** every task ends green; the full existing suite (`pytest -q`) must stay green.
- **Tests import from src** via `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))` — the established pattern (see `tests/test_investigate_steer.py`).
- **Gitignore** `mag_acts_*.npz` (large, regenerable). Commit code, tests, small CSVs, plots, the doc.
- **npz key names** are the contract between extract → directions/probes; the canonical set is defined in Task 4 and MUST be used verbatim downstream.

---

### Task 1: MAG package config + constants

**Files:**
- Create: `src/mag/__init__.py`
- Create: `src/mag/config.py`
- Test: `tests/test_mag_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Q_TRUTH: str`, `Q_SUFFIX: str`, `E_FEWSHOT: str`
  - `YES_VARIANTS: list[str]`, `NO_VARIANTS: list[str]`
  - `TAUS: list[float]` = `[-1.0, -0.3, 0.0, 0.3, 1.0]`
  - `OPERATOR_NAMES: list[str]` = the 8 operator names (order fixed):
    `["Direct","Prefixed","Answered","Verdict","InputDelta","QuestionDelta","Interaction","FewShot"]`
  - `DATASETS: list[str]` = `["cities","sp_en_trans","companies_true_false","common_claim_true_false"]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mag_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mag_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag'`

- [ ] **Step 3: Write the implementation**

```python
# src/mag/__init__.py
"""MAG (Mining via Activation Geometry) adapted to truth — see
docs/superpowers/specs/2026-07-16-mag-battery-truth-design.md."""
```

```python
# src/mag/config.py
"""Fixed strings and constants for the MAG battery (arXiv:2607.04222, adapted to truth).
Kept in one file so a reviewer can see and change the exact instruction Q and few-shot E."""

# Base gemma-2-2b: plain completion, not chat. Readout / y^M taken at the last token of
#   Q_TRUTH + statement + Q_SUFFIX  (the next token is the yes/no verdict).
Q_TRUTH = "Answer in yes or no only. Is the following statement true?\nStatement: "
Q_SUFFIX = "\nAnswer:"

E_FEWSHOT = (                      # fixed 2-shot preamble for phi_FewShot (one true, one false)
    "Answer in yes or no only. Is the following statement true?\n"
    "Statement: The sky is blue.\nAnswer: yes\n"
    "Statement: Fish can fly.\nAnswer: no\n"
)

# First-token variants summed for p_yes / p_no (Eq. 1). Space-prefixed forms matter for gemma's BPE.
YES_VARIANTS = [" yes", "yes", " Yes", "Yes"]
NO_VARIANTS = [" no", "no", " No", "No"]

# MAG sweeps tau in {0, 0.3, 1.0}; we add the negatives so the lie-asymmetry test still applies.
TAUS = [-1.0, -0.3, 0.0, 0.3, 1.0]

OPERATOR_NAMES = ["Direct", "Prefixed", "Answered", "Verdict",
                  "InputDelta", "QuestionDelta", "Interaction", "FewShot"]

DATASETS = ["cities", "sp_en_trans", "companies_true_false", "common_claim_true_false"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mag_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mag/__init__.py src/mag/config.py tests/test_mag_config.py
git commit -m "feat(mag): package config — Q/E strings, operator names, signed TAUS"
```

---

### Task 2: The eight operators (pure)

**Files:**
- Create: `src/mag/operators.py`
- Test: `tests/test_mag_operators.py`

**Interfaces:**
- Consumes: `config.OPERATOR_NAMES`.
- Produces:
  - `operator_features(name: str, cache, layer: int) -> np.ndarray` returning `(n, d)` float64. `cache` is a mapping with per-statement all-layer arrays `A_p, A_Qp, A_Qpv, A_verdict, A_EQp` each `(L+1, n, d)` and constants `A_Q, A_empty` each `(L+1, d)`.
  - `all_operator_features(cache, layer) -> dict[str, np.ndarray]` — every operator at `layer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mag_operators.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.operators import operator_features, all_operator_features


def _cache(L=3, n=5, d=4, seed=0):
    rng = np.random.default_rng(seed)
    per = lambda: rng.standard_normal((L, n, d))
    const = lambda: rng.standard_normal((L, d))
    return {"A_p": per(), "A_Qp": per(), "A_Qpv": per(), "A_verdict": per(),
            "A_EQp": per(), "A_Q": const(), "A_empty": const()}


def test_input_delta_is_prefixed_minus_direct():
    c = _cache(); L = 1
    got = operator_features("InputDelta", c, L)
    assert np.allclose(got, c["A_Qp"][L] - c["A_p"][L])


def test_interaction_adds_empty_constant():
    c = _cache(); L = 2
    got = operator_features("Interaction", c, L)
    assert np.allclose(got, c["A_Qp"][L] - c["A_p"][L] + c["A_empty"][L])


def test_question_delta_broadcasts_constant():
    c = _cache(); L = 0
    got = operator_features("QuestionDelta", c, L)
    assert np.allclose(got, c["A_Qp"][L] - c["A_Q"][L])   # (n,d) - (d,)


def test_direct_and_prefixed_passthrough():
    c = _cache(); L = 1
    assert np.allclose(operator_features("Direct", c, L), c["A_p"][L])
    assert np.allclose(operator_features("Prefixed", c, L), c["A_Qp"][L])


def test_operators_do_not_mutate_inputs():
    c = _cache(); L = 1
    before = c["A_Qp"][L].copy()
    operator_features("InputDelta", c, L)
    assert np.allclose(c["A_Qp"][L], before)


def test_all_operator_features_returns_eight_2d():
    c = _cache(); feats = all_operator_features(c, 1)
    assert len(feats) == 8
    assert all(v.ndim == 2 and v.shape == (5, 4) for v in feats.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mag_operators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag.operators'`

- [ ] **Step 3: Write the implementation**

```python
# src/mag/operators.py
"""The 8 MAG operators (Table 1) as pure functions over cached last-token readouts.

Each returns an (n, d) feature matrix at a chosen layer L. Per-statement caches are
(L+1, n, d); constants A_Q, A_empty are (L+1, d) and broadcast over n.
"""
import numpy as np

from .config import OPERATOR_NAMES


def operator_features(name, cache, layer):
    A_p = np.asarray(cache["A_p"][layer], dtype=np.float64)      # (n, d)
    A_Qp = np.asarray(cache["A_Qp"][layer], dtype=np.float64)    # (n, d)
    A_Qpv = np.asarray(cache["A_Qpv"][layer], dtype=np.float64)
    A_verd = np.asarray(cache["A_verdict"][layer], dtype=np.float64)
    A_EQp = np.asarray(cache["A_EQp"][layer], dtype=np.float64)
    A_Q = np.asarray(cache["A_Q"][layer], dtype=np.float64)      # (d,)
    A_empty = np.asarray(cache["A_empty"][layer], dtype=np.float64)  # (d,)

    if name == "Direct":
        return A_p
    if name == "Prefixed":
        return A_Qp
    if name == "Answered":
        return A_Qpv
    if name == "Verdict":
        return A_verd
    if name == "InputDelta":
        return A_Qp - A_p
    if name == "QuestionDelta":
        return A_Qp - A_Q            # (n,d) - (d,)
    if name == "Interaction":
        return A_Qp - A_p + A_empty  # (n,d) - (n,d) + (d,)
    if name == "FewShot":
        return A_EQp
    raise ValueError(f"unknown operator {name!r}")


def all_operator_features(cache, layer):
    return {name: operator_features(name, cache, layer) for name in OPERATOR_NAMES}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mag_operators.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mag/operators.py tests/test_mag_operators.py
git commit -m "feat(mag): 8 operators as pure functions over cached readouts"
```

---

### Task 3: Self-verdict `y^M` (MAG Eq. 1)

**Files:**
- Create: `src/mag/verdict.py`
- Test: `tests/test_mag_verdict.py`

**Interfaces:**
- Consumes: `config.YES_VARIANTS`, `config.NO_VARIANTS`.
- Produces:
  - `verdict_from_logits(logits_row, yes_ids, no_ids) -> dict` with keys `ymL(int), margin(float), conf(float), p_yes(float), p_no(float)`. `logits_row` is a 1-D `(vocab,)` array; softmax computed inside.
  - `first_token_ids(tokenizer, variants) -> list[int]` — dedup first-token id of each variant string.
  - `compute_verdicts(model, tokenizer, prompts, device, batch_size=16) -> dict` returning arrays `ymL(n,), margin(n,), conf(n,)` where each prompt is already the full `Q_TRUTH+p+Q_SUFFIX`. (I/O; verified in the Task 10 smoke run, not unit-tested.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mag_verdict.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.verdict import verdict_from_logits


def test_yes_wins():
    v = 8
    logits = np.full(v, -10.0); logits[2] = 5.0   # yes id
    r = verdict_from_logits(logits, yes_ids=[2], no_ids=[3])
    assert r["ymL"] == 1 and r["margin"] > 0 and 0 < r["conf"] <= 1


def test_no_wins():
    v = 8
    logits = np.full(v, -10.0); logits[3] = 5.0   # no id
    r = verdict_from_logits(logits, yes_ids=[2], no_ids=[3])
    assert r["ymL"] == 0 and r["margin"] < 0


def test_tie_defaults_to_zero():
    v = 8
    logits = np.zeros(v)                            # p_yes == p_no
    r = verdict_from_logits(logits, yes_ids=[2], no_ids=[3])
    assert r["ymL"] == 0                            # strict > means tie -> 0
    assert abs(r["margin"]) < 1e-9


def test_multi_variant_ids_are_summed():
    v = 8
    logits = np.full(v, -10.0); logits[2] = 2.0; logits[4] = 2.0   # two yes variants
    r_two = verdict_from_logits(logits, yes_ids=[2, 4], no_ids=[3])
    r_one = verdict_from_logits(logits, yes_ids=[2], no_ids=[3])
    assert r_two["p_yes"] > r_one["p_yes"]         # summing variants raises p_yes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mag_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag.verdict'`

- [ ] **Step 3: Write the implementation**

```python
# src/mag/verdict.py
"""y^M — the base model's own yes/no verdict (MAG Eq. 1):
    y^M(p) = 1[ Pr(yes | Q||p) > Pr(no | Q||p) ]
computed from the first-generated-token logits over Q_TRUTH + p + Q_SUFFIX.
"""
import numpy as np
import torch

from .config import YES_VARIANTS, NO_VARIANTS


def first_token_ids(tokenizer, variants):
    ids = []
    for s in variants:
        enc = tokenizer(s, add_special_tokens=False)["input_ids"]
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


def verdict_from_logits(logits_row, yes_ids, no_ids):
    x = np.asarray(logits_row, dtype=np.float64)
    x = x - x.max()                    # numerical stability
    probs = np.exp(x); probs /= probs.sum()
    p_yes = float(probs[yes_ids].sum())
    p_no = float(probs[no_ids].sum())
    return {"ymL": int(p_yes > p_no), "margin": p_yes - p_no,
            "conf": max(p_yes, p_no), "p_yes": p_yes, "p_no": p_no}


def compute_verdicts(model, tokenizer, prompts, device, batch_size=16):
    """prompts are full strings (Q_TRUTH + statement + Q_SUFFIX). Returns arrays of y^M/margin/conf.
    Uses right padding + attention mask so the final real-token logits are read per row."""
    yes_ids = first_token_ids(tokenizer, YES_VARIANTS)
    no_ids = first_token_ids(tokenizer, NO_VARIANTS)
    tokenizer.padding_side = "right"
    ym, marg, conf = [], [], []
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s:s + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=96).to(device)
            logits = model(**enc).logits                     # (B, seq, vocab)
            am = enc["attention_mask"]
            last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
            for b in range(len(batch)):
                row = logits[b, int(last[b].item()), :].float().cpu().numpy()
                r = verdict_from_logits(row, yes_ids, no_ids)
                ym.append(r["ymL"]); marg.append(r["margin"]); conf.append(r["conf"])
    return {"ymL": np.array(ym, dtype=int),
            "margin": np.array(marg, dtype=np.float64),
            "conf": np.array(conf, dtype=np.float64)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mag_verdict.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mag/verdict.py tests/test_mag_verdict.py
git commit -m "feat(mag): y^M self-verdict from first-token logits (Eq.1)"
```

---

### Task 4: Extraction → `mag_acts_<ds>.npz` (I/O)

**Files:**
- Create: `src/mag/extract.py`
- Test: none new (verified in Task 10 smoke run). Existing suite must stay green.

**Interfaces:**
- Consumes: `dct_steer_utils.load_model`, `config` (Q/E strings), `verdict.compute_verdicts`.
- Produces: `extract_mag(dataset_file, device, limit=None) -> str` (writes `mag_acts_<ds>.npz`, returns path). The npz keys are the **canonical contract** used by Tasks 5–7:
  - per-statement, all-layer `(L+1, n, d)`: `A_p, A_Qp, A_Qpv, A_verdict, A_EQp`
  - constants `(L+1, d)`: `A_Q, A_empty`
  - `(n,)`: `ymL, margin, conf, labels`
  - object `(n,)`: `statements`; scalars: `model`

**Reading detail:** for each of the six readout variants we run one forward with `output_hidden_states=True` and take the **last non-pad token** at every layer (identical index math to `extract.py`). The six input texts per statement `p` (label `y` known from CSV, `y^M` computed first):
- `A_p`: `p`
- `A_Qp`: `Q_TRUTH + p + Q_SUFFIX`
- `A_Qpv`: `Q_TRUTH + p + Q_SUFFIX + " " + ("yes" if ymL else "no")`
- `A_verdict`: `("yes" if ymL else "no")`
- `A_EQp`: `E_FEWSHOT + "Statement: " + p + Q_SUFFIX`
- constants `A_Q`: `Q_TRUTH.rstrip()`; `A_empty`: `tokenizer.bos` only (single-token empty input).

- [ ] **Step 1: Write the implementation**

```python
# src/mag/extract.py
"""extract.py — MAG forward passes -> mag_acts_<ds>.npz (all-layer last-token readouts for the
8 operators' ingredients + y^M). Run once per dataset. gemma-2-2b base, fp32.

    python -m mag.extract cities.csv --device mps
    python -m mag.extract cities.csv --limit 20 --device cpu   # smoke
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ on path
import dct_steer_utils as su
from mag.config import Q_TRUTH, Q_SUFFIX, E_FEWSHOT
from mag.verdict import compute_verdicts

DATASET_DIR = "got_datasets"
BATCH = 16
MAX_LENGTH = 96


def _readouts(model, tokenizer, texts, device):
    """Last non-pad-token hidden state at every layer for each text. Returns (L+1, n, d)."""
    tokenizer.padding_side = "right"
    per = []
    with torch.no_grad():
        for s in range(0, len(texts), BATCH):
            batch = texts[s:s + BATCH]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LENGTH).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states  # tuple (L+1) of (B,seq,d)
            am = enc["attention_mask"]
            last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
            for b in range(len(batch)):
                idx = int(last[b].item())
                per.append(torch.stack([h[b, idx, :] for h in hs]).float().cpu().numpy())
            print(f"    {min(s + BATCH, len(texts))}/{len(texts)}", flush=True)
    arr = np.stack(per, axis=0)               # (n, L+1, d)
    return np.transpose(arr, (1, 0, 2)).astype(np.float32)  # (L+1, n, d)


def _const_readout(model, tokenizer, text, device):
    """Single-text readout -> (L+1, d)."""
    return _readouts(model, tokenizer, [text], device)[:, 0, :]


def extract_mag(dataset_file, device, limit=None):
    ds = dataset_file.replace(".csv", "")
    df = pd.read_csv(f"{DATASET_DIR}/{dataset_file}")
    statements = df["statement"].astype(str).tolist()
    labels = df["label"].to_numpy().astype(int)
    if limit is not None:
        statements, labels = statements[:limit], labels[:limit]
    n = len(statements)
    print(f"[mag/extract] {ds}: {n} statements on {device}", flush=True)

    tok, model, dev = su.load_model(device)

    prefixed = [Q_TRUTH + p + Q_SUFFIX for p in statements]
    print("  y^M ...", flush=True)
    verd = compute_verdicts(model, tok, prefixed, dev)
    yesno = ["yes" if v else "no" for v in verd["ymL"]]

    print("  A_p ...", flush=True);        A_p = _readouts(model, tok, statements, dev)
    print("  A_Qp ...", flush=True);       A_Qp = _readouts(model, tok, prefixed, dev)
    print("  A_Qpv ...", flush=True)
    A_Qpv = _readouts(model, tok, [pf + " " + a for pf, a in zip(prefixed, yesno)], dev)
    print("  A_verdict ...", flush=True);  A_verd = _readouts(model, tok, yesno, dev)
    print("  A_EQp ...", flush=True)
    A_EQp = _readouts(model, tok, [E_FEWSHOT + "Statement: " + p + Q_SUFFIX for p in statements], dev)

    print("  constants ...", flush=True)
    A_Q = _const_readout(model, tok, Q_TRUTH.rstrip(), dev)
    A_empty = _const_readout(model, tok, tok.bos_token or tok.eos_token, dev)

    out = f"mag_acts_{ds}.npz"
    np.savez_compressed(out, A_p=A_p, A_Qp=A_Qp, A_Qpv=A_Qpv, A_verdict=A_verd, A_EQp=A_EQp,
                        A_Q=A_Q, A_empty=A_empty,
                        ymL=verd["ymL"], margin=verd["margin"], conf=verd["conf"],
                        labels=labels, statements=np.array(statements, dtype=object),
                        model=su.MODEL_NAME)
    agree = float((verd["ymL"] == labels).mean())
    print(f"[mag/extract] wrote {out}  shape A_Qp={A_Qp.shape}  agree(y^M,gold)={agree:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    extract_mag(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports cleanly (no model load)**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import mag.extract; print('ok', mag.extract.MAX_LENGTH)"`
Expected: `ok 96`

- [ ] **Step 3: Confirm the existing suite is still green**

Run: `pytest -q`
Expected: all pass (no regressions)

- [ ] **Step 4: Commit**

```bash
git add src/mag/extract.py
git commit -m "feat(mag): extraction -> mag_acts_<ds>.npz (operator ingredients + y^M)"
```

---

### Task 5: Directions — `v_Q`, `u_Q`, calibration constant

**Files:**
- Create: `src/mag/directions.py`
- Test: `tests/test_mag_directions.py`

**Interfaces:**
- Consumes: `funnel_utils.unit`, `operators.operator_features`, `config.OPERATOR_NAMES`.
- Produces (pure):
  - `v_Q(A_Qp_L, A_p_L) -> np.ndarray` — mean prefix shift `(d,)`, raw (not unit).
  - `class_mean_diff(feats, labels) -> np.ndarray` — `mean(feats[labels==0]) - mean(feats[labels==1])` `(d,)` (i.e. `v^- - v^+`).
  - `a_prefix_norm(A_Qp_L) -> float` — `mean_p ‖A_Qp_L‖`.
  - `build_directions(cache, layer, labels_gold, labels_yM) -> dict` — assembles `v_Q`, `u_Q_yM`, `u_Q_gold` (from `InputDelta` features), `a_prefix_norm`, and the funnel cosines when `mean_diff`/`grad`/`dct_V` are passed. (Impure wiring around the pure pieces; tested via the pure pieces + a small synthetic cache.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mag_directions.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.directions import v_Q, class_mean_diff, a_prefix_norm


def test_vq_is_mean_of_shifts():
    A_p = np.array([[0.0, 0.0], [1.0, 1.0]])
    A_Qp = np.array([[1.0, 0.0], [3.0, 1.0]])
    # shifts: [1,0] and [2,0] -> mean [1.5, 0]
    assert np.allclose(v_Q(A_Qp, A_p), [1.5, 0.0])


def test_class_mean_diff_is_neg_minus_pos():
    feats = np.array([[2.0, 0.0],    # label 0
                      [4.0, 0.0],    # label 0
                      [0.0, 0.0]])   # label 1
    labels = np.array([0, 0, 1])
    # mean(neg)=[3,0], mean(pos)=[0,0] -> [3,0]
    assert np.allclose(class_mean_diff(feats, labels), [3.0, 0.0])


def test_a_prefix_norm_positive_and_is_mean_row_norm():
    A_Qp = np.array([[3.0, 4.0], [0.0, 0.0]])   # norms 5 and 0 -> mean 2.5
    assert abs(a_prefix_norm(A_Qp) - 2.5) < 1e-9


def test_class_mean_diff_handles_missing_class():
    feats = np.array([[1.0, 1.0], [2.0, 2.0]])
    labels = np.array([0, 0])                    # no positives
    out = class_mean_diff(feats, labels)
    assert out.shape == (2,) and np.all(np.isfinite(out))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mag_directions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag.directions'`

- [ ] **Step 3: Write the implementation**

```python
# src/mag/directions.py
"""MAG directions: v_Q (Eq.2 mean prefix shift), u_Q (class-mean contrast v^- - v^+),
and A_prefix_norm (the alpha(tau) calibration constant). Plus assembly into mag_dir_<ds>.npz."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funnel_utils import unit
from mag.operators import operator_features


def v_Q(A_Qp_L, A_p_L):
    """Eq. 2 — average prefix-induced shift at a layer. (n,d),(n,d) -> (d,)."""
    return (np.asarray(A_Qp_L, np.float64) - np.asarray(A_p_L, np.float64)).mean(axis=0)


def class_mean_diff(feats, labels):
    """u_Q = v^- - v^+  (negative class mean minus positive class mean). (n,d),(n,) -> (d,)."""
    feats = np.asarray(feats, np.float64); labels = np.asarray(labels).astype(int)
    d = feats.shape[1]
    neg = feats[labels == 0].mean(axis=0) if (labels == 0).any() else np.zeros(d)
    pos = feats[labels == 1].mean(axis=0) if (labels == 1).any() else np.zeros(d)
    return neg - pos


def a_prefix_norm(A_Qp_L):
    """mean_p ||A(Q||p)[L]|| — the calibration magnitude for alpha(tau)."""
    return float(np.linalg.norm(np.asarray(A_Qp_L, np.float64), axis=1).mean())


def build_directions(cache, layer, labels_gold, labels_yM,
                     mean_diff=None, grad=None, dct_V_top=None):
    """Assemble the MAG directions at `layer` and their cosines to the funnel directions.
    Steering direction is built from the canonical MAG feature (InputDelta)."""
    A_p_L = np.asarray(cache["A_p"][layer], np.float64)
    A_Qp_L = np.asarray(cache["A_Qp"][layer], np.float64)
    feat = operator_features("InputDelta", cache, layer)     # (n,d)

    v = v_Q(A_Qp_L, A_p_L)
    u_gold = class_mean_diff(feat, labels_gold)
    u_yM = class_mean_diff(feat, labels_yM)
    apn = a_prefix_norm(A_Qp_L)

    out = {"v_Q": v.astype(np.float32), "v_Q_unit": unit(v).astype(np.float32),
           "u_Q_gold": u_gold.astype(np.float32), "u_Q_gold_unit": unit(u_gold).astype(np.float32),
           "u_Q_yM": u_yM.astype(np.float32), "u_Q_yM_unit": unit(u_yM).astype(np.float32),
           "A_prefix_norm": np.float32(apn), "layer": np.int64(layer)}

    def cos(a, b):
        return float(unit(a) @ unit(b)) if b is not None else float("nan")

    out["cos_vQ_mean_diff"] = np.float32(cos(v, mean_diff))
    out["cos_uGold_mean_diff"] = np.float32(cos(u_gold, mean_diff))
    out["cos_uGold_grad"] = np.float32(cos(u_gold, grad))
    out["cos_uGold_dctV"] = np.float32(cos(u_gold, dct_V_top))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mag_directions.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mag/directions.py tests/test_mag_directions.py
git commit -m "feat(mag): directions — v_Q, u_Q (both labels), A_prefix_norm + funnel cosines"
```

---

### Task 6: Probes E1–E3 + transfer + reconstruction error (pure)

**Files:**
- Create: `src/mag/probes.py`
- Test: `tests/test_mag_probes.py`

**Interfaces:**
- Consumes: `operators.operator_features`/`all_operator_features`, `config.OPERATOR_NAMES`, `funnel_utils` (`load_dct`, `top_k_by_potency`, `unit`).
- Produces (pure/testable):
  - `reconstruction_error(shifts, v) -> float` — Eq. 3: `mean‖shift - v‖ / mean‖shift‖`.
  - `wilson_ci(k, n, conf=0.95) -> (lo, hi)`.
  - `E1_readability(cache, layer, y_gold, y_yM, dct=None) -> list[dict]` rows `{operator,target,acc,roc,n}` (+ DCT-top-k and random-k baseline rows).
  - `E2_disagreement(feat_mag, feat_raw, y_gold, y_yM) -> list[dict]` rows `{feature,n_disagree,match_yM_rate,ci_lo,ci_hi}`.
  - `E3_linearity(cache, layer, extra_dirs) -> list[dict]` rows `{direction,mode,eps_Q,cos}`.
  - `transfer_rank(feats_by_ds, labels_by_ds) -> dict` `{top1, spearman, rows:[...]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mag_probes.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.probes import reconstruction_error, wilson_ci, E1_readability, E2_disagreement


def _cache(L=3, n=40, d=6, seed=1):
    rng = np.random.default_rng(seed)
    # make InputDelta linearly separable by gold so E1 acc > 0.5
    y = np.array([0, 1] * (n // 2))
    A_p = rng.standard_normal((L, n, d))
    A_Qp = A_p.copy()
    A_Qp[:, :, 0] += (y * 4.0 - 2.0)          # class signal in dim 0 of the shift
    per = lambda: rng.standard_normal((L, n, d))
    const = lambda: rng.standard_normal((L, d))
    c = {"A_p": A_p, "A_Qp": A_Qp, "A_Qpv": per(), "A_verdict": per(),
         "A_EQp": per(), "A_Q": const(), "A_empty": const()}
    return c, y


def test_reconstruction_error_zero_when_perfect():
    shifts = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert reconstruction_error(shifts, np.array([1.0, 0.0])) < 1e-12


def test_reconstruction_error_one_when_v_zero():
    shifts = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert abs(reconstruction_error(shifts, np.zeros(2)) - 1.0) < 1e-12


def test_wilson_ci_brackets_point():
    lo, hi = wilson_ci(7, 10)
    assert 0.0 <= lo < 0.7 < hi <= 1.0


def test_E1_returns_expected_keys_and_beats_half_on_signal():
    c, y = _cache()
    rows = E1_readability(c, layer=1, y_gold=y, y_yM=y, dct=None)
    assert rows and set(rows[0]) == {"operator", "target", "acc", "roc", "n"}
    idel = [r for r in rows if r["operator"] == "InputDelta" and r["target"] == "gold"]
    assert idel and idel[0]["acc"] > 0.6      # planted signal is decodable


def test_E2_match_rate_in_unit_interval():
    c, y = _cache()
    from mag.operators import operator_features
    feat = operator_features("InputDelta", c, 1)
    y_yM = y.copy(); y_yM[:5] = 1 - y_yM[:5]   # a few disagreements
    rows = E2_disagreement(feat, operator_features("Direct", c, 1), y, y_yM)
    for r in rows:
        assert 0.0 <= r["match_yM_rate"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mag_probes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag.probes'`

- [ ] **Step 3: Write the implementation**

```python
# src/mag/probes.py
"""MAG probes E1 (readability), E2 (disagreement), E3 (linearity/reconstruction), and the
§4 transfer ranking. Pure functions over cached readouts so they unit-test without gemma."""
import math
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funnel_utils import unit, load_dct, top_k_by_potency
from mag.operators import operator_features, all_operator_features
from mag.config import OPERATOR_NAMES

SEED = 42


def reconstruction_error(shifts, v):
    """Eq. 3 — mean||shift - v|| / mean||shift||. shifts (n,d), v (d,)."""
    shifts = np.asarray(shifts, np.float64); v = np.asarray(v, np.float64)
    denom = np.linalg.norm(shifts, axis=1).mean()
    if denom == 0:
        return 0.0
    num = np.linalg.norm(shifts - v, axis=1).mean()
    return float(num / denom)


def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return 0.0, 1.0
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _cv_acc_roc(X, y):
    """5-fold stratified CV mean accuracy + ROC-AUC on standardized logistic features."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    Xs = StandardScaler().fit_transform(np.asarray(X, np.float64))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000)
    pred = cross_val_predict(clf, Xs, y, cv=skf)
    try:
        proba = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y, proba))
    except Exception:
        roc = float("nan")
    return float((pred == y).mean()), roc


def E1_readability(cache, layer, y_gold, y_yM, dct=None):
    rows = []
    for name in OPERATOR_NAMES:
        X = operator_features(name, cache, layer)
        for tgt, y in (("gold", y_gold), ("yM", y_yM)):
            acc, roc = _cv_acc_roc(X, y)
            rows.append({"operator": name, "target": tgt, "acc": acc, "roc": roc, "n": len(y)})
    # DCT-top-k and random-k baselines on raw A(p)
    A_p = operator_features("Direct", cache, layer)
    if dct is not None:
        V, U, _ = dct
        for k in (10, 50):
            idx = top_k_by_potency(V, U, k)
            Xk = A_p @ V[:, idx]
            for tgt, y in (("gold", y_gold), ("yM", y_yM)):
                acc, roc = _cv_acc_roc(Xk, y)
                rows.append({"operator": f"DCT_top{k}", "target": tgt, "acc": acc, "roc": roc, "n": len(y)})
    rng = np.random.default_rng(SEED)
    Rk = A_p @ rng.standard_normal((A_p.shape[1], 10))
    for tgt, y in (("gold", y_gold), ("yM", y_yM)):
        acc, roc = _cv_acc_roc(Rk, y)
        rows.append({"operator": "random_10", "target": tgt, "acc": acc, "roc": roc, "n": len(y)})
    return rows


def E2_disagreement(feat_mag, feat_raw, y_gold, y_yM):
    """Fit on the agree-set (y^M==gold), predict y^M on the disagreement set D; report the
    fraction of D where the classifier sides with the MODEL (y^M) not the label."""
    y_gold = np.asarray(y_gold).astype(int); y_yM = np.asarray(y_yM).astype(int)
    agree = y_gold == y_yM
    D = ~agree
    rows = []
    for tag, feat in (("mag", feat_mag), ("raw", feat_raw)):
        n_dis = int(D.sum())
        if n_dis == 0 or len(np.unique(y_yM[agree])) < 2:
            rows.append({"feature": tag, "n_disagree": n_dis, "match_yM_rate": float("nan"),
                         "ci_lo": float("nan"), "ci_hi": float("nan")})
            continue
        sc = StandardScaler().fit(feat[agree])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(feat[agree]), y_yM[agree])
        pred = clf.predict(sc.transform(feat[D]))
        k = int((pred == y_yM[D]).sum())
        lo, hi = wilson_ci(k, n_dis)
        rows.append({"feature": tag, "n_disagree": n_dis, "match_yM_rate": k / n_dis,
                     "ci_lo": lo, "ci_hi": hi})
    return rows


def E3_linearity(cache, layer, extra_dirs=None):
    """eps_Q for v_Q (final-layer mode) and, for comparison, for each direction in extra_dirs
    (e.g. {'mean_diff':vec,'dct':vec}). cos = mean cosine of per-prompt shift to the direction."""
    A_p = operator_features("Direct", cache, layer)
    A_Qp = operator_features("Prefixed", cache, layer)
    shifts = A_Qp - A_p
    v = shifts.mean(axis=0)
    rows = []
    def cos_to(d):
        d = unit(np.asarray(d, np.float64))
        s = shifts / (np.linalg.norm(shifts, axis=1, keepdims=True) + 1e-8)
        return float((s @ d).mean())
    rows.append({"direction": "v_Q", "mode": "final",
                 "eps_Q": reconstruction_error(shifts, v), "cos": cos_to(v)})
    for name, d in (extra_dirs or {}).items():
        # scale the comparison direction to v_Q's magnitude so eps_Q is comparable
        d = np.asarray(d, np.float64)
        d_scaled = unit(d) * np.linalg.norm(v)
        rows.append({"direction": name, "mode": "final",
                     "eps_Q": reconstruction_error(shifts, d_scaled), "cos": cos_to(d)})
    return rows


def transfer_rank(feats_by_ds, labels_by_ds):
    """Leave-one-out: for each target T, rank candidates C by centroid-cosine and by realized
    transfer accuracy (train probe on C, test on T). Report Top-1 match and Spearman rho."""
    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score
    names = list(feats_by_ds)
    rows, geom_all, real_all = [], [], []
    top1 = 0
    for T in names:
        cands = [c for c in names if c != T]
        realized, geom = {}, {}
        cent_T = feats_by_ds[T].mean(axis=0)
        for C in cands:
            sc = StandardScaler().fit(feats_by_ds[C])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(feats_by_ds[C]), labels_by_ds[C])
            realized[C] = accuracy_score(labels_by_ds[T], clf.predict(sc.transform(feats_by_ds[T])))
            geom[C] = float(unit(feats_by_ds[C].mean(axis=0)) @ unit(cent_T))
        best_real = max(realized, key=realized.get)
        best_geom = max(geom, key=geom.get)
        top1 += int(best_real == best_geom)
        for C in cands:
            rows.append({"target": T, "candidate": C, "realized_delta": realized[C],
                         "geom_score": geom[C]})
            geom_all.append(geom[C]); real_all.append(realized[C])
    rho = float(spearmanr(geom_all, real_all).correlation) if len(geom_all) > 2 else float("nan")
    return {"top1": top1 / len(names), "spearman": rho, "rows": rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mag_probes.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mag/probes.py tests/test_mag_probes.py
git commit -m "feat(mag): probes E1-E3 + transfer + reconstruction error (pure)"
```

---

### Task 7: E4 steering — α(τ) calibration + generation

**Files:**
- Create: `src/mag/steer.py`
- Test: `tests/test_mag_steer.py`

**Interfaces:**
- Consumes: `dct_steer_utils` (`Steerer`, `generate`, `load_model`), `funnel_utils.unit`, `config.TAUS`.
- Produces:
  - `alpha(tau, a_prefix_norm) -> float` = `tau * a_prefix_norm`.
  - `injected_vector(tau, unit_dir, a_prefix_norm) -> np.ndarray` = `alpha(tau) * unit_dir`.
  - `run_e4(ds, directions, device, factual_prompts, yesno_prompts) -> tuple[str,str]` (writes `mag_steer_<ds>.csv` schema `direction,scale,prompt,completion` and `mag_verdict_flips_<ds>.csv` schema `direction,tau,flip_rate`; returns both paths). Impure; verified in smoke run.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mag_steer.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.steer import alpha, injected_vector


def test_alpha_zero_gives_zero():
    assert alpha(0.0, 12.3) == 0.0


def test_alpha_linear_in_tau():
    assert abs(alpha(0.3, 10.0) - 3.0) < 1e-12
    assert abs(alpha(-1.0, 10.0) + 10.0) < 1e-12


def test_injected_vector_zero_when_tau_zero():
    u = np.array([0.6, 0.8])            # unit
    assert np.allclose(injected_vector(0.0, u, 10.0), [0.0, 0.0])


def test_injected_vector_norm_is_calibrated():
    u = np.array([0.6, 0.8])            # ||u|| = 1
    v = injected_vector(0.3, u, 10.0)
    assert abs(np.linalg.norm(v) - 3.0) < 1e-9      # |tau| * A_prefix_norm


def test_injected_vector_sign_flips_with_tau():
    u = np.array([1.0, 0.0])
    assert np.allclose(injected_vector(-0.3, u, 10.0), -injected_vector(0.3, u, 10.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mag_steer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag.steer'`

- [ ] **Step 3: Write the implementation**

```python
# src/mag/steer.py
"""E4 — calibrated steering. Inject alpha(tau)*unit(d) at a direction's native layer via the
existing Steerer, on both a matched-format yes/no set (verdict flips) and the free-form factual
set (OLMo-judged). CSV schemas match judge_results.run_steer so scoring is unchanged.

    python -m mag.steer --dataset cities --device mps
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dct_steer_utils as su
from funnel_utils import unit
from mag.config import TAUS, Q_TRUTH, Q_SUFFIX

# free-form factual set (same 32 stems as the corrected steer rerun) — imported to stay in sync
from steer_supervised import FACTUAL_PROMPTS

# 24 neutral factual statements for the matched-format yes/no flip test (true statements)
YESNO_STATEMENTS = [
    "The capital of Japan is Tokyo.", "Paris is the capital of France.",
    "Water is made of hydrogen and oxygen.", "The Earth orbits the Sun.",
    "Two plus two equals four.", "The Pacific is the largest ocean.",
    "Mount Everest is the tallest mountain.", "The sun rises in the east.",
    "Gold's chemical symbol is Au.", "Humans breathe out carbon dioxide.",
    "The heart pumps blood.", "Ten minus four equals six.",
    "The freezing point of water is zero Celsius.", "Rome is in Italy.",
    "The square root of nine is three.", "Plants absorb carbon dioxide.",
    "Sydney is in Australia.", "The opposite of hot is cold.",
    "A week has seven days.", "George Washington was the first US president.",
    "Cairo is in Egypt.", "Mercury is closest to the Sun.",
    "The sky is blue.", "Shakespeare wrote Romeo and Juliet.",
]


def alpha(tau, a_prefix_norm):
    return float(tau) * float(a_prefix_norm)


def injected_vector(tau, unit_dir, a_prefix_norm):
    return alpha(tau, a_prefix_norm) * np.asarray(unit_dir, np.float64)


def _yes_no_answer(text):
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "?"


def run_e4(ds, directions, device):
    """directions: list of dicts {name, unit_dir (d,), layer, a_prefix_norm}."""
    tok, model, dev = su.load_model(device)
    steer_rows = [("direction", "scale", "prompt", "completion")]
    flip_rows = [("direction", "tau", "flip_rate")]

    for d in directions:
        uvec = unit(np.asarray(d["unit_dir"], np.float64))
        with su.Steerer(model, int(d["layer"])) as st:
            # baseline (tau=0) yes/no answers for the flip comparison
            base = {}
            st.set(None)
            for s in YESNO_STATEMENTS:
                base[s] = _yes_no_answer(su.generate(model, tok, Q_TRUTH + s + Q_SUFFIX, 3))
            for tau in TAUS:
                vec = torch.tensor(injected_vector(tau, uvec, d["a_prefix_norm"]), dtype=torch.float32)
                st.set(None if tau == 0 else vec)
                # free-form factual (judged)
                for p in FACTUAL_PROMPTS:
                    c = su.generate(model, tok, p, 8)
                    steer_rows.append((d["name"], tau, p, c))
                # matched-format yes/no flips vs baseline
                flips = 0
                for s in YESNO_STATEMENTS:
                    a = _yes_no_answer(su.generate(model, tok, Q_TRUTH + s + Q_SUFFIX, 3))
                    flips += int(a != base[s] and a != "?")
                flip_rows.append((d["name"], tau, flips / len(YESNO_STATEMENTS)))
                print(f"  {d['name']} tau={tau:+.1f} done", flush=True)

    sp = f"mag_steer_{ds}.csv"
    with open(sp, "w", newline="") as f:
        csv.writer(f).writerows(steer_rows)
    fp = f"mag_verdict_flips_{ds}.csv"
    with open(fp, "w", newline="") as f:
        csv.writer(f).writerows(flip_rows)
    print(f"[mag/steer] wrote {sp} and {fp}")
    return sp, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    ds = a.dataset
    md = np.load(f"mag_dir_{ds}.npz")
    directions = [
        {"name": "mag_u_gold", "unit_dir": md["u_Q_gold_unit"],
         "layer": int(md["layer"]), "a_prefix_norm": float(md["A_prefix_norm"])},
        {"name": "mag_u_yM", "unit_dir": md["u_Q_yM_unit"],
         "layer": int(md["layer"]), "a_prefix_norm": float(md["A_prefix_norm"])},
    ]
    # supervised directions at the same layer, calibrated with the same A_prefix_norm
    if os.path.exists(f"truth_dir_{ds}.npz"):
        td = np.load(f"truth_dir_{ds}.npz")
        for nm in ("mean_diff", "grad"):
            directions.append({"name": f"sup_{nm}", "unit_dir": unit(td[nm]),
                               "layer": int(td["layer"]), "a_prefix_norm": float(md["A_prefix_norm"])})
    run_e4(ds, directions, a.device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mag_steer.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mag/steer.py tests/test_mag_steer.py
git commit -m "feat(mag): E4 calibrated steering (alpha(tau)) + verdict-flip counting"
```

---

### Task 8: `run_mag.py` CLI — directions + probes from cached acts

**Files:**
- Create: `src/run_mag.py`
- Test: none new (smoke in Task 10).

**Interfaces:**
- Consumes: everything in `mag/` + `funnel_utils` (`resolve_layer`, `load_dct`, `mean_diff_dir`, `grad_dir`, `load_acts`), `mag.directions.build_directions`, `mag.probes.*`.
- Produces: CLI `python -m run_mag --dataset <ds> --probe {directions,e1,e2,e3,transfer,all} [--layer N] [--device ...]`. Writes `mag_dir_<ds>.npz`, `mag_readability_<ds>.csv`, `mag_disagreement_<ds>.csv`, `mag_linearity_<ds>.csv`, `mag_transfer.csv`.

- [ ] **Step 1: Write the implementation**

```python
# src/run_mag.py
"""run_mag.py — drive the MAG probes from cached mag_acts_<ds>.npz.

    python src/run_mag.py --dataset cities --probe directions
    python src/run_mag.py --dataset cities --probe all
    python src/run_mag.py --probe transfer            # across all 4 datasets
"""
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import funnel_utils as fu
from mag.config import DATASETS
from mag.directions import build_directions
from mag import probes
from mag.operators import operator_features


def _load(ds):
    return np.load(f"mag_acts_{ds}.npz", allow_pickle=True)


def _write_csv(path, rows):
    if not rows:
        print(f"[run_mag] no rows for {path}"); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[run_mag] wrote {path}")


def do_directions(ds, layer):
    c = _load(ds)
    y_gold = c["labels"].astype(int); y_yM = c["ymL"].astype(int)
    mean_diff = grad = dctV = None
    if os.path.exists(f"acts_{ds}.npz"):
        X, y = fu.load_acts(ds, layer)
        mean_diff = fu.mean_diff_dir(X, y); grad = fu.grad_dir(X, y)
    if os.path.exists(f"dct_V_{ds}.pt"):
        V, U, _ = fu.load_dct(ds)
        dctV = V[:, fu.top_k_by_potency(V, U, 1)[0]]
    out = build_directions(c, layer, y_gold, y_yM, mean_diff, grad, dctV)
    np.savez(f"mag_dir_{ds}.npz", **out)
    print(f"[run_mag] wrote mag_dir_{ds}.npz  layer={layer}  "
          f"cos(u_gold,mean_diff)={float(out['cos_uGold_mean_diff']):+.3f}")


def do_e1(ds, layer):
    c = _load(ds)
    dct = fu.load_dct(ds) if os.path.exists(f"dct_V_{ds}.pt") else None
    rows = probes.E1_readability(c, layer, c["labels"].astype(int), c["ymL"].astype(int), dct)
    _write_csv(f"mag_readability_{ds}.csv", rows)


def do_e2(ds, layer):
    c = _load(ds)
    feat_mag = operator_features("InputDelta", c, layer)
    feat_raw = operator_features("Direct", c, layer)
    rows = probes.E2_disagreement(feat_mag, feat_raw, c["labels"].astype(int), c["ymL"].astype(int))
    _write_csv(f"mag_disagreement_{ds}.csv", rows)


def do_e3(ds, layer):
    c = _load(ds)
    extra = {}
    if os.path.exists(f"truth_dir_{ds}.npz"):
        td = np.load(f"truth_dir_{ds}.npz"); extra["mean_diff"] = td["mean_diff"]
    if os.path.exists(f"dct_V_{ds}.pt"):
        V, U, _ = fu.load_dct(ds); extra["dct"] = V[:, fu.top_k_by_potency(V, U, 1)[0]]
    rows = probes.E3_linearity(c, layer, extra)
    _write_csv(f"mag_linearity_{ds}.csv", rows)


def do_transfer():
    feats, labels = {}, {}
    for ds in DATASETS:
        if not os.path.exists(f"mag_acts_{ds}.npz"):
            continue
        c = _load(ds); layer = fu.BEST_LAYER.get(ds, 11)
        feats[ds] = operator_features("InputDelta", c, layer)
        labels[ds] = c["labels"].astype(int)
    if len(feats) < 2:
        print("[run_mag] transfer needs >=2 datasets extracted"); return
    res = probes.transfer_rank(feats, labels)
    _write_csv("mag_transfer.csv", res["rows"])
    print(f"[run_mag] transfer Top-1={res['top1']:.2f}  Spearman={res['spearman']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--probe", required=True,
                    choices=["directions", "e1", "e2", "e3", "transfer", "all"])
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    if a.probe == "transfer":
        do_transfer(); return
    ds = a.dataset
    if ds is None:
        raise SystemExit("--dataset required for this probe")
    # Lead at the truth-peak layer (matches DCT/probe results); --layer overrides (e.g. 26 = final).
    layer = a.layer if a.layer is not None else fu.BEST_LAYER.get(ds, 11)
    if a.probe in ("directions", "all"):
        do_directions(ds, layer)
    if a.probe in ("e1", "all"):
        do_e1(ds, layer)
    if a.probe in ("e2", "all"):
        do_e2(ds, layer)
    if a.probe in ("e3", "all"):
        do_e3(ds, layer)
    if a.probe == "all":
        do_transfer()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports and shows help**

Run: `.venv/bin/python src/run_mag.py --help`
Expected: usage text listing `--probe {directions,e1,e2,e3,transfer,all}`

- [ ] **Step 3: Commit**

```bash
git add src/run_mag.py
git commit -m "feat(mag): run_mag.py CLI — directions + E1/E2/E3 + transfer from cached acts"
```

---

### Task 9: `viz_mag.py` — one plot per probe

**Files:**
- Create: `src/viz_mag.py`
- Test: none new (smoke in Task 10).

**Interfaces:**
- Consumes: the CSVs from Task 8 + `judge_mag_steer_<ds>.csv` (from E4 judging).
- Produces: `python src/viz_mag.py --dataset <ds>` writing `plot_mag_readability_<ds>.png`, `plot_mag_linearity_<ds>.png`, `plot_mag_flips_<ds>.png` (guards missing CSVs like `viz_spectrum`).

- [ ] **Step 1: Write the implementation**

```python
# src/viz_mag.py
"""viz_mag.py — plots for the MAG battery (mirrors viz_steer/viz_funnel). Guards missing CSVs.

    python src/viz_mag.py --dataset cities
"""
import argparse
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def plot_readability(ds):
    rows = _read(f"mag_readability_{ds}.csv")
    if not rows:
        return
    gold = [r for r in rows if r["target"] == "gold"]
    ops = [r["operator"] for r in gold]
    acc = [float(r["acc"]) for r in gold]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(ops)), acc, color="#4477aa")
    ax.axhline(0.5, color="k", ls="--", lw=0.7, label="chance")
    ax.set_xticks(range(len(ops))); ax.set_xticklabels(ops, rotation=45, ha="right")
    ax.set_ylabel("CV accuracy (gold)"); ax.set_ylim(0, 1.0)
    ax.set_title(f"MAG E1 readability — {ds}"); ax.legend(); fig.tight_layout()
    fig.savefig(f"plot_mag_readability_{ds}.png", dpi=150)
    print(f"wrote plot_mag_readability_{ds}.png")


def plot_linearity(ds):
    rows = _read(f"mag_linearity_{ds}.csv")
    if not rows:
        return
    labels = [f'{r["direction"]}/{r["mode"]}' for r in rows]
    eps = [float(r["eps_Q"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), eps, color="#ee6677")
    ax.axhline(1.0, color="k", ls="--", lw=0.7, label="no better than not steering")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("reconstruction error eps_Q")
    ax.set_title(f"MAG E3 linearity — {ds}"); ax.legend(); fig.tight_layout()
    fig.savefig(f"plot_mag_linearity_{ds}.png", dpi=150)
    print(f"wrote plot_mag_linearity_{ds}.png")


def plot_flips(ds):
    rows = _read(f"mag_verdict_flips_{ds}.csv")
    if not rows:
        return
    dirs = sorted({r["direction"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for dn in dirs:
        sub = sorted([r for r in rows if r["direction"] == dn], key=lambda r: float(r["tau"]))
        ax.plot([float(r["tau"]) for r in sub], [float(r["flip_rate"]) for r in sub],
                "o-", label=dn)
    ax.set_xlabel("tau (calibrated)"); ax.set_ylabel("verdict-flip rate"); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"MAG E4 verdict flips — {ds}"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"plot_mag_flips_{ds}.png", dpi=150)
    print(f"wrote plot_mag_flips_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_readability(a.dataset); plot_linearity(a.dataset); plot_flips(a.dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports**

Run: `.venv/bin/python src/viz_mag.py --help`
Expected: usage text with `--dataset`

- [ ] **Step 3: Commit**

```bash
git add src/viz_mag.py
git commit -m "feat(mag): viz_mag.py — readability / linearity / flip plots"
```

---

### Task 10: Wire MAG E4 into `investigate_steer.py` + gitignore + end-to-end smoke

**Files:**
- Modify: `src/investigate_steer.py` (add `--prefix`/`--datasets` so it can read `judge_mag_steer_<ds>.csv`)
- Modify: `.gitignore` (add `mag_acts_*.npz`)
- Test: extend `tests/test_investigate_steer.py` with a prefix-plumbing check.

**Interfaces:**
- Consumes: existing `investigate_steer.load`/`report`.
- Produces: `investigate_steer` gains module-level `INPUT_PREFIX` (default `"judge_steer"`) that `load(ds)` reads, plus CLI flags `--prefix` and `--datasets`. Default behavior (DCT arm) unchanged.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_investigate_steer.py
def test_load_respects_input_prefix(tmp_path, monkeypatch):
    import investigate_steer as isv
    monkeypatch.chdir(tmp_path)
    with open("judge_mag_steer_cities.csv", "w") as f:
        f.write("direction,scale,prompt,completion,verdict,reason\n"
                "mag_u_gold,-1.0,The capital of Japan is,Berlin,FALSE,x\n"
                "mag_u_gold,1.0,The capital of Japan is,Tokyo,TRUE,x\n")
    monkeypatch.setattr(isv, "INPUT_PREFIX", "judge_mag_steer")
    rows = isv.load("cities")
    assert len(rows) == 2 and rows[0]["scale"] == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_investigate_steer.py::test_load_respects_input_prefix -v`
Expected: FAIL (`AttributeError: module 'investigate_steer' has no attribute 'INPUT_PREFIX'`)

- [ ] **Step 3: Modify `investigate_steer.py`**

Add near the top (after `DATASETS = [...]`):

```python
INPUT_PREFIX = "judge_steer"   # overridable so the MAG arm can point at judge_mag_steer_<ds>.csv
```

Change `load` to use it:

```python
def load(ds):
    try:
        rows = list(csv.DictReader(open(f"{INPUT_PREFIX}_{ds}.csv")))
    except FileNotFoundError:
        return []
    for r in rows:
        r["scale"] = float(r["scale"]); r["abs"] = abs(r["scale"]); r["ds"] = ds
    return rows
```

Change `main` to accept the flags:

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="judge_steer",
                    help="CSV prefix: judge_steer (DCT arm) or judge_mag_steer (MAG arm)")
    ap.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    args = ap.parse_args()
    global INPUT_PREFIX
    INPUT_PREFIX = args.prefix
    dsets = [d for d in args.datasets.split(",") if d]
    report({ds: load(ds) for ds in dsets})
```

Rename the existing module-level `DATASETS = ["cities", "common_claim_true_false"]` to `DATASETS_DEFAULT` and update its one use in `main`. (Grep first: `grep -n "DATASETS" src/investigate_steer.py` — only the definition and `main` reference it.)

- [ ] **Step 4: Run the new test + full suite**

Run: `pytest tests/test_investigate_steer.py -q && pytest -q`
Expected: all pass (including the pre-existing investigate_steer tests)

- [ ] **Step 5: Update .gitignore**

Add the line `mag_acts_*.npz` to `.gitignore` (confirm `acts_*.npz`/`dct_*.pt` are already ignored; match that style).

- [ ] **Step 6: End-to-end smoke on 20 rows (CPU)**

Run:
```bash
.venv/bin/python -m mag.extract cities.csv --limit 20 --device cpu
.venv/bin/python src/run_mag.py --dataset cities --probe all --layer 11
.venv/bin/python src/viz_mag.py --dataset cities
```
Expected: `mag_acts_cities.npz` written (A_Qp shape ~ `(27, 20, 2304)`); `mag_dir_cities.npz`, `mag_readability_cities.csv`, `mag_disagreement_cities.csv`, `mag_linearity_cities.csv` written; `plot_mag_readability_cities.png`, `plot_mag_linearity_cities.png` written. (With n=20 some CV/AUC cells may be `nan` — acceptable for a smoke; the pipeline completing end-to-end is the check.)

- [ ] **Step 7: Commit**

```bash
git add src/investigate_steer.py tests/test_investigate_steer.py .gitignore
git commit -m "feat(mag): plumb investigate_steer prefix for MAG E4; ignore mag_acts; smoke green"
```

---

## Post-implementation (run-time, not part of this plan's code tasks)

These are the actual research runs the code enables — executed after the code lands, tracked in the spec's §12 compute plan, not as TDD tasks here:

1. Full extraction on all 4 datasets (laptop `--device mps`), then `run_mag.py --probe all` per dataset + `--probe transfer`.
2. E4 generation (`python -m mag.steer --dataset <ds> --device mps`) → judge `mag_steer_<ds>.csv` with the OLMo backend on the GH200 (writing `judge_mag_steer_<ds>.csv`), then `investigate_steer.py --prefix judge_mag_steer --datasets cities,common_claim_true_false`.
3. Write `docs/DCT_VS_MAG_ON_TRUTH.md` per spec §16 success criteria.

---

## Self-Review

**Spec coverage:**
- 8 operators → Task 2 ✓ | `y^M` Eq.1 → Task 3 ✓ | extraction/npz → Task 4 ✓ | `v_Q`/`u_Q` both labels + `A_prefix_norm` + funnel cosines → Task 5 ✓ | E1 readability (incl. DCT-top-k & random-k) → Task 6/`E1_readability` ✓ | E2 disagreement + Wilson CI → Task 6 ✓ | E3 linearity/`ε_Q` + comparison directions → Task 6 ✓ | §4 transfer (Top-1 + Spearman) → Task 6/`transfer_rank` ✓ | E4 α(τ) calibration + generation + verdict flips → Task 7 ✓ | reuse of `Steerer`/judge/`investigate_steer` → Tasks 7 & 10 ✓ | both-layer directions → `run_mag --layer` (peak default; final via `--layer 26`) ✓ | signed TAUS → Task 1 ✓ | gitignore `mag_acts` → Task 10 ✓ | plots → Task 9 ✓.
- Spec §10 injection-layer note: `steer.py` injects each direction at its native layer (`md["layer"]` peak for MAG/supervised; DCT top-V would use layer 13 — added to the E4 direction list only if `dct_V_<ds>.pt` present, deferred to run-time since DCT injection wiring already exists in the DCT arm). The head-to-head at peak is fully covered; DCT-at-13 is a run-time addition, consistent with the spec's "native layer" primary.
- Spec E3 "layer-wise secondary" mode: `E3_linearity` implements the final-layer primary + comparison rows; the layer-wise sweep is available from the all-layer `mag_acts` but is left to the run-time analysis (not a separate code path needed for the headline). Noted so it isn't mistaken for a gap.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test shows assertions. ✓

**Type consistency:** `operator_features(name, cache, layer)` signature identical across Tasks 2/5/6/8. `injected_vector(tau, unit_dir, a_prefix_norm)` identical Task 7 test + impl. `build_directions(...)` keys (`u_Q_gold_unit`, `A_prefix_norm`, `layer`) consumed verbatim in `steer.main` and `run_mag.do_directions`. CSV schema `direction,scale,prompt,completion` matches `judge_results.run_steer` reader. `INPUT_PREFIX` consumed by `load` and set by `main`/tests. ✓

One consistency fix applied inline: `run_mag.main` computes `layer` via `BEST_LAYER` (peak) by default rather than `resolve_layer` (which would return DCT source layer 13), so E1/E2/E3/directions lead at the truth-peak layer as the spec's readout decision requires; `--layer` overrides for the final-block (26) reporting pass.
