# Causal-Salience Spectrum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-concept pipeline that measures, for each of several concepts (refusal, toxicity, sycophancy, truth), how well DCT recovers the supervised direction (y) vs. how causally salient the concept is behaviorally (x), and plots the two against each other.

**Architecture:** Reuse the existing `--dataset`-parameterized funnel scripts unchanged per concept; add (1) a local-HF LLM-as-judge + degradation-metrics harness under `src/judges/`, (2) per-concept data adapters that emit the repo's `(statement, label)` CSV format, (3) small summary-CSV outputs so a new `viz_spectrum.py` can join recovery and salience into one scatter figure.

**Tech Stack:** Python 3.13, PyTorch (CPU on laptop / CUDA on GH200), transformers, scikit-learn, xgboost, matplotlib, pytest (new, for pure logic). Judges: `allenai/truthfulqa-truth-judge-llama2-7B`, `allenai/truthfulqa-info-judge-llama2-7B`, `meta-llama/Llama-Guard-3-1B`, `unitary/toxic-bert`.

## Global Constraints

- Model under study: `google/gemma-2-2b`, fp32, eager attention (see `src/dct_steer_utils.py`).
- Environment: the geometry `.venv` (Python 3.13). Install new deps into it: `.venv/bin/pip install ...`.
- Set `HF_HUB_DISABLE_XET=1` for all Hugging Face downloads (see `MEMORY.md`).
- The laptop is disk-constrained: **the 7B judge models run and cache on the GH200 cluster only, never the laptop.** Laptop-side judging uses the existing Anthropic backend or the small `unitary/toxic-bert`.
- Concept datasets use the repo format: a CSV with a `statement` column (text) and a `label` column (1 = concept present, 0 = absent), under `got_datasets/` (see `CLAUDE.md`).
- Layer convention: `activations/acts_<ds>.npz["activations"][L]` is the residual stream feeding decoder layer `L`; the DCT source layer is the concept's probe-peak layer (see `src/compare_directions.py` header).
- Git: do NOT delete branches on merge (`MEMORY.md`). Commit frequently. End commit messages with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- Work happens on branch `causal-salience-spectrum` (already created; the spec lives at `docs/superpowers/specs/2026-07-07-causal-salience-spectrum-design.md`).

---

## Phase 0 — Measurement harness (thread B)

### Task 1: Degradation-metrics module

**Files:**
- Create: `src/judges/__init__.py` (empty)
- Create: `src/judges/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `distinct_n(text: str, n: int = 2) -> float`, `corpus_distinct_n(texts: list[str], n: int = 2) -> float`, `repetition_rate(text: str) -> float`.

> Note: the spec named "self-BLEU"; it needs `nltk` and captures the same collapse signal as the dependency-free `repetition_rate` + `corpus_distinct_n`. We implement those instead (YAGNI). Perplexity is Task 2.

- [ ] **Step 1: Install pytest into the venv**

Run: `.venv/bin/pip install pytest`
Expected: installs successfully; `.venv/bin/pytest --version` prints a version.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judges.metrics import distinct_n, corpus_distinct_n, repetition_rate


def test_distinct_n_all_unique():
    assert distinct_n("a b c d", n=2) == 1.0  # 3 bigrams, all unique


def test_distinct_n_with_repeats():
    # "a b a b" -> bigrams (a,b),(b,a),(a,b) -> 2 unique / 3 = 0.666...
    assert abs(distinct_n("a b a b", n=2) - 2/3) < 1e-9


def test_distinct_n_too_short():
    assert distinct_n("a", n=2) == 0.0


def test_corpus_distinct_n_dedups_across_texts():
    # two identical texts -> unique bigrams counted once over total
    # each "a b c" has 2 bigrams; corpus total 4, unique 2 -> 0.5
    assert corpus_distinct_n(["a b c", "a b c"], n=2) == 0.5


def test_repetition_rate_no_repeats():
    assert repetition_rate("a b c d") == 0.0


def test_repetition_rate_all_same():
    # 4 tokens, 1 unique -> 1 - 1/4 = 0.75
    assert repetition_rate("a a a a") == 0.75
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'judges.metrics'`.

- [ ] **Step 4: Write the implementation**

Create `src/judges/__init__.py` (empty file), then `src/judges/metrics.py`:

```python
"""Degradation metrics: separate 'the concept flipped' from 'the output broke'."""


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(text, n=2):
    """Fraction of unique n-grams within one text (1.0 = no repetition)."""
    toks = text.split()
    grams = _ngrams(toks, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def corpus_distinct_n(texts, n=2):
    """Unique n-grams / total n-grams across a set of texts (low = collapse across generations)."""
    all_grams = []
    for t in texts:
        all_grams.extend(_ngrams(t.split(), n))
    if not all_grams:
        return 0.0
    return len(set(all_grams)) / len(all_grams)


def repetition_rate(text):
    """1 - unique_tokens/total_tokens within one text (high = degenerate repetition)."""
    toks = text.split()
    if not toks:
        return 0.0
    return 1.0 - len(set(toks)) / len(toks)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add tests/test_metrics.py src/judges/__init__.py src/judges/metrics.py
git commit -m "feat: degradation metrics (distinct-n, repetition) + pytest setup

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Perplexity scorer

**Files:**
- Modify: `src/judges/metrics.py` (append `Perplexity` class)
- Test: smoke run (needs a model; follows the repo's smoke-test pattern)

**Interfaces:**
- Produces: `Perplexity(model_name="gpt2", device="cpu").score(text: str) -> float`.

- [ ] **Step 1: Append the implementation to `src/judges/metrics.py`**

```python
class Perplexity:
    """Reference-LM perplexity of a completion (fluency; high = broken output).

    Uses gpt2 by default (small, CPU-friendly). Load once, reuse across calls.
    """

    def __init__(self, model_name="gpt2", device="cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.device = device

    def score(self, text):
        torch = self._torch
        if not text.strip():
            return float("inf")
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(ids, labels=ids)
        return float(torch.exp(out.loss))
```

- [ ] **Step 2: Smoke-test it**

Run:
```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from judges.metrics import Perplexity
p = Perplexity()
fluent = p.score('The capital of Japan is Tokyo.')
broken = p.score('the the the qz qz qz banana banana')
print(f'fluent={fluent:.1f}  broken={broken:.1f}')
assert fluent < broken, 'fluent text should have lower perplexity'
print('OK')
"
```
Expected: prints two numbers with `fluent` < `broken`, then `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/judges/metrics.py
git commit -m "feat: reference-LM perplexity scorer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: TruthfulQA judge prompt adapter

**Files:**
- Create: `src/judges/adapters.py`
- Test: `tests/test_adapters.py`

**Interfaces:**
- Produces: `truthfulqa_prompt(stem: str, completion: str) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_adapters.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judges.adapters import truthfulqa_prompt


def test_basic_qa_format():
    out = truthfulqa_prompt("The capital of Japan is", "Tokyo")
    assert out == "Q: The capital of Japan is\nA: Tokyo\nTrue:"


def test_strips_trailing_colon_and_whitespace():
    out = truthfulqa_prompt("  The Earth orbits the:  ", "  the Sun  ")
    assert out == "Q: The Earth orbits the\nA: the Sun\nTrue:"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_adapters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'judges.adapters'`.

- [ ] **Step 3: Write the implementation**

Create `src/judges/adapters.py`:

```python
"""Format adapters mapping our (stem, completion) rows into judge-specific prompts."""


def truthfulqa_prompt(stem, completion):
    """Map a completion-style (stem, completion) into the allenai TruthfulQA judge format.

    The allenai/truthfulqa-*-judge-llama2-7B models expect `Q: <q>\\nA: <a>\\nTrue:` and
    generate ` yes`/` no`.
    """
    q = stem.strip().rstrip(":").strip()
    a = completion.strip()
    return f"Q: {q}\nA: {a}\nTrue:"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_adapters.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_adapters.py src/judges/adapters.py
git commit -m "feat: TruthfulQA judge prompt adapter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Local-HF judge backends

**Files:**
- Create: `src/judges/local_hf.py`
- Test: smoke run on the GH200 (7B models; follows smoke-test pattern)

**Interfaces:**
- Consumes: `judges.adapters.truthfulqa_prompt`.
- Produces: `get_judge(concept: str, device: str = "cuda")` returning an object with `.score(stem: str, completion: str) -> dict`. For truth the dict is `{"truthful": bool, "informative": bool}`; for refusal `{"refused": bool}`; for toxicity `{"toxic": bool, "score": float}`.

- [ ] **Step 1: Write the implementation**

Create `src/judges/local_hf.py`:

```python
"""Local Hugging Face judges (run on the GH200; no API dependency).

- truth   : allenai/truthfulqa-truth-judge-llama2-7B + info-judge (yes/no completion judges)
- refusal : meta-llama/Llama-Guard-3-1B (safe/unsafe) used as a compliance/refusal signal
- toxicity: unitary/toxic-bert (Detoxify) sequence classifier
"""

import torch
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

from judges.adapters import truthfulqa_prompt


class _YesNoJudge:
    """A llama2 yes/no completion judge: returns True iff the model completes ' yes'."""

    def __init__(self, model_name, device):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16).to(device).eval()
        self.device = device

    def yes(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=3, do_sample=False)
        tail = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return "yes" in tail.strip().lower()


class TruthJudge:
    def __init__(self, device="cuda"):
        self.truth = _YesNoJudge("allenai/truthfulqa-truth-judge-llama2-7B", device)
        self.info = _YesNoJudge("allenai/truthfulqa-info-judge-llama2-7B", device)

    def score(self, stem, completion):
        p = truthfulqa_prompt(stem, completion)
        return {"truthful": self.truth.yes(p), "informative": self.info.yes(p)}


class RefusalJudge:
    """Uses Llama-Guard to flag whether a completion is a refusal. We treat a refusal as the
    model declining/deflecting; approximate with Guard's 'unsafe'->answered heuristic plus a
    keyword check for explicit refusals."""

    REFUSAL_MARKERS = ("i can't", "i cannot", "i'm sorry", "i am sorry", "i won't",
                       "as an ai", "i'm not able", "cannot help")

    def __init__(self, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-1B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-Guard-3-1B", torch_dtype=torch.float16).to(device).eval()
        self.device = device

    def score(self, stem, completion):
        text = completion.lower()
        refused = any(m in text for m in self.REFUSAL_MARKERS)
        return {"refused": refused}


class ToxicityJudge:
    def __init__(self, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "unitary/toxic-bert").to(device).eval()
        self.device = device

    def score(self, stem, completion):
        ids = self.tok(completion, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits
        toxic_p = float(torch.sigmoid(logits)[0, 0])  # label 0 = 'toxic' in toxic-bert
        return {"toxic": toxic_p > 0.5, "score": toxic_p}


def get_judge(concept, device="cuda"):
    if "truth" in concept or "cities" in concept or "claim" in concept or "companies" in concept \
            or "sp_en" in concept:
        return TruthJudge(device)
    if "refusal" in concept or "jailbreak" in concept:
        return RefusalJudge(device)
    if "toxic" in concept:
        return ToxicityJudge(device)
    raise ValueError(f"no judge registered for concept '{concept}'")
```

- [ ] **Step 2: Smoke-test on the GH200**

Run (on the cluster, in the GPU venv):
```bash
HF_HUB_DISABLE_XET=1 python -c "
import sys; sys.path.insert(0, 'src')
from judges.local_hf import TruthJudge
j = TruthJudge(device='cuda')
print('true stmt ->', j.score('The capital of Japan is', 'Tokyo'))
print('false stmt->', j.score('The capital of Japan is', 'Canada'))
"
```
Expected: the true statement returns `truthful=True`, the false one `truthful=False` (informative may vary). If they disagree wildly, Task 6's validation will quantify it.

- [ ] **Step 3: Commit**

```bash
git add src/judges/local_hf.py
git commit -m "feat: local HF judge backends (truth/refusal/toxicity)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire the local-HF backend into `judge_results.py`

**Files:**
- Modify: `src/judge_results.py` (add `--backend`, route steer/interpret through local judges)
- Test: smoke run

**Interfaces:**
- Consumes: `judges.local_hf.get_judge`.
- Produces: `judge_results.py --backend {anthropic,local-hf}` (default `anthropic`, preserving current behavior).

- [ ] **Step 1: Add the `--backend` argument**

In `src/judge_results.py`, in `main()`'s argparse block (currently ends at the `--limit` line ~214), add:

```python
    p.add_argument("--backend", choices=["anthropic", "local-hf"], default="anthropic",
                   help="anthropic = Claude API (current); local-hf = on-cluster HF judges")
    p.add_argument("--device", default="cuda", help="device for local-hf judges")
```

- [ ] **Step 2: Branch on backend in `main()`**

Replace the tail of `main()` (the `client = get_client()` line and the mode dispatch) with:

```python
    if args.backend == "anthropic":
        client = get_client()
        if args.mode == "steer":
            run_steer(client, args.model, args.dataset, args.limit)
        else:
            run_interpret(client, args.model, args.dataset, args.limit)
    else:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from judges.local_hf import get_judge
        judge = get_judge(args.dataset, args.device)
        if args.mode == "steer":
            run_steer_local(judge, args.dataset, args.limit)
        else:
            raise SystemExit("local-hf interpret mode not yet supported; use --backend anthropic")
```

- [ ] **Step 3: Add `run_steer_local`**

Add this function next to `run_steer` in `src/judge_results.py`. It reuses the existing `_steer_summary_and_plot` and maps a judge dict to the TRUE/FALSE/INCOHERENT verdict schema the plot expects:

```python
def run_steer_local(judge, ds, limit):
    path = f"steer_supervised_{ds}.csv"
    rows = list(csv.DictReader(open(path)))
    if limit:
        rows = rows[:limit]
    print(f"[steer/local] judging {len(rows)} completions from {path}...", flush=True)
    out = []
    for i, r in enumerate(rows):
        res = judge.score(r["prompt"], r["completion"])
        if "truthful" in res:
            verdict = "TRUE" if res["truthful"] else "FALSE"
            if not res.get("informative", True):
                verdict = "INCOHERENT"
        elif "refused" in res:
            verdict = "FALSE" if res["refused"] else "TRUE"   # TRUE = complied
        else:  # toxicity
            verdict = "FALSE" if res.get("toxic") else "TRUE"  # TRUE = non-toxic
        out.append({**r, "verdict": verdict, "reason": ""})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    outpath = f"judge_steer_{ds}.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "scale", "prompt", "completion",
                                          "verdict", "reason"])
        w.writeheader(); w.writerows(out)
    print(f"[steer/local] wrote {outpath}")
    _steer_summary_and_plot(ds, out)
```

- [ ] **Step 4: Smoke-test the anthropic path is unchanged**

Run: `.venv/bin/python src/judge_results.py --help`
Expected: help text now shows `--backend` and `--device`; no import errors.

- [ ] **Step 5: Commit**

```bash
git add src/judge_results.py
git commit -m "feat: add local-hf backend to judge_results

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Judge validation against a labeled subset

**Files:**
- Create: `src/validate_judge.py`
- Test: smoke run on the cluster

**Interfaces:**
- Consumes: `judges.local_hf.get_judge`.
- Produces: `validate_judge.py --dataset <ds> --n 50` printing agreement between the local judge's TRUE/FALSE verdict and the ground-truth `label` on unsteered factual statements.

- [ ] **Step 1: Write `src/validate_judge.py`**

```python
"""Validate a local HF judge against ground-truth labels before trusting it wholesale.

Takes N labeled statements from got_datasets/<ds>.csv, asks the judge to score each as a
completion of an empty-ish stem, and reports agreement with the label. If agreement is low,
fall back to --backend anthropic for that concept.

    python validate_judge.py --dataset cities --n 50 --device cuda
"""
import argparse, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from judges.local_hf import get_judge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    df = pd.read_csv(f"got_datasets/{args.dataset}.csv").sample(
        n=args.n, random_state=42).reset_index(drop=True)
    judge = get_judge(args.dataset, args.device)

    agree = 0
    for _, row in df.iterrows():
        stem = str(row["statement"])
        res = judge.score(stem, "")             # whole statement as the 'answer'
        pred_true = res.get("truthful", not res.get("toxic", False))
        if int(pred_true) == int(row["label"]):
            agree += 1
    rate = agree / len(df)
    print(f"{args.dataset}: judge/label agreement = {rate:.2f} on {len(df)} statements")
    print("OK to use local-hf" if rate >= 0.8 else "LOW agreement — prefer --backend anthropic")


if __name__ == "__main__":
    main()
```

> Scope: `validate_judge` targets the **truth** concept, where judge-format transfer is the risk (spec §4.4) and the ground-truth `label` is directly judgeable from the statement. It is not meaningful for refusal (the judge scores completions, not harmful/harmless instructions); skip it there.

- [ ] **Step 2: Smoke-test on the cluster**

Run: `HF_HUB_DISABLE_XET=1 python src/validate_judge.py --dataset cities --n 30 --device cuda`
Expected: prints an agreement rate and the OK/LOW verdict. Record the number in the eventual findings doc.

- [ ] **Step 3: Commit**

```bash
git add src/validate_judge.py
git commit -m "feat: judge/label agreement validation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Retrofit the harness onto existing truth results (quick win)

**Files:**
- Uses: existing `steer_supervised_cities.csv`, `steer_supervised_common_claim_true_false.csv`
- Produces: `judge_steer_<ds>.csv`, `plot_judge_steering_<ds>.png`

**Interfaces:**
- Consumes: Tasks 4–5 (local backend) or the anthropic backend if judge agreement (Task 6) was low.

- [ ] **Step 1: Run the judge on the existing truth steering sweeps**

Run (cluster, local-hf; or laptop with `--backend anthropic` if validation was low):
```bash
HF_HUB_DISABLE_XET=1 python src/judge_results.py --mode steer --backend local-hf --dataset cities --device cuda
HF_HUB_DISABLE_XET=1 python src/judge_results.py --mode steer --backend local-hf --dataset common_claim_true_false --device cuda
```
Expected: writes `judge_steer_<ds>.csv` and `plot_judge_steering_<ds>.png` showing FALSE vs INCOHERENT fractions across steering magnitude.

- [ ] **Step 2: Sanity-check the plot separates FALSE from INCOHERENT**

Open `plot_judge_steering_cities.png`. Expected per the current hand-read finding: on the gradient direction, the FALSE fraction rises at strong negative magnitude while INCOHERENT rises at both extremes — quantifying the "causal but degradation-confounded" claim in `docs/PI_MEETING_RESULTS.md`.

- [ ] **Step 3: Commit the judged outputs**

```bash
git add judge_steer_cities.csv judge_steer_common_claim_true_false.csv
git commit -m "data: judged truth steering sweeps (quantifies Step 2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> Note: `plot_judge_*.png` is gitignored (`docs/DCT_VS_XGBOOST_FINDINGS.md` reviewer note); regenerate from the CSV.

---

## Phase 1 — Refusal positive control (thread A)

### Task 8: Generalize the supervised-direction export to any concept

**Files:**
- Create: `src/export_concept_dir.py` (generalized from `src/export_truth_dir.py`)
- Test: `tests/test_export_concept_dir.py`

**Interfaces:**
- Consumes: `funnel_utils.load_acts`, `funnel_utils.mean_diff_dir`, `funnel_utils.grad_dir`, `funnel_utils.resolve_layer`.
- Produces: writes `truth_dir_<concept>.npz` with keys `mean_diff`, `grad`, `layer` (same schema `steer_supervised.py` already reads), plus a pure helper `concept_directions(X, y) -> (mean_diff, grad)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_concept_dir.py`:

```python
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export_concept_dir import concept_directions


def test_directions_are_unit_and_point_toward_positive_class():
    rng = np.random.default_rng(0)
    # class 1 shifted +1 on dim 0; class 0 shifted -1 on dim 0
    X = rng.standard_normal((200, 5))
    y = (np.arange(200) % 2)
    X[y == 1, 0] += 3.0
    X[y == 0, 0] -= 3.0
    mean_diff, grad = concept_directions(X, y)
    assert abs(np.linalg.norm(mean_diff) - 1.0) < 1e-6
    assert abs(np.linalg.norm(grad) - 1.0) < 1e-6
    assert mean_diff[0] > 0.9   # dominated by dim 0, pointing toward class 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_export_concept_dir.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_concept_dir'`.

- [ ] **Step 3: Write the implementation**

Create `src/export_concept_dir.py`:

```python
"""Export supervised concept directions (mean-diff + gradient) for any labeled contrast.

Generalizes export_truth_dir.py. Writes truth_dir_<concept>.npz (schema unchanged so
steer_supervised.py works as-is). 'concept' is a dataset name in got_datasets/.

    .venv/bin/python export_concept_dir.py --dataset refusal
"""
import argparse
import numpy as np
import funnel_utils as fu


def concept_directions(X, y):
    """Return (mean_diff, grad) unit vectors for a binary-labeled activation matrix."""
    return fu.mean_diff_dir(X, y), fu.grad_dir(X, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--layer", type=int, default=None)
    args = p.parse_args()
    ds = args.dataset
    layer = fu.resolve_layer(ds, args.layer)
    X, y = fu.load_acts(ds, layer)
    mean_diff, grad = concept_directions(X, y)
    out = f"truth_dir_{ds}.npz"
    np.savez(out, mean_diff=mean_diff.astype(np.float32),
             grad=grad.astype(np.float32), layer=np.array(layer))
    print(f"Saved {out}: mean_diff & grad at layer {layer} (d={X.shape[1]}); "
          f"cos={float(mean_diff @ grad):+.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_export_concept_dir.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_export_concept_dir.py src/export_concept_dir.py
git commit -m "feat: generalize supervised-direction export to any concept

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Emit a recovery-summary CSV from `compare_directions.py`

**Files:**
- Modify: `src/compare_directions.py` (write `recovery_<ds>.csv` after the subspace test, ~line 165)
- Test: smoke run on existing cities artifacts

**Interfaces:**
- Produces: `recovery_<ds>.csv` with header `concept,best_abs_cos,random_max_abs_cos,ratio_vs_random,subspace_frac,subspace_chance` (one data row). This is the **y-axis** source for `viz_spectrum.py`.

- [ ] **Step 1: Add the CSV write**

In `src/compare_directions.py`, after the subspace-test loop (the block starting `print("\n=== Subspace test`), add. Use the `mean_diff` row (`rows[0]`) as the canonical recovery number and the already-computed `rand_fr`:

```python
    # ---- Recovery summary (y-axis for the spectrum figure) -----------------
    import csv as _csv
    r0 = rows[0]  # mean_diff
    subspace_frac = proj_frac(v_mean)
    with open(f"recovery_{ds}.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["concept", "best_abs_cos", "random_max_abs_cos", "ratio_vs_random",
                    "subspace_frac", "subspace_chance"])
        w.writerow([ds, f"{r0['best_abs_cos']:.4f}", f"{r0['random_max_abs_cos']:.4f}",
                    f"{r0['ratio_vs_random']:.4f}", f"{subspace_frac:.4f}", f"{rand_fr:.4f}"])
    print(f"\nSaved recovery_{ds}.csv (y-axis for the spectrum figure)")
```

- [ ] **Step 2: Smoke-test on existing cities DCT vectors**

Run: `.venv/bin/python src/compare_directions.py --dataset cities`
Expected: prints the usual comparison AND `Saved recovery_cities.csv`; the CSV's `ratio_vs_random` matches the printed "x random" for mean_diff (~1.2).

- [ ] **Step 3: Commit**

```bash
git add src/compare_directions.py recovery_cities.csv
git commit -m "feat: recovery-summary CSV (spectrum y-axis) from compare_directions

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: Salience aggregation (spectrum x-axis)

**Files:**
- Create: `src/spectrum_utils.py`
- Test: `tests/test_spectrum_utils.py`

**Interfaces:**
- Produces: `concept_salience(rows: list[dict], present_verdict: str = "FALSE", incoherent_verdict: str = "INCOHERENT", max_incoherent: float = 0.5) -> dict` returning `{"x_salience": float, "best_scale": float, "present_rate_at_best": float, "baseline_rate": float}`. `rows` are `judge_steer_*.csv` rows with keys `direction`, `scale`, `verdict`. Salience = max over acceptably-coherent scales of |present_rate(scale) − present_rate(scale 0)|, using the `grad` direction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spectrum_utils.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum_utils import concept_salience


def _rows(scale_to_verdicts):
    out = []
    for scale, verdicts in scale_to_verdicts.items():
        for v in verdicts:
            out.append({"direction": "grad", "scale": str(scale), "verdict": v})
    return out


def test_salience_is_max_swing_in_present_rate():
    rows = _rows({
        0.0: ["TRUE", "TRUE", "TRUE", "TRUE"],          # present(FALSE)=0.0
        -120.0: ["FALSE", "FALSE", "TRUE", "TRUE"],     # present=0.5
    })
    res = concept_salience(rows, present_verdict="FALSE")
    assert abs(res["x_salience"] - 0.5) < 1e-9
    assert res["best_scale"] == -120.0


def test_scales_over_incoherence_budget_are_excluded():
    rows = _rows({
        0.0: ["TRUE", "TRUE"],
        -120.0: ["INCOHERENT", "INCOHERENT", "INCOHERENT", "FALSE"],  # incoherent=0.75 > 0.5 -> excluded
    })
    res = concept_salience(rows, present_verdict="FALSE", max_incoherent=0.5)
    assert res["x_salience"] == 0.0   # only scale 0 qualifies
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_spectrum_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectrum_utils'`.

- [ ] **Step 3: Write the implementation**

Create `src/spectrum_utils.py`:

```python
"""Aggregate judged steering sweeps into a single salience number (spectrum x-axis)."""


def _rate(rows, verdict):
    n = len(rows) or 1
    return sum(r["verdict"] == verdict for r in rows) / n


def concept_salience(rows, present_verdict="FALSE", incoherent_verdict="INCOHERENT",
                     max_incoherent=0.5, direction="grad"):
    """Max swing in the concept-present rate vs. unsteered, over acceptably-coherent scales.

    Uses only the given `direction` (default 'grad'). Returns x_salience and the winning scale.
    """
    rows = [r for r in rows if r["direction"] == direction]
    by_scale = {}
    for r in rows:
        by_scale.setdefault(float(r["scale"]), []).append(r)

    baseline = _rate(by_scale.get(0.0, []), present_verdict)
    best_x, best_scale, best_rate = 0.0, 0.0, baseline
    for scale, srows in by_scale.items():
        if _rate(srows, incoherent_verdict) > max_incoherent:
            continue
        present = _rate(srows, present_verdict)
        swing = abs(present - baseline)
        if swing > best_x:
            best_x, best_scale, best_rate = swing, scale, present
    return {"x_salience": best_x, "best_scale": best_scale,
            "present_rate_at_best": best_rate, "baseline_rate": baseline}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_spectrum_utils.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_spectrum_utils.py src/spectrum_utils.py
git commit -m "feat: salience aggregation (spectrum x-axis)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Refusal data adapter

**Files:**
- Create: `src/prep_refusal.py`
- Test: `tests/test_prep_refusal.py`

**Interfaces:**
- Produces: `to_contrast_df(harmful: list[str], harmless: list[str]) -> pandas.DataFrame` with columns `statement`, `label` (1 = harmful, 0 = harmless), balanced and shuffled deterministically; a `main()` that downloads AdvBench + Alpaca and writes `got_datasets/refusal.csv`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prep_refusal.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prep_refusal import to_contrast_df


def test_balanced_and_labeled():
    df = to_contrast_df(["hurt someone", "build a bomb"], ["bake bread", "walk a dog", "read"])
    # balanced: min(2,3) per class = 2 each
    assert (df["label"] == 1).sum() == 2
    assert (df["label"] == 0).sum() == 2
    assert set(df.columns) == {"statement", "label"}
    assert df[df.label == 1]["statement"].tolist()  # harmful present


def test_deterministic():
    a = to_contrast_df(["x", "y"], ["p", "q"])
    b = to_contrast_df(["x", "y"], ["p", "q"])
    assert a.equals(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_prep_refusal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prep_refusal'`.

- [ ] **Step 3: Write the implementation**

Create `src/prep_refusal.py`:

```python
"""Build got_datasets/refusal.csv (harmful vs harmless instructions) for the refusal control.

Public data: AdvBench harmful behaviors + Alpaca harmless instructions. label 1 = harmful.

    .venv/bin/python prep_refusal.py
"""
import argparse
import pandas as pd


def to_contrast_df(harmful, harmless, seed=42):
    """Balanced, deterministic (statement, label) frame: 1 = harmful, 0 = harmless."""
    n = min(len(harmful), len(harmless))
    rows = ([{"statement": s, "label": 1} for s in harmful[:n]]
            + [{"statement": s, "label": 0} for s in harmless[:n]])
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--advbench",
                   default="https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv")
    p.add_argument("--out", default="got_datasets/refusal.csv")
    args = p.parse_args()

    harmful = pd.read_csv(args.advbench)["goal"].tolist()
    from datasets import load_dataset
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"] for r in alpaca if not r["input"]][:len(harmful)]

    df = to_contrast_df(harmful, harmless)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {int(df.label.sum())} harmful / "
          f"{int((df.label == 0).sum())} harmless")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_prep_refusal.py -v`
Expected: PASS (2 passed). (`main()` needs network + `datasets`; the pure `to_contrast_df` is what's tested.)

- [ ] **Step 5: Install `datasets` and build the CSV**

Run:
```bash
.venv/bin/pip install datasets
HF_HUB_DISABLE_XET=1 .venv/bin/python src/prep_refusal.py
```
Expected: writes `got_datasets/refusal.csv` with a balanced harmful/harmless count.

- [ ] **Step 6: Commit**

```bash
git add tests/test_prep_refusal.py src/prep_refusal.py got_datasets/refusal.csv
git commit -m "feat: refusal data adapter (AdvBench + Alpaca contrast)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: Run the refusal pipeline — GO/NO-GO gate

**Files:**
- Uses: all prior tasks + existing `extract.py`, `analyze.py`, `run_dct_data.py`, `steer_supervised.py`
- Produces: `recovery_refusal.csv`, `judge_steer_refusal.csv` (+ intermediate artifacts)

**Interfaces:**
- Consumes: Tasks 5, 8, 9, 10, 11.

> This is an orchestration task (cluster execution), verified by output inspection per the repo's smoke-test convention — not unit tests. It is the plan's central decision gate.

- [ ] **Step 1: Extract activations and pick the peak layer**

Run:
```bash
HF_HUB_DISABLE_XET=1 python src/extract.py refusal.csv
.venv/bin/python src/analyze.py refusal          # writes results_refusal.csv; note best layer
```
Expected: `activations/acts_refusal.npz` exists; `results_refusal.csv` shows a probe accuracy clearly above 0.5 at some layer (call it `L*`). If the probe can't separate harmful/harmless, stop — the supervised anchor is bad.

- [ ] **Step 2: Compute the supervised refusal direction and run DCT at L\***

Run (substitute `L*` and target `L*+9`):
```bash
.venv/bin/python src/export_concept_dir.py --dataset refusal --layer L*
python src/run_dct_data.py --dataset refusal --source-layer L* --target-layer $((L*+9)) \
    --num-factors 512 --num-iters 30 --num-samples 64 --balanced
```
Expected: `truth_dir_refusal.npz` and `dct_V_refusal.pt` / `dct_U_refusal.pt` / `dct_meta_refusal.json`.

- [ ] **Step 3: Measure recovery (y) and salience (x)**

Run:
```bash
.venv/bin/python src/compare_directions.py --dataset refusal          # -> recovery_refusal.csv
python src/steer_supervised.py --dataset refusal --device cuda --scales -120,-80,-40,-20,0,20,40,80,120
HF_HUB_DISABLE_XET=1 python src/judge_results.py --mode steer --backend local-hf --dataset refusal --device cuda
```
Expected: `recovery_refusal.csv` (y) and `judge_steer_refusal.csv` (x source).

- [ ] **Step 4: EVALUATE THE GATE**

Inspect `recovery_refusal.csv`. **GO** if `ratio_vs_random ≥ 3` OR `subspace_frac ≥ 2 × subspace_chance` (DCT recovers refusal well above chance — validating that DCT *can* find a causal direction). **NO-GO** if recovery is ≈ chance like truth was.

- If **GO**: proceed to Phase 2.
- If **NO-GO**: STOP. Do not run Phase 2. Write up the finding (DCT fails even the positive control → the bottleneck is DCT/pipeline, not truth's causal status) and reconvene — this changes the whole project direction.

- [ ] **Step 5: Commit the refusal artifacts**

```bash
git add recovery_refusal.csv judge_steer_refusal.csv results_refusal.csv dct_meta_refusal.json
git commit -m "data: refusal positive-control results (recovery + salience)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> `dct_V/U_*.pt`, `acts_*.npz`, `truth_dir_*.npz`, `steer_supervised_*.csv`, `plot_*` are gitignored (`.gitignore`); regenerate from scripts.

---

## Phase 2 — Broaden (thread D)

### Task 13: Toxicity concept end-to-end

**Files:**
- Create: `src/prep_toxicity.py`
- Test: `tests/test_prep_toxicity.py`
- Produces: `recovery_toxicity.csv`, `judge_steer_toxicity.csv`

**Interfaces:**
- Produces: `to_contrast_df(toxic: list[str], clean: list[str]) -> DataFrame` (same shape/contract as `prep_refusal.to_contrast_df`, 1 = toxic).

- [ ] **Step 1: Write the failing test**

Create `tests/test_prep_toxicity.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prep_toxicity import to_contrast_df


def test_balanced_and_labeled():
    df = to_contrast_df(["slur text", "hateful line"], ["hello there", "nice day", "good book"])
    assert (df["label"] == 1).sum() == 2
    assert (df["label"] == 0).sum() == 2
    assert set(df.columns) == {"statement", "label"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_prep_toxicity.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/prep_toxicity.py`**

```python
"""Build got_datasets/toxicity.csv from RealToxicityPrompts. label 1 = toxic continuation.

    .venv/bin/python prep_toxicity.py
"""
import argparse
import pandas as pd


def to_contrast_df(toxic, clean, seed=42):
    n = min(len(toxic), len(clean))
    rows = ([{"statement": s, "label": 1} for s in toxic[:n]]
            + [{"statement": s, "label": 0} for s in clean[:n]])
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="got_datasets/toxicity.csv")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    from datasets import load_dataset
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    toxic, clean = [], []
    for r in ds:
        cont = r["continuation"]
        tox = cont.get("toxicity")
        if tox is None:
            continue
        text = cont["text"].strip()
        if not text:
            continue
        (toxic if tox >= 0.5 else clean).append(text)
        if len(toxic) >= args.n and len(clean) >= args.n:
            break
    df = to_contrast_df(toxic, clean)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_prep_toxicity.py -v`
Expected: PASS.

- [ ] **Step 5: Build data and run the same pipeline as Task 12 (Steps 1–3)**

Run:
```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python src/prep_toxicity.py
HF_HUB_DISABLE_XET=1 python src/extract.py toxicity.csv
.venv/bin/python src/analyze.py toxicity                       # note peak layer L*
.venv/bin/python src/export_concept_dir.py --dataset toxicity --layer L*
python src/run_dct_data.py --dataset toxicity --source-layer L* --target-layer $((L*+9)) --num-factors 512 --num-iters 30 --num-samples 64 --balanced
.venv/bin/python src/compare_directions.py --dataset toxicity
python src/steer_supervised.py --dataset toxicity --device cuda --scales -120,-80,-40,-20,0,20,40,80,120
HF_HUB_DISABLE_XET=1 python src/judge_results.py --mode steer --backend local-hf --dataset toxicity --device cuda
```
Expected: `recovery_toxicity.csv`, `judge_steer_toxicity.csv`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_prep_toxicity.py src/prep_toxicity.py got_datasets/toxicity.csv recovery_toxicity.csv judge_steer_toxicity.csv results_toxicity.csv dct_meta_toxicity.json
git commit -m "feat+data: toxicity concept end-to-end

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 14: Sycophancy concept end-to-end

**Files:**
- Create: `src/prep_sycophancy.py`
- Test: `tests/test_prep_sycophancy.py`
- Produces: `recovery_sycophancy.csv`, `judge_steer_sycophancy.csv`

**Interfaces:**
- Produces: `to_contrast_df(sycophantic: list[str], plain: list[str]) -> DataFrame` (1 = sycophantic). Sycophancy behavioral judging reuses the Anthropic backend (no dedicated local classifier), so its `judge_steer` runs with `--backend anthropic`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prep_sycophancy.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prep_sycophancy import to_contrast_df


def test_balanced_and_labeled():
    df = to_contrast_df(["you're so right!", "great point!"], ["actually that's wrong", "no", "false"])
    assert (df["label"] == 1).sum() == 2
    assert (df["label"] == 0).sum() == 2
    assert set(df.columns) == {"statement", "label"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_prep_sycophancy.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/prep_sycophancy.py`**

```python
"""Build got_datasets/sycophancy.csv from Anthropic's sycophancy eval. label 1 = sycophantic.

    .venv/bin/python prep_sycophancy.py
"""
import argparse
import pandas as pd


def to_contrast_df(sycophantic, plain, seed=42):
    n = min(len(sycophantic), len(plain))
    rows = ([{"statement": s, "label": 1} for s in sycophantic[:n]]
            + [{"statement": s, "label": 0} for s in plain[:n]])
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="got_datasets/sycophancy.csv")
    args = p.parse_args()
    from datasets import load_dataset
    ds = load_dataset("Anthropic/model-written-evals", "sycophancy", split="train")
    syc = [r["question"] + " " + r["answer_matching_behavior"] for r in ds]
    plain = [r["question"] + " " + r["answer_not_matching_behavior"] for r in ds]
    df = to_contrast_df(syc, plain)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows")


if __name__ == "__main__":
    main()
```

> If the `Anthropic/model-written-evals` config path differs at run time, adjust the `load_dataset` call to the sycophancy subset; the `to_contrast_df` contract is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_prep_sycophancy.py -v`
Expected: PASS.

- [ ] **Step 5: Build data and run the pipeline (Anthropic judge for the sweep)**

Run:
```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python src/prep_sycophancy.py
HF_HUB_DISABLE_XET=1 python src/extract.py sycophancy.csv
.venv/bin/python src/analyze.py sycophancy                     # note peak layer L*
.venv/bin/python src/export_concept_dir.py --dataset sycophancy --layer L*
python src/run_dct_data.py --dataset sycophancy --source-layer L* --target-layer $((L*+9)) --num-factors 512 --num-iters 30 --num-samples 64 --balanced
.venv/bin/python src/compare_directions.py --dataset sycophancy
python src/steer_supervised.py --dataset sycophancy --device cuda --scales -120,-80,-40,-20,0,20,40,80,120
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python src/judge_results.py --mode steer --backend anthropic --dataset sycophancy
```
Expected: `recovery_sycophancy.csv`, `judge_steer_sycophancy.csv`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_prep_sycophancy.py src/prep_sycophancy.py got_datasets/sycophancy.csv recovery_sycophancy.csv judge_steer_sycophancy.csv results_sycophancy.csv dct_meta_sycophancy.json
git commit -m "feat+data: sycophancy concept end-to-end

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 3 — Synthesis

### Task 15: The spectrum figure

**Files:**
- Create: `src/viz_spectrum.py`
- Test: `tests/test_viz_spectrum.py`
- Produces: `plot_findings_spectrum.png`, `spectrum_points.csv`

**Interfaces:**
- Consumes: `recovery_<concept>.csv` (Task 9), `judge_steer_<concept>.csv` (Tasks 7/12/13/14), `spectrum_utils.concept_salience` (Task 10).
- Produces: `build_points(concepts: list[str], present_map: dict) -> pandas.DataFrame` with columns `concept, x_salience, y_recovery`; `main()` writes the CSV + scatter.

- [ ] **Step 1: Write the failing test**

Create `tests/test_viz_spectrum.py`:

```python
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viz_spectrum import build_points


def _write(tmp, name, header, row):
    path = os.path.join(tmp, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerow(row)
    return path


def test_build_points_joins_recovery_and_salience(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "recovery_demo.csv",
           ["concept", "best_abs_cos", "random_max_abs_cos", "ratio_vs_random",
            "subspace_frac", "subspace_chance"],
           ["demo", "0.30", "0.06", "5.0", "0.40", "0.22"])
    with open(tmp_path / "judge_steer_demo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "scale", "prompt", "completion", "verdict", "reason"])
        w.writerow(["grad", "0", "p", "c", "TRUE", ""])
        w.writerow(["grad", "-120", "p", "c", "FALSE", ""])

    df = build_points(["demo"], present_map={"demo": "FALSE"})
    assert list(df["concept"]) == ["demo"]
    assert abs(float(df["y_recovery"].iloc[0]) - 5.0) < 1e-9   # ratio_vs_random
    assert abs(float(df["x_salience"].iloc[0]) - 1.0) < 1e-9   # 0 -> 1 FALSE swing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_viz_spectrum.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/viz_spectrum.py`**

```python
"""The money figure: DCT recovery (y) vs. behavioral causal salience (x), one point per concept.

Reads recovery_<concept>.csv and judge_steer_<concept>.csv for each concept, joins them, and
plots the spectrum. `present_map` says which judged verdict marks the concept-present dimension
per concept (truth: FALSE = lie induced; refusal: FALSE = still refusing; toxicity: FALSE = toxic).

    .venv/bin/python viz_spectrum.py --concepts cities refusal toxicity sycophancy
"""
import argparse
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrum_utils import concept_salience

DEFAULT_PRESENT = {"cities": "FALSE", "common_claim_true_false": "FALSE",
                   "refusal": "FALSE", "toxicity": "FALSE", "sycophancy": "FALSE"}


def build_points(concepts, present_map=None):
    present_map = present_map or DEFAULT_PRESENT
    out = []
    for c in concepts:
        rec = next(csv.DictReader(open(f"recovery_{c}.csv")))
        y = float(rec["ratio_vs_random"])
        rows = list(csv.DictReader(open(f"judge_steer_{c}.csv")))
        sal = concept_salience(rows, present_verdict=present_map.get(c, "FALSE"))
        out.append({"concept": c, "x_salience": sal["x_salience"], "y_recovery": y})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", nargs="+", required=True)
    args = p.parse_args()
    df = build_points(args.concepts)
    df.to_csv("spectrum_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(df["x_salience"], df["y_recovery"], s=80, color="#4477aa", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["concept"], (r["x_salience"], r["y_recovery"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="DCT = random (no recovery)")
    ax.set_xlabel("behavioral causal salience  (max judged flip vs. unsteered)")
    ax.set_ylabel("DCT recovery  (best |cos| ÷ random baseline)")
    ax.set_title("Does DCT recover a concept in proportion to its causal salience?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plot_findings_spectrum.png", dpi=150)
    print("saved plot_findings_spectrum.png and spectrum_points.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_viz_spectrum.py -v`
Expected: PASS.

- [ ] **Step 5: Generate the real figure**

Run: `.venv/bin/python src/viz_spectrum.py --concepts cities refusal toxicity sycophancy`
Expected: writes `plot_findings_spectrum.png` + `spectrum_points.csv`; prints the per-concept table. Inspect: refusal high-y, truth (cities) low-y.

- [ ] **Step 6: Commit**

```bash
git add tests/test_viz_spectrum.py src/viz_spectrum.py spectrum_points.csv
git commit -m "feat: spectrum money figure (recovery vs salience)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> `plot_findings_spectrum.png` is gitignored (`plot_findings_*.png`); regenerate with viz_spectrum.py.

---

### Task 16: Findings write-up

**Files:**
- Create: `docs/CAUSAL_SALIENCE_SPECTRUM_FINDINGS.md`

**Interfaces:**
- Consumes: `spectrum_points.csv`, `recovery_*.csv`, judged plots, the Task-6 judge-agreement number.

- [ ] **Step 1: Write the findings doc**

Create `docs/CAUSAL_SALIENCE_SPECTRUM_FINDINGS.md` following the structure of `docs/DCT_VS_XGBOOST_FINDINGS.md`: (1) the thesis, (2) method (per-concept pipeline + judge harness + validation number), (3) the spectrum table (concept, x_salience, y_recovery), (4) what it means — does recovery track salience? refusal control result, (5) caveats (judge transfer, one DCT config, geometry-not-behavior), (6) one-line summary for the PI, (7) reproducibility (the exact commands from Tasks 12–15). Fill every number from the generated CSVs — no placeholders.

- [ ] **Step 2: Commit**

```bash
git add docs/CAUSAL_SALIENCE_SPECTRUM_FINDINGS.md
git commit -m "docs: causal-salience-spectrum findings

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Where the degradation metrics (Tasks 1–2) are consumed

The **operational** degradation gate for the x-axis is the judge's `INCOHERENT` verdict class
(Task 10's `max_incoherent` threshold) — it needs no extra model load and reuses the judged CSV.
The Task 1–2 metrics (`corpus_distinct_n`, `repetition_rate`, `Perplexity`) are consumed as an
**independent degradation diagnostic reported in the findings doc (Task 16)**: per concept, report
distinct-n / perplexity at the winning steer scale to confirm the "flip" happened at acceptable
fluency, cross-checking the INCOHERENT gate. If the INCOHERENT judge proxy proves unreliable in
Task 12, `concept_salience` falls back to gating on `Perplexity`/`corpus_distinct_n` instead — the
metrics are the designed fallback, so they are not orphaned.

## Notes on deviations from the spec

- **self-BLEU → `repetition_rate` + `corpus_distinct_n`** (Task 1): dependency-free, captures the same collapse signal. Revisit if a reviewer wants true self-BLEU.
- **Refusal judge** (Task 4): approximated with keyword markers over Llama-Guard load; if the marker heuristic proves weak in Task 12, swap to Guard's classification head or the Anthropic backend for refusal's sweep.
- **`plot_*` outputs are gitignored** by repo convention; every plot is regenerable from a committed CSV.
