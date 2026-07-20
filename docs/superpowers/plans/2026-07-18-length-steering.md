# Length-Steering (Uncapped Tokens) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show what a steering vector does over *long* (~96-token) generations via a per-token intrinsic coherence trajectory plus an OLMo judge run on length-cutoff prefixes, contrasting the inert `mean_diff` truth axis against the known-degrader `resid_pc1`.

**Architecture:** Two pure, unit-tested modules (`length_prompts`, `length_intrinsics`) feed one generation script (`length_steer.py`) that injects a calibrated steering vector via the existing `dct_steer_utils.Steerer` and logs per-token signals during a single greedy `model.generate`. It writes two CSVs: a per-position intrinsic CSV and a cutoff-prefix CSV whose schema (`direction,scale,prompt,completion`) is exactly what the existing `judge_results.py --mode steer` consumes, so the OLMo judge is reused unchanged. A plotting script turns the two CSVs into the coherence and verdict figures. Runs on the GH200 via two jobs (gen, then judge).

**Tech Stack:** Python 3.13, PyTorch (fp32, eager attention), transformers, numpy, pandas, matplotlib (Agg). Model `google/gemma-2-2b` (base). OLMo judge (`allenai/Olmo-3-7B-Instruct`) via the existing `judge_results.py --backend olmo`.

## Global Constraints

- Model is `google/gemma-2-2b`, a **base** model, loaded fp32 + `attn_implementation="eager"` via `dct_steer_utils.load_model` — never re-instantiate the model differently.
- Generation is **greedy**: `do_sample=False, repetition_penalty=1.3` (identical to `dct_steer_utils.generate`), `max_new_tokens=96`.
- Steering is injected via `dct_steer_utils.Steerer(model, layer)` as `τ · A_prefix_norm · unit(dir)`, matching `src/mag/steer.py::injected_vector`. `A_prefix_norm` and `layer` come from `mag_dir_<ds>.npz`.
- Both directions inject at the **same layer** (11 for cities, 13 for common_claim — this equals `mag_dir.layer == truth_dir.layer`) with the **same** `mag_dir_<ds>.npz["A_prefix_norm"]`.
- `û` convention: `mean_diff` is stored as `mean(true) − mean(false)` (points toward TRUE, matching `src/mag/steer.py`, which loads it unnegated), so **+τ pushes `mean_diff` toward TRUE and −τ toward FALSE (lying)**. `resid_pc1_unit` sign is as stored in `mag_dir`.
- Taus: `{-1.0, -0.3, 0.0, 0.3, 1.0}` — two-sided so the run probes both toward-FALSE (−τ) and toward-TRUE (+τ); a one-sided push could never observe a truth→false flip. τ=0 ⇒ no vector injected (`Steerer.set(None)`).
- Judge cutoffs (cumulative prefix token counts): `[8, 16, 32, 64, 96]`.
- Datasets: `cities` and `common_claim_true_false`.
- The cutoff-prefix CSV MUST use header `direction,scale,prompt,completion` (the schema `judge_results.run_steer` reads), with `direction = "<dir>_tau<τ>"` and `scale = <cutoff>` (a number), so the judge and its grouping work unmodified.
- Tests run with the repo venv: `PYTHONPATH=src .venv/bin/python -m pytest`. Pure-function tests must not load the model.

---

### Task 1: Prompt sets (`length_prompts.py`)

**Files:**
- Create: `src/length_prompts.py`
- Test: `tests/test_length_prompts.py`

**Interfaces:**
- Produces:
  - `cities_stems(n=300, seed=0) -> list[tuple[str, str]]` — returns `(stem, expected_answer)` pairs built from `got_datasets/cities.csv` **true** rows only. A cities statement `"The city of Paris is in the country of France."` becomes stem `"The city of Paris is in the country of"` and answer `"France"`.
  - `COMMON_CLAIM_STEMS: list[tuple[str, str]]` — curated `(stem, expected_answer)` pairs (~100).
  - `common_claim_stems(n=100) -> list[tuple[str, str]]` — returns `COMMON_CLAIM_STEMS[:n]`.
  - `get_prompt_set(dataset, n=None, seed=0) -> list[tuple[str, str]]` — dispatch: `cities` → `cities_stems`; `common_claim_true_false` → `common_claim_stems`.

- [ ] **Step 1: Write the failing test for the cities stemmer**

```python
# tests/test_length_prompts.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import length_prompts as lp


def test_cities_stem_strips_answer_and_period():
    stems = lp.cities_stems(n=50, seed=0)
    assert 0 < len(stems) <= 50
    for stem, ans in stems:
        # stem ends right before the country, no trailing period, answer is non-empty
        assert stem.endswith("the country of")
        assert not stem.endswith(".")
        assert ans and ans[0].isupper()
        # the answer must NOT appear in the stem (no leakage)
        assert ans not in stem


def test_cities_stems_deterministic():
    a = lp.cities_stems(n=20, seed=7)
    b = lp.cities_stems(n=20, seed=7)
    assert a == b
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_length_prompts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'length_prompts'`.

- [ ] **Step 3: Implement `length_prompts.py`**

```python
# src/length_prompts.py
"""Completion-stem prompt sets for the length-steering experiment.

cities: derived programmatically from got_datasets/cities.csv (regular template, reliable to
stem). common_claim: curated, because the free-form claims do not cut into clean stems."""
import re
import pandas as pd

DATASET_DIR = "got_datasets"

# cities rows look like: "The city of Paris is in the country of France."
_CITY_RE = re.compile(r"^(The city of .+ is in the country of) ([A-Z][^.]*)\.?$")


def cities_stems(n=300, seed=0):
    df = pd.read_csv(f"{DATASET_DIR}/cities.csv")
    df = df[df["label"] == 1]                      # true statements only
    df = df.sample(min(n, len(df)), random_state=seed)
    out = []
    for s in df["statement"].astype(str):
        m = _CITY_RE.match(s.strip())
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


# ~100 curated (stem, expected_answer) pairs — unambiguous world claims with a crisp answer.
COMMON_CLAIM_STEMS = [
    ("The number of legs a spider has is", "eight"),
    ("The chemical symbol for gold is", "Au"),
    ("The planet known as the Red Planet is", "Mars"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The number of continents on Earth is", "seven"),
    ("The tallest land animal is the", "giraffe"),
    ("The gas humans need to breathe to survive is", "oxygen"),
    ("The organ that pumps blood through the body is the", "heart"),
    ("The freezing point of water in Celsius is", "zero"),
    ("The speed of light is approximately 300,000 kilometers per", "second"),
    ("The author of Romeo and Juliet is William", "Shakespeare"),
    ("The currency used in Japan is the", "yen"),
    ("The largest ocean on Earth is the", "Pacific"),
    ("The number of sides a triangle has is", "three"),
    ("The primary language spoken in Brazil is", "Portuguese"),
    ("The metal that is liquid at room temperature is", "mercury"),
    ("The closest star to Earth is the", "Sun"),
    ("The number of days in a leap year is", "366"),
    ("The bone that protects the brain is the", "skull"),
    ("The process plants use to make food is called", "photosynthesis"),
    ("The country that gifted the Statue of Liberty to the US is", "France"),
    ("The hardest natural substance on Earth is", "diamond"),
    ("The number of players on a soccer team on the field is", "eleven"),
    ("The vitamin produced by the skin in sunlight is vitamin", "D"),
    ("The largest mammal on Earth is the blue", "whale"),
    ("The layer of gas that protects Earth from UV rays is the", "ozone"),
    ("The number of strings on a standard guitar is", "six"),
    ("The chemical formula for water is", "H2O"),
    ("The first man to walk on the moon was Neil", "Armstrong"),
    ("The powerhouse of the cell is the", "mitochondria"),
    ("The tallest mountain on Earth is Mount", "Everest"),
    ("The number of colors in a rainbow is", "seven"),
    ("The study of living organisms is called", "biology"),
    ("The largest desert on Earth is the", "Sahara"),
    ("The human body has how many pairs of ribs:", "twelve"),
    ("The inventor of the telephone was Alexander Graham", "Bell"),
    ("The longest river in the world is the", "Nile"),
    ("The number of teeth in a healthy adult human is", "thirty-two"),
    ("The gas that makes up most of Earth's atmosphere is", "nitrogen"),
    ("The country with the largest population is", "India"),
    ("The smallest prime number is", "two"),
    ("The organ responsible for filtering blood is the", "kidney"),
    ("The number of hours in a day is", "twenty-four"),
    ("The painter of the Mona Lisa was Leonardo da", "Vinci"),
    ("The largest internal organ in the human body is the", "liver"),
    ("The number of planets in our solar system is", "eight"),
    ("The primary gas that plants absorb is carbon", "dioxide"),
    ("The continent that is also a country is", "Australia"),
    ("The number of legs an insect has is", "six"),
    ("The unit used to measure electrical resistance is the", "ohm"),
    ("The largest species of shark is the whale", "shark"),
    ("The capital city where the Eiffel Tower stands is", "Paris"),
    ("The frozen form of water is called", "ice"),
    ("The number of zeros in one thousand is", "three"),
    ("The scientist who developed the theory of relativity was Albert", "Einstein"),
    ("The tallest building material grown as grass is", "bamboo"),
    ("The organ used for breathing in fish is the", "gills"),
    ("The number of minutes in an hour is", "sixty"),
    ("The metal most commonly used in electrical wiring is", "copper"),
    ("The natural satellite of Earth is the", "Moon"),
    ("The number of legs a dog has is", "four"),
    ("The largest country by land area is", "Russia"),
    ("The color obtained by mixing blue and yellow is", "green"),
    ("The instrument used to measure temperature is a", "thermometer"),
    ("The number of sides a hexagon has is", "six"),
    ("The first element on the periodic table is", "hydrogen"),
    ("The type of animal a frog is classified as is an", "amphibian"),
    ("The number of legs a horse has is", "four"),
    ("The ship that sank in 1912 after hitting an iceberg was the", "Titanic"),
    ("The number of weeks in a year is", "fifty-two"),
    ("The largest bird in the world is the", "ostrich"),
    ("The gas released by plants during photosynthesis is", "oxygen"),
    ("The number of degrees in a right angle is", "ninety"),
    ("The country where the pyramids of Giza are located is", "Egypt"),
    ("The organ that produces insulin is the", "pancreas"),
    ("The number of players on a basketball team on the court is", "five"),
    ("The nearest planet to the Sun is", "Mercury"),
    ("The word for a baby dog is a", "puppy"),
    ("The number of sides a square has is", "four"),
    ("The largest state in the United States by area is", "Alaska"),
    ("The natural process by which water turns to vapor is", "evaporation"),
    ("The number of legs a spider has is not six but", "eight"),
    ("The metal with the chemical symbol Fe is", "iron"),
    ("The number of months in a year is", "twelve"),
    ("The primary source of energy for Earth is the", "Sun"),
    ("The animal known as man's best friend is the", "dog"),
    ("The number of letters in the English alphabet is", "twenty-six"),
    ("The chemical symbol for oxygen is", "O"),
    ("The country famous for the Great Wall is", "China"),
    ("The organ that allows humans to see is the", "eye"),
    ("The number of seconds in a minute is", "sixty"),
    ("The largest planet's most famous feature is its", "rings"),
    ("The name of the galaxy Earth is in is the Milky", "Way"),
    ("The number of legs an octopus has is", "eight"),
    ("The primary ingredient in bread is", "flour"),
    ("The number of colors on a traffic light is", "three"),
    ("The scientist known for the laws of motion was Isaac", "Newton"),
    ("The frozen continent at the southern pole is", "Antarctica"),
    ("The number of eyes a typical human has is", "two"),
]


def common_claim_stems(n=100):
    return COMMON_CLAIM_STEMS[:n]


def get_prompt_set(dataset, n=None, seed=0):
    if dataset == "cities":
        return cities_stems(n or 300, seed)
    if dataset == "common_claim_true_false":
        return common_claim_stems(n or 100)
    raise ValueError(f"unknown dataset {dataset!r}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_length_prompts.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/length_prompts.py tests/test_length_prompts.py
git commit -m "feat(length): prompt stems for cities (programmatic) + common_claim (curated)"
```

---

### Task 2: Intrinsic per-token signals (`length_intrinsics.py`)

**Files:**
- Create: `src/length_intrinsics.py`
- Test: `tests/test_length_intrinsics.py`

**Interfaces:**
- Consumes: per-step logit vectors (numpy arrays, shape `(vocab,)`) and the list of generated token ids.
- Produces:
  - `token_max_prob(logits) -> float` — max softmax probability.
  - `token_entropy(logits) -> float` — Shannon entropy (nats) of the softmax.
  - `rep3_flags(token_ids) -> list[int]` — 0/1 per position; 1 when the trigram ending at position `i` repeats a trigram that ended earlier.
  - `intrinsic_signals(scores, token_ids) -> dict` — `{"max_prob": [...], "entropy": [...], "rep3": [...]}`, all lists of length `len(token_ids)`; `scores` is a list of `(vocab,)` numpy arrays aligned with `token_ids`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_length_intrinsics.py
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import length_intrinsics as li


def test_max_prob_peaked_vs_flat():
    peaked = np.array([100.0, 0.0, 0.0, 0.0])   # near one-hot after softmax
    flat = np.zeros(4)                            # uniform
    assert li.token_max_prob(peaked) > 0.99
    assert abs(li.token_max_prob(flat) - 0.25) < 1e-6


def test_entropy_flat_is_log_vocab():
    flat = np.zeros(8)
    assert abs(li.token_entropy(flat) - math.log(8)) < 1e-6
    peaked = np.array([100.0, 0.0, 0.0])
    assert li.token_entropy(peaked) < 1e-3


def test_rep3_flags_detects_repeated_trigram():
    # tokens: a b c a b c  -> the second "c" (idx 5) completes a repeated trigram (a,b,c)
    ids = [1, 2, 3, 1, 2, 3]
    flags = li.rep3_flags(ids)
    assert flags == [0, 0, 0, 0, 0, 1]


def test_intrinsic_signals_lengths_and_keys():
    scores = [np.zeros(5) for _ in range(4)]
    ids = [0, 1, 2, 2]
    out = li.intrinsic_signals(scores, ids)
    assert set(out) == {"max_prob", "entropy", "rep3"}
    assert all(len(out[k]) == 4 for k in out)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_length_intrinsics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'length_intrinsics'`.

- [ ] **Step 3: Implement `length_intrinsics.py`**

```python
# src/length_intrinsics.py
"""Cheap per-token coherence signals computed from generation logits.

max_prob (confidence), entropy (uncertainty), and a 3-gram repetition flag (looping).
Pure numpy so it is unit-testable without loading the model."""
import numpy as np


def _softmax(logits):
    x = np.asarray(logits, dtype=np.float64).ravel()
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def token_max_prob(logits):
    return float(_softmax(logits).max())


def token_entropy(logits):
    p = _softmax(logits)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def rep3_flags(token_ids):
    ids = list(token_ids)
    seen = set()
    flags = [0] * len(ids)
    for i in range(len(ids)):
        if i >= 2:
            tri = (ids[i - 2], ids[i - 1], ids[i])
            if tri in seen:
                flags[i] = 1
            seen.add(tri)
    return flags


def intrinsic_signals(scores, token_ids):
    return {
        "max_prob": [token_max_prob(s) for s in scores],
        "entropy": [token_entropy(s) for s in scores],
        "rep3": rep3_flags(token_ids),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_length_intrinsics.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/length_intrinsics.py tests/test_length_intrinsics.py
git commit -m "feat(length): pure per-token intrinsic signals (max-prob, entropy, rep3)"
```

---

### Task 3: Generation + logging script (`length_steer.py`)

**Files:**
- Create: `src/length_steer.py`
- Test: `tests/test_length_steer.py` (pure helpers only; full gen is smoke-run, not unit-tested)

**Interfaces:**
- Consumes: `length_prompts.get_prompt_set`, `length_intrinsics.intrinsic_signals`, `dct_steer_utils.load_model/Steerer`, `funnel_utils.unit`, `mag_dir_<ds>.npz` (`resid_pc1_unit`, `A_prefix_norm`, `layer`), `truth_dir_<ds>.npz` (`mean_diff`, `layer`).
- Produces:
  - `load_directions(ds) -> (layer:int, a_prefix_norm:float, dirs:list[dict])` — `dirs` items `{name, unit_dir(np,2304)}` for `mean_diff` and `resid_pc1`. Asserts `mag_dir.layer == truth_dir.layer`.
  - `injected_vector(tau, unit_dir, a_prefix_norm) -> np.ndarray` — `tau * a_prefix_norm * unit(unit_dir)` (identical formula to `mag/steer.py`).
  - `prefixes_for_cutoffs(token_ids, cutoffs, tokenizer) -> dict[int,str]` — decode of the first `k` generated tokens for each cutoff `k` (skips cutoffs beyond the generated length).
  - `generate_with_logging(model, tok, prompt, max_new_tokens) -> (text, scores_list, gen_ids)` — one greedy generate with `output_scores=True`.
  - CLI writing `length_steer_<ds>.csv` and `length_prefixes_<ds>.csv`.

- [ ] **Step 1: Write failing tests for the pure helpers**

```python
# tests/test_length_steer.py
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import length_steer as ls


def test_injected_vector_scales_and_normalizes():
    d = np.array([3.0, 4.0, 0.0])          # norm 5
    v = ls.injected_vector(2.0, d, 10.0)   # 2 * 10 * unit(d)
    assert np.allclose(np.linalg.norm(v), 20.0)
    assert np.allclose(v, np.array([12.0, 16.0, 0.0]))


def test_injected_vector_tau_zero_is_zero():
    d = np.array([1.0, 2.0, 2.0])
    assert np.allclose(ls.injected_vector(0.0, d, 99.0), 0.0)


class _FakeTok:
    # decode returns "t<id> " per token so cutoffs are countable
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


def test_prefixes_for_cutoffs_truncates():
    ids = list(range(10))
    out = ls.prefixes_for_cutoffs(ids, [4, 8, 16], _FakeTok())
    assert set(out) == {4, 8}          # 16 > 10 generated tokens -> skipped
    assert out[4] == "t0 t1 t2 t3"
    assert out[8] == "t0 t1 t2 t3 t4 t5 t6 t7"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_length_steer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'length_steer'`.

- [ ] **Step 3: Implement `length_steer.py`**

```python
# src/length_steer.py
"""Length-steering generation: one long greedy completion per (prompt, direction, tau), logging
per-token intrinsic signals, plus cutoff-prefix rows for the OLMo judge.

    python length_steer.py --dataset cities --device cuda
    python length_steer.py --dataset cities --device cpu --limit 4   # smoke

Writes:
    length_steer_<ds>.csv     one row per (direction, tau, prompt): completion + per-position signals
    length_prefixes_<ds>.csv  one row per (direction, tau, prompt, cutoff): prefix text for the judge
                              (schema direction,scale,prompt,completion — direction="<dir>_tau<tau>",
                               scale=<cutoff> — so judge_results.py --mode steer reads it unchanged)
"""
import argparse
import csv
import json
import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from length_prompts import get_prompt_set
from length_intrinsics import intrinsic_signals

TAUS = [0.0, 0.3, 1.0]
CUTOFFS = [8, 16, 32, 64, 96]
MAX_NEW_TOKENS = 96


def injected_vector(tau, unit_dir, a_prefix_norm):
    return float(tau) * float(a_prefix_norm) * unit(np.asarray(unit_dir, np.float64))


def load_directions(ds):
    md = np.load(f"mag_dir_{ds}.npz")
    td = np.load(f"truth_dir_{ds}.npz")
    layer = int(md["layer"])
    assert layer == int(td["layer"]), (
        f"mag_dir.layer={layer} != truth_dir.layer={int(td['layer'])}; "
        "mean_diff and resid_pc1 must live at the same layer to inject with a shared norm")
    apn = float(md["A_prefix_norm"])
    dirs = [
        {"name": "mean_diff", "unit_dir": unit(np.asarray(td["mean_diff"], np.float64))},
        {"name": "resid_pc1", "unit_dir": unit(np.asarray(md["resid_pc1_unit"], np.float64))},
    ]
    return layer, apn, dirs


def prefixes_for_cutoffs(token_ids, cutoffs, tokenizer):
    out = {}
    for k in cutoffs:
        if k <= len(token_ids):
            out[k] = tokenizer.decode(token_ids[:k], skip_special_tokens=True).replace("\n", " ").strip()
    return out


def generate_with_logging(model, tok, prompt, max_new_tokens):
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3,
            pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    gen_ids = out.sequences[0][inp["input_ids"].shape[1]:].tolist()
    scores = [s[0].float().cpu().numpy() for s in out.scores]   # list of (vocab,)
    # scores and gen_ids are aligned and equal-length (one score per generated token)
    n = min(len(scores), len(gen_ids))
    text = tok.decode(gen_ids[:n], skip_special_tokens=True).replace("\n", " ").strip()
    return text, scores[:n], gen_ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke test)")
    ap.add_argument("--n-prompts", type=int, default=0, help="0 = dataset default (cities 300 / cc 100)")
    a = ap.parse_args()
    ds = a.dataset

    layer, apn, dirs = load_directions(ds)
    prompts = get_prompt_set(ds, n=a.n_prompts or None)
    if a.limit:
        prompts = prompts[:a.limit]
    print(f"[length] {ds}: layer={layer} A_prefix_norm={apn:.3f} "
          f"{len(prompts)} prompts x {len(dirs)} dirs x {len(TAUS)} taus", flush=True)

    tok, model, dev = su.load_model(a.device)
    steer_rows = [("direction", "tau", "prompt", "completion", "max_prob", "entropy", "rep3")]
    prefix_rows = [("direction", "scale", "prompt", "completion")]   # judge schema

    for d in dirs:
        with su.Steerer(model, layer) as st:
            for tau in TAUS:
                vec = None if tau == 0.0 else torch.tensor(
                    injected_vector(tau, d["unit_dir"], apn), dtype=torch.float32)
                st.set(vec)
                for stem, _answer in prompts:
                    text, scores, gen_ids = generate_with_logging(model, tok, stem, MAX_NEW_TOKENS)
                    sig = intrinsic_signals(scores, gen_ids)
                    steer_rows.append((d["name"], tau, stem, text,
                                       json.dumps([round(x, 4) for x in sig["max_prob"]]),
                                       json.dumps([round(x, 4) for x in sig["entropy"]]),
                                       json.dumps(sig["rep3"])))
                    for k, ptext in prefixes_for_cutoffs(gen_ids, CUTOFFS, tok).items():
                        prefix_rows.append((f"{d['name']}_tau{tau}", k, stem, ptext))
                print(f"  {d['name']} tau={tau:+.1f} done ({len(prompts)} prompts)", flush=True)

    with open(f"length_steer_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(steer_rows)
    with open(f"length_prefixes_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(prefix_rows)
    print(f"[length] wrote length_steer_{ds}.csv and length_prefixes_{ds}.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the helper tests to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_length_steer.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: CPU smoke run (end-to-end, tiny)**

Run: `PYTHONPATH=src .venv/bin/python src/length_steer.py --dataset cities --device cpu --limit 2`
Expected: prints the per-dir/tau progress lines and writes `length_steer_cities.csv` (rows: 1 header + 2 dirs×3 taus×2 prompts = 12) and `length_prefixes_cities.csv`. Confirm both files exist and the prefix CSV header is exactly `direction,scale,prompt,completion`.

- [ ] **Step 6: Commit**

```bash
git add src/length_steer.py tests/test_length_steer.py
git commit -m "feat(length): greedy 96-token gen with per-token logging + cutoff-prefix CSV"
```

---

### Task 4: Plots (`viz_length.py`)

**Files:**
- Create: `src/viz_length.py`
- Test: `tests/test_viz_length.py`

**Interfaces:**
- Consumes: `length_steer_<ds>.csv` (per-position signals), `judge_length_<ds>.csv` (produced by the judge in Task 5; schema `direction,scale,prompt,completion,verdict,reason`).
- Produces:
  - `mean_trajectory(rows, signal) -> dict[(direction,tau), list[float]]` — averages the per-position `signal` array across prompts, per (direction, τ), padding/truncating to the common min length.
  - `plot_coherence(ds)` → `plot_length_coherence_<ds>.png`.
  - `plot_verdict(ds)` → `plot_length_verdict_<ds>.png`.
  - CLI: `python viz_length.py --dataset cities`.

- [ ] **Step 1: Write the failing test for `mean_trajectory`**

```python
# tests/test_viz_length.py
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import viz_length as vz


def test_mean_trajectory_averages_and_truncates():
    rows = [
        {"direction": "mean_diff", "tau": "0.0", "entropy": json.dumps([1.0, 2.0, 3.0])},
        {"direction": "mean_diff", "tau": "0.0", "entropy": json.dumps([3.0, 2.0])},   # shorter
    ]
    traj = vz.mean_trajectory(rows, "entropy")
    # common length is 2; averaged: [(1+3)/2, (2+2)/2] = [2.0, 2.0]
    assert traj[("mean_diff", "0.0")] == [2.0, 2.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_viz_length.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'viz_length'`.

- [ ] **Step 3: Implement `viz_length.py`**

```python
# src/viz_length.py
"""Length-steering figures: coherence trajectory (intrinsic) + verdict-vs-cutoff (judge).

    python viz_length.py --dataset cities
Reads length_steer_<ds>.csv and judge_length_<ds>.csv; writes two PNGs."""
import argparse
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = {"mean_diff": "#228833", "resid_pc1": "#cc3311"}
_STYLES = {"0.0": ":", "0.3": "--", "1.0": "-"}


def mean_trajectory(rows, signal):
    groups = {}
    for r in rows:
        arr = json.loads(r[signal])
        groups.setdefault((r["direction"], str(float(r["tau"]))), []).append(arr)
    out = {}
    for key, arrs in groups.items():
        m = min(len(a) for a in arrs)
        out[key] = [sum(a[i] for a in arrs) / len(arrs) for i in range(m)]
    return out


def plot_coherence(ds):
    rows = list(csv.DictReader(open(f"length_steer_{ds}.csv")))
    traj = mean_trajectory(rows, "entropy")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (dirn, tau), ys in sorted(traj.items()):
        ax.plot(range(1, len(ys) + 1), ys, _STYLES.get(tau, "-"),
                color=_COLORS.get(dirn, "#333333"), label=f"{dirn} τ={tau}")
    ax.set_xlabel("generated token position")
    ax.set_ylabel("mean next-token entropy (nats) — higher = degrading")
    ax.set_title(f"{ds}: coherence trajectory under steering")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"plot_length_coherence_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_length_coherence_{ds}.png")


def plot_verdict(ds):
    rows = list(csv.DictReader(open(f"judge_length_{ds}.csv")))
    # direction column is "<dir>_tau<tau>", scale column is the cutoff
    cutoffs = sorted({int(float(r["scale"])) for r in rows})
    dirs = sorted({r["direction"] for r in rows})
    fig, axes = plt.subplots(1, len(dirs), figsize=(4.2 * len(dirs), 4.2), squeeze=False)
    for ax, dirn in zip(axes[0], dirs):
        fr = {"TRUE": [], "FALSE": [], "INCOHERENT": []}
        for k in cutoffs:
            sub = [r for r in rows if r["direction"] == dirn and int(float(r["scale"])) == k]
            n = len(sub) or 1
            for v in fr:
                fr[v].append(sum(r["verdict"] == v for r in sub) / n)
        ax.plot(cutoffs, fr["TRUE"], "o-", color="#228833", label="TRUE")
        ax.plot(cutoffs, fr["FALSE"], "s-", color="#cc3311", label="FALSE")
        ax.plot(cutoffs, fr["INCOHERENT"], "^--", color="#999999", label="INCOHERENT")
        ax.set_title(dirn); ax.set_xlabel("cutoff (tokens)"); ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("fraction"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"{ds}: verdict vs generation length", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"plot_length_verdict_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_length_verdict_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_coherence(a.dataset)
    plot_verdict(a.dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_viz_length.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/viz_length.py tests/test_viz_length.py
git commit -m "feat(length): coherence-trajectory + verdict-vs-cutoff plots"
```

---

### Task 5: Cluster job scripts + runbook

**Files:**
- Create: `deltaai/run_length_steer.slurm`
- Create: `deltaai/run_length_judge.slurm`
- Create: `deltaai/LENGTH_STEER_RUN.md`

**Interfaces:**
- Consumes: the gen env `.venv-dct-gpu`, the judge env `.venv-judge-gpu`, and the existing `judge_results.py --mode steer --backend olmo` with `--steer-input length_prefixes_<ds>.csv --steer-output judge_length_<ds>.csv`.

- [ ] **Step 1: Write `run_length_steer.slurm` (clone of `run_mag_steer.slurm`)**

```bash
#!/bin/bash
#SBATCH --job-name=length_steer
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=length_steer_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== length_steer $ds $(date) ==="
  PYTHONPATH=src python3 src/length_steer.py --dataset "$ds" --device cuda \
    || echo "!!!! $ds FAILED — continuing"
done
echo "=== length steer done $(date) ==="
```

- [ ] **Step 2: Write `run_length_judge.slurm` (clone of `run_mag_judge.slurm`, pointed at the length CSVs)**

```bash
#!/bin/bash
#SBATCH --job-name=length_judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=length_judge_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== length_judge $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "length_prefixes_${ds}.csv" \
    --steer-output "judge_length_${ds}.csv" \
    --steer-plot "plot_judge_length_${ds}.png" \
    || echo "!!!! $ds FAILED — continuing"
done
echo "=== length judge done $(date) ==="
```

- [ ] **Step 3: Write `LENGTH_STEER_RUN.md`**

Write a self-contained runbook modeled on `deltaai/MAG_E4_RUN.md` with these steps (no placeholders — real commands):
1. **Laptop:** confirm `mag_dir_<ds>.npz` and `truth_dir_<ds>.npz` exist (they carry `resid_pc1_unit`, `A_prefix_norm`, `layer`, `mean_diff`). rsync code up excluding `*.npz`, then a second rsync of `mag_dir_*.npz truth_dir_*.npz`. (Copy the two-rsync pattern and the `--exclude '*.npz'` warning verbatim from `MAG_E4_RUN.md`.)
2. **Cluster:** put the account into both scripts: `ACC=$(grep -o -- '--account=[^ ]*' deltaai/run_dct.slurm | head -1 | cut -d= -f2); sed -i "s/ACCOUNT_NAME/$ACC/" deltaai/run_length_steer.slurm deltaai/run_length_judge.slurm`.
3. `sbatch deltaai/run_length_steer.slurm` — done when `length_steer_*.csv` and `length_prefixes_*.csv` (2 of each) exist.
4. After step 3's CSVs exist: `sbatch deltaai/run_length_judge.slurm` — done when `judge_length_*.csv` (2) exist.
5. **Laptop:** rsync back `length_steer_*.csv length_prefixes_*.csv judge_length_*.csv`, then `PYTHONPATH=src .venv/bin/python src/viz_length.py --dataset cities` and `--dataset common_claim_true_false`.
Include the same "If something goes wrong" table shape as `MAG_E4_RUN.md`, adapted (stale `mag_dir` → `KeyError: 'resid_pc1_unit'`; judge input missing → confirm `length_prefixes_*.csv` exist first).

- [ ] **Step 4: Verify the slurm scripts are syntactically valid**

Run: `bash -n deltaai/run_length_steer.slurm && bash -n deltaai/run_length_judge.slurm && echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add deltaai/run_length_steer.slurm deltaai/run_length_judge.slurm deltaai/LENGTH_STEER_RUN.md
git commit -m "feat(length): GH200 gen+judge slurm scripts and runbook"
```

---

## Self-Review

- **Spec coverage:** goal (long-gen steering, two-signal) → Tasks 3+4; directions mean_diff+resid_pc1 calibrated injection → Task 3 `load_directions`/`injected_vector`; both datasets, taus {0,.3,1}, 96 tokens, cutoffs → Global Constraints + Task 3; prompt-set split (cities programmatic / common_claim curated) → Task 1; intrinsic signals → Task 2; judge-at-cutoffs reusing OLMo judge → Task 3 prefix CSV + Task 5 judge slurm; artifacts (2 CSVs + 2 plots + runbook) → Tasks 3/4/5. All spec sections map to a task.
- **Placeholder scan:** none — every code step has full code; the runbook step (Task 5 Step 3) specifies exact commands to copy from `MAG_E4_RUN.md`.
- **Type consistency:** `injected_vector(tau, unit_dir, a_prefix_norm)` signature identical in Task 3 code and tests; the prefix CSV schema `direction,scale,prompt,completion` is consistent across Task 3 (writer), Task 5 (judge `--steer-input`), and Task 4 (`plot_verdict` reader, which reads the judge output `judge_length_<ds>.csv`). `mean_trajectory` reads `entropy` column written by Task 3.
- **Note vs spec:** spec §method mentions per-position `max_prob`/`entropy`/`rep3` all logged (done, Task 3), and the coherence plot uses entropy (Task 4) with max_prob/rep3 available in the CSV for follow-up — consistent with the spec's "entropy and/or max-prob".
