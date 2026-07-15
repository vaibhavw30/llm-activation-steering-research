# Implementation Plan — OLMo-3 as the Local LLM-as-Judge

*Goal: give the funnel/spectrum an open, local, no-API-key judge by adding an **OLMo-3-7B**
backend to `judge_results.py`, alongside the existing Anthropic and `local-hf` (TruthfulQA-7B)
backends. Written to be executed task-by-task on the M5 Pro (24 GB unified, MPS). Last updated:
2026-07-14.*

Companion docs: `STATUS_SINCE_LAST_MEETING.md` (why we need a judge), `PROJECT_CONTEXT_AND_ROADMAP.md`
(§7B — make the qualitative results quantitative), the causal-salience-spectrum plan
(`docs/superpowers/plans/2026-07-07-causal-salience-spectrum.md`, Tasks 4–7).

---

## 0. Why OLMo-3, and why now

We have three judge options; each has a problem this plan fixes:

| Backend | State | Problem |
|---|---|---|
| Anthropic (Haiku) | built, default | **needs an API key we don't have**; billed; not reproducible/citable |
| `local-hf` (allenai TruthfulQA-7B yes/no judges) | built (Task 4/5), untested | narrow **yes/no completion judges in TruthfulQA Q/A format** — a format mismatch with our completions (flagged as a validation risk); **can't do interpret mode** |
| **OLMo-3-7B-Instruct** (this plan) | to build | — |

**Why OLMo-3-7B-Instruct is the right judge:**
- **Fully open** — weights, training data (Dolma3 / Dolci mixtures), code, and recipe all public
  ([allenai.org/olmo](https://allenai.org/olmo)). Reproducible and citable for the PI; no gating
  friction (unlike Llama-Guard).
- **Local + free** — runs on the M5 Pro; removes the no-API-key blocker entirely.
- **A general chat judge**, not a narrow yes/no head. It reuses our *existing* Anthropic prompts
  (`STEER_SYS` asks for `{"verdict": TRUE|FALSE|INCOHERENT}` JSON; `INTERPRET_SYS` asks for a
  label + `manipulates_truth`), so it supports **both `steer` and `interpret` modes** — the
  TruthfulQA backend only did `steer`.
- **Modern instruction-following** (2025–26 release) — far better at emitting clean JSON and
  honoring a rubric than llama2-7B.

Model IDs (HuggingFace, `allenai/`):
[`Olmo-3-7B-Instruct`](https://huggingface.co/allenai/Olmo-3-7B-Instruct) (primary judge),
[`Olmo-3-7B-Think`](https://huggingface.co/allenai/Olmo-3-7B-Think) (reasoning variant, optional —
see Task 5).

**Where the Dolci datasets fit (honest scope).**
[`Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) (2.15 M
instruction pairs) and [`Dolci-Think-SFT-7B`](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B)
(1.8 M reasoning traces) are the **post-training data for OLMo-3-7B** — *training* mixtures, not
evaluation sets. They matter here two ways, both secondary:
1. **Provenance** — they tell us what the OLMo judge was trained on (reasoning, math, coding,
   multilingual, **safety**, tool-use), i.e. its coverage/biases when it judges our completions.
2. **Concept-contrast source (optional, Task 6)** — the **safety** subset of `Dolci-Instruct-SFT`
   is a candidate source of refusal contrast pairs for the spectrum's positive control, an
   alternative/supplement to the AdvBench+Alpaca build in `prep_refusal.py`.

They are **not** a judge-validation label set — for that we use our own true/false CSVs (Task 4).

---

## 1. Hardware fit (M5 Pro: 24 GB unified, MPS, ~600 GB disk)

- `Olmo-3-7B-Instruct` in fp16 ≈ **14 GB** weights → fits in 24 GB with room for KV cache. Use
  `--device mps`. (Only *one* 7B loads at a time here, unlike the TruthfulQA `TruthJudge` which
  loads two.)
- `steer` mode: `max_new_tokens ≈ 80`, ~144 completions/dataset → minutes on MPS.
- `interpret` mode: 10 vectors/dataset, longer prompts, `max_new_tokens ≈ 200` → also minutes.
- `Olmo-3-7B-Think` emits long `<think>…</think>` traces → larger KV cache + slower; keep it for
  the hard FALSE-vs-INCOHERENT calls only (Task 5), not the default.
- transformers 5.12.1 (already in `.venv`) supports OLMo-3 chat templates. Keep
  `HF_HUB_DISABLE_XET=1` for the download.

---

## 2. Tasks

### Task 1 — OLMo chat-judge module

**Files:** create `src/judges/olmo_judge.py`.

**Interface:** `OlmoJudge(model="allenai/Olmo-3-7B-Instruct", device="mps")` with
`.chat(system: str, user: str, max_tokens: int = 200) -> str` returning the model's text (from
which the existing `extract_json` pulls the verdict). Mirrors the Anthropic `ask()` contract so the
existing `run_steer` / `run_interpret` can drive it.

```python
"""Local OLMo-3 chat judge (open, no API key). Mirrors the Anthropic `ask()` contract so
run_steer/run_interpret in judge_results.py work unchanged."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_OLMO = "allenai/Olmo-3-7B-Instruct"


class OlmoJudge:
    def __init__(self, model=DEFAULT_OLMO, device="mps"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16).to(device).eval()

    def chat(self, system, user, max_tokens=200):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
```

**Verify (on the M5 Pro):**
```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from judges.olmo_judge import OlmoJudge
j = OlmoJudge(device='mps')
from judge_results import STEER_SYS, extract_json
for stem, comp in [('The capital of Japan is','Tokyo'), ('Two plus two equals','three')]:
    print(comp, '->', extract_json(j.chat(STEER_SYS, f'Stem: \"{stem}\"\nCompletion: \"{comp}\"', 80)))
"
```
Expected: `Tokyo → verdict TRUE`, `three → verdict FALSE`.

### Task 2 — Wire `--backend olmo` into `judge_results.py`

**Files:** modify `src/judge_results.py`.

**Design:** OLMo is a general chat judge, so route it through the *existing* `run_steer` /
`run_interpret` (both already take a `client` and call `ask(client, model, system, user)`). Add a
tiny adapter so `ask` dispatches to `OlmoJudge.chat` when the client is an `OlmoJudge`.

- **Step 1** — argparse: extend the existing `--backend` choices to include `"olmo"`, add
  `--olmo-model` (default `allenai/Olmo-3-7B-Instruct`). `--device` already exists (default it to
  `mps` in docstrings/help for this machine).
- **Step 2** — make `ask()` backend-aware (minimal change):
  ```python
  def ask(client, model, system, user, max_tokens=200, retries=4):
      from judges.olmo_judge import OlmoJudge
      if isinstance(client, OlmoJudge):
          return client.chat(system, user, max_tokens)
      # ... existing Anthropic retry loop unchanged ...
  ```
- **Step 3** — in `main()`, add the branch:
  ```python
  elif args.backend == "olmo":
      sys.path.insert(0, os.path.dirname(__file__))
      from judges.olmo_judge import OlmoJudge
      client = OlmoJudge(args.olmo_model, args.device)
      if args.mode == "steer":
          run_steer(client, args.olmo_model, args.dataset, args.limit)
      else:
          run_interpret(client, args.olmo_model, args.dataset, args.limit)
  ```
  This gives OLMo **both** modes for free, reusing `STEER_SYS`, `INTERPRET_SYS`, `extract_json`,
  the CSV writers, and `_steer_summary_and_plot`.
- **Step 4** — `--help` shows `olmo` in `--backend`; Anthropic/local-hf paths unchanged.

### Task 3 — Run the quick-win (make funnel Test 2 quantitative)

The payoff: turn the hand-read/keyword-scored steering sweep into a judged FALSE-vs-INCOHERENT
curve, closing the biggest caveat in `PI_MEETING_RESULTS.md`.

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python src/judge_results.py --mode steer --backend olmo \
    --dataset cities --device mps --limit 24        # smoke first
HF_HUB_DISABLE_XET=1 .venv/bin/python src/judge_results.py --mode steer --backend olmo \
    --dataset cities --device mps                    # full
HF_HUB_DISABLE_XET=1 .venv/bin/python src/judge_results.py --mode steer --backend olmo \
    --dataset common_claim_true_false --device mps
```
Outputs `judge_steer_<ds>.csv` + `plot_judge_steering_<ds>.png` — the curve that separates
"steering made it *lie*" (FALSE = causal truth effect) from "steering made it *gibberish*"
(INCOHERENT = degradation). Then `interpret` mode → the 0/10 top-vector truth-manipulation count:
```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python src/judge_results.py --mode interpret --backend olmo \
    --dataset cities --device mps
```

### Task 4 — Validate the OLMo judge before trusting it

Reuses the roadmap's Task 6 idea, but with data we have. The judge is only useful if it agrees with
ground truth on unambiguous cases.

**Files:** create `src/validate_judge.py` (or extend the existing stub if present).
**Method:** sample N statements from `got_datasets/cities.csv` (which have gold `label`), form the
`(stem, completion)` the judge sees, ask the OLMo judge, and compute agreement between its
TRUE/FALSE verdict and the gold label on the *unsteered* (scale-0) rows. Report accuracy + a small
confusion table.
**Gate:** if agreement ≥ 0.85 → trust OLMo for that concept; if low → fall back to Anthropic (once a
key exists) or the TruthfulQA judge, and note it. This is the honest guard the PI will ask for.

### Task 5 (optional) — OLMo-3-7B-Think for hard degradation calls

The FALSE-vs-INCOHERENT boundary at extreme steering is genuinely ambiguous. `Olmo-3-7B-Think`
reasons before answering, which may adjudicate it better. Add `--olmo-model allenai/Olmo-3-7B-Think`
support: parse the verdict from *after* the `</think>` tag (strip the reasoning trace before
`extract_json`). Use only for a re-judge of the ambiguous high-|scale| rows, not the full sweep
(slower, larger KV cache).

### Task 6 (optional) — Dolci safety subset as a refusal-contrast source

For the spectrum's refusal positive control (roadmap Task 11), the `Dolci-Instruct-SFT` **safety**
subset is an alternative to AdvBench+Alpaca. **Files:** extend `src/prep_refusal.py` with a
`--source dolci` path that streams the safety rows via `datasets.load_dataset(
"allenai/Dolci-Instruct-SFT", streaming=True)`, filters to safety-tagged examples, and emits the
same `(statement, label)` contrast CSV the pipeline expects (1 = harmful/refusable, 0 = benign).
Keep AdvBench+Alpaca as the default; Dolci is a second, differently-sourced refusal point that
strengthens the control. (Verify the subset has a usable source/tag column first — it's a 2 M-row
mixture; stream, don't download.)

---

## 3. Execution order & decision points

1. **Task 1 → Task 2 → Task 3 (cities smoke)** — the critical path. Gets a real judged steering
   plot on the laptop with no API key. Stop and eyeball the smoke output before the full run.
2. **Task 4** — validate before believing any judged number. This gates everything downstream.
3. **Task 3 full** (cities + common_claim, steer + interpret) — closes the funnel Test 1/2 caveats.
4. **Tasks 5–6** — optional, only if (5) the degradation boundary stays noisy, or (6) you want a
   second refusal point for the spectrum.

**Do not commit** `judge_steer_*.csv` if they're large/regenerable (the roadmap keeps them
gitignored); commit the code, the plots, and small aggregates.

---

## 4. What this unblocks

- **Funnel Test 2 → quantitative**: FALSE-vs-INCOHERENT curve instead of an 8-prompt keyword score.
- **Funnel Test 1 → quantitative**: a 0/10 "manipulates truthfulness" count over the top DCT
  vectors (interpret mode — newly possible because OLMo is a general chat judge).
- **Spectrum x-axis**: the judge is the behavioral-salience scorer (`concept_salience` consumes
  `judge_steer_*.csv`), so an open local judge unblocks the whole spectrum measurement without a key
  or the cluster.
- **Reproducibility for the PI**: a fully-open judge with public weights + training data, citable,
  no billing.

## 5. Sources

- OLMo overview & releases: [allenai.org/olmo](https://allenai.org/olmo)
- Judge model: [allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct);
  reasoning variant [allenai/Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think)
- Post-training data (provenance / concept-contrast source):
  [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT),
  [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B)
