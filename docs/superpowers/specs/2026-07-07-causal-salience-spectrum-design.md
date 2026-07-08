# Causal-Salience Spectrum: Does DCT Recover a Concept in Proportion to Its Causal Dominance?

*Design spec — 2026-07-07*

## 1. Motivation & thesis

The existing funnel established a **null for truth**: DCT (the model's most causally-salient
directions, found label-free) does not recover the supervised truth direction — not as a single
vector, a combination, a transferable feature, or the non-linear XGBoost structure
(`docs/DCT_VS_TRUTH_FINDINGS.md`, `docs/DCT_VS_XGBOOST_FINDINGS.md`). Supervised truth steers only
weakly (`plot_findings_steering.png`).

The null has one unclosed hole and one missed opportunity:

- **Hole (positive control):** does DCT fail on truth because truth isn't causal, or because
  DCT/our pipeline recovers *nothing* supervised? Without a concept where DCT *does* recover a
  known causal direction, a skeptic can dismiss the whole comparison.
- **Opportunity (generality):** "truth" is one point. If we measure several concepts spanning a
  range of causal salience, the binary null becomes a **quantified law.**

**Thesis to test:**

> An unsupervised causal method (DCT) recovers a concept's supervised direction *to the degree that
> concept is causally load-bearing.* Refusal (behaviorally dominant) → recovered and steerable;
> truth (a decodable correlate) → not recovered. Decodable-vs-causal is a measurable spectrum, not
> a truth-specific quirk.

**The money figure** (`plot_findings_spectrum.png`): one point per concept.

- **x-axis — causal salience:** behavioral steering-effect size of the concept's *supervised*
  direction (judge-scored, normalized against degradation). Machinery: `src/steer_supervised.py`.
- **y-axis — DCT recovery:** max single-vector cosine + subspace fraction between DCT's top
  directions and the supervised direction, over the random baseline. Machinery:
  `src/compare_directions.py`.

Prediction: monotonic positive. Truth ≈ (low-x, low-y) is already measured; refusal should be
(high-x, high-y); toxicity/sycophancy fill the middle.

**Scope decision (from brainstorming):** this program pursues the *unified spectrum thesis*
(positive control + harden + broaden). It explicitly does **not** build a new steering method
(no control-theory / A-LQR / semantic-supervision work in this spec — deferred).

## 2. Key insight: everything is one pipeline, re-parameterized by concept

The existing scripts are already `--dataset`-parameterized. A "concept" is just a
`(text, label)` contrast CSV in the same format the truth datasets already use (`statement`,
`label` — see `CLAUDE.md`). Once each concept's data is shaped that way, the *entire* existing
pipeline runs unchanged per concept:

| Step | Existing script | Per-concept role |
|---|---|---|
| Extract activations | `src/extract.py` | last-token residual stream, all layers → `activations/acts_<concept>.npz` |
| Peak-layer probe sweep | `src/analyze.py` | pick the concept's best probe layer (the DCT source layer) |
| Supervised direction | `src/export_truth_dir.py` → generalize | `mean_diff` + `grad` via `funnel_utils.mean_diff_dir` / `grad_dir` |
| DCT run | `src/run_dct_data.py` | 512 factors at `source→source+9` → `dct_V/U_<concept>.pt` |
| **y-axis** (recovery) | `src/compare_directions.py` | max-cos + subspace vs random baseline |
| **x-axis** (salience) | `src/steer_supervised.py` | ± magnitude sweep of the supervised direction |
| Quantify behavior | `src/judge_results.py` → extend (§4) | judge completions; degradation metrics |
| Spectrum figure | **new** `src/viz_spectrum.py` | scatter x vs y, one point per concept |

The only genuinely new code is: (a) the judge harness build-out (§4), (b) `viz_spectrum.py`, and
(c) small per-concept data-prep adapters (§3).

## 3. Concepts & data

Each concept becomes a balanced `(text, label)` CSV under `got_datasets/`, `label=1` = concept
present. Behavioral prompts for the steering sweep live alongside (analogous to
`steer_supervised.py`'s `FACTUAL_PROMPTS`).

| Concept | label=1 / label=0 | Data source (public, unblocked) | Behavioral (x-axis) judge |
|---|---|---|---|
| **Refusal** (positive control) | harmful instruction / harmless instruction | AdvBench harmful + Alpaca harmless | refusal classifier |
| **Toxicity** | toxic continuation / clean | RealToxicityPrompts (+ labels) | toxicity classifier |
| **Sycophancy** | sycophantic / non-sycophantic | Anthropic sycophancy eval set | LLM-judge |
| **Truth** (done) | true / false statement | existing `got_datasets/` | truth judge (§4) |

**Refusal is the go/no-go control** and starts on public data now. Akshal's jailbreaking dataset
is added later as a *second* refusal point (same pipeline), not a blocker.

Per-concept data prep is a small adapter script (`src/prep_<concept>.py`) that emits the contrast
CSV + a behavioral-prompt list. This is the only concept-specific logic.

## 4. Measurement harness (thread B) — the shared, reusable layer

This is the enabling infrastructure; every concept and the existing truth results plug into it.

### 4.1 Local HF judges (explicit requirement: run locally)

Current `src/judge_results.py` calls the **Anthropic API** (Haiku) — see its `get_client()` /
`ask()`. We add a **local Hugging Face backend** so judging runs on the GH200 with no API
dependency, selected via `--backend {anthropic, local-hf}`:

- **Truth:** `allenai/truthfulqa-truth-judge-llama2-7B` (truthfulness) and
  `allenai/truthfulqa-info-judge-llama2-7B` (informativeness). Llama-2-7B fine-tunes; load with
  `transformers` on GPU. They expect the TruthfulQA judge prompt format (`Q: ... A: ... True:` →
  the model completes `yes`/`no`); a small adapter maps our `(stem, completion)` rows into that
  format.
- **Refusal:** a local refusal classifier — `meta-llama/Llama-Guard-3-1B` (or a keyword+small
  classifier fallback on "I can't/I'm sorry" patterns).
- **Toxicity:** `unitary/toxic-bert` (Detoxify), local.

All judges expose one interface: `judge(concept, stem, completion) -> {verdict, score}`.

### 4.2 Degradation metrics (separate "flipped" from "broke")

The open caveat in the current steering result is that hard steering degrades output, confounding
"made it false" with "made it gibberish." Add per-completion:

- **Perplexity** under a reference LM (GPT-2 or gemma-2-2b itself) — fluency.
- **distinct-n** and **self-BLEU** — diversity / model-collapse.

These normalize the x-axis: causal salience = behavioral flip *rate at fixed degradation*, not raw
flip rate at a magnitude that also destroys the output.

### 4.3 Quick win — retrofit onto existing truth results first

Run the built-out harness on the **existing** `steer_supervised_<ds>.csv` and
`interpret_top10_<ds>.md` to produce judged versions (`judge_steer_*`, `judge_interpret_*` — the
plot/CSV shapes already exist in `judge_results.py`). This makes the current truth story
quantitative and closes the "hand-read" caveat in `docs/PI_MEETING_RESULTS.md` before any new
concept is run.

### 4.4 Judge validation

The allenai judges were trained on TruthfulQA Q/A, not our completion format. Before trusting them,
validate agreement against a hand-labeled subset (~50 completions); report the agreement rate in
the harness output. If agreement is poor, fall back to the Anthropic backend for truth and note it.

## 5. Phase sequence & gates

- **Phase 0 — harness (B):** build the local-HF judge backend + degradation metrics; download the
  7B judges *on the cluster* (not the disk-constrained laptop — see `MEMORY.md`); validate (§4.4);
  retrofit onto existing truth results (§4.3). Deliverable: judged truth steering plot + validated
  harness.
- **Phase 1 — refusal positive control (A):** prep public refusal data → extract → peak layer →
  supervised direction → DCT run → `compare_directions` (y) + judged `steer_supervised` (x).
  **GO/NO-GO GATE:** if DCT does **not** recover refusal above the random baseline, halt the
  broadening and pivot to diagnosing DCT/pipeline (the null's premise would be in question).
- **Phase 2 — broaden (D):** toxicity, then sycophancy, same pipeline. Add Akshal's jailbreaking
  data as a second refusal point when available.
- **Phase 3 — synthesis:** `viz_spectrum.py` → the money figure; findings doc
  `docs/CAUSAL_SALIENCE_SPECTRUM_FINDINGS.md`.

## 6. New / changed files (surface summary)

- **New:** `src/prep_<concept>.py` (data adapters), `src/viz_spectrum.py` (money figure),
  `src/judges/` (local-HF judge backends + degradation metrics),
  `docs/CAUSAL_SALIENCE_SPECTRUM_FINDINGS.md`.
- **Changed (generalize, keep truth working):** `src/export_truth_dir.py` →
  `src/export_concept_dir.py` (any labeled contrast; truth stays a special case);
  `src/judge_results.py` (add `--backend local-hf`, route to `src/judges/`).
- **Unchanged, run per concept:** `src/extract.py`, `src/analyze.py`, `src/run_dct_data.py`,
  `src/compare_directions.py`, `src/steer_supervised.py`, `src/interpret_top10.py`,
  `src/funnel_utils.py`.

## 7. Risks & mitigations

- **Judge/format transfer:** allenai judges expect TruthfulQA format → §4.4 validation gate;
  Anthropic fallback.
- **Phase-1 gate is load-bearing:** the whole program assumes DCT can recover *some* causal
  direction; Phase 1 tests exactly that, early and cheaply, before further investment.
- **Disk/compute:** two Llama-2-7B judges are large; run + cache them on the GH200, never the
  disk-constrained laptop (`MEMORY.md`). Each concept = one ~9-min GPU DCT run + judge inference.
- **Concept data quality:** RealToxicityPrompts / sycophancy sets are messier than curated truth
  data; report per-concept probe accuracy so a weak supervised direction (bad x/y anchor) is
  visible, not silent.

## 8. Success criteria

- **Primary:** the spectrum figure exists with ≥3 concepts, and refusal (positive control) lands
  clearly above the random baseline on the y-axis — validating that DCT *can* recover a causal
  direction.
- **Thesis-confirming (hoped):** y-axis recovery increases monotonically with x-axis salience;
  truth remains the low-low corner.
- **Thesis-negating (still valuable):** if DCT recovers nothing (even refusal), that redirects the
  project to DCT diagnosis — a real finding, caught at the Phase-1 gate.
- **Secondary:** the existing truth steering result is quantitative (judge-scored) and the
  "hand-read" caveat is removed from the PI doc.

## 9. Out of scope (explicitly deferred)

New steering/decoding *methods*: control-theory / reachability, A-LQR behavioral eval, SHAP-derived
XGBoost steering directions, semantic supervision. These are the "constructive pivot" the user
deprioritized; revisit after the spectrum result.
