# Status — What We've Done Since the Last Meeting

*Progress summary for the next PI check-in. Organized around the three active threads:
generating the labels (LLM-as-judge), interpreting DCT's top vectors, and steering with the
supervised truth direction. Last updated: 2026-07-14 — **the judge is now built, validated, and run
twice (a first pass, then a corrected rerun); both hand-read caveats below are CLOSED with numbers.***

Companion docs: `INVESTIGATION_steering_validity.md` (**the full judge results + statistics — §6 has
the corrected-rerun numbers this update summarizes**), `PI_MEETING_RESULTS.md` (the master funnel
writeup — the last meeting's deliverable), `DCT_VS_XGBOOST_FINDINGS.md` (the non-linear extension),
`PROJECT_CONTEXT_AND_ROADMAP.md` (the full roadmap + resume instructions).

> **Headline since 2026-07-07:** the LLM-judge went from "wired but never run" to **run end-to-end on
> the DeltaAI GH200 with the OLMo-3-7B backend, validated at a 0.970 gate, and rerun with a corrected
> design.** Both qualitative caveats are now quantitative: interpret is **0/10** (was hand-read), and
> steering shows **a bounded lie-asymmetry null (Δ=−0.010, TOST-equivalent to 0) plus significant,
> symmetric degradation** (was a keyword heuristic). Detail: `INVESTIGATION_steering_validity.md` §6.

---

## 0. The through-line

The last meeting established the **DCT-vs-truth funnel**: truth is ~99% linearly decodable but is
**not** among gemma-2-2b's most causally-salient directions — *easy to read, hard to push*. That
conclusion rested on two pieces of **qualitative, hand-read** evidence (the DCT vector
interpretations and the steering completions) plus quantitative subspace/transfer math.

Everything since has pushed on two fronts:
1. **Harden the qualitative evidence into numbers** — build an automated judge so "interpret" and
   "steer" produce scores instead of hand-reads.
2. **Generalize the binary null into a spectrum** — *DCT recovers a concept in proportion to how
   causally load-bearing it is* (the causal-salience-spectrum initiative).

---

## 1. Thread — Interpreting DCT's top vectors (funnel Test 1)

**What it is.** DCT returns 512 directions: `V` are the input directions you steer with, `U` the
downstream effects. Rank the top-10 by potency `‖U‖` (the model's strongest causal levers). Inject
each during generation across 10 probe prompts — 5 factual ("The capital of Japan is") + a few
open-ended tone/format ("I think that") — and read steered-vs-unsteered completions to label what
each direction *does*.

**Status: DONE and now JUDGED — 0/10 both datasets (0/20 pooled).**

**Finding.** The top vectors are about **geography / format / tone**, never a clean truthfulness
knob. Falsehoods ("the capital of Japan is in Germany") appear only as a *byproduct* of a geography
vector scrambling entities — not a true↔false switch.

**Caveat CLOSED.** The hand-reading is now a machine count. Under a strict flip-only rubric the OLMo
judge flags **0/10 vectors on each dataset** (0/20 pooled; 95% upper bound ≤0.14) as manipulating
truthfulness — reproducing our hand-read exactly, without a human in the loop. (A first-pass loose
rubric had over-counted 3/10; drilling into those six confirmed all were confabulation / register /
incoherence, and the strict rubric erased them. Full record: `INVESTIGATION_steering_validity.md`
§1, §6.4.)

---

## 2. Thread — Steering the supervised truth direction (funnel Test 2)

**What it is.** Take the probe's own truth direction — computed two ways, `mean_diff` (difference of
class means) and `grad` (logistic-probe gradient) — and inject it at the truth-peak layer across a
signed magnitude sweep (−120 … +120). Negative pushes away from truth, positive toward. Read whether
factual completions flip.

**Status: DONE and now JUDGED — `plot_judge_steering_*.png` (TRUE/FALSE/INCOHERENT curves, 576
completions/dataset). The keyword heuristic is retired.**

**Finding (revised by the judge).** When the confound is resolved, the effect is **degradation, not
lie-induction**. The apparent "real falsehoods" the keyword heuristic flagged at large negative scale
turn out to be **symmetric**: steering hard in *either* direction lowers the TRUE rate (pooled
Cochran-Armitage z = **−5.06, p < 1e-6**) the same on both sides, while there is **no −vs+ FALSE
asymmetry** at all — pooled Δ = **−0.010, 95% CI [−0.042, +0.023]**, formally equivalent to zero
(TOST, every slice), corroborated by a prompt-clustered bootstrap. Amplifying the truth axis **breaks**
the model; it does not make it **lie**. So the direction is a *degradation* lever, not a *truth* lever
— a sharper, and cleaner, version of "real but minor."

**Caveat CLOSED.** The noisy 1/8 keyword heuristic is replaced by 576 judged completions/dataset behind
a 0.970-validated judge. We also found and fixed a *measurement-validity* artifact in the first pass
(gemma's rambling 24-token completions were scored whole, failing correct answers on their tails); the
corrected design (8-token completions, answer-only rubric, 32 factual prompts) raised the unsteered
baseline TRUE rate 0.50 → 0.81. Full statistics: `INVESTIGATION_steering_validity.md` §3, §6.

---

## 3. Thread — Generating the labels (the LLM-as-judge) — *the main recent work*

Threads 1 and 2 are only as strong as how their outputs are scored, and both were hand-read /
keyword-scored. The fix is an **LLM-as-judge** that classifies each completion automatically, turning
the qualitative story into numbers. Two modes:

- **steer mode** — label each steered completion **TRUE / FALSE / INCOHERENT**. This is the crucial
  one: it separates "the direction made the model *lie*" (FALSE = a genuine causal truth effect) from
  "it just made the model produce *gibberish*" (INCOHERENT = mere degradation) — the precise
  confound the keyword heuristic couldn't resolve in Test 2.
- **interpret mode** — for each top-10 DCT vector, auto-decide "does this manipulate truthfulness?"
  turning Test 1's hand-reading into a count (0/10 → the null is legitimate).

**Status of the judge, precisely:**

| Piece | State |
|---|---|
| Harness `judge_results.py`, Anthropic API backend | Pre-existed; **unusable for us** (no API key) |
| **Local HF judge `src/judges/local_hf.py`** | **Built today.** `allenai/truthfulqa-truth-judge-llama2-7B` + info-judge (truth), Llama-Guard-3-1B (refusal), toxic-bert (toxicity) |
| **`--backend local-hf` wired into `judge_results.py`** | **Built today** (`run_steer_local`, maps judge dicts → TRUE/FALSE/INCOHERENT) |
| Verification | Imports clean, dispatch works, `--help` shows flags, **test suite green** |
| **`--backend olmo` (OLMo-3-7B-Instruct)** | **Built + shipped since.** Open model, no API key; the backend actually used for the run. `src/judges/olmo_judge.py` + a validation gate `src/validate_judge.py` |
| **Actual label generation** | **DONE** — run on the DeltaAI GH200 (validation gate **0.970 → PASS**), then rerun with the corrected design. Steer curves + interpret counts + full statistics all produced |

**How it actually ran (superseding the "run it locally" plan below).** The 7B judge swap-thrashed the
24 GB Mac under load, so we moved to the **DeltaAI GH200** cluster with a separate `.venv-judge-gpu`
(the OLMo backend needs transformers ≥5, which conflicts with DCT's pinned 4.51.3 — hence two venvs).
The run: 24-row smoke → **validation gate** (`validate_judge.py`, aborts if <0.85 on gold cities
labels; it hit **0.970**) → full cities + common_claim steer sweeps → both interpret runs. A first pass
exposed a measurement-validity artifact (Thread 2); the **corrected rerun** (chained steer→judge via
SLURM `--dependency=afterok`) produced the final numbers folded into this update.

---

## 4. Other progress under the hood

- **DCT-vs-XGBoost non-linear extension (merged).** XGBoost opens a **+0.057** truth gap over a
  linear probe on the *full* residual stream (common_claim), but that gap **collapses to ≈0** inside
  DCT's top-k causal subspace (no better than random projections). So the null holds *non-linearly*
  as well as linearly. See `DCT_VS_XGBOOST_FINDINGS.md`.
- **Causal-salience-spectrum initiative (spec + plan + foundation merged).** Thesis: *DCT recovers a
  concept's supervised direction in proportion to how causally load-bearing the concept is.* Truth
  sits at (low salience, low recovery); **refusal** is the planned positive control expected to sit
  high on both — and doubles as a **go/no-go gate** for the whole premise. The judge harness above is
  a prerequisite for the spectrum's behavioral **x-axis**. The pure-logic half (6 modules + tests) is
  implemented and green; the GPU/cluster half is precisely staged in the roadmap.
- **Housekeeping (today).** Installed `pytest` + `datasets`; fixed two latent bugs (a silent
  bogus-baseline in the salience aggregation; unguarded file reads in the spectrum plot builder);
  added regression tests.

---

## 5. One-sentence version for the PI

> Since last meeting we extended the null into the non-linear regime, reframed decodable-vs-causal as
> a measurable **spectrum**, and **built, validated (0.970 gate), and ran** the LLM-judge on the GH200
> — converting our two hand-read results into numbers: **interpret is 0/10** (no DCT vector flips a
> fact) and **steering shows no lie-asymmetry** (Δ=−0.010, TOST-equivalent to zero) but **significant,
> symmetric degradation** (P(TRUE) falls with |scale|, z=−5.06). Decodable ≠ causal — now with a
> *bounded* null, not a hand-read one.

---

## 6. Immediate next steps

1. ~~Run the judge (steer/interpret) and validate it~~ — **DONE** (GH200, gate 0.970, corrected rerun;
   see `INVESTIGATION_steering_validity.md` §6). Both funnel caveats in `PI_MEETING_RESULTS.md` are
   closed with numbers.
2. **Fold the judged numbers into `PI_MEETING_RESULTS.md`** — swap its "Test 1/2 were hand-read /
   keyword-scored" caveats for the quantitative versions (0/10 interpret; bounded lie-asymmetry +
   symmetric degradation).
3. **Optional deeper tests if the PI wants them** (Tier-2/3, all cheap to add): dose-response odds
   ratio on the degradation trend; cross-dataset homogeneity (is the −vs+ null the same in both?);
   human-vs-judge κ on a sample of the *actual steering completions* (the 0.970 gate validated the
   judge on clean single-claim statements, not on these completions).
4. Then proceed to the spectrum's **refusal positive-control** and its go/no-go gate (roadmap §4) — the
   judge that gates it is now proven to work.
