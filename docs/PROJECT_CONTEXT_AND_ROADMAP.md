# Project Context & Roadmap — READ THIS FIRST

*The single onboarding + resume + research-direction document. Written to be self-contained so
work can continue on a **different machine** with zero prior session context. If you are picking
this up on a fresh laptop, read sections 0, 4, and 5 first, then 7.*

*Last updated: 2026-07-07 (end of the "pure-logic foundation" implementation session).*

---

## 0. TL;DR — what this repo is and where it's going

**What it is.** An interpretability research project on `google/gemma-2-2b` asking whether "truth"
(and, more generally, a concept) is not just *readable* from the residual stream but *causally
load-bearing* in the model's computation. Core tension: **decodability ≠ causal salience.**

**What we found (truth).** Truth is ~99% linearly decodable but is **not** among the model's most
causally-salient directions (DCT, an unsupervised causal method, doesn't recover it as a single
vector, a combination, a transferable feature, or the non-linear structure XGBoost finds). Truth is
easy to read, hard to push.

**Where it's going.** Generalize that binary null into a **quantified spectrum**: *DCT recovers a
concept in proportion to how causally load-bearing that concept is.* Refusal (behaviorally dominant)
should be recovered; truth (a decodable correlate) is not. That's the current initiative. Longer
term: understand *why* truth isn't causal, and build supervision/control-theory tools to steer into
truthful behavior (see §7).

**State right now.** The design spec and implementation plan for the spectrum are written and
committed. The **pure-logic foundation** (6 modules + tests) is implemented, reviewed, and passing
(14/14). The GPU/cluster half is intentionally deferred and precisely staged (see §4).

---

## 1. The research question (north star)

Prior work (Marks & Tegmark 2023): a **linear probe** reads "is this statement true?" off
gemma-2-2b's hidden activations at ~99%. That proves truth is *linearly represented* — the info is
present and separable. It does **not** prove the model *uses* that direction to compute anything.

- **Decodable** = a linear read-out recovers the label (info is present).
- **Causally salient / high-gain** = perturbing along the direction strongly changes downstream
  computation (the model routes behavior through it).

These can diverge (superposition: many features are linearly present; only some are load-bearing).
The whole project measures that gap, concretely and quantitatively.

**DCT (Deep Causal Transcoding)** is our causal instrument: an unsupervised method that finds the
directions the model is *most causally sensitive to* (the biggest steering levers), label-free, as
~512 vectors. If an unsupervised *causal* method independently lands on the *supervised* truth
direction, that's strong evidence truth is causal. It doesn't — that's the finding.

---

## 2. What we've established (findings so far)

Read these committed docs for the full story (they are thorough and meeting-ready):

| Doc | What it establishes |
|---|---|
| `docs/PI_MEETING_RESULTS.md` | **The master writeup.** The whole DCT-vs-truth funnel: all 5 tests, every number, plain-English meaning, statistical baselines, caveats. Start here. |
| `docs/DCT_VS_TRUTH_FINDINGS.md` | The core **null**: best DCT vector ≈ 1.2× random vs the truth direction; ≈ chance in the 512-dim span, even at the truth-peak layer. |
| `docs/DCT_VS_XGBOOST_FINDINGS.md` | The **non-linear** extension: XGBoost opens a +0.057 truth gap on the full residual stream (common_claim), but that gap collapses to ≈0 inside DCT's causal subspace. Null holds non-linearly too. |
| `docs/DIRECTIONS_FINDINGS.md` | `mean_diff` vs `grad` truth-direction agreement (cosine tracks dataset cleanliness: cities 0.41, common_claim 0.09). |
| `docs/MEETING_SUMMARY.md` | Short version: linear-vs-XGBoost accuracy gap + directions. |
| `docs/MASTER_EXPLAINER.md`, `docs/EXPLAINER.md` | Longer teaching writeups of DCT and the geometry-of-truth setup. |

**The five funnel tests and their verdicts (all point the same way):**

1. **Interpret DCT's top-10 directions** (`interpret_top10_*.md`, `src/interpret_top10.py`): they're
   about **geography / format / tone**, not truth. Falsehoods appear only as a *byproduct* of
   geography-scrambling, never a clean true↔false knob.
2. **Steer the supervised truth direction** (`plot_findings_steering.png`, `src/steer_supervised.py`):
   the direction **is** causal (strong negative gradient steering induces real falsehoods: "two plus
   two = three", "sun rises in the west") but the effect is **weak and degradation-confounded** —
   accuracy falls at *both* extremes, not just one sign. A **real but minor** lever. **Caveat: this
   is still hand-read / keyword-scored — the LLM-judge to make it quantitative is built but not yet
   run (see §7 direction B).**
3. **Subspace / combination test** (`src/subspace_top_k.py`, `src/compare_directions.py`): truth is
   not a combination of top-k DCT vectors either (in-span ≈ chance). The 0.94 classify-from-DCT number
   is a Johnson–Lindenstrauss trap — random features score 0.97.
4. **Cross-dataset transfer** (`src/cross_dataset.py`): DCT "truth" transfers at chance (0.50);
   supervised transfers 0.60–0.76. DCT's directions are input-distribution-specific.
5. **DCT-vs-XGBoost non-linear subspace** (`src/subspace_xgb.py`, this branch's predecessor):
   see `DCT_VS_XGBOOST_FINDINGS.md`.

**The one-line takeaway:** truth is decodable-but-not-causally-dominant in gemma-2-2b — a clean
counterexample to the common inference "a probe finds feature X ⟹ the model uses X."

---

## 3. The current initiative — the causal-salience spectrum

**Spec:** `docs/superpowers/specs/2026-07-07-causal-salience-spectrum-design.md`
**Plan:** `docs/superpowers/plans/2026-07-07-causal-salience-spectrum.md`

**Thesis:** *An unsupervised causal method (DCT) recovers a concept's supervised direction to the
degree that concept is causally load-bearing.* Decodable-vs-causal is a measurable spectrum, not a
truth-specific quirk.

**The money figure** (`plot_findings_spectrum.png`, produced by `src/viz_spectrum.py`) — one point
per concept:

- **x-axis = causal salience:** behavioral steering-effect size of the concept's *supervised*
  direction (judge-scored, normalized against degradation). Machinery: `src/steer_supervised.py` +
  the judge harness.
- **y-axis = DCT recovery:** max single-vector cosine + subspace fraction between DCT's top
  directions and the supervised direction, over the random baseline. Machinery:
  `src/compare_directions.py`.

**Prediction:** monotonic positive. Truth ≈ (low-x, low-y), already measured. **Refusal** should be
(high-x, high-y) — it's the **positive control**: if DCT can't recover even refusal, the whole
premise ("DCT can find *a* causal direction") is wrong, and that's a Phase-1 go/no-go gate.

**Concepts:** refusal (control, public data now), toxicity, sycophancy, truth (done). Akshal's
jailbreaking dataset is a *second* refusal point when it lands, not a blocker.

**Why this design works:** every concept is just a `(statement, label)` contrast CSV in the repo's
existing format, so the *entire* existing pipeline (`extract → analyze → run_dct_data →
compare_directions → steer_supervised`) runs per concept unchanged. The only new code is the judge
harness, per-concept data adapters, and `viz_spectrum.py`.

---

## 4. Implementation state — EXACTLY where to resume

> ⚠️ The subagent-driven-development progress ledger lives in `.git/sdd/progress.md`, which is **NOT
> pushed to the remote and does NOT clone to another machine.** This section is the durable,
> committed replacement. Trust this + `git log` on the `causal-salience-spectrum` branch.

**Branch:** `causal-salience-spectrum` (off `main` at merge-base `5ef4ea3`).

### DONE this session — the pure-logic foundation (6 tasks, each committed + reviewed, 14/14 tests pass)

| Task | Module | What it provides | Commit |
|---|---|---|---|
| 1 | `src/judges/metrics.py` | `distinct_n`, `corpus_distinct_n`, `repetition_rate`, `Perplexity` (degradation metrics) + pytest setup | `cbfb675` |
| 3 | `src/judges/adapters.py` | `truthfulqa_prompt(stem, completion)` — maps to the allenai judge format | `32c2a91` |
| 8 | `src/export_concept_dir.py` | `concept_directions(X, y)` + CLI — supervised mean_diff/grad for ANY labeled contrast (generalizes `export_truth_dir.py`; writes `truth_dir_<ds>.npz`) | `97cc11d` |
| 10 | `src/spectrum_utils.py` | `concept_salience(rows, present_verdict, ...)` — the spectrum **x-axis** aggregation | `1ed2e38` |
| 11 | `src/prep_refusal.py` | `to_contrast_df(harmful, harmless)` — refusal contrast CSV (the `main()` data-build is DEFERRED) | `955a492` |
| 15 | `src/viz_spectrum.py` | `build_points(concepts, present_map)` + `main()` plot (the real-figure generation is DEFERRED) | `3e394b4` |

Tests live in `tests/` (pytest was added to the venv this session). Run them with `.venv/bin/pytest tests/ -q`.

Final whole-branch review verdict: **merge as-is** — no Critical/Important; all cross-task interface
contracts verified to hold. Two cleanups to fold in when the deferred producers land:
- `build_points` uses bare `open()` — switch to `with`, and guard `StopIteration` on empty/malformed CSV.
- `concept_salience` seeds `best_scale=0.0` — assert a scale-0 baseline row exists before trusting the number.

### DEFERRED — need the GH200/DeltaAI cluster, 7B judge models, or network (NOT yet implemented)

These are fully specified in the plan (`docs/superpowers/plans/2026-07-07-causal-salience-spectrum.md`);
each task there has complete code and exact commands. Resume order:

| Task | What it does | Blocker |
|---|---|---|
| 2 | Smoke-test `Perplexity` (gpt2) | small model download |
| 4 | `src/judges/local_hf.py` — local HF judges: `allenai/truthfulqa-truth-judge-llama2-7B` + `...info-judge...`, Llama-Guard refusal, toxic-bert | 7B models on GPU |
| 5 | Wire `--backend local-hf` into `src/judge_results.py` (+ `run_steer_local`) | needs Task 4 |
| 6 | `src/validate_judge.py` — judge/label agreement check | 7B models |
| 7 | **Quick win:** retrofit the judge harness onto EXISTING truth steering sweeps → makes funnel Test 2 quantitative, closes the "hand-read" caveat | needs Tasks 4–5 |
| 9 | Add `recovery_<ds>.csv` output to `src/compare_directions.py` (the spectrum **y-axis** source) | smoke on cluster/existing DCT vectors |
| 11 (rest) | Run `prep_refusal.py` to build `got_datasets/refusal.csv` (AdvBench + Alpaca) | network + `pip install datasets` |
| 12 | **Refusal pipeline + GO/NO-GO gate:** extract → analyze peak layer → export_concept_dir → run_dct_data → compare_directions (y) → steer_supervised + judge (x) | GH200 |
| 13, 14 | Toxicity, sycophancy concepts end-to-end | GH200 + network |
| 15 (rest) | Generate the real `plot_findings_spectrum.png` from all concepts | needs 9,12,13,14 |
| 16 | `docs/CAUSAL_SALIENCE_SPECTRUM_FINDINGS.md` writeup | needs the numbers |

**Decision already made:** `judge_steer_*.csv` stays gitignored (regenerable via 7B judge inference);
commit only the small aggregated results `spectrum_points.csv` + `recovery_*.csv`. Drop
`got_datasets/refusal.csv` etc. from commit steps only if you keep the convention that concept CSVs
are regenerable — but the concept CSVs ARE cheap and reproducible, so committing them is fine and
recommended for provenance.

---

## 5. Environment setup on a NEW laptop

The repo gitignores all heavy/derived artifacts (`.venv*/`, `activations/`, `dct_*.pt`, `*.npz`,
`plot_*.png`, `truth_dir_*.npz`, `steer_supervised_*.csv`, `judge_*.csv`). So a fresh clone has the
code, datasets (`got_datasets/*.csv`), and docs — but NOT the activations, DCT vectors, or plots.
You regenerate those with the scripts.

**CPU laptop setup (for analysis + the pure-logic work in this branch):**

```bash
git clone <this-repo-url> llm-activation-steering-research
cd llm-activation-steering-research

# Python 3.13 (NOT 3.14 — no torch wheels). Create the venv:
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
# CPU-only torch, then the rest:
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers accelerate pandas numpy scikit-learn xgboost matplotlib pytest

# Verify the pure-logic tests pass:
.venv/bin/pytest tests/ -q          # expect 14 passed
```

**Model:** `google/gemma-2-2b` is **gated** — accept the license once at
<https://huggingface.co/google/gemma-2-2b> and `hf auth login` with a read token. It's ~9.7 GB fp32
across 3 shards. **Always `export HF_HUB_DISABLE_XET=1`** before any HF download (the xet backend
needs large scratch space and fails on constrained disks). Check `df -h` before downloading.

**Disk:** the original laptop was severely disk-constrained. On a new machine confirm you have ~15 GB
free for the model + activations before extracting. Do NOT delete dev/build caches to make room; use
re-downloadable installers instead.

**To regenerate the truth artifacts locally** (activations → analysis), see `CLAUDE.md` (the original
build guide) — `python src/extract.py <dataset>.csv` then `python src/analyze.py <dataset>`.

---

## 6. The GH200 / DeltaAI cluster workflow (for the deferred GPU tasks)

The DCT runs, the 7B judges, and the steering generations need a GPU. We use **NCSA DeltaAI (GH200,
ARM64 + Hopper)**. Full instructions are in **`deltaai/GPU_SETUP.md`** (committed) and
`deltaai/setup_env.sh` (committed). Note `deltaai/MY_RUN_STEPS.md` is **gitignored** (personal
account details) — recreate your own from `GPU_SETUP.md` if it's not present on the new machine.

Key facts:
- ARM CPU ⇒ do **not** pip-install torch; use the module `python/miniforge3_pytorch` via
  `deltaai/setup_env.sh` (creates `.venv-dct-gpu`). Pin `transformers==4.51.3` (required for DCT's
  `SlicedModel`; eager attention required — flash/SDPA break the calibrator's forward-AD).
- Pre-download gemma-2-2b on the **login node** (compute nodes have no internet); `export
  HF_HOME=$HOME/hf_cache`.
- DCT run command (per concept), from the plan:
  `python src/run_dct_data.py --dataset <ds> --source-layer <L*> --target-layer <L*+9>
  --num-factors 512 --num-iters 30 --num-samples 64 --balanced`
  where `L*` = the concept's probe-peak layer (from `analyze.py`).
- Budget: ~312 GPU-hours were left; each DCT run ≈ 0.15 GPU-hr. The 7B judges add inference time but
  fit easily on one GH200 (~96 GB HBM).
- Bring results back to the laptop with `rsync` (see `deltaai/GPU_SETUP.md` / `MY_RUN_STEPS.md`).

---

## 7. The specific research directions we actually want to pursue

*This is the part that matters most for continuation. Prioritized. Each item says what it establishes
and which code/tasks it touches.*

### A. Finish the causal-salience spectrum (immediate — the current plan)
Execute the deferred tasks (§4) on the cluster, in order, respecting the **Phase-1 refusal go/no-go
gate** (Task 12). Outcome either way is publishable:
- If recovery tracks salience and refusal lands high-y → the null becomes a **law** (decodable-vs-causal
  is a spectrum).
- If DCT recovers *nothing*, even refusal → the bottleneck is DCT/our pipeline, not truth's causal
  status → pivot to direction D.

### B. Make the qualitative results quantitative (cheap, high-value, do early)
Build out the LLM-as-judge harness (Tasks 4–7) and **retrofit it onto the existing truth steering
sweeps first** (Task 7 — the quick win). This closes the single biggest caveat in
`PI_MEETING_RESULTS.md`: funnel Test 1 (interpret) and Test 2 (steering) are currently **hand-read**.
- Local HF judges (run on the cluster, no API): `allenai/truthfulqa-truth-judge-llama2-7B` (truth) +
  `allenai/truthfulqa-info-judge-llama2-7B` (informativeness). `src/judge_results.py` currently uses
  the **Anthropic API** (Haiku) — Task 5 adds the local backend.
- **Model-collapse / degradation metrics** to separate "made it false" from "made it gibberish":
  perplexity (built: `judges/metrics.Perplexity`), distinct-n and repetition_rate (built), and
  optionally self-BLEU. Also look at **dist(n) diversity** of generations and **perplexity** as a
  quantitative model-collapse signal under hard steering.
- **Validate the judge** against ground-truth labels before trusting it (Task 6) — the allenai judges
  were trained on TruthfulQA Q/A format, not our completions.

### C. Broaden concepts, then reach out for adversarial validation
- Add **toxicity** and **sycophancy** (Tasks 13–14) to fill the spectrum's middle.
- **Reach out to Akshal** for his **jailbreaking / behavior dataset** — validate the spectrum thesis
  in a domain we didn't design, and add a second refusal-family point. This is explicitly a wanted
  direction: test whether DCT's top causal directions become concept-related on a messier, more
  behaviorally-loaded dataset.

### D. Understand WHY truth isn't causally salient, and integrate supervision (the deeper arc)
This is the real intellectual payoff and where the project is ultimately headed.
- **Diagnose the null.** Scale DCT: `--num-factors 1024`, deeper/other target layers — does a truth
  lever appear lower in the ranking or in a different slice? Try an **instruction-tuned model** or a
  **fact-requiring task** where asserting truth actually drives behavior.
- **Place the supervised truth direction into DCT's causal frame** and measure how much it actually
  shifts computation — quantify "real but minor lever" precisely.
- **Integrate supervision / control theory.** If the unsupervised causal method won't surface truth,
  motivate *supervised* causal targeting:
  - Tools from **control theory / reachability** for a truthfulness objective — search for how to
    steer the model *into* a truthful region of activation space, then discover directions from
    there (rather than from the raw input distribution DCT samples).
  - The **A-LQR** approach (Julian's paper) — local-linear behavioral control; get access to the
    A-LQR code for a proper **behavioral** (not geometric) evaluation. Everything so far is
    representation-level geometry; A-LQR is the gold-standard causal test.
  - Add **semantic supervision** to the DCT-style discovery so the high-gain directions it finds are
    tied to a target concept, not just to what varies in the input prompts.
- **Automate target-feature discovery** — build the LLM-as-judge into a loop that scores and selects
  which directions/features actually manipulate the target behavior, closing the loop from
  "discover directions" → "test them behaviorally" → "keep the causal ones."

### E. Open questions to bring to the PI (Julian)
- Which concepts to prioritize beyond factual truth (toxicity? sycophancy? refusal is the control).
- Access to the **A-LQR code** for behavioral evaluation.
- Whether to move to an instruction-tuned model where truth may be more load-bearing.

---

## 8. How to resume the subagent-driven execution

The remaining plan tasks (§4 deferred list) were being executed via the
**superpowers:subagent-driven-development** flow (fresh implementer subagent per task, task review
after each, final whole-branch review). To resume:

1. Read the plan: `docs/superpowers/plans/2026-07-07-causal-salience-spectrum.md` — each task has
   complete code + exact commands + its own commit step.
2. The pure-logic tasks (1, 3, 8, 10, 11, 15) are DONE (§4). Start at the first deferred task that
   fits your environment (on a CPU laptop with network: Task 2, Task 9 smoke if DCT vectors are
   present, Task 11 data-build; on the cluster: Tasks 4–7, 12–14).
3. Respect the **Task 12 go/no-go gate** before running Phase 2 (Tasks 13–14).
4. Fold in the two cleanups noted in §4 when you touch `viz_spectrum.py` / `spectrum_utils.py`.

---

## 9. Key files map

```
CLAUDE.md                         original build guide (extraction/analysis of the truth datasets)
docs/
  PROJECT_CONTEXT_AND_ROADMAP.md  ← you are here
  PI_MEETING_RESULTS.md           the master funnel writeup (start here for findings)
  DCT_VS_TRUTH_FINDINGS.md        the core null
  DCT_VS_XGBOOST_FINDINGS.md      the non-linear extension
  DIRECTIONS_FINDINGS.md          mean_diff vs grad agreement
  MASTER_EXPLAINER.md / EXPLAINER.md / MEETING_SUMMARY.md
  superpowers/
    specs/2026-07-07-causal-salience-spectrum-design.md   the initiative's spec
    plans/2026-07-07-causal-salience-spectrum.md          the 16-task implementation plan
src/
  extract.py analyze.py           activation extraction + per-layer probe/XGBoost (truth pipeline)
  funnel_utils.py                 load_acts / mean_diff_dir / grad_dir / resolve_layer / top_k_by_potency
  run_dct_data.py dct*.py         DCT training (GH200)
  compare_directions.py           DCT-vs-supervised cosine + subspace (spectrum y-axis; add recovery CSV = Task 9)
  steer_supervised.py             ± steering sweep of a supervised direction (spectrum x-axis)
  interpret_top10.py              interpret DCT's top-10 causal directions
  subspace_top_k.py subspace_xgb.py   the combination / non-linear subspace tests
  judge_results.py                LLM-as-judge (Anthropic now; add local-hf backend = Task 5)
  export_concept_dir.py           [new] supervised direction for any concept (Task 8)
  judges/metrics.py adapters.py   [new] degradation metrics + judge format adapter (Tasks 1,3)
  spectrum_utils.py               [new] concept_salience x-axis aggregation (Task 10)
  prep_refusal.py                 [new] refusal contrast CSV (Task 11)
  viz_spectrum.py                 [new] the money figure (Task 15)
tests/                            [new] pytest suite for the pure-logic modules
deltaai/                          GH200 setup + slurm scripts (GPU_SETUP.md is the tracked guide)
got_datasets/                     the concept CSVs (statement,label) — committed, small
```
