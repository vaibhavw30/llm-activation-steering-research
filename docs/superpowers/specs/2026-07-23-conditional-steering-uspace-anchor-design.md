# Conditional Steering + U-Space Anchored DCT — Design

**Date:** 2026-07-23
**Branch:** `feat/mag-e4-steering`
**Model:** `google/gemma-2-2b` base, fp32. Datasets: `cities`, `common_claim_true_false`.

## Motivation

Two results set this up:

1. **MAG E4's verdict-flip lead.** On matched-format yes/no prompts, `sup_mean_diff` flipped the
   model's self-verdict antisymmetrically (79–100% of statements flip to "no" at τ=−1 vs ~0% at
   τ=+1), while the same direction is inert on free-form factual stems. Hypothesis: **the truth
   axis is causally potent only conditional on question-mode** — the verdict circuit reads it,
   free-form generation ignores it. The current evidence is weak (24 hand-written all-true
   statements, 3-token string match, no controls). Arms A0–A2 upgrade it.
2. **Warm-DCT's V-space negative.** Anchoring DCT's *input* direction toward the supervised truth
   axis (λ ∈ {0..3}) never produced a truth lever (`docs/WARM_DCT_RESULTS.md`). Arm B inverts the
   constraint: leave the input free and anchor the *effect* toward the truth readout — "find any
   input direction whose causal effect moves the truth axis."

Sign convention throughout: directions point toward TRUE; +τ→TRUE, −τ→FALSE. Calibration for the
MAG-layer arms: `α(τ) = τ · A_prefix_norm` (from `mag_dir_<ds>.npz`), injection at the MAG layer
(cities 11, common_claim 13), exactly as in `src/mag/steer.py`.

---

## Arm A0 — Backfill: OLMo-judge the existing MAG free-form completions

**What:** run the already-written `mag_steer_{ds}.csv` (schema matches `judge_results.run_steer`)
through the existing OLMo judge, producing `judge_mag_steer_{ds}.csv`.

**Why:** makes MAG E4's free-form claim ("mean_diff is inert in free-form") apples-to-apples with
the DCT judge results instead of resting on manual reading of raw completions.

**Where:** wherever the judge venv lives (GH200 judge job, same shape as job `2703220`; the two
CSVs rsync up, judge CSVs rsync back). No new generation.

## Arm A1 — Verdict-mode steering at scale with logit readout (main event)

**New script:** `src/mag/steer_verdict.py` (runs locally, `--device mps`).

- **Statements:** 64 true + 64 false per dataset, drawn from the dataset's **test split**
  (80/20, `random_state=42`, stratified — the standard split convention) so they were not used to
  fit `mean_diff`/`grad`. If a dataset's test split has fewer than 64 of a class, take all of them.
- **Prompt:** `Q_TRUTH + statement + Q_SUFFIX` from `src/mag/config.py`, unchanged.
- **Measurement:** single forward pass per (direction, τ, statement); read the first-token
  distribution at the answer position and compute `p_yes` / `p_no` by summing
  `YES_VARIANTS` / `NO_VARIANTS` token probabilities (the MAG battery's Eq.-1 readout — reuse that
  code). Record continuous margin `p_yes − p_no`. No sampling, no judge.
- **Directions (5):** `sup_mean_diff`, `sup_grad`, `mag_resid_pc1` (causal-but-not-truth control),
  `random_unit` (fresh seeded random vector, same calibration — any-vector control), `cold_top`
  (DCT's most potent factor, injected at the MAG layer with the same α for comparability; noted
  as a layer-transplant caveat).
- **Sweep:** full 7-point τ grid {−1, −0.6, −0.3, 0, +0.3, +0.6, +1}.
- **Output:** `mag_verdict_logits_{ds}.csv` — columns
  `direction, tau, statement, label, p_yes, p_no, margin`.
- **Analysis/figure (`src/mag/viz_verdict.py` or extension of existing viz):** margin-vs-τ curves
  **split by gold label**, per direction; plus verdict accuracy vs τ.
- **Discriminating hypothesis:** if −τ shoves every statement toward "no" regardless of label →
  content-blind verdict lever (real but shallow). If steering moves true vs false statements
  *differentially* — especially if some τ makes the model more *accurate* (yes to true, no to
  false) — that is latent truth knowledge amplified along `mean_diff`: the headline positive.

**Cost:** ~5 directions × 7 τ × ≤128 statements × 2 datasets ≈ 9k single-token forwards. Local,
an evening.

## Arm A2 — Composed steering: v_Q gate + mean_diff content, OLMo-judged

**New script:** `src/mag/steer_conditional.py` (local generation; judging on cluster).

- **Prompts:** the 32 `FACTUAL_PROMPTS` free-form stems from `src/steer_supervised.py` (where
  `mean_diff` was inert).
- **Injection at the MAG layer:**
  `g · A_prefix_norm · v_Q_unit + τ · A_prefix_norm · mean_diff_unit`, gate `g ∈ {0, 1}`,
  τ over the 7-grid.
  - `g=0` sweep reproduces the inert baseline in-run (shared-baseline hygiene).
  - `g=1, τ=0` isolates v_Q alone on free-form text — required sanity row: if v_Q alone destroys
    coherence, the composition is uninterpretable.
- **Generation:** same `su.generate` parameters as the existing steer pipeline so the judge CSV
  schema (`direction, scale, prompt, completion`) is unchanged; condition encoded in the
  direction name (e.g. `meandiff_gate0`, `meandiff_gate1`).
- **Judge:** existing OLMo pipeline → `judge_mag_conditional_{ds}.csv`.
- **Hypothesis:** if truth-causality is gated on question-mode, `g=1` shows a sign-dependent FALSE
  rise at −τ where `g=0` stays flat. If both flat → the gate is not injectable as a residual-stream
  vector (it lives in the prompt/attention pattern) — itself informative.

## Arm B — U-space anchored DCT (cluster)

**Target-layer seeds (local, first):** new small script `src/export_target_dir.py` (standalone,
so `export_concept_dir.py` and its output schema stay untouched): compute `mean_diff`
(and `grad`, stored but unused this round) at the DCT **target** layers — cities 20,
common_claim 22 — from `activations/acts_{ds}.npz` via `fu.load_acts(ds, layer)`. Write
`truth_dir_tgt_{ds}.npz` (keys `mean_diff, grad, layer`; must NOT overwrite `truth_dir_{ds}.npz`).

**`src/dct.py` change:** symmetric to the existing V-anchor at `dct.py:812-818`. Add
`u_anchor` / `u_anchor_lambda` params to `fit`; apply `anchor_step` to the U update's column 0
(pre-normalize) toward the target-layer seed, scaled by ‖G_U[:,0]‖ (mirroring the V anchor's
scale-relative form). Note: with β=1, U is re-derived from G_U each iteration, so the U-anchor is
a per-iteration bias on the effect direction that shapes which V direction wins — the intended
mechanism. V column 0 is initialized **random** in this mode (input direction fully free).

**`src/dct_warm.py` change:** `--anchor-space {v,u}` (default `v`, existing behavior). In `u`
mode: load seed from `truth_dir_tgt_{ds}.npz` (assert its `layer` == target_layer), pass as
`u_anchor`; V warm-seeding disabled. Output tag: `dct_uwarm_V_{ds}_{seed}_{lamtag}.pt` / `_U.pt`
so nothing collides with the V-warm artifacts.

**Sweep:** λ_u ∈ {0.3, 1, 3} × 2 datasets × seed `mean_diff@tgt` = **6 fits** (λ_u=0 ≡ cold run,
already have it). No grad seeds this round (u_gold ≈ mean_diff; marginal value doesn't justify
doubling).

**Geometry deliverables (findings before any steering):**
1. cos(U₀, mean_diff@tgt) vs λ_u — does the anchor knob work in effect-space?
2. cos(V₀, mean_diff@src) — what input does an effect-anchored search *choose*: truth-aligned, or
   does the model route "move the truth readout" through an off-axis input?
3. Drift/objective curves as in the V-warm run.

**Behavior:** steer each fit's V₀ through the unchanged steer + OLMo judge pipeline (same τ grid,
`input_scale`, CSV schema; directions named `uwarm_mean_diff_lam{tag}`), figures via the
`viz_dct_warm` panel machinery extended with the new direction names. Two slurm jobs (train+steer,
judge), same shape as jobs `2701296` / `2703220`; existing runbook pattern.

---

## Interpretation matrix

| Result | Reading |
|---|---|
| A1: differential-by-label margin movement | Truth knowledge causally amplifiable — headline positive |
| A1: content-blind shove only; A2 flat | Verdict lever real but shallow; question-mode gate not injectable |
| A2: g=1 unlocks FALSE at −τ | Conditional causality confirmed — gate+content composition works |
| B: U-anchor knob works, steering still no FALSE | Strongest negative: no input direction has a truth-aligned causal effect in this framework |
| B: V₀ off-axis but steering moves truth | Truth causally reachable only via off-axis inputs — nonlinear routing |

## Execution order

1. **A0 + A1** (A1 local; A0 needs one judge job — can ride along with B's judge job if batching
   Duo pushes matters).
2. **A2** (local generation; judge with the same batch).
3. **B** (one cluster train+steer round + judge round).

A1's outcome may reshape investment in B; the order is chosen so the cheap discriminator lands
first.

## Constraints

- No commits/pushes without explicit user request.
- Uncommitted `spectrum_*` changes must not be swept into any commit for this work.
- Cluster access: NCSA password + Duo push per ssh/rsync — batch transfers.
- Disk-constrained laptop: no new large artifacts locally beyond the 6 U-warm `.pt` pairs'
  metadata/CSVs (the `.pt` files stay on cluster; only CSVs/figures rsync back, matching the
  established two-rsync pattern).
- Sign convention and τ grid identical to prior runs for cross-run comparability.
