# Warm-Started / Semi-Supervised DCT — Design

**Date:** 2026-07-18
**Status:** Design approved, ready for planning
**Origin:** PI meeting notes #2 (does the top DCT vector = truth?), #3 (make DCT more truth-actionable), #5 (address DCT's limitations), #6 (semi-supervised — warm-start / bias the gradient search around a known truthfulness direction from mean-diff / xgboost truth directions).

---

## Goal

Bias DCT's causal gradient search *toward the known supervised truth axis* and test whether the refined direction becomes a **causal truth-lever** where the raw supervised axis is causally inert.

**Headline (primary):** *The causal truth-lever lives near — but off — the inert supervised axis.* MAG proved the contrastive `mean_diff` axis **reads** truth (decodable) but is causally **inert**: steering it degrades the model (verdict → INCOHERENT), it does not flip truth→lie (verdict → FALSE). Warm-start DCT's causal search *at* `mean_diff`, hold it loosely there with a soft anchor, and measure whether the refined direction tips the OLMo judge toward FALSE where raw `mean_diff` only degrades.

**Supporting (note #2 audit):** Does vanilla (cold-start) DCT's *top* discovered vector correspond to truth? Reuse the **existing** cold-start `dct_V_<ds>.pt`, rank its factors by causal effect, measure each top factor's cosine with `mean_diff@src`, and behaviorally steer the top one. Yields a yes/no-with-evidence answer to "is the dominant DCT/MELBO vector a truth direction?"

Both questions are answered through the **same E4 behavioral test** the MAG battery used (calibrated injection + OLMo FALSE/INCOHERENT read), so the results are read the same way as MAG and Project A — FALSE-rising = causal lever, INCOHERENT-rising = degrader. (Absolute fractions sit on the DCT `input_scale` calibration, so the cross-project comparison is qualitative, not a same-axis numeric match — see Calibration comparability.)

## Context / what already exists

- `src/dct.py` — `ExponentialDCT` (dct.py:445). `fit(init=...)` supports `"random"` (dct.py:449 `_init_rand`) and `"rand_backward"` (dct.py:456). The training loop (dct.py:719-803) is a **power-iteration** update, not SGD: each iteration does `V ← normalize(β·G_V + (1−β)·V)` (dct.py:790), with an optional QR orthogonalization of `V` (dct.py:723) and optional soft-ortho deflation. `G_V` is the gradient of the objective `Σ‖Δ‖` w.r.t. the source-layer bias. `rank(...)` (dct.py:587) scores factors by causal effect (`alphas`, solved from the Gram system) or by projection onto a `target_vec`.
- `src/dct_train.py` — the full training entry: loads gemma-2-2b, builds `SlicedModel(source_layer → target_layer)`, extracts `X` (source acts) + `Y` (unsteered target acts) for `num_samples` prompts, calibrates `input_scale` via `SteeringCalibrator(target_ratio=0.5)`, calls `fit(init="rand_backward")`, and saves `V` (scaled by `input_scale`), `U`, and a metadata JSON.
- **Existing cold-start artifacts (on disk, both datasets):** `dct_V_<ds>.pt`, `dct_U_<ds>.pt`, `dct_meta_<ds>.json`.
  - cities: `source_layer=11`, `target_layer=20`, `num_factors=512`, `num_iters=30`, `num_samples=64`, `input_scale≈47.716`, `token_idxs="-3:"`.
  - common_claim_true_false: `source_layer=13`, `target_layer=22`, `num_factors=512`, `num_iters=30`, `num_samples=64`, `input_scale≈86.733`, `token_idxs="-3:"`.
- `mag_acts_<ds>.npz` — residual-stream activations at **every** layer, with labels. Source for computing the seed axes *at the DCT source layer*.
- `truth_dir_<ds>.npz` — supervised `mean_diff`, `grad`, and `layer` (the truth-peak layer). NOTE: this `mean_diff` lives at the **truth-peak** layer, which is **not** the DCT source layer — it cannot be used as the seed directly (see Key subtlety).
- `src/mag/steer.py` — the E4 calibrated-injection harness: `injected_vector(τ, û, a_prefix_norm) = τ · a_prefix_norm · unit(û)`, injected at a direction's layer via `su.Steerer`, generating the free-form factual set (OLMo-judged) and the yes/no flip set. `û = mean(false) − mean(true)` points toward FALSE, so **+τ pushes toward lying**.
- `src/judge_results.py` / `src/judges/` — the OLMo TRUE / FALSE / INCOHERENT judge. `run_mag_judge.slurm` routes MAG output to `judge_mag_steer_<ds>.csv`.
- `deltaai/run_dct.slurm` (`.venv-dct-gpu`), `deltaai/run_mag_steer.slurm`, `deltaai/run_mag_judge.slurm` (`.venv-judge-gpu`), and the runbooks `deltaai/MAG_E4_RUN.md`, `deltaai/MAG_EXTRACT_RUN.md` — the GH200 job patterns to clone.

gemma-2-2b is a BASE model. "Truthfulness behavior" is read from free-form completions scored by the judge.

## Key subtlety (load-bearing)

DCT's `V` (and its factors) live in **source-layer** space (layer 11 for cities, 13 for common_claim). The supervised `mean_diff` in `truth_dir_<ds>.npz` is computed at the **truth-peak** layer, a different space. Therefore the warm-start seed and soft-anchor target must be `mean_diff` / `grad` **recomputed at the DCT source layer** from `mag_acts_<ds>.npz`:

- `mean_diff@src = unit( mean(acts[src][label==1]) − mean(acts[src][label==0]) )`
- `grad@src = unit( coef / scaler.scale_ )` from a LogisticRegression trained on `acts[src]` (standardized), matching the `analyze.py` convention.

All three compared directions (raw `mean_diff@src`, warm-refined, cold-top) are then injected **at the DCT source layer** with **one shared calibration**, so they share a single space and scale.

## Scope (locked)

| Axis | Decision |
|---|---|
| Warm-start mechanism | New `init="warm"` seeds the primary V factor at the seed axis; a **soft anchor** term `+ λ·V_anchor` is added inside the power-iteration update (dct.py:790), re-applied each iteration |
| λ=0 ablation | `λ=0` = init-only, free-drift baseline (seed then unconstrained search) |
| Seed axes | `mean_diff@src` **and** `grad@src` — run as **two separate warm runs** (not one pool: near-parallel seeds would be collapsed by the QR step) |
| λ sweep | `{0, 0.1, 0.3}` per seed — traces the drift-vs-causality tradeoff |
| Datasets | `cities` **and** `common_claim_true_false` |
| Cold-start | **Reuse** the existing `dct_V_<ds>.pt` for the note-#2 audit + cold baseline (re-run once only if `dct_meta_<ds>.json` does not match the warm-run config) |
| DCT config (warm) | Same `source_layer`, `target_layer`, `input_scale`, `num_iters`, `token_idxs`, `num_samples` as the existing cold run (read from `dct_meta_<ds>.json`) |
| Warm factor pool | Modest (`num_factors=64`): factor 0 seeded at the axis + anchored; remaining factors `rand_backward` (they provide the orthogonal context DCT's ranking needs) |
| Injection layer | **DCT source layer** for all directions (not truth-peak) |
| Behavioral read | E4 harness: τ sweep, OLMo judge — **FALSE-rising = causal lever**, **INCOHERENT-rising = mere degrader** |
| Compute | GH200 cluster, three-job pattern (warm-train → steer-gen → judge) |

**Out of scope (parked):** `resid_pc1` as a seed (different question — turning a known degrader into a lever); the note-#6 "fine-grained truth *decomposition*" into multiple distinct sub-levers (a possible byproduct of the anchored pool, but not a success criterion here); multiple non-truth contexts (note #4, explicitly deferred); a hard cone constraint on the search.

## Method

### 1. Seed axes at the source layer
For each dataset, read `source_layer` from `dct_meta_<ds>.json`, load `mag_acts_<ds>.npz`, and compute `mean_diff@src` and `grad@src` (formulas above). Persist both to `dct_warm_seeds_<ds>.npz` (keys: `mean_diff_src`, `grad_src`, `source_layer`). Sanity check: cosine of `mean_diff@src` with `truth_dir`'s `mean_diff` (different layers, so a moderate — not near-1 — cosine is expected; the check is that it is a coherent truth-reading direction, verified by a quick linear-probe accuracy at `src`).

### 2. `init="warm"` + soft anchor (the only changes to `dct.py`)
- **New init path** in `ExponentialDCT.fit`, alongside `"random"` / `"rand_backward"`: `init="warm"` accepts a `warm_seed` vector (d_source,) and `anchor_lambda` (float). It sets `V[:,0] = unit(warm_seed)`, initializes `U[:,0]` via one Jacobian-vector product `U[:,0] ← normalize(J · V[:,0])` (reusing the backward-init Jacobian machinery at dct.py:475-535), and initializes the remaining columns exactly as `rand_backward` does. Stores `self.V_anchor = unit(warm_seed)` and `self.anchor_lambda`.
- **Anchor in the update:** at dct.py:790, when `anchor_lambda > 0`, the factor-0 update becomes `V[:,0] ← normalize(β·G_V[:,0] + (1−β)·V[:,0] + λ·V_anchor)` before the QR/normalize step (anchor applied only to the seeded column). `λ=0` reduces exactly to the current behavior (free drift).
- Everything else in `fit` (QR, deflation, calibration coupling via `input_scale`) is unchanged.

### 3. Warm training runs
A new entry `src/dct_warm.py` (thin wrapper over the `dct_train.py` extraction/calibration path) that: reads `dct_meta_<ds>.json` for the config, reuses/recomputes `X`/`Y`, loads a seed from `dct_warm_seeds_<ds>.npz`, and calls `fit(init="warm", warm_seed=..., anchor_lambda=λ, input_scale=<meta>, num_factors=64, ...)`. Saves the fitted `V`/`U` to `dct_warm_V_<ds>_<seed>_lam<λ>.pt` (+ `_U`), and an objective-values trace. Run for `seed ∈ {mean_diff, grad}` × `λ ∈ {0, 0.1, 0.3}` per dataset = 6 warm runs/dataset.

### 4. Direction extraction + geometry
For each dataset, assemble the candidate directions (all d_source unit vectors at `source_layer`):
- `raw_mean_diff` = `mean_diff@src`
- `raw_grad` = `grad@src`
- `warm_<seed>_lam<λ>` = the **factor-0** column of each warm run (the refined seed)
- `cold_top` = the top-ranked factor of the existing cold `dct_V_<ds>.pt`, ranked by `ExponentialDCT.rank(...)` causal score
Record, per direction: cosine to its seed axis (`drift = 1 − cos`), causal rank/score among the warm pool, and — for `cold_top` — cosine to `mean_diff@src`.

### 5. Behavioral evaluation (E4, reused)
Extend the `mag/steer.py` pattern so it can take an explicit list of `{name, unit_dir, layer=source_layer, norm}` directions (rather than only the `mag_dir` keys). Inject each candidate at `source_layer` via `su.Steerer` as `τ · norm · unit_dir`, where **`norm` is the dataset's DCT `input_scale`** (read from `dct_meta_<ds>.json` — already the calibrated source-layer steering magnitude, identical across every direction, so magnitudes are comparable). Sweep **τ ∈ {0, +0.3, +0.6, +1.0}** (same sign convention as MAG: +τ → toward FALSE; τ=0 = no vector = baseline), generate the free-form factual set, and run the OLMo judge (`run_mag_judge`-style) to get TRUE / FALSE / INCOHERENT fractions per (direction, τ). Also run the yes/no flip set for the matched-format flip rate.

### 6. Artifacts
- `dct_warm_seeds_<ds>.npz` — the source-layer seed axes.
- `dct_warm_V_<ds>_<seed>_lam<λ>.pt` (+ `_U`, + objective trace) — the 6 warm runs/dataset.
- `dct_warm_steer_<ds>.csv` — one row per (direction, τ, prompt): completion (schema matches `judge_results` so scoring is unchanged).
- `judge_dct_warm_steer_<ds>.csv` — the judge verdicts per (direction, τ).
- `dct_warm_summary_<ds>.csv` — one row per direction: `drift`, `causal_rank`, `cos_to_mean_diff`, and FALSE / INCOHERENT / TRUE fractions at each τ.
- Plots (`matplotlib.use("Agg")`):
  - `plot_dct_warm_drift_<ds>.png` — drift (1−cos to seed) vs λ, per seed.
  - `plot_dct_warm_verdict_<ds>.png` — head-to-head FALSE vs INCOHERENT bars: `raw_mean_diff` vs `warm_*` vs `cold_top`, at the strongest τ.
  - `plot_dct_warm_audit_<ds>.png` — note-#2 audit: cold factors ranked by causal score, colored by cosine to `mean_diff@src`.
- A short curated markdown table of vivid raw-vs-warm completion examples for the meeting.

### 7. Compute / files
- New: `src/dct_warm.py` (seed computation + warm training entry), a small extension to `src/dct.py` (`init="warm"` + anchor), an eval/plot script (`src/viz_dct_warm.py` or an extension of the existing viz), and slurm scripts cloned from the MAG/DCT patterns (`deltaai/run_dct_warm.slurm` for train+gen, reuse `run_mag_judge.slurm`-style for the judge).
- Runbook doc analogous to `deltaai/MAG_E4_RUN.md` for the three-job cluster run.

## Success criteria

1. Both datasets produce `dct_warm_seeds_<ds>.npz`, the 6 warm `V` files, `dct_warm_steer_<ds>.csv`, `judge_dct_warm_steer_<ds>.csv`, `dct_warm_summary_<ds>.csv`, and the three plots.
2. **Headline:** the verdict plot shows at least one `warm_*` direction tipping toward **FALSE** (or measurably more FALSE than `raw_mean_diff` at matched τ), i.e. a causal lever emerges near the inert axis. The drift plot quantifies how far off `mean_diff` that lever sits and at which λ.
3. **Audit (note #2):** the audit plot + `cos_to_mean_diff` answer whether cold DCT's top causal factor is truth-aligned.
4. A handful of clean qualitative raw-vs-warm examples suitable to show the PI.

(2)–(3) are the *expected* narrative. If instead **every** warm direction stays INCOHERENT (no FALSE emerges at any λ), that is a real, reportable finding — it extends the MAG null ("no causal truth-lever near the axis") from the observational regime into the causal-search regime, and is a legitimate result, not a failure.

## Risks / notes

- **Seed lives in the wrong space if taken from `truth_dir`.** Mitigated by recomputing `mean_diff@src` / `grad@src` at the DCT source layer from `mag_acts` (step 1). This is the single most important correctness point — a test must assert the seed's dimensionality and layer provenance.
- **QR collapses near-parallel seeds.** Mitigated by running `mean_diff` and `grad` as separate warm runs and anchoring only factor 0.
- **Anchor interacts with QR/deflation.** The anchor is applied to factor 0 *before* the existing QR/normalize; verify on a smoke run that factor 0 stays near its seed for λ>0 and drifts for λ=0 (a direct unit check on the mechanism).
- **Calibration comparability.** All directions use one shared norm per dataset — the DCT `input_scale` from `dct_meta_<ds>.json` — so FALSE/INCOHERENT fractions are comparable across directions. Do **not** mix in MAG's truth-peak `A_prefix_norm` (a different layer's scale); `input_scale` is the source-layer steering magnitude and is the correct single norm here. Note the absolute fractions are therefore on the DCT-`input_scale` scale, not MAG's `A_prefix_norm` scale, so cross-project comparison is qualitative (does FALSE rise?), not a same-axis numeric match.
- **Reusing the existing cold `dct_V`.** Guard: assert `dct_meta_<ds>.json` matches the warm config (`source_layer`, `target_layer`, `num_factors`, `input_scale`, `num_iters`, `token_idxs`) before reusing; if it mismatches, re-run cold once with the warm config so the head-to-head is matched.
- **Judge on short free-form completions.** Same regime as MAG E4 (short factual stems, `max_new_tokens`≈8), where the judge is reliable; the length-cutoff concern is Project A's problem, not this one.
