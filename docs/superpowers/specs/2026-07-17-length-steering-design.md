# Steering Under Length (Uncapped Tokens) — Design

**Date:** 2026-07-17
**Status:** Design approved, ready for planning
**Origin:** PI meeting note #1 — "Uncap the token output... look at larger length output, what happens when applying a steering vector."

---

## Goal

Show what a steering vector does over *long* generations, two ways at once:

1. A smooth per-token **coherence trajectory** (intrinsic signals, cheap, computed during generation) — a real position-by-position curve.
2. A **semantic truth anchor** (OLMo judge run on length-cutoff prefixes) — the TRUE/FALSE/INCOHERENT verdict as a function of how much text we let the model produce.

Contrast an **inert-truth** direction against a **known degrader**, on both datasets, so the plot tells a clean story:

> *The mean_diff truth axis tracks the unsteered coherence baseline and stays TRUE at every length — it never lies, even given room to. The resid_pc1 degrader's coherence collapses smoothly starting mid-generation and its verdict tips to INCOHERENT by a modest length.*

This makes the MAG finding ("decodable ≠ causal; the off-axis structure is a degrader, not a lie-lever") **visible at length**, not just at 8 tokens.

## Context / what already exists

- `src/steer_supervised.py` — injects the supervised `mean_diff`/`grad` truth direction at the truth-peak layer, sweeping magnitude, generating short completions. Has a `--max-new-tokens` flag **defaulted to 8**, and its own code comment documents *why*: gemma-2-2b (a base model) rambles past the answer, and a long tail made the OLMo judge score the whole paragraph (a correct `"four..."` got marked FALSE on trailing text). **This interpretation problem is exactly what this project designs around** — we never judge one long blob; we judge cutoff prefixes and separately track intrinsic coherence.
- `src/dct_steer_utils.py` (`su`) — `load_model`, `Steerer` (adds a vector at a layer's residual stream via a hook), `generate`.
- `mag_dir_<ds>.npz` — carries `resid_pc1_unit`, `A_prefix_norm`, `layer`. `truth_dir_<ds>.npz` — carries `mean_diff`, `grad`, `layer`.
- `src/mag/steer.py` — the calibrated-injection pattern `injected_vector(tau, unit_dir, a_prefix_norm) = tau * a_prefix_norm * unit(dir)`.
- `deltaai/run_mag_steer.slurm` + `deltaai/run_mag_judge.slurm` — the two-job (gen then judge) GH200 pattern to clone.
- `src/judge_results.py` / `src/judges/` — the OLMo TRUE/FALSE/INCOHERENT judge.

gemma-2-2b is a BASE model. "Truthfulness behavior" is read from free-form completions.

## Scope (locked)

| Axis | Decision |
|---|---|
| Directions | `mean_diff` (inert truth axis) **and** `resid_pc1` (known degrader) |
| Calibration | Both injected via MAG-style `α(τ) = τ · A_prefix_norm · û`, so magnitudes are comparable across directions |
| Taus | `{0, +0.3, +1.0}` — 0 = unsteered baseline; `û` points toward FALSE, so +τ pushes toward lying |
| Datasets | `cities` **and** `common_claim_true_false` |
| Max new tokens | ~96 (greedy) |
| Judge cutoffs | `[8, 16, 32, 64, 96]` cumulative-prefix lengths |
| Compute | GH200 cluster (two jobs: gen, then judge), to afford a few-hundred-prompt holistic set |

**Out of scope (parked):** negative taus (add only if positive curves are interesting), the `grad` direction, multiple non-truth contexts (PI note #4, explicitly deferred).

## Prompt set (holistic, per-dataset split)

Approved split — build each dataset's prompt set the way that keeps the truth signal clean:

- **cities → (i) programmatic stems.** Sample a few hundred *true* statements from `cities.csv` and cut each into a completion stem before the answer token. cities statements are regular (`"The city of Paris is in the country of France."` → stem `"The city of Paris is in the country of"`), so stemming is reliable and gives a large grounded set.
- **common_claim → (ii) curated stems.** common_claim is free-form ("Most spiders have eight legs") and does not cut into a clean stem, so hand-grow the existing curated `FACTUAL_PROMPTS` style to ~100 clean, unambiguous completion stems, each with a crisp correct answer. Lower N, but the verdict signal stays clean — and common_claim's role has always been the messy contrast, so a curated messy-topic set is on-theme.

Target sizes: cities ≈ few hundred (e.g. 300); common_claim ≈ 100. Each prompt stored with its expected correct answer where known (cities stems have it structurally; curated set carries it), so the judge and any answer-match check have ground truth.

## Method

### 1. Directions & calibrated injection
Load `resid_pc1_unit` + `A_prefix_norm` + `layer` from `mag_dir_<ds>.npz`; load `mean_diff` (unit-normalized) from `truth_dir_<ds>.npz`, injected at the **same layer** with the **same** `A_prefix_norm`. Injection vector `= τ · A_prefix_norm · û`, added at the layer's residual stream via `su.Steerer`. τ=0 → no vector (baseline).

### 2. Generation + intrinsic per-token logging
One long **greedy** completion of ~96 new tokens per (prompt, direction, τ). During generation, at each generated position record:
- **max-prob** — `softmax(logits).max()` at that step (model confidence).
- **entropy** — entropy of the next-token distribution (flatter = degradation).
- **rep3** — a boolean 3-gram repetition flag (the emitted token completes a repeated trigram = looping).

Averaged over prompts, these give the coherence-vs-position backbone (steered vs baseline, per direction).

### 3. Judge-at-cutoffs
Truncate each stored 96-token completion to `[8, 16, 32, 64, 96]` tokens; run the OLMo judge on each cumulative prefix. Yields verdict fraction {TRUE / FALSE / INCOHERENT} vs length, per (direction, τ). No re-generation — one long gen is truncated.

### 4. Artifacts
- `length_steer_<ds>.csv` — one row per (prompt, direction, τ): the full 96-token completion + per-position intrinsic arrays (max-prob, entropy, rep3).
- `length_judge_<ds>.csv` — one row per (prompt, direction, τ, cutoff): the prefix verdict.
- Plots (`matplotlib.use("Agg")`):
  - `plot_length_coherence_<ds>.png` — mean intrinsic signal (entropy and/or max-prob) vs token position, one line per (direction, τ), baseline overlaid.
  - `plot_length_verdict_<ds>.png` — stacked TRUE/FALSE/INCOHERENT fractions vs cutoff length, per (direction, τ).
- A short curated markdown table of vivid unsteered-vs-steered *paragraph* examples for the meeting (hand-picked from the CSV).

### 5. Compute / files
- New: `src/length_steer.py` (generation + intrinsic logging + writes `length_steer_<ds>.csv`), a judge entry that reads `length_steer_<ds>.csv` and writes `length_judge_<ds>.csv`, a plotting script, and two slurm scripts cloned from the MAG pattern (`deltaai/run_length_steer.slurm`, `deltaai/run_length_judge.slurm`).
- Runbook doc analogous to `deltaai/MAG_E4_RUN.md` for the two-job cluster run.

## Success criteria

1. Both datasets produce `length_steer_<ds>.csv`, `length_judge_<ds>.csv`, and the two plots.
2. The coherence plot shows `resid_pc1` at τ=+1 degrading (entropy up / max-prob down / rep3 up) as position grows, while `mean_diff` at τ=+1 tracks the τ=0 baseline.
3. The verdict plot shows `mean_diff` staying majority-TRUE across cutoffs while `resid_pc1` shifts toward INCOHERENT at longer cutoffs.
4. A handful of clean qualitative paragraph examples suitable to show the PI.

(2)–(3) are the *expected* narrative; if the data disagrees — e.g. mean_diff *does* start lying given length — that is a real, reportable finding, not a failure.

## Risks / notes

- **Greedy vs sampled:** greedy is deterministic and reproducible; entropy is still meaningful as the model's uncertainty even though we take the argmax. Keep greedy.
- **Judge on mid-sentence prefixes:** cutoffs land mid-sentence; INCOHERENT is the correct verdict for a broken tail, which is the signal we want. Acceptable.
- **cities stemming:** must strip the trailing answer robustly (regex on the known template); verify on a sample before the full run.
- **Magnitude comparability:** relies on `A_prefix_norm` being the right shared scale; it is the same calibration E4 used, so results are directly comparable to the existing MAG steering numbers.
