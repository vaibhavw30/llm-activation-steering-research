# Warm-Started DCT — Results

**Run date:** 2026-07-22 (DeltaAI GH200; train+steer job `2701296`, judge job `2703220`).
**Model:** `google/gemma-2-2b` base, fp32. Source/target layers: cities 11/20, common_claim 13/22.
**Judge:** OLMo-3, TRUE / FALSE / INCOHERENT, 32 prompts per (direction × τ).

## The question this run asked

In §5 of the since-last-meeting doc we found the supervised truth axis (`mean_diff`) is
**behaviorally inert-to-degrading**: steering along it does not make the model lie, it just
(sometimes) makes it incoherent. DCT's *own* most-potent factor (`cold_top`) is causally potent
but is **orthogonal to truth** (cos ≈ 0). So we have a decodable direction that isn't causal, and a
causal direction that isn't truth.

**Warm-started DCT** was the attempt to bridge them: seed DCT's causal-gradient factor search at the
supervised truth axis with a soft anchor of strength λ, and sweep λ ∈ {0, 0.3, 1, 3}. The hope: at
some λ the refined direction inherits DCT's causal potency *and* stays truth-aligned, becoming a real
**truth lever** — one where −τ makes the model assert falsehoods.

We tested 11 directions × 7 τ ∈ {−1, −0.6, −0.3, 0, +0.3, +0.6, +1} × 2 datasets. Sign convention:
directions point toward TRUE, so **+τ → TRUE, −τ → FALSE (lying)**. `input_scale` cities ≈ 47.7,
common_claim ≈ 86.7. τ=0 is a single shared untouched baseline (verified: all 11 directions produce
identical completions at τ=0).

## Part 1 — The anchor knob works exactly as designed (geometry)

`cos(warm factor 0, its seed)` climbs the intended ladder with λ, on both datasets — the scale-relative
anchor is a clean, monotone control:

| λ | cos(warm mean_diff, mean_diff) cities | common_claim |
|---|---|---|
| 0.0 | 0.03 | 0.02 |
| 0.3 | 0.28 | 0.25 |
| 1.0 | 0.72 | 0.66 |
| 3.0 | 0.95 | 0.95 |

So behaviorally we are genuinely interpolating from "free DCT search" (λ=0) to "essentially the
supervised axis" (λ=3). Any behavioral change across λ is attributable to the anchor, not to noise.
(`plot_dct_warm_drift_{ds}.png`.)

## Part 2 — The behavioral answer: **NO truth lever appears at any λ**

This is the headline. **In all 11 directions, on both datasets, the FALSE fraction never rises in a
sign-dependent way with −τ.** It sits at the ~0.03–0.09 baseline everywhere (see the red line pinned
to the floor in every panel of `plot_dct_warm_curves_{ds}.png`). Anchoring an inert supervised axis
toward DCT's causal search does **not** turn it into a lie lever — the interpolation runs from *inert*
(high λ) to *mild degrader* (low λ) and never passes through *truth lever*.

What steering *does* do, when it does anything, is **destroy coherence** — and only on the messy
dataset, only for the DCT-derived (low-λ / free) directions:

- **`cold_top` on common_claim is the cleanest signal** — and it's a **degrader, not a liar**:
  as τ goes +1 → −1, INCOHERENT climbs 0.03 → 0.06 → 0.06 → 0.16 → 0.12 → **0.47 → 0.66**, while
  FALSE stays flat (≤ 0.16). Positive τ *sharpens* the model (INCOH → 0.03, TRUE → 0.94); negative τ
  *breaks* it. Sign-dependent in **coherence**, not in **truth**.
- **`warm_grad_lam0` on common_claim** degrades at +τ (τ=+1: TRUE 0.25, INCOH 0.53), and
  **`warm_mean_diff_lam0`** degrades mildly at −τ (τ=−1: INCOH 0.31). Both are low-λ (near-free) fits.
- **The higher you crank λ, the more inert the direction becomes.** `warm_mean_diff_lam3`,
  `warm_grad_lam1/lam3`, and both raw supervised axes are flat across the whole τ sweep — TRUE stays
  ~0.8, nothing moves. Anchoring harder toward the supervised seed makes steering *safer/more inert*,
  the opposite of unlocking a causal lever.
- **cities is inert across the board** — at these τ magnitudes the well-baked city facts don't move in
  either coherence or truth for any direction. The degradation story is specific to the fragile,
  messy claims.

`plot_dct_warm_effect_{ds}.png` makes the dissociation one bar chart: per direction, the max rise
above baseline in FALSE (would-be lie lever, red) vs in INCOHERENT (degradation lever, grey). FALSE
bars are ~0 everywhere; the only bars that move are INCOH, on the low-λ DCT directions in common_claim.

## What it means

The experiment's own question — *does warm-starting close the decodable-≠-causal gap?* — gets a clean
**no**. The result is a **stronger negative than §5**: it's not just that the raw supervised axis is
inert; it's that **no point on the geometric path from the supervised axis to DCT's free causal search
is a truth lever.** The most causally potent direction we can find (`cold_top`, orthogonal to truth)
moves **coherence**, not truth. Pushing the model "away from true" doesn't produce confident
falsehoods — it produces incoherent text. That is the decodable-≠-causal dissociation, sharpened:
truth is linearly *decodable* from these activations (the probes work) but is not *steerable* as truth
by any direction in this family — the causal axis that exists is a fluency/coherence axis that happens
to be orthogonal to the truth axis.

Caveat worth stating in the meeting: the τ sweep tops out at ±1·input_scale at a single injection
layer; a stronger or multi-layer intervention might eventually force falsehoods (at the cost of ever
more incoherence). And the anchor is scale-relative (an engineering choice, commit `8d20fb0`), so
"λ=3 ≈ the supervised axis" is a cosine-0.95 approximation, not identity. But within the regime that
keeps generations coherent, the answer is unambiguous: **warm-starting does not manufacture a truth
lever.**

## Figures (in repo root)

| File | Shows |
|---|---|
| `plot_dct_warm_curves_{ds}.png` | **money figure** — TRUE/FALSE/INCOH vs τ, all 11 directions. FALSE flat everywhere. |
| `plot_dct_warm_effect_{ds}.png` | per-direction max ΔFALSE vs ΔINCOH — steering degrades, never lies. |
| `plot_dct_warm_drift_{ds}.png` | drift vs λ — the anchor knob is monotone. |
| `plot_dct_warm_verdict_{ds}.png` | (canonical viz) FALSE vs INCOH at the most-negative τ. |
| `plot_dct_warm_audit_{ds}.png` | cold DCT factors ranked by ‖U‖ potency, colored by |cos| to mean_diff — top factor is not truth-aligned. |
