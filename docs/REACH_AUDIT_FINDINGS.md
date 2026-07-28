# Backward-Reachability Audit — Findings (gemma-2-2b, cities 11→20, common_claim 13→22)

**Run:** DeltaAI GH200, 2026-07-24. All five phases complete; analysis on the merged
`feat/mag-e4-steering`. Figures: `plot_reach_audit_dissociation.png`,
`plot_reach_audit_certificate.png`, plus the standard `plot_reach_{margins,curves,geometry,svd,jlens}_<ds>.png`.

## Headline

**The FALSE-probe halfspace is "reachable" in the linear J-space sense, but that reachability is
behaviorally inert.** Steering a source-layer activation along `Jᵀw` toward the target-layer FALSE
set does **not** make the model assert falsehoods — the behavioral lie-rate stays at baseline
(~3%, vs ~3% unsteered) at every magnitude, and the completions stay coherent and *true*
("The city of Busan is in South Korea", "Abu Dhabi is in the United Emirates") even at the
strongest push. The linear reachability certificate does not transfer to behavior.

This is the LiSeCo / non-identifiability caveat, confirmed with a behavioral ground truth: a
readout being movable (or even *predicted* movable) is not the same as a behavioral lever.

## The five-phase chain

| Phase | Question | cities (11→20) | common_claim (13→22) |
|---|---|---|---|
| **P1** linear reachability | Is the FALSE set reachable within a small budget? | verdict **reachable-candidate**; median eps\*=2.80, all statements "reachable" | **reachable-candidate**; median eps\*=10.69 |
| **P2** Jacobian SVD | Do Jᵀ's dominant directions align with DCT's concept subspace? | overlap **0.062** (≪0.3), eff_rank 489/2304, s64/s1=0.11 | overlap **0.047**, eff_rank 539/2304 |
| **P3** behavioral steering | Does steering along Jᵀw flip the model's verdict? | **No** — lie-rate 3.5% at strongest push (vs 3% baseline); completions coherent+true | **No** — lie-rate 5.5% (vs 3% baseline) |
| **P4** per-layer J-lens | (context) margins by layer, decl vs quest | computed (`reach_jlens_cities.csv`) | computed |
| **P5** linearization trust | Is eps\* inside the region where the linearization holds? | eps\*=2.80 vs trust radius **2.99** → 0.94× (**inside**) | eps\*=10.69 vs trust radius **4.87** → **2.20× (OUTSIDE)** |

## Two distinct failure modes, both fatal to naive "reachable"

- **common_claim — the certificate is extrapolation.** eps\* (10.69) is 2.2× the linearization
  trust radius (4.87), so the "reachable" claim is evaluated far outside where the first-order
  model is valid. Empirically the per-statement readout never crosses the FALSE boundary under the
  predicted push (0% of statements cross g≤0; median g stays +60), matching the P5 diagnosis. The
  mean-arm readout is additionally *non-monotonic/unstable* at large scale (see the jump at
  ‖Δ‖≈5 in the figure) — off-manifold perturbation, not control.

- **cities — genuine readout↔behavior dissociation.** eps\* (2.80) is *inside* the trust radius
  (0.94×), and the readout does sit on the FALSE side — yet behavior still does not move. This one
  cannot be blamed on linearization error: the target-layer truth readout is (weakly) controllable
  where the linear model is valid, but controlling it buys **no** behavioral change. Readout ≠
  behavior.

In both cases **P2 already warned** the linearization was untrustworthy (Jᵀw's top-16 singular
subspace overlaps DCT-V at only ~0.05, and the Jacobian is near-isotropic at effective rank ~500 of
2304). P3 is the behavioral confirmation; P5 is the quantitative explanation.

## What the numbers were (for the writeup)

- **Behavioral lie-rate under strongest toward-FALSE steering** (per-statement arm, 200 true
  statements each): cities **3.5%**, common_claim **5.5%** — both within noise of the ~3% unsteered
  baseline. Incoherent fraction 18–26% (mild degradation, *not* asserted falsehoods).
- **Readout displacement vs prediction:** cities median |Δg_read| = 3.2 for eps\*=2.87 (the readout
  barely moves under the shared mean direction); common_claim median |Δg_read| = 21 for eps\*=12,
  never crossing the boundary.
- **P2 overlap:** 0.062 (cities), 0.047 (common_claim) — both fire the spec-§4 stop-and-diagnose
  (threshold 0.3).

## Caveats / honest limits

1. **Population mismatch between P1 and P3.** P1's eps\* is computed on full-statement target
   activations `h_tgt`; the steer arm reads the probe on generation-*prompt* prefixes ("The city of
   Busan is in South…"). Baseline readout signs differ (cities prefixes read g≈−52 at baseline;
   common_claim statements read +81), so the steer arm is **not** a direct falsification of the
   specific eps\* values — it is a behavioral test on a related population. The behavioral null is
   population-independent and stands on its own; the exact eps\*-vs-crossing comparison should be
   read qualitatively.
2. **Mean-arm readout is numerically unstable at large scale** (common_claim). Trust the
   per-statement arm and the behavioral fractions; treat the mean-arm readout curve as illustrative.
3. Single model, two hops, one probe family. The dissociation is demonstrated, not yet
   characterized across layers/probes.

## Deep audit (four parallel analyses, 2026-07-24 evening)

Four independent deep-dives (judge forensics, per-statement actuation, J-lens depth profiles,
SVD energy) sharpened — and in two places **corrected** — the story above.

### D1. Actuation is linear, correctly signed, and ~10–35× attenuated
Per-statement fits of `g_read ~ scale` (stmt arm): median R² = **0.999** (cities) / 0.990
(common_claim), **0% wrong-sign** in both datasets. But realized slope is far below the predicted
‖Jᵀw‖: realized crossing budget eps_realized ≈ **97 in both datasets** vs eps\* = 2.9 / 12.3 — a
**35× (cities) / 8× (common_claim)** shortfall. Since eps_realized ≈ 1–2× `input_scale`, a *real*
crossing needs a perturbation comparable to the activation's own norm — "reachable" is false in
any practical sense. Worse: for cities the predicted margins don't even rank-order actuation
(Spearman ρ = 0.002) — the certificate loses its *ordering* content, not just its gain
calibration. Attenuation is unimodal/uniform (no well-actuated subpopulation). Crucially the
protocol shift is tiny: eps\* was computed at the full-statement last token, steering/readout
happen at the stem (one word removed). **One word of context shift collapses Jacobian gain ~10×
while preserving linearity and sign.**

### D2. Behavior is inert — strictly (judge forensics)
The earlier "mild coherence degradation" read was too generous. INCOHERENT rate is **not**
dose-dependent (3 of 4 arms n.s.; one marginal p≈0.10). The 131 FALSE verdicts concentrate on a
handful of problem prompts that recur **identically at scale 0** (e.g. an MCQ-style "a. heart.
b. aorta" completion judged FALSE at every scale; judge factual errors on Abidjan/Maturín). The
single significant trend (common_claim mean-arm FALSE vs |scale|, p=0.012) runs **backwards** —
driven by the *toward-TRUE* extremes degenerating on 2–3 fixed prompts. Incoherence, where
scale-correlated at all (cities stmt arm), is topic-derailment into generic institutional filler
("Budapest is in → the process of building a new…"), sign-symmetric — an off-manifold nudge into
a high-frequency subspace, not negation. **Verdict: H1, inert. Not even an incoherence knob.**

### D3. The Jacobian is structured and concept-blind on the input side (corrects "near-isotropic")
The spectrum is NOT flat: eff_rank ~21–23% of 2304 with a two-regime decay (s64/s1 ≈ 0.11–0.16 —
too steep for a flat block, too shallow for one power law). The hop map routes through a compact
channel, and the channel's **output** contains the truth axis: `mean_diff_tgt` puts **74%
(cities) / 51% (common_claim)** of its energy in the top-64 left-singulars (17–125× random). But
the channel's **input** is concept-blind: `md_src` only 1.8–7× random in top-V, DCT overlap 0.05.
So the truth readout at the target is *downstream-visible* but not *upstream-addressable* by any
source-side concept direction — mean-diff, probe-grad, or DCT alike. On common_claim even the
probe-gradient direction is at/below random in top-U (0.58–0.73×). Any linear concept-steering
method operating in this band hits the same structural ceiling. (Side finding: cities' local
Jacobian geometry splits by label — true statements have higher eff_rank, lower s1, p<0.001 —
absent in common_claim.)

### D4. Controllability is front-loaded; we steered past the peak (J-lens)
All readouts (truth, v_q, verdict) peak at layers 0–8 and decay monotonically — no late
resurgence. At the target layers every readout has lost 65–87% of peak. The behavioral verdict
readout has **2–7× smaller absolute margin than the truth probe at every layer**, and quest-mode
verdict controllability (the pathway the judge actually elicits) crosses *below* decl-mode right
at the source layers (L13/L15). In 3 of 4 dataset×mode combos there is a distinct verdict-margin
peak at **layers ~5–9**, 1.5–3× better-conditioned than the layers we used (11/13). The
dissociation is thus over-determined: smaller lever (verdict ≪ truth), wrong depth (past peak),
wrong mode (quest collapsed), and D3's input-side misalignment.

### The refined headline
The linear-reachability certificate fails behaviorally not because linearization breaks (it
stays R²≈0.99!) but because of **three transfer failures**: (1) *context transfer* — Jacobian
gain collapses ~10–35× across a one-word context shift while sign/linearity survive; (2)
*population transfer* — the probe's t02 boundary doesn't transfer from statement tokens to
generation prefixes (cities prefixes read g≈−52, "deep FALSE", while the model completes
truthfully — the probe itself dissociates from behavior at baseline); (3) *input-side concept
misalignment* — J's high-gain channel is not addressable from source-layer concept directions.
And behavioral validation *inherently forces* transfer (1)–(2): behavior only exists during
generation, which is never the point where the certificate is computed. A behaviorally meaningful
reachability certificate must therefore be robust to exactly the shifts this one is not.

## So what — next steps

- The reachability certificate needs a **behavioral validity gate**: only report a hop as
  "reachable" when eps\* is inside the P5 trust radius **and** a behavioral spot-check moves. cities
  passes the first, fails the second → the interesting scientific case.
- Ask Julian: is the readout↔behavior gap expected for this probe/hop, and does A-LQR's behavioral
  objective close it? cities (readout-controllable, behavior-inert) is the cleanest test case to
  hand to A-LQR.

### Necessary next tests (from the deep audit; roughly in order of information per GPU-hour)

1. **Same-point control (the missing experiment, ~minutes of GPU).** Teacher-forced forward on the
   *full statement* with the injection hook active, read g at the statement's last token — the
   exact linearization point of eps\*. If realized slope ≈ ‖Jᵀw‖ there, the 10–35× attenuation is
   purely context/token shift (clean decomposition); if attenuated there too, the vjp margins are
   miscalibrated even locally. Either answer is decisive for the writeup.
2. **Probe recalibration on the generation population.** Fit/threshold the truth probe on
   prefix-token activations (labels = truth of the model's own completions). Quantifies transfer
   failure (2) and re-answers "did steering cross the *right* boundary?" — the −52 cities baseline
   says the current t02 is meaningless on prefixes.
3. **In-channel steering.** Steer along top right-singular vectors v₁..v_k of J (the directions J
   actually amplifies), or Jᵀw projected onto top-V. Tests D3 directly: does input-side alignment
   restore gain across the context shift, and does *anything* behavioral move when you push the
   high-gain channel?
4. **Early-window, verdict-direction steering.** Source at L5–8 (the J-lens peak), direction =
   pullback of the *verdict* (yes/no logit-diff) readout, quest-mode prompts. The J-lens says this
   is the best-conditioned behavioral configuration by 1.5–3×. If behavior moves here but not in
   the original configuration, the audit's mechanistic story is confirmed causally.
5. **Iterative/receding-horizon steering (bridge to A-LQR).** Recompute Jᵀw at the steered state
   every step (2–3 Newton-style iterations within the P5 trust radius). Tests whether one-shot gain
   collapse is direction *rotation* under context shift — and is exactly the open-loop→closed-loop
   upgrade A-LQR formalizes.
6. **Judge hardening (cheap, local).** Drop/flag the ~15 identified problem prompts (MCQ-style
   completions, Abidjan/Maturín-class judge errors) or require 2-judge agreement; rerun the
   fractions. The null is already robust but this removes the noise floor for future arms.
