# DCT vs. MAG on Truth — Findings

*Is the funnel's "decodable ≠ causal" null a fact about **truth in gemma-2-2b**, or an artifact of
the one unsupervised miner we used (DCT)? We built a **second, mechanically unrelated** miner — MAG,
which reads features from the **prefix-induced activation shift** rather than from a steering
objective — and ran it head-to-head on the same four truth datasets. **Short answer: the null
replicates.** MAG's truth direction recovers the supervised direction almost perfectly (cos ≈ 0.98–1.00
on all four datasets) yet is **orthogonal to DCT's top causal lever** (cos ≈ 0 on both datasets where
DCT exists), and DCT's causal subspace shows **no truth advantage over random directions**. A separate,
unexpected finding: MAG's **self-verdict channel is dead on a base model** — gemma-2-2b answers "yes"
to essentially every "Is this true?" question — so the *fully unsupervised* arm of MAG does not work
here without an instruction-tuned model.*

Companion to `DCT_VS_TRUTH_FINDINGS.md` (the linear funnel), `DCT_VS_XGBOOST_FINDINGS.md` (the
non-linear extension), and `PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md` §8 (the plain-language MAG
narrative). Code: `src/mag/`, `src/run_mag.py`, `src/viz_mag.py`. Per-dataset artifacts:
`mag_readability_<ds>.csv`, `mag_linearity_<ds>.csv`, `mag_dir_<ds>.npz`, `mag_transfer.csv`,
`plot_mag_*_<ds>.png`.

**Status:** Complete. E1–E3 + directions + transfer (the *geometric* battery, §3–§5) and the E4
*behavioral* run (§7) are done on the truth datasets. E4 ran on a GH200 (cities + common_claim,
9 directions × 5 τ, OLMo-judged free-form + a matched-format yes/no flip test) and **closes both
leads negatively**: lead #1's divergent operators are decodable-but-inert or merely format-biasing,
lead #3's residual-PC1 is a non-directional degradation lever — consistent with the decisive rank-1
result (§6). Causal/behavioral control over truth collapses onto the single supervised axis.

---

## 1. Why this experiment exists

The funnel established that gemma-2-2b's supervised truth direction — ~99% linearly decodable on clean
data — is **not** among the directions DCT flags as most causally salient, and steering it barely moves
factual behavior. Every subsequent test hardened that null, but all of them shared one dependency: the
unsupervised miner was always **DCT**. The cleanest objection a skeptic can raise is *"maybe truth just
isn't the kind of structure DCT's objective finds — a different miner would recover it."*

MAG (Mining via Activation Geometry) is that different miner. It does not optimize a steering objective.
It asks the model a question about its own input — "Is this statement true?" — and measures how the
residual stream **moves** when the question is prepended: Δ<sup>Q</sup>(p) = m(Q‖p) − m(p). The premise
is that if the model internally represents the answer, that representation should show up in the shift.
This is a mechanically independent route to the same target, so agreement between DCT and MAG is
evidence about **truth in the model**, not about either method.

---

## 2. Method (what actually ran)

Per dataset, at the truth-peak layer (`BEST_LAYER`: sp_en_trans 7, cities 11, companies 14,
common_claim 13), over gemma-2-2b **base**, fp32:

- **Extraction** (`mag.extract`): 8 operator ingredients + the self-verdict `y^M` (the model's own
  first-token yes/no to "Is this true?"). Caches: `mag_acts_<ds>.npz`. common_claim extracted on the
  DeltaAI GH200 (CUDA); the other three on the laptop (MPS) — fp32 throughout, floating-point
  differences far below probe scale.
- **E1 readability**: 5-fold stratified CV (per-fold scaling, no leakage) linear probe on each of the 8
  operators, plus DCT-top-k and random-k baselines on raw activations. Reported for both **gold** labels
  and the self-verdict **y^M**.
- **E3 linearity**: ε_Q, the residual fraction of the prefix shift a single direction fails to explain
  (0 = perfectly one-dimensional).
- **Directions**: `u_Q` (contrast of shifts) vs. the supervised `mean_diff` and `grad` directions and
  DCT's top-potency vector `V`, by cosine.
- **Transfer**: does the MAG (InputDelta) direction found on one dataset rank the others' truth signal?
- **E2 disagreement**: on statements where `y^M` ≠ gold, which does the geometry track? — *not
  measurable here* (see §5).

---

## 3. Result 1 — MAG's geometry recovers truth (gold-label arm)

The prefix shift carries essentially the **full** linear truth signal. On every dataset, the best MAG
operator (InputDelta = A_Qp − A_p, or Interaction) matches raw activations to within ~1 point of
accuracy:

| Dataset | Layer | Direct (raw) acc / ROC | InputDelta acc / ROC | random-10 acc |
|---|---|---|---|---|
| sp_en_trans | 7 | 0.980 / 0.995 | 0.972 / 0.994 | 0.791 |
| cities | 11 | 0.993 / 1.000 | 0.993 / 1.000 | 0.968 |
| companies | 14 | 0.914 / 0.975 | 0.908 / 0.972 | 0.681 |
| common_claim | 13 | 0.714 / 0.788 | 0.720 / 0.794 | 0.633 |

And the direction MAG builds from those shifts **is** the supervised truth direction:

| Dataset | cos(u_Q, mean_diff) | cos(u_Q, grad) |
|---|---|---|
| sp_en_trans | **0.998** | 0.562 |
| cities | **0.988** | 0.408 |
| companies | **0.988** | 0.199 |
| common_claim | **0.981** | 0.088 |

Two methods that never share a line of code — supervised mean-difference and MAG's contrast-of-shifts —
land on the same axis. (The lower `grad` cosines are expected: the logistic gradient is whitened by the
feature covariance, so it points differently from a raw mean-difference; `mean_diff` is the
apples-to-apples comparison.)

**E3**: the prefix shift is close to one-dimensional, and gets *less* so as the concept gets messier —
a clean monotonic trend: sp_en_trans ε_Q = 0.18 < cities 0.24 < companies 0.48 < common_claim 0.55.
Truth lives in the shift's **class contrast**, not its mean: the raw shift direction `v_Q` is
orthogonal to `mean_diff` (cos ≈ 0 on all datasets) — the *average* effect of prepending the question
is generic "now I'm answering a question" processing; the *difference between true and false* shifts is
where truth sits.

---

## 4. Result 2 — the funnel null replicates under MAG

MAG gives us a second, independent handle on the "decodable ≠ causal" claim, and it agrees with DCT.

**Orthogonality.** On the two datasets where DCT vectors exist, MAG's truth direction is orthogonal to
DCT's strongest causal lever:

| Dataset | cos(u_Q, DCT top-V) |
|---|---|
| cities | **+0.002** |
| common_claim | **−0.096** |

The direction the model is *most causally driven by* and the direction that *reads truth* are unrelated
— exactly the funnel's finding, now reached from the activation-shift side instead of the steering side.

**No subspace advantage.** Projecting raw activations onto DCT's top-k causal directions does not
concentrate truth beyond what random directions of the same rank achieve:

| Dataset | DCT top-10 | DCT top-50 | random-10 | full-space (Direct) |
|---|---|---|---|---|
| cities | 0.947 | 0.989 | 0.968 | 0.993 |
| common_claim | 0.669 | 0.722 | 0.633 | 0.714 |

On cities, DCT's top-10 subspace reads truth *worse* than 10 random directions (0.947 < 0.968); on
common_claim it is marginally above random but DCT-top-50 only reaches full-space parity. Either way,
**DCT's causal subspace holds no special truth signal** — the linear echo of the XGBoost-subspace
result in `DCT_VS_XGBOOST_FINDINGS.md`, now confirmed by a second miner.

**Transfer is weak.** Across all four datasets the MAG (InputDelta) direction transfers only modestly
(Top-1 = 0.25, Spearman = 0.404 over the 12 ordered dataset pairs) — the truth axis is somewhat
domain-specific, not a single portable direction.

---

## 5. Result 3 (unexpected) — the self-verdict channel is dead on a base model

MAG's *fully unsupervised* promise rests on `y^M`: the model labeling its own inputs, so no gold labels
are needed. **On gemma-2-2b base, that channel is degenerate — the model answers "yes" to essentially
every statement:**

| Dataset | y^M distribution | agree(y^M, gold) |
|---|---|---|
| sp_en_trans | 354 yes / 0 no | 0.500 |
| cities | 1496 yes / 0 no | 0.500 |
| companies | 1200 yes / 0 no | 0.500 |
| common_claim | 4449 yes / 1 no | 0.500 |

Because the datasets are balanced, "always yes" scores exactly 0.500 — chance. Consequently **every
y^M-target probe and E2 (disagreement) is undefined** (reported as NaN): there is no meaningful
disagreement set when the model never says "no." The `Verdict` operator corroborates from the geometry
side — activations read at the verdict position separate true from false at chance (0.48–0.63 across
datasets), so the model's yes/no machinery simply isn't engaging the truth of the claim.

This is a genuine finding, not a bug: it says the base model has no usable *behavioral* truth verdict,
even though the *geometry* (§3) clearly separates true from false. The unsupervised-labeling arm of MAG
would need an **instruction-tuned** model to be testable. The gold-label (semi-supervised) arm — which
is what §3–§4 report — is unaffected.

*(Implementation note: this degeneracy surfaced a robustness bug — the E1 CV guard rejected the
all-one-class case but not the 4449/1 near-degenerate case, which crashed logistic regression on a
single-class training fold. Fixed to require ≥ n_splits minority samples; regression test added.)*

---

## 6. E4 setup — the two remaining leads, and a rank-1 result

E4 injects `α(τ)·û` at a direction's native layer (`α(τ) = τ·A_prefix_norm`, τ ∈ {−1, −0.3, 0, +0.3, +1}),
generates, and scores TRUE/FALSE/INCOHERENT with the OLMo judge — the *same* battery as the supervised
steering arm, so every candidate is directly comparable to the known supervised null (steering the truth
direction degrades symmetrically; it does not make the model lie). Beyond the canonical `u_gold`/`u_yM`
and the supervised `mean_diff`/`grad` baselines, E4 now also tests two leads that a skeptic could raise
against "MAG just re-finds the supervised axis."

**Lead #1 — the divergent operators.** The operators that read truth well but whose class contrast points
*off* the supervised `mean_diff` axis (Prefixed, Answered, QuestionDelta, FewShot) each contribute a
steering candidate `u_op = class_mean_diff(operator_features(op))`. Their 1-D contrast directions read
truth above chance yet sit ≈ orthogonal to the primary truth axis:

| Operator | dir. readout acc (cities) | cos(u_op, primary u_Q) |
|---|---|---|
| Prefixed | 0.749 | +0.04 |
| Answered | 0.897 | −0.02 |
| QuestionDelta | 0.749 | +0.04 |
| FewShot | 0.921 | −0.04 |

These are genuinely different directions that still carry truth — the necessary precondition for "a chance"
at a *different* (possibly more causal) lever. E4 will say whether any of them induces directional lying
rather than the supervised degradation.

**Lead #3 — the residual, and why the "second truth axis" is provably zero.** On messy data the prefix
shift is multidimensional (ε_Q ≈ 0.55 on common_claim), which invites the hope of a *second*, more causal
truth direction hiding off the primary axis. It does not exist as a linear contrast: the primary axis
`û_Q` **is** the class-mean-difference of the shift, so residualising the shift against it and re-taking the
class-mean-diff is identically zero. Measured on real activations, `‖resid class-mean-diff‖ / ‖u_primary‖`
= **7.9 × 10⁻¹⁰ (cities)** and **1.1 × 10⁻⁹ (common_claim)** — truth is **exactly rank-1** in the shift's
mean structure; one direction captures the entire linear signal. The dominant direction that *does* remain
after removing `û_Q` is residual-PC1, and it reads truth at **0.477 / 0.498 = chance** — it is not a truth
axis. E4 steers it anyway, reframed honestly as the probe *"is any large non-truth component of the prefix
shift causal, or is only the (impotent) truth axis?"*.

**What this changes.** MAG offers no *new* linear truth direction distinct from the supervised one on the
clean axis (the well-reading operators land on `mean_diff` at cos ≈ 0.99; the shift is rank-1 for truth).
The open behavioral question is entirely whether the *off-axis* candidates — lead #1's divergent operators
and lead #3's residual-PC1 — behave any differently under steering than the supervised direction's known
degradation. That is exactly what the pending GH200 E4 run measures.

## 7. Result 4 — E4 behavioral test: causal control collapses onto the supervised axis

The GH200 sweep ran both datasets: 9 directions × 5 τ, free-form completions scored
TRUE/FALSE/INCOHERENT by the OLMo judge, plus a **matched-format yes/no flip test** on 24 obviously
true statements (baseline answer "yes"; a *flip* = the model now wrongly answers "no" = an induced lie).
Figure: `plot_mag_e4_headtohead.png`.

**The discriminator.** Since `u = mean(false) − mean(true)` points toward FALSE, +τ pushes the model
toward lying and −τ toward truth. A *genuine directional lie-lever* flips the yes/no verdict at **one**
sign and leaves true statements alone at the other (antisymmetric). A *degradation lever* flips at
**both** signs — a large-norm injection breaks generation regardless of direction. That single test
separates the whole panel:

| direction | flip τ=−1 / τ=+1 (cities) | flip τ=−1 / τ=+1 (common_claim) | reading |
|---|---|---|---|
| `sup_mean_diff` | 0.79 / 0.00 | 1.00 / 0.00 | antisymmetric → **directional** (truth axis) |
| `mag_u_gold` | 0.83 / 0.00 | 1.00 / 0.00 | antisymmetric → **directional** |
| `sup_grad` | 0.29 / 0.92 | 0.04 / 0.83 | antisymmetric (opp. sign) → **directional** |
| `Answered` (lead #1) | 0.00 / 0.00 | 0.00 / 0.00 | **inert** — despite 0.90 readout acc |
| `FewShot` (lead #1) | 0.00 / 0.00 | 0.00 / 0.33 | **inert** — despite 0.92 readout acc |
| `Prefixed` / `QuestionDelta` (lead #1) | 0.17 / 0.50 | 0.00 / 1.00 | flips lie-side only, but uncorroborated (below) |
| `mag_u_yM` | 0.00 / 0.00 | 0.00 / 0.50 | ~inert |
| `mag_resid_pc1` (lead #3) | **0.96 / 0.96** | 0.42 / 1.00 | flips **both** signs → **degradation** |

**Lead #1 closes — decodability does not predict causality.** The two *best* truth readers, Answered
(0.90) and FewShot (0.92), have **zero** behavioral effect: perfectly decodable, causally inert.
Prefixed/QuestionDelta (identical directions) do move the forced verdict at the lie-sign (up to 1.00 on
common_claim), but the flip is **uncorroborated by free-form generation**: at that same τ their output is
**84% TRUE / 9% FALSE / 6% incoherent** — the model still *writes* true statements, it just says "no" in
the yes/no slot. That is a forced-answer/format bias, not a coherent lie-lever. No divergent operator
gives a cleaner causal handle than the supervised axis; the strongest readers are the weakest levers.

**Lead #3 closes — no second causal component.** residual-PC1 flips the verdict at *both* τ signs
(0.96 / 0.96 on cities), the signature of magnitude-driven degradation rather than directional control —
exactly what the proven rank-1 result (§6) predicts. The large off-axis component of the prefix shift
carries no directional truth signal; injecting it hard only disrupts.

**And the funnel null replicates behaviorally.** Free-form FALSE never exceeds ~0.25 for *any* direction,
including the supervised truth axis — no direction makes gemma-2-2b *coherently* assert falsehoods; they
degrade (INCOHERENT) or, in the yes/no format, bias the answer token. The truth axis has *directional
verdict control* but not *coherent lie generation*, and every MAG candidate is either inert,
format-biasing, or disruptive. The behavioral picture matches the geometry: decodability is spread across
many operators, causality is rank-1 and supervised.

**Caveats.** n = 24 yes/no + 32 free-form prompts; one model, one layer per dataset; τ calibrated to full
prefix strength (|τ| = 1), which the U-shaped free-form FALSE curve suggests is already the degradation
regime — a finer sub-prefix τ grid could expose directional control before degradation sets in. On a base
(non-instruction-tuned) model the yes/no format is weakly grounded, likely why the forced-answer flip is
so easily biased. An instruction-tuned model + finer τ sweep is the natural follow-up.

## 8. Bottom line for the PI

> **The funnel's null is not a DCT artifact.** A second, mechanically unrelated miner (MAG, reading the
> prefix-induced activation shift) recovers the supervised truth direction almost perfectly
> (cos ≈ 0.98–1.00 on all four datasets) — truth is *there* in the geometry — yet that direction is
> **orthogonal to DCT's top causal lever** (cos ≈ 0 on both datasets where DCT exists), and DCT's causal
> subspace carries **no truth signal beyond random**. Two independent unsupervised methods agree that
> truth is easy to read and not causally load-bearing.
>
> One bonus result cuts the other way and is worth flagging: MAG's **self-labeling premise fails on a
> base model** — gemma-2-2b answers "yes" to every truth question, so the geometry knows what the model
> cannot say. Fully-unsupervised MAG needs an instruction-tuned model.
>
> **The behavioral E4 run (§7) confirms it causally.** Steering MAG's off-axis candidates does *not*
> beat the supervised null: the divergent operators that read truth best (Answered 0.90, FewShot 0.92)
> are behaviorally **inert**, the ones that move the forced answer do so as a format bias (84% of their
> free-form output stays true), and the off-axis residual only *degrades* (flips at both τ signs). Causal
> control over truth is rank-1 and lands on the single supervised axis — decodable ≠ causal, now shown
> from both the geometry and the behavior.

**What's next.** The battery is complete — the geometric arm (§3–§6) and the behavioral arm (§7) agree:
truth is easy to read, spread across many operators, and causally rank-1 on the supervised axis. The two
open follow-ups both come straight from §7's caveats: (1) **a finer sub-prefix τ grid** — |τ| = 1 sits in
the degradation regime (U-shaped free-form FALSE), so a sweep over small τ could reveal whether any
direction has *directional* control before degradation swamps it; (2) **an instruction-tuned model**,
which fixes both the dead self-verdict channel (§5) and the weakly-grounded yes/no format that let the
forced-answer flip be biased (§7). The reproducible assets are `deltaai/run_mag_steer.slurm` +
`run_mag_judge.slurm` (E4 generation + judge) and `plot_mag_e4_headtohead.png` (the §7 figure).
