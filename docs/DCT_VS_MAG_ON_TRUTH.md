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

**Status:** E1–E3 + directions + transfer (the *geometric* battery) are complete on all four datasets,
reported below. **E4 (calibrated behavioral steering → OLMo judge) is now built and gated on the two
remaining leads** (§6): lead #1, the *divergent operators*, and lead #3, the *off-truth-axis residual*.
The E4 *geometry* is settled here (including a decisive rank-1 result, §6); the E4 *behavioral* numbers
(FALSE-vs-INCOHERENT curves) come from the GH200 run driven by `deltaai/run_mag_steer.slurm` +
`deltaai/run_mag_judge.slurm` and are the only piece still pending.

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

## 7. Bottom line for the PI

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

**What's still open:** the E4 *behavioral* run. The code is complete — `mag.steer` now emits the
lead #1 and lead #3 candidates (§6) alongside `u_gold`/`u_yM`/supervised, the judge I/O-prefix plumbing
is fixed (`judge_results.py --mag` → `judge_mag_steer_<ds>.csv`, closing the item flagged in
`PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md` §8), and the injection path is smoke-tested. What remains is
the GH200 sweep itself: `deltaai/run_mag_steer.slurm` (gemma generation) then `deltaai/run_mag_judge.slurm`
(OLMo TRUE/FALSE/INCOHERENT), scored on the **same** battery as the DCT steering arm. That produces the
behavioral head-to-head; §3–§6 are the geometric one. The prior is strong that the off-axis candidates
inherit the supervised degradation, but the rank-1 result (§6) makes them the *only* remaining place a
distinct causal truth-lever could hide, so the test is worth running.
