# The 20-minute version

*What I ran on the cluster since the last meeting, why, what came back, and what it means.
Five figures, five numbers. The full walkthrough is
[`EXPLAINER_FOR_THE_PI.md`](EXPLAINER_FOR_THE_PI.md); this is the talk track.*

**Suggested timing:** 2 min setup, 4 min what I ran, 9 min results, 3 min what it means,
2 min asks. Everything after §5 is backup for questions.

---

## 1. Where we left off, and the one-line result (2 min)

Last meeting you reframed the problem: stop hunting for a single steering **direction**,
define a **target set** and compute its backward-reachable set. Because the map is locally
linear, the tools are just linear algebra.

I did that. Here is the whole result in one line:

> **The certificate is exactly right and the model does not budge. The reason is that the
> truth direction is no better aligned with the decision that picks the next word than a
> randomly drawn vector is. When I redefined the target set in token space, as you suggested,
> the same machinery started working.**

The five numbers I want you to leave with:

| Number | What it is |
|---:|---|
| **0.9997** | The certificate's calibration at its own linearization point. The math is exact |
| **3.5%** | The model's lie rate under maximum push, against a 3% unsteered baseline. The behavior is inert |
| **1.7%** | The fraction of the *deciding* logit margin our old steering direction was actually consuming |
| **200/200** | Token flips using the token-space target set at the output layer, both datasets |
| **0.340** | Token flip rate once that is pulled back through the Jacobian to layer 16, at the certified budget, where the old direction gets 0.000 |

---

## 2. The idea, in one slide (part of the 2 min)

Pick a site to inject at, pick a scalar you want to move. Then:

```
g     = how far you are from the boundary          (the gap)
m     = how much the readout moves per unit push   (the controllability margin)
eps*  = g / m                                      (the certified budget)
```

`m = ||A^T w||`, where `A` is the Jacobian from the injection site to the readout and `w` is
the readout direction. That transpose is the whole method: it costs **one backward pass per
statement**, so we never build a 2304x2304 Jacobian and a per-statement certificate over
hundreds of statements is affordable.

The analogy: **`g` is how far the goalpost is, `m` is how fast your car goes in top gear, and
`eps*` is how long the drive takes.** What we found is that the drive is short, the car works
exactly as the spec says, and the goalpost is not where the game is being played.

---

## 3. What I ran, and why (4 min)

Three rounds on DeltaAI GH200, about 475 GPU-hr remaining in the allocation.

| Round | Date | Why I ran it | What it cost |
|---|---|---|---|
| **Reachability audit**, 5 phases | 07-24 | Compute the certificate and test it from five angles at once: are the margins real, is the high-gain channel the concept channel, does behavior move, is the depth right, is the linearization valid | 5 jobs |
| **Horizon 0**, 7 validations | 07-29 | The audit came back null. Before believing a null I wanted every loophole a reviewer would poke closed: is the math wrong, is the probe threshold just shifted, are the bad prompts driving it, is it curvature | 2 jobs |
| **Token-space program**, 8 experiments | 08-04 | Your note 6, plus the controls that could have invalidated the whole negative result | 4 jobs, ~8 GPU-hr, **no judge job** |

**Two design points worth 30 seconds each.**

*The token-space round needed no LLM judge at all.* The outcome variable is "does the
next-token argmax equal the target token id", which is a mechanical comparison. That is a
direct consequence of your note 6 and it is most of why the round was cheap.

*Every run carries an `oracle` arm.* At the output layer I solve for the least-norm
displacement into the target token's cone and verify it in closed form against all 256,000
vocabulary entries. **Its flip is a theorem, not a hypothesis.** So if the running model does
not emit the target, the injection code is wrong and I throw the run away. This is your note 3
turned into something that stays checked, and it is the single best decision in the program.

**Before the GPU rounds I also ran two free CPU checks**, because each could have killed a
downstream program for zero cost:

- **Is the token-space target set even non-empty?** There is a known result that tokens whose
  embedding sits inside the convex hull of all embeddings can never be the argmax for any
  hidden state. If our false-answer tokens were interior, the proposal was dead. Answer:
  **182 of 182 achievable**, proved with an explicit witness for each, not estimated.
- **Is the null just aggregate cancellation?** The literature reports that half of examples can
  be anti-steerable, which would flatten an average while individuals move both ways. Answer:
  **no, for cities.** 103 of 199 statements are byte-identical at every steering scale, and
  among those that move the effect goes the *right* way 24 times against 4 (p = 1.8e-4).
  Cancellation cannot explain a null when nothing moves in either direction.

---

## 4. Results (9 min)

### 4.1 The certificate is real, and behavior does not move

The truth readout is genuinely controllable on paper: margin **9.07** against a random-direction
null of **1.10**, so 8.3x, and the budget to flip it is a small fraction of the activation's own
norm.

Then we steered at that budget and generated:

| | cities | common_claim |
|---|---:|---:|
| lie rate at the strongest toward-FALSE push | **3.5%** | 5.5% |
| unsteered baseline | ~3% | ~3% |

Figure: [`plot_judge_reach_stmt_cities.png`](../plot_judge_reach_stmt_cities.png). **The flat
lines are the finding.** Even at maximum push the completions stay coherent and true: "The city
of Busan is in South Korea."

### 4.2 The math is not the problem, and I checked that hard

This is the part that turns a bug into a result. Three of the seven Horizon-0 validations:

| Check | Question | Answer |
|---|---|---|
| Same-point control | Is the certificate right where it is defined? | **Calibration 0.9997** on cities, exact to four significant figures |
| Newton re-steering | Is the problem curvature? | No. Residual gap drops **six orders of magnitude in 3 steps**, final direction at cosine 0.971 to where we started |
| Judge hardening | Are ~15 bad prompts driving the null? | No. After removing them the baseline is a **clean 100% true** and the strongest push barely dents it |

The Newton check is worth calling out because it was a **deliberate negative control**: the
diagnosis predicted it would not help, and it did not.

So the loss is not the math. It is the **transfer**. The certificate is computed on the full
statement; behavior only exists during generation, which starts one word earlier. Across that
one-word shift the gain collapses **13.7x**.

### 4.3 The strongest supporting number: the probe knows nothing at generation time

I took the token where generation actually begins, labelled each one for free using judge
verdicts we already had, and tried four ways to read truth off it.

| | cities | common_claim |
|---|---:|---:|
| old threshold, **balanced** accuracy | **0.500** | **0.500** |
| recalibrated threshold | = base rate | = base rate |
| refit logistic probe, 5-fold CV | 0.930 | 0.944 |
| **XGBoost**, 5-fold CV | gap **+0.013** | gap **+0.000** |

Recalibrating recovers nothing. Refitting recovers nothing. **XGBoost recovers nothing**, which
kills the obvious "the boundary is just nonlinear here" objection. This is not a threshold
shift, it is an **absence of signal**, and I have not found it claimed in the literature.

The feature-level version of the same thing, for the talk: in SAE feature space, removing **one
word** of context replaces the actuator's features almost entirely. Jaccard **0.049**, 3 shared
features of 32, against a control measuring the same quantity on a different sample of
statements at **0.684**. Figure:
[`plot_sae_overlap_cities.png`](../plot_sae_overlap_cities.png). One word does not weaken the
direction, it substitutes a different one.

### 4.4 The actual diagnosis: alignment, not linearity

The number nobody had ever computed is the **logit margin**, the gap between the token the
model is about to emit and the one we want. Flipping cities to a false country costs a median
**3.4% of the activation norm**. It is cheap.

So why did nothing move? Because of where we were pushing. Against a chance floor of
`sqrt(2/(pi d)) = 0.0166` for a random unit vector in 2304 dimensions:

| Direction | alignment with the token decision | vs chance |
|---|---:|---:|
| final-layer mean-difference truth direction | 0.00398 | **0.24x** |
| the direction we actually steered with | 0.01325 | 0.80x |
| probe-gradient direction | 0.01355 | 0.82x |

**All of them at or below chance.** On cities all three are significantly *worse* than random
(p < 1e-6). Figure: [`plot_token_alpha_cities.png`](../plot_token_alpha_cities.png).

Stated in the unit that matters: **at the budget the old certificate calls a full trip into the
FALSE halfspace, our direction consumes 1.7% of the deciding margin.** We were not pushing
weakly on the right thing. We were pushing on something statistically indistinguishable from
noise.

### 4.5 Your token-space fix works

Define the target set as the set of activations that emit the target token. At temperature 0
that is exactly a convex cone in the unembedding basis, so the *same* `g/m` formula applies with
`w = E[target] - E[current]` and `t = 0`. The gap `g` is now literally the logit margin, and
**the readout cannot dissociate from behavior because it is the decision.**

| site | direction | flip rate at certified budget | at 2x |
|---|---|---:|---:|
| output layer | `oracle` (closed-form cone solution) | **1.000** (200/200) | **1.000** |
| layer 16, pulled back through the Jacobian | token direction | **0.340** | **0.675** |
| every arm, every budget | **old probe direction** | **0.000** | **0.000** |

Figure: [`plot_token_steer_cities.png`](../plot_token_steer_cities.png).

And it is a *semantic* flip, not just noise: the fraction of completions naming the target false
country goes from **0.010** unsteered to **0.075** and **0.120**, at a degeneracy rate of 0.005
and 0.020. Compare the old truth direction at the output layer, which produces target 0.000 at a
degeneracy of 0.885. **The token direction produces a lie; the truth direction produces
silence.**

---

## 5. What it means (3 min)

**What went wrong, in one sentence.** I certified arrival at a fitted probe's halfspace at layer
20 while the judge measured the argmax at layer 26, with five unmodeled operations in between.

**Why your suggestion fixes it structurally, not incrementally.** Token space collapses that
gap. Membership in the target set *is* the behavior, so the dissociation is impossible by
construction rather than merely unlikely.

**What the negative result is actually a result about.** It is not "truth is not steerable in
gemma-2-2b." It is: *a certificate whose target set is defined by a fitted readout can be
arithmetically exact and behaviorally meaningless, and the diagnostic that tells you which you
have is alignment against the chance floor.* That is a methodological claim about the whole
activation-steering literature, and every steering paper that reports a budget without reporting
alignment is exposed to it.

**The honest boundary, which I would rather say than be asked.** Everything in the token program
is a **first-token** claim, and the completion recovers: `oracle` flips the first token on 200 of
200 statements, and yet only 0.085 of those completions go on to name the target country while
0.700 still name the correct one. **Reaching the target at position `t` does not keep the model
there at `t+1`.**

**And that is exactly where this hands off to your work.** Extending the target set over a
horizon, with a value recursion or a receding-horizon controller that re-plans each token, is the
structurally correct object. It is also precisely where our failures point: our gain collapses
across one word of context, which is the canonical **open-loop** failure mode that a closed-loop
design answers. Separately, our R² = 0.999 is a direct quantitative corroboration of A-LQR's
local-linearity assumption.

---

## 6. What I want from you (2 min)

**Three decisions:**

1. **arXiv:1910.13272** from your notes is a feedback-linearization paper, not a reachability
   one. Intended as the "invert the output map" formalization, or a typo?
2. **The 2026-07-23 conditional-steering round** is built and was never submitted before the
   pivot. My read is retire it and say so in the writeup. Overrule me if you disagree.
3. **Is the refusal positive control still worth 3 to 6 GPU-hr?** It validates the
   *probe-halfspace* instrument, which the audit chapters rest on. But the token-space program
   now carries a positive control by construction, and it passes.

**What I would run next, in order:** the pre-norm arm swept further (one job, and it is a
falsifiable prediction I have written down in advance); the token direction re-run at layers 0 to
8, where the budget is 2.6x cheaper than where we steered; and an oracle arm at the layer sites so
that 0.340 becomes a fraction of what was provably available rather than a bare number.

---

## Backup: likely questions

**"Could the null be a bug in your steering hook?"** The `oracle` arm is the answer. At the output
layer the flip is provable in closed form and it fires on 200 of 200 statements. Separately, the
realized gain at that site must be exactly 1 by arithmetic, and it measures 1.0000 across 13
directions and 4 budgets on both datasets.

**"You broadcast the perturbation to every position, including BOS."** Correct, and I found that
by reading our own code. It is worth **2.5x** on the realized flip rate at cities layer 16 (0.340
broadcast against 0.135 at the last position only). Both conventions now run side by side. It also
means our budgets are not directly comparable to a published ActAdd budget without dividing
through by that ratio.

**"You had `repetition_penalty = 1.3` on during a truth experiment."** Also correct, also found by
reading our own code, and it means the decoder was not a pure argmax so the cone geometry did not
literally hold for the older runs. Quantified: it changed degeneracy from 0.885 to 0.000 and empty
completions from 740 to 20, and it changed the **flip counts not at all**. It inflated what we saw
downstream; it did not change the token we were measuring.

**"Is the latent space chaotic, so a norm budget is meaningless?"** Two different claims, and they
get opposite answers. Scale dependence is **absent**: the linearity ratio is 1.0000 exactly at the
output layer and within 6.1% of 1 at layer 16, so a first-order certificate is valid across the
whole range we sweep. Anisotropy is **present** but structured: exactly one direction stands out at
depth, at 8.7x the random median, and everything else including both truth directions sits inside a
1.74x band with the random directions. An anisotropic but linear site needs different **units**, not
a different theory, and `eps/alpha` is exactly those units.

**"Did you sweep layers?"** All 26, on 200 statements, both injection conventions. Control authority
is **front-loaded**: layer 0 is the cheapest and cities' layer 11 costs 2.6x it. We steered past the
peak. This independently reproduces the audit's Phase 4 finding with a completely different readout.

**"How much does this rest on one dataset?"** The two datasets are never pooled and they fail by
different mechanisms. **cities is the discriminating one** and it carries every argument: its target
is a semantically false country, whereas common_claim's target is the runner-up token, so on
common_claim generic disruption can land on target by accident. common_claim's certificate is also
outside its own linear validity radius (2.20x) where cities' is inside (0.94x).

**Caveats I will state unprompted:** one model; the refusal positive control has not run, so for the
probe-halfspace certificate "the instrument works" and "the instrument is broken" are not yet
separated; the 0.500 result rests on 9 and 10 false examples at a 0.94 base rate; n = 200 per
dataset; temperature 0 only; no judge ran in the token program, so nothing there is a truth verdict.
