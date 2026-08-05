# Next Steps Brief, and a Deep Research Prompt

**Date:** 2026-08-03
**Status:** post-audit, post-PI-meeting. Supersedes the "where this goes" section of
[RESULTS_SINCE_LAST_MEETING_PART3.md](RESULTS_SINCE_LAST_MEETING_PART3.md) §10.
**Audience:** you first, then the PI.

This document has four parts.

- **Part I** writes out every math operation from token input to token output and
  justifies each one, because the PI asked for exactly that.
- **Part II** decodes the PI's notes into eight testable claims, and reports three
  things I found in our own code today that those notes implicate.
- **Part III** is the experiment program: eight experiments, each with a cost, a
  prediction, and a decision gate.
- **Part IV** is a self-contained deep research prompt to paste into Claude chat.

---

## Part 0. Where we are, in one page

We built a backward-reachability certificate for a linear truth readout. Pick a
source layer (11) and a target layer (20). Define a readout `w` at layer 20 and a
target set `{u : w·u <= t02}` called the FALSE halfspace. Linearize the layer 11 to
layer 20 map, get `J`, and compute for each statement the minimum perturbation norm
that provably lands the readout inside the target set:

```
g = w·h_tgt − t02        the gap
m = ‖Jᵀw‖                the controllability margin
ε* = g / m               the certified budget
```

The certificate is correct. We then steered along `−Jᵀŵ` at `1×` and `2×` that
budget and judged the generated text with an independent model.

**The readout moved exactly as predicted and the behavior did not move at all.**

The linear prediction is excellent (per-statement R² of readout versus scale: cities
median 0.9991, 97% above 0.99). The behavior is a flat line. On cities, **51.8% of
statements emit a byte-identical completion at all five steering scales**, and 70.4%
are byte-identical to baseline at twice the certified budget. On common_claim the
text does change (only 4.1% identical) but the truth verdict does not (FALSE rate
11, 11, 10, 12, 11 across the sweep, trend p = 0.92).

We diagnosed four contributing mechanisms (D1 context transfer, D2 population
transfer, D3 input-side concept misalignment, D4 controllability front-loading). But
the PI's notes reframe all four as symptoms of one design decision: **we certified
arrival at a probe's halfspace, not at a behavior.** Everything below follows from
taking that seriously.

---

## Part I. The full math, input space to output space

The PI asked for every operation written out and justified. Here it is for
gemma-2-2b. Config facts are read from the local checkpoint
(`config.json`, snapshot `c5ebcd40`): `hidden_size = 2304`, `num_hidden_layers = 26`,
`vocab_size = 256000`, `final_logit_softcapping = 30.0`, `attn_logit_softcapping =
50.0`, `rms_norm_eps = 1e-6`.

### I.1 The forward map, operation by operation

Let the prompt be token ids `t_1..t_L`.

**(1) Embedding, with a scale factor.**

```
x_i = E[t_i] · sqrt(d),     d = 2304, sqrt(d) = 48
```

`E` is the (256000, 2304) embedding matrix. Gemma multiplies embeddings by
`sqrt(d)`. *Why it matters to us:* this is the exact factor that bit us in the
`SlicedModel` landmine (transformers >= 5 stopped re-applying it to `inputs_embeds`),
and it is why every reach script now prints `[reach] slice fidelity cos=1.000000` at
startup. It is a scalar, so it does not change any direction, only magnitudes.

**(2) The residual stream.** Write `h^(0) = x`. For each layer `l = 1..26`:

```
h^(l) = h^(l−1) + Attn_l( RMSNorm(h^(l−1)) ) + MLP_l( RMSNorm(·) )
```

(gemma-2 additionally post-norms each sublayer output; that detail does not change
the argument.) *Why it matters:* the residual stream is a running sum, so an additive
intervention `h^(l) ← h^(l) + Δ` is exactly the kind of perturbation the architecture
is built to carry forward. This is the entire justification for additive steering.

**(3) Our intervention.** We add a fixed vector `Δ ∈ R^2304` to the input of layer 11,
**broadcast across every sequence position**, on every forward pass including every
generation step (`dct_steer_utils.py:39-48`, a `forward_pre_hook` that returns
`args[0] + v` where `args[0]` has shape (batch, seq, d) and `v` has shape (d,)).

**(4) The hop map and its Jacobian.** Define

```
F(Δ) = h^(20)_last(Δ)         the target-layer activation at the last prompt token
J    = ∂F/∂Δ |_(Δ=0)          shape (2304_tgt, 2304_src)
```

`J` is obtained by one vector-Jacobian product (`torch.func.vjp`), never by 2304
forward passes. *Justification:* we only ever need `Jᵀw` for a fixed `w`, and a VJP
computes exactly that in one backward pass. This is the single most important
efficiency decision in the pipeline.

**(5) The readout.** `w` is a unit vector at layer 20. `s = w·h^(20)_last` is a scalar.

**(6) The certificate.** With `g = s − t02` and `m = ‖Jᵀw‖`,

```
F(Δ) ≈ F(0) + JΔ    ⟹    Δs ≈ w·JΔ = (Jᵀw)·Δ
```

so the minimum-norm `Δ` that achieves `Δs = −g` is `Δ = −g·(Jᵀw)/‖Jᵀw‖²`, with norm
`ε* = g/m`. This is the point-to-hyperplane distance formula. It is exact for the
linearized system and we measured where it stops holding (`reach_linerr`, 20%
relative error threshold).

**Everything above is what we built. Everything below is what we never modeled.**

**(7) Final norm.** `z = RMSNorm(h^(26)) = h^(26) / sqrt(mean(h²) + 1e-6) ⊙ (1 + γ)`.

*This is a nonlinearity, and a consequential one.* RMSNorm is **homogeneous of degree
zero**: scaling `h` by any `c > 0` leaves `z` essentially unchanged. Consequence: at
the final layer, **the radial component of any steering vector is discarded**. Only
the angular change survives to the logits. Nobody in this project has checked how
much of `Jᵀw` is radial.

**(8) Unembedding.** `u = W_U z`, where `W_U` is (256000, 2304) and tied to `E`.

**(9) Logit softcapping.** `û_j = 30 · tanh(u_j / 30)`.

*Justification for ignoring it in one place and not another:* `tanh` is strictly
increasing, applied coordinatewise with the same cap, so it **preserves the argmax
ordering exactly**. Any temperature-0 statement is unaffected by softcapping. But it
strongly compresses probabilities, so any temperature > 0 or probability-margin
statement must include it.

**(10) Repetition penalty.** Our generation calls
`model.generate(..., do_sample=False, repetition_penalty=1.3)`
(`dct_steer_utils.py:66`). For any token already present in the sequence, the logit is
divided by 1.3 if positive and multiplied by 1.3 if negative.

**(11) Decoding.** `do_sample=False`, so the next token is `argmax` over the penalized
logits. `max_new_tokens = 8`.

### I.2 The one-sentence version of what went wrong

We certified arrival at step (6). The judge measures step (11). **Steps (7) through
(11) are five operations we never modeled**, including one degree-zero homogeneity
(the norm), one 256000-way discrete argmax, and one prefix-dependent reweighting.
The audit's D1 through D4 are all descriptions of what happens inside that gap.

### I.3 The reformulation the PI is proposing, written out

Move the readout to the end of the chain. Then the gap is zero operations wide.

**At temperature 0, the model's output token is a deterministic, piecewise-constant
function of the final activation.** The set of final activations that emit token `j`
is exactly

```
C_j = { z : (W_U[j] − W_U[k]) · z ≥ 0   for all k ≠ j }
```

an intersection of 255,999 halfspaces: a convex polyhedral cone. (Softcapping drops
out by monotonicity, per step 9.) Pulled back through RMSNorm, which is degree-zero
homogeneous, `C_j` in pre-norm residual space is a **cone**, not a halfspace.

Now define the target set behaviorally. For the stem "The city of Busan is in", let
`T` be the set of first tokens that begin a false answer (" Norway", " Brazil", ...).
Let `k*` be the current argmax. For a candidate target token `j ∈ T`, define

```
a_j = W_U[j] − W_U[k*]           the verdict direction for this token pair
M_j = u_{k*} − u_j = −a_j · z    the logit margin (how far from flipping)
```

The minimum-norm move in `z`-space that makes `j` beat `k*` is `M_j / ‖a_j‖`, and
pulling back to source layer `l` with the same VJP machinery gives

```
ε*_token(l, j) = M_j / ‖J_lᵀ a_j‖
```

**Compare that to what we already compute:**

| our certificate | token-space certificate |
|---|---|
| `w` = probe direction at layer 20 | `a_j = W_U[j] − W_U[k*]`, an unembedding difference |
| `t02` = calibrated probe threshold | `0`, by definition of argmax |
| `g = w·h − t02` | `M_j` = the logit margin |
| `m = ‖Jᵀw‖` | `‖J_lᵀ a_j‖` |
| `ε* = g/m` | `M_j / ‖J_lᵀ a_j‖` |

**It is the same formula with a different `w` and `t02 = 0`.** We do not throw away
the pipeline. We change the readout, and the dissociation we spent three weeks
measuring becomes impossible by construction, because the readout *is* the behavior.

### I.4 Why this also explains the byte-identical result

At temperature 0 the token map is piecewise constant. A perturbation does **nothing
at all** until it crosses a face of the cone, then flips discontinuously. So "51.8%
of completions are byte-identical across all five scales" is not mysterious: those
statements' perturbations never consumed the logit margin `M`. The right measurement
was never "did the text change." It was **"what fraction of `M` did we consume."**
That number has never been computed. It is cheap. It is E1 below.

### I.5 Temperature, written out

```
P(j) = exp(û_j / T) / Σ_k exp(û_k / T)
```

- `T → 0`: all mass on the argmax; the map is deterministic and piecewise constant.
  This is our setting.
- `T = 1`: the model's native distribution.
- `T > 1`: flattens toward uniform while preserving the ordering.

*Why the PI raised it:* at `T = 0` the input-to-output map is a genuine deterministic
function, so "invert the mapping" is a well-posed question with a closed-form answer
(the cone `C_j`). At `T > 0` the target set softens into `{z : P(j) ≥ τ}`, which is a
smooth sublevel set (and here softcapping does NOT drop out). Starting at `T = 0` is
the right call: the geometry is exact and convex.

---

## Part II. The PI's notes, decoded, plus three findings from our code

### II.1 The eight claims

| # | PI's note | What it asserts | Status |
|---|---|---|---|
| 1 | "naively steer on output, if it causes the same problem, mostly making it incoherent" | A no-Jacobian control isolates *formulation error* from *linear-steering weakness* | **Never run.** E1. |
| 2 | "if incoherent, issue with linear feature in the final layer" | If even direct final-layer steering only degrades text, the truth feature is not behaviorally linear there | E1 decides |
| 3 | "check that steering vector is what you want it to be" | Verify the injected vector is what we think, where we think | Partially done; **gaps found**, see II.2 |
| 4 | "small perturbations in different directions can have vastly different chains (chaotic latent space)" | Directional sensitivity is wildly heterogeneous; a norm budget is the wrong currency | Not measured. E6. |
| 5 | "SAE is a useful alternative to classify the target sets" | Define the target set by interpretable features, not a probe hyperplane | Machinery exists, unrun. E5. |
| 6 | "temperature 0, deterministic, then invert that mapping of the tokens" | **Define the target set in token space and pull it back.** | The main idea. E3. Math in §I.3. |
| 7 | "write out all the math from input to output and justify" | Conceptual writeup | Part I above |
| 8 | "ActAdd does a full search over all layers; over some layers the perturbation does nothing; mean difference is heuristic" | Our layer choice (11 → 20) came from DCT and was **never swept** | E2 |

### II.2 Three things I found in our code today

**Finding A: the perturbation is broadcast to every position, including BOS.**

`dct_steer_utils.Steerer._hook` adds `v` of shape (d,) to `args[0]` of shape
(batch, seq, d). NumPy/torch broadcasting means **every token position gets the same
`Δ`**, including the BOS token. gemma-2 uses BOS as a strong attention sink. ActAdd
and CAA add at a small span of positions, not globally.

This is self-consistent with our Jacobian (which was computed under the same
broadcast, and `reach_jlens.py:97` sums gradients over positions), so the certificate
is not wrong. But two things follow that nobody has checked:

1. The *total* injected energy scales with sequence length: `L` positions each
   receiving norm `ε` is a perturbation of Frobenius norm `ε·sqrt(L)`, not `ε`.
   Whether `input_scale` (47.72 / 86.73, from DCT's `SteeringCalibrator`) was
   calibrated under the same convention determines whether our "we stayed inside the
   trusted budget" claim is even in the right units.
2. Corrupting the attention sink is a plausible, cheap-to-test source of the
   incoherence we attributed to over-steering.

**This is a candidate confound for the entire negative result and it costs one
ablation to rule out.** It is E7.

**Finding B: `repetition_penalty = 1.3` is on, in a truth experiment, at 8 tokens.**

`dct_steer_utils.py:66` sets `repetition_penalty=1.3`, inherited from
`apply_dct_vector.py`. On an 8-token factual completion this is aggressive and it is
a confound in three ways: it makes the decoder not a pure argmax (so §I.3's clean
cone geometry does not literally hold for our existing runs); it penalizes tokens
that appear in the prompt, which for "The city of X is in" includes common
continuation tokens; and it interacts with steering in a way nobody modeled. **Turn
it off for all reachability experiments.** There is no reason for it here.

**Finding C: the layer-sweep and unembedding-readout infrastructure already exists.**

`reach_jlens.py` already builds

```python
w_verdict = unit(W_U[yes].mean(0) − W_U[no].mean(0))     # reach_jlens.py:65-66
LAYERS = list(range(0, 26))                              # reach_jlens.py:35
```

and computes `‖J_lᵀ w‖` for every layer `l` from 0 to 25, for a readout defined in
the **unembedding basis**. That is 80% of the machinery E2 and E3 need. The PI's
proposal is an extension of code we have already written and already run, not a
rewrite. `plot_reach_jlens_<ds>.png` is the existing figure.

---

## Part III. The experiment program

Ordered by information per GPU-hour. E1, E4, E6, E7 are cheap and should run as one
batch before anything expensive.

### E1. Naive output-layer steering (the PI's first ask, and the triage)

**Do:** Skip the Jacobian entirely. Intervene directly at the target layer (20), then
at the final layer (26), adding `−δ·w` for a grid of `δ`. At layer 20 the readout
crosses `t02` exactly by construction, with zero linearization error, zero hop, zero
context transfer. Generate, judge, and also record the logit margin `M` consumed.

**Why it is decisive:** it removes every explanation we have been offering.

| outcome | conclusion |
|---|---|
| readout crosses, behavior unchanged | The FALSE halfspace is **behaviorally meaningless**. D1 and D2 are irrelevant. The target set definition is the bug. Go straight to E3. |
| behavior changes | The **hop is the problem**. Our formulation or the source layer is at fault. D1/D4 are real. |
| output becomes incoherent | The truth feature at the final layer is **not behaviorally linear at the magnitude required**. This is the PI's claim 2. Go to E5 and E6. |

**Cost:** ~1 GPU-hr. No new math. **Run this first.**

**Key auxiliary measurement:** for every generation, log `M = u_top1 − u_top2` at the
stem's last token, unsteered and steered. The ratio `ΔM / M` is the missing number
from §I.4. If we never consumed more than a few percent of the margin, the negative
result was never about truth at all.

### E2. Full layer sweep (ActAdd)

**Do:** For source layers 0 through 25, steer along the layer-appropriate direction
(mean-diff at that layer, and `J_lᵀw`) at a few magnitudes. Produce a
26 × n_scales heatmap of (a) readout change, (b) verdict change, (c) coherence rate.

**Why:** ActAdd's central empirical result is that the layer matters enormously, and
that at many layers the model simply absorbs the perturbation with no behavioral
effect. Our source layer 11 was chosen by DCT for a different objective. We have
never checked whether it is a layer where steering does anything at all.
`reach_jlens.py` already sweeps all 26 layers for margins; this extends it to
behavior.

**Cost:** ~4 to 8 GPU-hr. **Highest value per hour of anything on this list.**

### E3. Token-space target sets (the reformulation)

**Do:** Implement §I.3.

1. Confirm bit-exact determinism at `T = 0` for fixed batch size, and measure how
   much it breaks when batch size changes (this is a known, real effect).
2. For each stem, define `T` = first tokens of false answers, and `k*` = the current
   argmax.
3. Compute `a_j = W_U[j] − W_U[k*]`, `M_j`, and `ε*_token(l, j) = M_j / ‖J_lᵀa_j‖`
   for every layer `l`.
4. Steer along `−J_lᵀ â_j` at that budget. Verify the token actually flips.

**Why:** the certificate now certifies a behavior. The readout-versus-behavior
dissociation cannot occur by construction. If this works, the paper changes from "we
found a negative result" to "we found the negative result, diagnosed it, and fixed
it," which is a much better paper.

**The check that could kill it, and that would itself be a finding:** *stolen
probability.* There is a known structural result that tokens whose output embedding
lies in the interior of the convex hull of all embeddings **can never be the argmax
for any hidden state**. If the false-answer tokens are interior points, the target
set is literally empty and no amount of steering can ever produce them. Test this
before running anything: it is a pure linear-algebra check on `W_U`, costs no GPU,
and either result is publishable.

**Cost:** CPU for the geometry, ~4 GPU-hr for the steering. **The main line.**

### E4. Decoding audit

**Do:** Turn off `repetition_penalty`. Re-run a small slice of the Phase 3 steering
with it off and compare. Confirm greedy determinism. Sweep temperature 0, 0.7, 1.0
at fixed steering to separate "steering did nothing" from "sampling noise swamped
steering."

**Why:** Finding B. Also, the common_claim result ("text changes, truth does not") has
an alternative explanation we have not excluded: some of that text change may be
decoding artifact rather than steering effect.

**Cost:** ~1 GPU-hr. Run in the E1 batch.

### E5. SAE-defined target sets

**Do:** Run `sae_decompose.py` (written, never run, laptop-only, free) on `w`,
`Jᵀw`, and the `V64` subspace. Then define the target set as "GemmaScope feature `f`
active above `τ`" rather than as a probe halfspace, and use the feature's decoder
direction as the actuator.

**Why:** the PI's claim 5, plus it directly tests D3 (input-side concept
misalignment). If `w` decomposes onto features that are not about truth, D3 is
confirmed with interpretable evidence rather than a cosine.

**Caveat to state up front:** the literature has a strong negative prior on SAE
features as *steering* directions (simple baselines beat them). Use SAEs to **define
and diagnose** the target set, not to actuate it.

**Cost:** free, CPU, today.

### E6. Directional sensitivity spectrum (the "chaos" claim)

**Do:** At fixed `ε`, for many directions (random, `Jᵀw`, top-`V64` singular vectors,
SAE decoder atoms), measure the realized `‖Δh^(26)‖ / ‖Δ‖` and the realized `ΔM`.
Report the full distribution, not the mean.

**Why:** claim 4. If gain varies by orders of magnitude across directions, then a
norm budget `ε` is the wrong currency for a certificate, and the right one is
something like a directional gain-normalized budget. This would be a genuine
contribution to the reachability framing.

**Cost:** ~2 GPU-hr. Pairs naturally with the existing `reach_svd` artifacts.

### E7. Injection-site ablation

**Do:** Compare four injection conventions at matched total perturbation energy:
all positions (current), all except BOS, last prompt token only, prompt only (not
generated tokens).

**Why:** Finding A. This is the cheapest possible check on a candidate confound for
the whole negative result.

**Cost:** ~1 GPU-hr. Run in the E1 batch.

### E8. Verify the actuator is what we think it is

**Do:** Assert, at runtime, that immediately after the hook fires,
`‖h_perturbed − h_clean‖` at the source layer equals `‖Δ‖·sqrt(L)` to numerical
precision; that the hook fires on the layer index we intend; and that the realized
`Δs` matches `ε·‖Jᵀw‖` at small `ε` (we know it matches at small `ε` and falls 13×
short at `1ε*`, which is D1, but the small-`ε` assertion has never been an automated
test).

**Why:** claim 3, done properly. This is a unit test, not an experiment, and it
should be permanent.

**Cost:** free.

### Recommended sequencing

```
Now (free, laptop):     E5 SAE decomposition, E8 assertions,
                        E3 step 0 (stolen-probability check on W_U)
Batch 1 (~4 GPU-hr):    E1 naive steering + E4 decoding audit + E7 injection ablation
                        → this batch alone can invalidate or rescue the whole result
Batch 2 (~8 GPU-hr):    E2 layer sweep
Batch 3 (~6 GPU-hr):    E3 token-space certificate, E6 sensitivity spectrum
```

Total roughly 18 GPU-hr against ~475 remaining. The refusal positive control
(19 to 30 GPU-hr, already fully implemented and gated) should wait until E1 reports,
because E1 may change what the positive control needs to test.

---

## Part IV. Deep research prompt

Paste everything in the block below into Claude chat with research enabled.

---

````
I am a researcher working on activation steering and reachability analysis in
language models. I need a thorough literature review to guide the next phase of an
experimental program. Please search deeply, prioritize primary sources, and give me
exact arXiv IDs and publication venues. Flag any ID you are not certain about.

## My setup

Model: gemma-2-2b (26 layers, d_model 2304, vocab 256000, final logit softcapping
30.0, tied embeddings, RMSNorm). Greedy decoding.

What I built: a backward-reachability certificate for a linear "truth" probe. I pick
a source layer (11) and a target layer (20), linearize the map between them to get a
Jacobian J, define a unit readout direction w at the target layer and a threshold
t02, and compute for each statement the minimum-norm perturbation at the source layer
that provably moves the readout into the target halfspace {u : w·u <= t02}:

    g = w·h_tgt − t02,    m = ‖J^T w‖,    ε* = g/m

I steer along −J^T w at 1x and 2x that certified budget, broadcast to all sequence
positions, and judge the generated text with an independent model.

What I found: the readout moves exactly as predicted (per-statement R² of readout
versus steering scale, median 0.999) and the behavior does not move at all. On a
clean dataset, 51.8% of completions are byte-identical across all five steering
scales. On a messier dataset the text changes but the truth verdict does not
(trend p = 0.92). I call this a readout-versus-behavior dissociation.

My advisor's diagnosis: I certified arrival at a probe's halfspace, not at a
behavior. The proposed fix is to define the target set in TOKEN space (at temperature
0 the model is deterministic, so the set of final activations emitting a given token
is exactly the polyhedral cone {z : (W_U[j] − W_U[k])·z >= 0 for all k}) and pull that
set back through the same Jacobian machinery, so that the readout IS the behavior.

## What I need from you

### 1. Activation steering: what is known about when it works and when it does not

Find and summarize the primary literature on additive activation steering, with
particular attention to NEGATIVE and reliability results. Specifically:

- ActAdd / activation engineering (Turner et al., I believe arXiv:2308.10248). I am
  told it does a full sweep over layers and finds that at many layers the
  perturbation has no behavioral effect, that the model "just eats" the feature
  direction. Confirm this, and tell me exactly what their layer-sweep protocol was
  and what they found.
- Representation Engineering (Zou et al., ~arXiv:2310.01405)
- Inference-Time Intervention (Li et al., ~arXiv:2306.03341), which targets
  truthfulness specifically and is the closest prior work to mine
- Contrastive Activation Addition (Rimsky et al., ~arXiv:2312.06681)
- Reliability and generalization of steering vectors (Tan et al., ~arXiv:2407.12404).
  I understand this reports high variance and brittleness. What exactly fails?
- Refusal is mediated by a single direction (Arditi et al., ~arXiv:2406.11717)
- Any survey of activation steering from 2025 or 2026

Key questions: (a) Is there a published account of steering that moves a probe
readout but not behavior? (b) What is the standard protocol for choosing the
intervention layer, and is mean-difference known to be a weak heuristic? (c) What is
known about injecting at all sequence positions versus a subset, and does anyone
report that perturbing the BOS token or attention sink causes degeneration?

### 2. The token-space / unembedding geometry angle

This is the most important section for me. I want everything relevant to defining a
target set in output-token space and pulling it back into activation space.

- The softmax bottleneck (Yang et al., ~arXiv:1711.03953): what does the rank
  constraint imply about which token distributions are even expressible?
- "Stolen probability" (Demeter et al., ACL 2020, ~arXiv:2005.02433): I understand
  this shows that tokens whose output embedding lies in the interior of the convex
  hull of the embedding set can NEVER be the argmax for any hidden state. Confirm
  this, give me the precise statement and conditions, and tell me what fraction of
  vocabulary is typically affected. This would mean some behavioral target sets are
  literally empty, which is critical to my experiment.
- Logit lens and tuned lens (Belrose et al., ~arXiv:2303.08112): what do these say
  about reading final-basis directions at intermediate layers?
- Any work on the geometry of the argmax decision regions of the unembedding, on
  logit margins, or on the polyhedral structure of the output map.
- Any work analyzing RMSNorm/LayerNorm as a degree-zero homogeneous map and what that
  implies for steering (specifically: at the final layer, is the radial component of
  a steering vector discarded?)

### 3. Determinism, temperature, and sampling

- Formal or empirical work on determinism of LLM inference at temperature 0. I am
  aware of a 2025 result on batch-invariance and nondeterminism in LLM inference
  (I think from Thinking Machines Lab) showing that identical inputs give different
  outputs when batch composition changes. Find it and tell me the mechanism and the
  fix.
- The effect of temperature on task performance and on output diversity (e.g.
  Renze & Guven, ~arXiv:2402.05201; Holtzman et al. nucleus sampling
  ~arXiv:1904.09751).
- Any work on repetition penalty as a confound in interpretability or steering
  experiments. My pipeline had repetition_penalty=1.3 on during a truth-completion
  experiment and I want to know if this is a known trap.

### 4. Sparse autoencoders for defining target sets (not for steering)

- Gemma Scope (~arXiv:2408.05147) and JumpReLU SAEs (~arXiv:2407.14435)
- AxBench (~arXiv:2501.17148), which I understand reports that simple baselines beat
  SAEs for steering. Confirm and give the exact claim.
- Sparse probing with SAEs (Kantamneni et al., ~arXiv:2502.16681): are SAE features
  useful for CLASSIFICATION even where they fail for steering? This distinction is
  the crux for me, because I want to use SAEs to define and diagnose a target set,
  not to actuate it.
- Improving steering vectors by targeting SAE features (Chalnev et al.,
  ~arXiv:2411.02193)
- Correct methodology for decomposing a fixed DIRECTION (not an activation) into SAE
  atoms. I use orthogonal matching pursuit against the decoder because I understand
  naive encoder application is wrong for directions. Find the source for that claim
  and any better method.

### 5. Sensitivity, chaos, and certified perturbation analysis

- Is there work measuring how much the behavioral effect of a fixed-norm activation
  perturbation varies across directions? I want to know whether "perturbation norm"
  is the wrong budget currency.
- Lipschitz constant estimation and certified robustness for deep networks
  (e.g. Fazlyab et al. ~arXiv:1906.04893, alpha-beta-CROWN), and whether any of it
  has been applied to transformer residual streams.
- Adversarial suffix work (GCG, ~arXiv:2307.15043) as evidence for extreme
  input sensitivity, and prompt-perturbation sensitivity results.

### 6. Control theory and reachability applied to LLMs

- Hamilton-Jacobi reachability background (Bansal, Chen, Herbert, Tomlin,
  ~arXiv:1709.07523) and DeepReach (~arXiv:2011.02082)
- "Taming AI Bots: Controllability of Neural States in Large Language Models"
  (Soatto et al., ~arXiv:2305.18449). This appears to be the closest formal work to
  my framing. Summarize its controllability result precisely.
- Any work applying MPC, LQR, or formal reachability to steering language model
  activations
- Probe-constrained decoding or safety-constrained generation using linear probes

### 7. Probing versus causality

- The Linear Representation Hypothesis (Park et al., ~arXiv:2311.03658)
- The Geometry of Truth (Marks & Tegmark, ~arXiv:2310.06824), which is my direct
  baseline
- LEACE concept erasure (~arXiv:2306.03819), amnesic probing (~arXiv:2006.00995),
  INLP (~arXiv:2004.07667). Key question: is there a documented pattern where erasing
  or moving a linearly-decodable concept fails to change model behavior? That is
  precisely my result and I want to know if it is already named in the literature.
- Probing methodology critiques (control tasks, Hewitt & Liang ~arXiv:1909.03368)

### 8. Synthesis, which is what I most need

After the review, answer these directly:

a) Is my readout-versus-behavior dissociation already a known and named phenomenon?
   If so, what is it called and who found it first? If not, is it a genuine
   contribution?
b) Is the token-space target-set reformulation novel, or has someone already
   certified reachability against an unembedding-defined target region?
c) What is the strongest published counter-explanation for my negative result that I
   have not considered? Be adversarial about this.
d) Given all of the above, what are the three highest-value experiments to run next,
   and what would each one rule out?
e) What is the single most likely reason my experiment failed, according to the
   literature rather than according to me?

Please organize by section, give exact citations, and clearly separate what the
literature establishes from what remains open.
````

---

## Part V bis. Results of the two free checks (run 2026-08-03)

Both zero-GPU checks from the sequencing table have now been run. Both returned
usable answers, and one of them clears the main risk to E3.

### V bis.1 SAE feature forensics (E5), both datasets

Command: `PYTHONPATH=src .venv/bin/python src/sae_decompose.py --dataset <ds>`, then
`src/viz_sae.py`. GemmaScope 16k, `average_l0` 80/71 (cities L11/L20) and 83/72
(common_claim L13/L22), OMP with least-squares refit, 32 atoms drawn from 16384.

**Result 1: the readout is feature-like, the actuator is not.**

| vector | cumulative explained @ 32 atoms, cities | common_claim |
|---|---|---|
| `w_mean_diff_tgt` (the readout `w`) | **0.504** | **0.649** |
| `jtw_mean` (the direction we actually inject) | **0.254** | 0.381 |
| top SINGLE atom, `w` vs `jtw_mean` | 0.060 vs 0.036 | 0.165 vs 0.065 |

The truth readout at the target layer is a reasonably concentrated object in feature
space. The actuator we inject at the source layer is about half as concentrated. This
is **D3 (input-side concept misalignment) restated in interpretable terms**, and it is
a stronger statement than the cosine we had been quoting
(`cos(Jᵀw, mean_diff@src) = 0.0955`): the readout looks like a concept, the actuator
looks like a diffuse mixture spread across the dictionary.

**Result 2: D2 quantified against a proper control. This is the strong finding.**

| pair | cities | common_claim |
|---|---|---|
| `jtw_full_matched` vs `jtw_mean` (same quantity, different statement subsets) | **0.684** | **0.684** |
| `jtw_full_matched` vs `jtw_stem` (**one word of context removed**) | **0.049** | 0.143 |
| two random 32-subsets of 16384 (chance floor) | ~0.001 | ~0.001 |

The 0.684 row is what makes the 0.049 row interpretable, and it was not part of the
original design. Measuring the *same* quantity on a *different sample of statements*
reproduces 68% of the feature support. Removing *one word* from the context reproduces
**5%**, which is 14x below that noise ceiling and only 50x above pure chance.

Deleting the final word of the statement nearly completely replaces the set of
features the actuator is built from. This is the mechanism behind the stem-population
probe's balanced accuracy of 0.500, and it is a far more legible figure for a talk.

**Result 3:** `common_v1`, the top singular vector of the stacked full and stem
directions (nominally "what survives the context shift"), sits at Jaccard 0.362 with
full and 0.164 with stem on cities. It is essentially the full-context direction under
another name, so there is little genuinely shared structure to steer along.

Artifacts: `sae_features_<ds>.csv`, `sae_overlap_<ds>.json`,
`plot_sae_explained_<ds>.png`, `plot_sae_overlap_<ds>.png`.

### V bis.2 Stolen-probability check (E3 step 0)

Script: `src/stolen_probability.py`. Output: `stolen_probability_cities.csv`.

**The test, and why it is exact in both directions.** By minimax on a compact convex
set,

```
max_{||z||<=1} min_{k!=j} (W[j] − W[k])·z
  = min_{λ in simplex} || Σ_k λ_k (W[j] − W[k]) ||
  = dist( W[j], conv{W[k] : k != j} )
```

so achievability of token `j` as an argmax is *exactly* the statement that `W[j]` is a
vertex of the convex hull of the unembedding rows. A positive value yields a **witness
`z`** whose full-vocabulary argmax is `j` (a proof of achievability); a value driven to
zero yields **convex weights reconstructing `W[j]` from the other rows** (a proof of
non-achievability). Solved by Frank-Wolfe, batched so that one GEMM
`W @ P.T` per iteration serves all candidates at once.

Two modelling points: the effective unembedding is `W = E ⊙ (1+γ)` because the final
RMSNorm gain sits between the residual stream and the tied unembedding; and logit
softcapping drops out entirely, since a coordinatewise strictly-increasing `tanh`
preserves argmax ordering.

**Result: 182 of 182 candidate tokens are ACHIEVABLE. Zero non-achievable.**

| group | n | verdict |
|---|---|---|
| `true_country` (first tokens of all 108 country names in cities.csv) | 102 distinct | 102 ACHIEVABLE |
| `random_control` | 64 | 64 ACHIEVABLE |
| `lowest_norm` (the 16 smallest-norm rows in the whole 256000 vocabulary) | 16 | 16 ACHIEVABLE |

97 of the 182 were resolved immediately by the self-witness `z = unit(W[j])`; the
remaining 85 were resolved by Frank-Wolfe, whose distances plateaued in the range
1.196 to 4.643 and never approached zero. Every one of the 182 passed the explicit
full-vocabulary witness check, so **the conclusion is a proof, not an estimate**,
independent of how well Frank-Wolfe converged.

Three things follow.

1. **E3 is viable. The token-space target set is non-empty.** The single risk that
   could have killed the reformulation before it started is now ruled out for the
   tokens we care about.
2. **The false-answer tokens are a strict subset of the true-answer tokens.**
   `cities.csv` draws its 89 wrong countries from the same pool as its 108 correct
   ones (`set(false) − set(true)` is empty), so the same 102 first-tokens serve both
   roles. Nothing about the target set is special or degenerate.
3. **Embedding norm is a poor proxy for interiority in gemma-2-2b.** The 16
   lowest-norm rows in the entire vocabulary (norm percentile 0.0, all Ethiopic
   script) are all achievable, with hull distances 2.59 to 2.85. If the literature's
   norm-based heuristic is what we had relied on, we would have concluded the
   opposite. Worth asking the deep-research pass about, since it bears on how the
   original result generalizes to a 256k-token tied-embedding vocabulary.

**The caveat that must travel with this result.** It shows the target set is non-empty
in the *unconstrained* sense: there exists some final activation making each token the
argmax. It does **not** show that such an activation is reachable from where the model
actually is, under a bounded perturbation, through the layers we can intervene on.
That is precisely the quantity `ε*_token(l, j) = M_j / ‖J_lᵀ a_j‖` from §I.3, and
measuring it is E3 proper. This check clears a necessary condition, not a sufficient
one.

**One reporting flaw to fix if this is published:** `margin_or_dist` in the CSV mixes
two different quantities. For self-witness rows it holds the top1-minus-top2 logit gap
at `z = unit(W[j])`; for Frank-Wolfe rows it holds the hull distance. Both are
positive and both certify achievability, but they are not comparable to each other and
the column should be split before the number appears in a figure.

---

## Part V. What to say to the PI

Three sentences, in this order.

1. "The certificate was correct and the behavior did not move, and the reason is that
   I certified arrival at a probe's halfspace at layer 20 while the judge measured the
   argmax at layer 26, with five unmodeled operations in between."
2. "Your token-space suggestion collapses that gap: at temperature 0 the target set
   is exactly a polyhedral cone in the unembedding basis, so the same `g/m` formula
   applies with `w = W_U[j] − W_U[k*]` and `t02 = 0`, and the dissociation becomes
   impossible by construction."
3. "Before I spend GPU-hours on that, I am running three cheap controls that could
   invalidate the negative result outright: naive steering at the output layer with no
   Jacobian, an injection-site ablation because I discovered we broadcast the
   perturbation to every position including BOS, and a decoding audit because we had
   `repetition_penalty=1.3` on during a truth experiment."

That third sentence is the one that shows maturity. Lead with the certificate being
correct, and do not bury the two confounds.
