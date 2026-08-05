# The token-space program: what the deep research changed, and what to run

Companion to `docs/NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md` (the brief that generated
the research prompt) and `docs/math_map.tex` (the full derivation, input space to
output space, which is what the PI asked for in writing).

Date: 2026-08-04. Nothing here has run on the cluster yet. Everything marked
**measured** ran locally today on artifacts already on disk.

---

## 0. The one-paragraph version

The deep research pass says our headline finding is not new: the probing-versus-causality
gap is Elazar 2021, LEACE 2023, and AxBench 2025, the last of those on our exact model.
What is ours is the certificate. It also proposed a counter-explanation we had not
considered (anti-steerability cancellation) and ranked it first. We tested that today
against data already on disk, and for `cities` it is **wrong**: the effect is
directionally correct and highly significant, just tiny. Then we computed the number
nobody had computed, the logit margin, and it explains everything: flipping the model
to a false country costs about 3.4 percent of the activation norm, and every truth
direction we have is aligned with the direction that buys it at or below the level a
randomly drawn vector would achieve. We were not pushing weakly on the right thing. We
were pushing on something statistically indistinguishable from noise.

---

## 1. Three things the deep research changed

**1.1 The framing of the contribution.** Stop claiming "decodable is not causal" as
the finding. Elazar et al. (amnesic probing, TACL 2021) said it, LEACE section 5.3
showed a decodability peak at a different layer from the causal peak, and AxBench
(ICML 2025) states on Gemma-2-2B that better classification does not yield better
steering, while also finding difference-in-means is the *best* detector. The claim
becomes: we make that gap quantitative and certifiable. A reviewer who knows Elazar
rejects the other framing on sight.

**1.2 A counter-explanation to test.** Tan et al. (NeurIPS 2024) report that in many
datasets nearly half of examples are anti-steerable, which would flatten an aggregate
while individual statements move both ways. Our headline was an aggregate FALSE rate,
so this had to be answered. Tested below.

**1.3 Two prior-art corrections.** INVPROP is arXiv:2302.01404 (our brief had no ID).
Grivas et al. (arXiv:2203.06462) supersedes Demeter for the unargmaxability test with
an exact algorithm, and reports the effect is rare above d around 200, which is why
our 182/182 achievable result came back clean at d = 2304.

---

## 2. What we measured today, on data already on disk

### 2.1 Signed steerability: the anti-steerability objection fails for `cities`

`src/signed_steer_audit.py`, no GPU. The per-statement arm already swept signed
scales, so this needed nothing new.

| | `cities` | `common_claim` |
|---|---|---|
| statements | 199 | 197 |
| **inert** (byte-identical at every scale) | **103 (51.8%)** | 8 (4.1%) |
| **churn** (text moved, verdict did not) | 68 (34.2%) | 133 (67.5%) |
| **mover** (verdict moved) | 28 (14.1%) | 56 (28.4%) |
| movers with the direction / against | 24 / 4 | 35 / 21 |
| exact two-sided sign test | **p = 0.00018** | p = 0.081 |

Read it as two different failures, which is why they must never be pooled:

- `cities` is **inert**, not cancelling. Half the statements do not move a single
  byte. Among those that move, the effect goes the *right* way 24 times against 4,
  which is significant at p = 1.8e-4. Anti-steerability cannot explain a null when
  nothing moves in either direction. This is the budget problem, and section 2.2
  says exactly how large it is.
- `common_claim` **is** partly the Tan regime: almost everything moves textually,
  and 21 of 56 verdict-movers go the wrong way. Here the aggregate genuinely does
  hide a split, and the distribution must be reported instead of the mean.

This partly overturns the deep research pass's ranking, which put aggregate
cancellation first for both datasets.

### 2.2 The margin: the number that was never computed

`src/token_geom.py`. At temperature 0 the logits are exactly linear in the post-final-norm
activation, so the set of activations emitting token j is a convex cone with 255,999
faces, and the least-norm displacement into it is a solvable QP. n = 200 statements from
`cities`, false-country targets, **all 200 certificates verified against the full
256,000-token vocabulary**:

| quantity | median | reading |
|---|---|---|
| norm of z | 189.2 | scale of the activation |
| logit margin M to the cheapest false country | 13.76 | |
| optimal budget, post-norm site | 6.37 | |
| **optimal budget / norm of z** | **0.034** | **a 3.4% nudge flips the country** |
| cone budget / single-face bound | 1.088 | the cone is 2 to 3 faces, not 1 |
| optimal budget, pre-norm site | 29.3 | |
| **RMSNorm penalty (pre / post)** | **4.4x** | cost of injecting before the norm |

Then the alignments. The comparison that matters is not against 1 but against what a
**random** direction would score, which in d = 2304 is `sqrt(2/(pi d)) = 0.0166`:

| direction | alpha | vs chance | required budget |
|---|---|---|---|
| final-layer mean-diff, fitted at the full-statement position | 0.0040 | **0.24x** | 239x optimal |
| target-layer mean-diff (**the one we steered with**) | 0.0133 | **0.80x** | 79x optimal |
| probe-gradient direction | 0.0135 | **0.81x** | 74x optimal |

**All three sit at or below chance** (Wilcoxon against the chance floor, p < 1e-6 in
every case; the fitted final-layer direction is four times worse than chance). They are
not weakly aligned with the decision that changes the emitted token. They are
uninformative about it.

So the flip is cheap and our actuator points essentially nowhere near it. A behavioral
null at a budget calibrated to a probe halfspace is not evidence about truth; it is
evidence that the budget was denominated in the wrong units. The right unit is the
fraction of the logit margin consumed, and that number has never appeared in any
results table we have produced.

One caveat travels with the two `_asis` rows: those directions were fitted at the
*target layer* and are read here in post-norm coordinates, so their alpha is a
cross-layer carryover. It answers the question we need ("how aligned is the direction
we actually steered with to the thing that moves the next token") but it is not a claim
that the two live in the same basis. The exact per-layer version is E2/E3. The
`md_full` row has no such caveat: it is fitted at the final layer, in the same
coordinates, and it is the worst of the three.

### 2.2b `common_claim` replicates it

n = 200, `--target runnerup` (the cheapest token flip of *any* kind, so these budgets
are a hard lower bound on the cost of a semantic flip). All 200 certificates verified.

| quantity | median |
|---|---|
| logit margin M to the runner-up token | 3.60 |
| optimal budget / norm of z | 0.011 |
| RMSNorm penalty (pre / post) | 5.2x |

| direction | alpha | vs chance | Wilcoxon vs chance |
|---|---|---|---|
| `md_full` | 0.0094 | 0.57x | p = 1.3e-13, below |
| `mean_diff_tgt` | 0.0156 | 0.94x | p = 0.10, **indistinguishable from random** |
| `probe_grad_tgt` | 0.0147 | 0.89x | p = 0.96, **indistinguishable from random** |

Same conclusion, arrived at slightly differently. On `cities` all three directions are
significantly *worse* than chance; on `common_claim` two of them are statistically
indistinguishable from a randomly drawn vector. Either way, none of them carries
information about the decision that changes the emitted token, and the required budget
is 59x to 97x optimal.

### 2.3 Already banked from the previous session

- **Stolen probability: 182/182 achievable**, including all 102 country first-tokens
  and the 16 lowest-norm rows in the vocabulary. Target cones are non-empty, so E3 is
  viable. This clears a necessary condition only, not reachability under a bounded
  perturbation.
- **SAE forensics**: the readout is more SAE-explainable than the actuator (0.50 vs
  0.25 on cities at k = 32), and the full-context versus stem-context Jaccard is 0.049
  against a same-quantity control of 0.684, which is what makes 0.049 interpretable.

---

## 3. Two code defects that must be off before any token-space claim

Both are in the existing runs and both are now flags in the new scripts.

**A. The steering vector was broadcast to every position, including BOS.**
`dct_steer_utils.Steerer` adds a `(d,)` vector to a `(batch, seq, d)` tensor. ActAdd
adds at one aligned position; CAA spreads across tokens. Broadcasting injects
`eps * sqrt(T)` of energy rather than `eps`, and perturbs the attention sink. Nothing
in the literature reports what perturbing gemma-2's sink does, so this is both a
confound and possibly a result. Flag: `--positions all|last`.

**B. `repetition_penalty = 1.3` was active during an 8-token factual completion.**
At a nonzero repetition penalty the decoder is not an argmax, so the cone geometry of
the whole token-space argument does not literally hold for the runs we already have.
There is no defensible reason for it in this experiment. New default: 1.0. Flag:
`--rep-penalty`.

---

## 4. The math, in one box

Full derivation in `docs/math_map.tex`. The compression:

Fix an injection site S, let `A_S` be the Jacobian from that site to the readout
activation, and let the property be `w . z >= t`. Then

```
g = t - w.z          gap
m = ||A_S^T w||      controllability margin
eps* = g / m         certified budget (a point-to-hyperplane distance)
```

Our old certificate: `w` = fitted mean-difference direction, `t` = `t02` from a 1-D
logistic calibration, `A_S` = the layer-to-layer Jacobian. A *readout*, which a
network is free not to consume. That is the structural source of the dissociation.

The token-space certificate: put `j_top` = the current argmax, `j_tgt` = the token we
want, and

```
a    = E[j_tgt] - E[j_top]
M    = -a.z = logit(j_top) - logit(j_tgt) > 0
w := a ,  t := 0  ,  eps*_token = M / ||A_S^T a||
```

**Same formula, same single VJP, same code.** What changes is that `a` is read off the
unembedding rather than fitted, and `a . z >= 0` is not a proxy for the behavior but a
restatement of it. The readout-versus-behavior dissociation becomes impossible by
construction. That is the whole idea from the meeting, and it costs no new machinery.

And for any other unit actuator u at the same site,

```
eps(u) = eps* / alpha ,   alpha = |(A_S^T a).u| / ||A_S^T a||
```

`alpha` is the entire diagnosis. Any steering experiment that fixes a budget without
reporting `alpha` cannot distinguish "not causally actuable" from "we pushed sideways."

---

## 5. The experiment table

| | script | site | cost | prediction | decision gate |
|---|---|---|---|---|---|
| **E0** geometry | `token_geom.py` | none | **done, both datasets** | flip costs a few percent of the norm; alpha at chance | confirmed: alpha 0.004 to 0.014 on `cities` and 0.009 to 0.016 on `common_claim`, against a chance floor of 0.0166. 400/400 cone certificates verified against the full 256k vocabulary |
| **E1a** naive steer, post-norm | `token_steer.py --site postnorm` | after the norm | 1 GPU-hr | oracle flips 100%; mean-diff flips near 0 at the old budget and flips at `eps/alpha` | oracle below 0.9 means the harness is broken, stop |
| **E1b** naive steer, pre-norm | `token_steer.py --site prenorm` | before the norm | 1 GPU-hr | about 4.6x worse than E1a | isolates how much of the loss is RMSNorm |
| **E2** layer sweep | `token_jac.py` | all 26 | 2 GPU-hr | the cheapest layer is not 11 or 13 | if it is, our layer choice was fine and the direction is the whole problem |
| **E3** certified steer | `token_steer.py --site layer:L` | cheapest L | 2 GPU-hr | flip at the certified budget | this is the paper if it works |
| **E4** decoding audit | `--rep-penalty 1.3` | post-norm | folded into E1 | rp=1.3 suppresses flips | quantifies how much the old runs were confounded |
| **E7** injection ablation | `--positions last` | post-norm | folded into E1 | broadcast gain is large | tells us what "budget eps" ever meant |
| **E5** SAE forensics | `sae_decompose.py` | none | done | | overlap 0.05 to 0.09, pre-warned |
| **E6** sensitivity spectrum | `token_sens.py` | any | 2 GPU-hr | gain varies across directions by more than one order of magnitude | if the spread is near 1, the latent space is not chaotic and a scalar budget is a fair currency after all |
| **E8** actuator assertions | `tests/test_token_{geom,steer}.py` | none | done, 31 tests | | permanent; runs in 4 seconds |

Total new GPU: roughly 8 hours against about 475 remaining. The sequencing is E0 (done),
then E1 plus E4 plus E7 in one job, then E2, then E3, with E6 free-riding on whichever
job holds the model.

**Correction to an earlier version of this table.** The row that read "E6
unargmaxability, done, 182/182" conflated two different things. The 182/182 result is
the *stolen-probability* check, which is step 0 of E3: it asks whether the target
tokens are argmax-achievable at all. E6 is the PI's claim 4, the directional
sensitivity spectrum, and until `token_sens.py` it had no implementation. The
stolen-probability check is also now largely subsumed: token_geom exhibits, for each of
400 statements, an explicit activation at which the target token wins a full-vocabulary
argmax, which is a constructive achievability proof for exactly the targets we use.

**The oracle arm is the part we have never had.** At the post-norm site with the
least-norm displacement, the token flip is a theorem: we verified in closed form that
it makes the target the full-vocabulary argmax. If the running model does not emit it,
the injection code is wrong and nothing else in the output means anything. Every
steering run from here carries this assertion.

---

## 6. Run order

```bash
pytest tests/test_token_geom.py tests/test_token_steer.py -q   # E8, 4 seconds, do this first
sbatch deltaai/run_token_geom.slurm          # E0, minutes
sbatch deltaai/run_token_steer.slurm         # E1a, E1b, E4, E7
sbatch deltaai/run_token_jac.slurm           # E2 + E3 Jacobians
sbatch deltaai/run_token_sens.slurm          # E6, independent of the rest
# read "cheapest layer" off the E2 stdout, then:
sbatch --export=ALL,TOKEN_STEER_LAYER=<L> deltaai/run_token_jac.slurm   # E3 steering
```

E8 goes first because it is free and because it is the only thing in the list that can
tell you the harness is broken before you spend GPU-hours on output you would then have
to throw away.

Judging is only needed for the arms that generate text. The next-token argmax is
logged directly and needs no judge, which is a large saving over the previous design:
`hit_target` is a mechanical check against the target token id, not an LLM verdict.

Before submitting, remember gemma's tokenizer left-pads. `attention_mask.sum(1) - 1`
silently reads a pad row; use the flip-argmax form in `reach_hop.last_nonpad_index`.
This bit the new extractor once already.

---

## 7. What to say to the PI

Four sentences.

1. The naive last-layer steer is built and it comes with an oracle arm whose flip is a
   closed-form theorem, so for the first time we can tell a broken harness from a real
   negative result.
2. We computed the logit margin, which is the denominator we were missing: flipping
   the country costs about three percent of the activation norm, and every truth
   direction we have is aligned with the direction that buys it at or *below* what a
   random vector in 2304 dimensions would score, so the budget we spent bought on the
   order of one percent of a flip.
3. Your token-space target set is the same `g/m` certificate with the readout swapped
   for a difference of unembedding rows and the threshold set to zero, which means it
   reuses the existing Jacobian code and makes the readout-versus-behavior gap
   impossible by construction.
4. Two things I had not accounted for: the dissociation itself is already published
   (Elazar, LEACE, AxBench on our exact model), so the contribution has to be the
   certificate rather than the phenomenon, and Tan et al. warn about anti-steerability
   cancellation, which I tested on data we already had and it does not explain
   `cities` (24 of 28 movers go the right way, p = 0.0002) but it does look like part
   of the story for `common_claim`.
