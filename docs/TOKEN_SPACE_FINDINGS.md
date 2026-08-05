# Token space: the PI's eight claims, answered

*Every claim from the last meeting, given a verdict against the finished token-space
runs. Written to be read cold. Every number below was read off
[`TOKEN_SPACE_RAW_FINDINGS.md`](TOKEN_SPACE_RAW_FINDINGS.md) (the Pass 1 extraction
dump) or off a named artifact on disk. Nothing is from memory, and where a figure in an
earlier document disagrees with the artifact, the artifact wins and the correction is
stated in place.*

**Scope of the record:** 2 datasets, 200 statements each, 12 steering arms, 26 layers
swept, 1 model (`google/gemma-2-2b`, fp32), temperature 0. No judge ran in this program:
`hit_target` is a mechanical argmax check against the target token id, and the coherence
readings are string-level. Date: 2026-08-05.

---

## 0. The one-paragraph version

The meeting's central proposal was to stop defining the target set with a fitted probe
and start defining it in token space, where membership *is* the behavior. That proposal
is now tested and it works: at the output layer the closed-form cone displacement flips
the emitted token on every statement it was scored on, 200 of 200 on cities and 200 of
200 on common_claim (101 of 101 and 44 of 44 inside the label-1 cross-tab that §3 and §5
report), and pulled back through the Jacobian to layer 16 it flips the token on
34.0% of cities statements at the certified budget and 67.5% at twice it, while the old
probe-halfspace direction flips nothing at all, a hit rate of 0.000 in every arm, at
every budget, on both datasets. The reason the old direction fails is now a number
rather than a story: on cities, at the budget the old certificate
calls a full trip into the FALSE halfspace, it consumes a median **1.7%** of the logit
margin that actually decides the token. The PI's prediction that a naive output-layer
steer would reproduce the failure and mostly produce incoherence held exactly, and the
inference he flagged as the next step ("then the issue is the linear feature in the
final layer") is refuted by the same arm: at the output layer the map from perturbation
to logits is *exactly* linear up to a monotone final-logit softcap that preserves the
argmax and therefore the cone, and a linear intervention there does flip the token
cleanly when it is the right one. What fails is alignment, not linearity. Of the three
claims that were never measured, two are now measured and the third, the SAE claim, is
not answerable from the files we have; and the request to write the algebra out end to
end is now met in one tracked document, with a single quantity defined there but not yet
computed, the basis-correct transported form of the cross-layer alpha.

---

## 1. The eight claims, and the verdicts

The claims are the table at
[`NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md` §II.1](NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md).

| # | The PI's note, compressed | Experiment | Verdict | Section |
|---|---|---|---|---|
| 1 | Naively steer on the output; if it causes the same problem, mostly making it incoherent | E1a, post-norm steer | **CONFIRMED** | §3 |
| 2 | If incoherent, the issue is the linear feature in the final layer | E1a, oracle vs `md_full` in one arm | **REFUTED** | §4 |
| 3 | Check the steering vector is what you want it to be | Oracle assertion, E4 decoding audit, E7 position ablation | **ANSWERED**: verified at the post-norm site only, where an oracle arm exists. There is no oracle assertion at layer 16 or layer 8, and the position set turned out to be the under-specified half | §5 |
| 4 | Small perturbations in different directions have vastly different chains (chaotic latent space) | E6, `token_sens` | **ANSWERED**, and the two halves get opposite answers: the anisotropy half holds (9.91x spread across directions at cities layer 16), the chaos half does not (`kappa` within 6.1% of 1 across a 400x budget range on cities) | §6 |
| 5 | SAE is a useful alternative for classifying the target set | E5, `sae_decompose` | **NOT ANSWERABLE FROM THESE FILES** | §7 |
| 6 | Temperature 0, deterministic, invert the token mapping | E0 plus E1 plus E3 | **CONFIRMED** | §8 |
| 7 | Write out all the math from input to output and justify | Documentation | **ANSWERED**: every object is named and justified in one tracked place, [`RESULTS_SINCE_LAST_MEETING_PART3.md`](RESULTS_SINCE_LAST_MEETING_PART3.md) §2.1 to §2.7, with the proofs in [`math_map.tex`](math_map.tex). One quantity is defined there and not computed: the transported cross-layer `alpha` | §9 |
| 8 | ActAdd sweeps all layers; some layers do nothing; mean difference is heuristic | E2, `token_jac`, plus the layer arms | **ANSWERED** | §10 |

---

## 2. What was run, and what each number is

**The sites.** A perturbation is injected at one of three places. *Post-norm* is the
activation `z` after the final RMSNorm, where the pre-softcap logits are exactly
`u = E z`, so the map from perturbation to logits is linear with no approximation at all
up to gemma-2's `final_logit_softcapping = 30.0`. That softcap is strictly increasing, so
it preserves the argmax and therefore the cone, and the margin the certificate spends is
the pre-softcap one: the code computes margins against the uncapped unembedding rows and
takes the argmax from the model's own capped head precisely because the two agree
(`src/token_geom.py:58`, `src/token_steer.py:153`). *Pre-norm* is
the residual stream before that norm. *Layer L* is the input of decoder block L, reached
by one vector-Jacobian product. cities was steered at layer 16, common_claim at layer 8.

**The directions.**

| name | what it is |
|---|---|
| `oracle` | the least-norm displacement into the target token's argmax cone, solved as a QP and verified in closed form against the full 256,000-token vocabulary. Its flip is a theorem, so it is the harness assertion, not a competitor |
| `md_full` | the final-layer contrastive mean-difference truth direction |
| `jtw_legacy` | the old probe-halfspace pullback, `J^T w` with `w` the target-layer truth direction |
| `jtw_token` | the token-space pullback, `J^T a` with `a = E[j_tgt] - E[j_top]` |

**The budget unit.** `frac` is the fraction of the certified budget spent, so `frac=1.0`
is by construction the displacement predicted to consume exactly one logit margin along
the target face. `frac_margin` is the fraction actually consumed, measured. A second
column in the CSVs, `readout_delta` (renamed `tgt_minus_top_delta` at the source), is the
same measurement rescaled per statement by the unperturbed margin, so it is never
reported beside `frac_margin`: printing both would show one number twice. The duplicate
check in `token_conclusions.proportional_per_statement` reads **True on all twelve arms**,
which confirms the CSVs were not regenerated and that A1 and A2 stand as written.

**What is measured and what is reconstructed.** Every flip count, every hit rate, every
`frac_margin`, every coherence figure and every country outcome below is a direct
measurement. The `crossed` column of the A1 cross-tab is **not**. `token_steer` logs no
truth readout, so whether the legacy readout crossed its own threshold is reconstructed
as `|scale| >= eps*_legacy`, and **two caveats travel with every number derived from
it**:

1. The per-statement `R^2` that licenses the reconstruction, **0.9991 on cities and
   0.990 on common_claim**, licenses **linearity** of readout against scale, **not
   gain**. The same measurement
   ([`REACH_AUDIT_FINDINGS.md:79`](REACH_AUDIT_FINDINGS.md), the D1 result) found the
   realized slope **8x to 35x below** the predicted `||J^T w||`. So `crossed` is an
   optimistic upper bound on how often the readout truly crossed, not a count of
   crossings.
2. `eps*_legacy` is calibrated at the post-norm site. Applied unchanged to the pre-norm
   and layer arms it is nominal, not calibrated: median `|frac_margin|` for the same
   direction falls from 0.0174 post-norm to 0.0029 pre-norm, a factor of 6.0. Read
   `crossed` as calibrated **only for the post-norm rows**.

**Two rules that govern every table below.** First, cities and common_claim are never
pooled. They fail by different mechanisms and their targets are different: cities uses a
semantic false-country target, common_claim uses the runner-up token, so on common_claim
generic disruption can land on target. **cities is the discriminating dataset and leads
every comparison.** Second, wherever a common_claim cross-direction number appears,
`jtw_legacy` covers **90 of 200 statements, restricted to 44 at label 1**, because
`reach_margins` capped common_claim at 2,000 of its 4,450 rows. Cross-direction rates on
that dataset are computed on the 90 shared statements.

**Figures.** Five per dataset, already generated by `src/viz_token.py`:
`plot_token_steer_<ds>.png` (the result), `plot_token_budget_<ds>.png` (cost of a flip),
`plot_token_alpha_<ds>.png` (alignment against the chance floor),
`plot_token_layers_<ds>.png` (the layer sweep), `plot_signed_steer_<ds>.png` (the
per-statement split).

---

## 3. Claim 1: naive output-layer steering reproduces the failure, and mostly makes it incoherent

**CONFIRMED.**

**Method.** E1a. Skip the Jacobian entirely and inject at the post-norm site, where
nothing nonlinear survives to be blamed. Sweep nine signed budgets per direction on 200
statements per dataset. A1's cross-tab is restricted to label 1 (n = 101 cities, n = 44
common_claim); the coherence and country tables are the full 200.

**cities, post-norm, repetition penalty 1.0, at `frac = 1.0`:**

| direction | flips (label 1, n=101) | median &#124;`frac_margin`&#124; (n=200) | degenerate | edit ratio |
|---|---:|---:|---:|---:|
| `oracle` | **101 / 101** | 1.0010 | 0.035 | 0.568 |
| `md_full` | 1 / 101 | 1.0877 | **0.885** | 0.862 |
| `jtw_legacy` | 0 / 101 | 0.0174 | 0.000 | 0.057 |

Country outcome on the same arm (unsteered baseline: 0.515 correct, 0.010 target):
`md_full` at `frac = 1.0` gives **0.015 correct, 0.985 none, 0.000 target**. At
`frac = -2.0` its degenerate fraction reaches 0.980, and **740 of its 1,800 completions
in this arm are empty**: the model emits a newline and stops.

**Reading.** The PI's conditional held exactly. Steering naively on the output along the
truth direction reproduces the original failure and mostly produces incoherence, in the
literal sense that the text stops existing. The same site, same hook and same budget unit
with the `oracle` direction flips every statement at a degeneracy of 0.035, so the
incoherence is a property of the direction and not of steering at the last layer. That is
precisely the isolation the claim asserted this control would provide.

**The pre-norm arm is a null for an arithmetic reason, not a behavioral one.** On the
shared statements that govern every cross-direction rate in this document (§2), the arm is
a clean null: every direction including `oracle` has hit rate **0.000** at `frac` 1.0 and
2.0 on cities (n = 200) and **0.000** on common_claim (n = 90). The one non-zero cell
anywhere in the arm is `md_full` at 0.005 on common_claim, and that is an *unrestricted*
per-direction rate over all 200 statements: the single hit lies outside the 90 shared
statements, so it does not enter any cross-direction comparison. A displacement certified after the norm must
survive the norm, which costs a median `rmsnorm_penalty` of **4.380** on cities and
**5.177** on common_claim, while the sweep stops at `frac = 2.0`. The pre-norm arm was
never given enough budget to be informative. §13 turns this into a cheap falsifiable test.

---

## 4. Claim 2: incoherence does not indict the linear feature at the final layer

**REFUTED**, and the pre-registration matched.

**The pre-registered expectation, quoted from the dump before the measurement:**

> PRE-REGISTERED EXPECTATION: claim 2 will be REFUTED. Recorded before the systematic
> measurement. If the measurement disagrees, it is reported as-is.

The measurement agreed. It is recorded here that the expectation was set first.

**Method.** The decisive comparison needs no new arm. It is three directions inside a
single arm, at a single budget, at the one site where the linearity question has an
exact answer: `u(z + d) = E z + E d`, up to the monotone final-logit softcap, which
preserves the argmax and therefore the cone.

**The sharpest fact in the dump.** On cities at `frac = 1.0`, `md_full` consumes a median
**1.0877** of the logit margin, *more* than `oracle`'s **1.0010**, and flips **1** of 101
statements against `oracle`'s **101**. Consuming the margin along the target face is
necessary and not sufficient. `md_full` has to spend roughly `1/alpha` of the optimal
norm to buy that margin, and a displacement that large moves the rest of the 256,000
logits too, so a newline wins the argmax instead of the target country.

**Reading.** If incoherence indicted "the linear feature in the final layer", then no
linear intervention at that layer should produce a clean targeted flip. One does, on
every statement, in the same arm, at the same budget scale. So the incoherence indicts
the *direction*: `md_full`'s median alignment with the token decision is
`alpha = 0.00398` on cities, below the **0.0166** a random unit direction in d = 2304
would score. The final layer's linear structure is not the problem; it is the one part of
the pipeline that is provably exactly linear, up to a monotone softcap that leaves the
argmax, and therefore the cone, untouched.

---

## 5. Claim 3: the injected vector is verified at the output site only, and the position set is half its specification

**ANSWERED.** The vector is verified where an oracle arm exists, which is the post-norm
site and only the post-norm site. There is no oracle assertion at cities layer 16 or at
common_claim layer 8 (§12), so this is not a general verification of the injected vector
at the layer sites, and the second half of the section is a defect report rather than a
confirmation.

**Check 1, the oracle assertion.** At the post-norm site with the cone QP solution, the
flip is a theorem verified in closed form against the full vocabulary. If the running
model does not emit the target, the injection code is wrong. It emits it on every
statement of both datasets: **200/200** on cities and **200/200** on common_claim at
`frac = 1.0`, in every post-norm arm including the repetition-penalty 1.3 arm. Restricted
to the label-1 subsets that A1's cross-tab uses, the same rows read 101/101 and 44/44.

**Check 2, the arithmetic assertion.** At the post-norm site `A_S = I`, so the realized
gain must be exactly 1 for every direction at every budget. Measured across 13 directions
(5 candidates plus 8 random) at 4 budgets spanning `frac` 0.01 to 4.0, on 40 statements:
**1.0000 everywhere, both datasets** (read from `token_sens_<ds>_postnorm_all.csv`).

**Check 3, the two known defects, now quantified rather than argued.**

*Repetition penalty 1.3 changed what we saw downstream and not the token we were
measuring.* cities post-norm `md_full` at `frac = 1.0`: degenerate **0.885** at rp 1.0
versus **0.000** at rp 1.3, and empty completions **740** versus **20** across the arm.
But the A1 flip counts are identical cell by cell between the two arms, and the `oracle`
country outcome barely moves (target 0.085 both, correct 0.700 versus 0.685). The old
runs' incoherence readings were inflated by the decoder; their flip counts were not.

*Broadcasting to every position was worth about 2.5x on cities.* At layer 16, `jtw_token`
hits **0.340** broadcast against **0.135** at the last position only at `frac = 1.0`, and
0.675 against 0.285 at `frac = 2.0`. On common_claim at layer 8 the same comparison is
flat: 0.811 against 0.800 at `frac = 1.0` (90 shared statements; `jtw_legacy` covers 90 of
200 there, 44 at label 1). Mean stem length is 6.32 words on cities and 9.74 on
common_claim.

**Reading.** The vector is what we think it is at the post-norm site, which is the only
site carrying an arm that can assert it. What was
under-specified was never the vector: it was the position set it is added at, which
changes the realized flip rate by a factor of 2.5 on cities while the first-order
`broadcast_gain` predicts only 1.355 (§10). A steering specification is a direction *and*
a position set, and only the first half was ever written down.

---

## 6. Claim 4: the site is anisotropic, not chaotic, and the currency problem is real for a different reason

**ANSWERED.**

**Method.** E6, `token_sens`, the two statistics `math_map.tex` defines: the realized gain
`G_S(u, eps) = ||z(eps u) - z(0)|| / eps` measured with no Jacobian anywhere in it, and
`kappa_S(u)`, the ratio of `G` at the largest budget to `G` at the smallest, which is
exactly 1 for a linear site. 40 statements, 4 budgets, 13 or 14 directions per site.
These are read from `token_sens_<ds>_*.csv` directly; the dump carries only the pre-norm
medians, through A8.

| site | spread of median gain across directions | `kappa` |
|---|---:|---|
| cities, post-norm | 1.000 | 1.000 exactly, all directions |
| cities, pre-norm | 1.31 | 0.997 to 1.001 |
| cities, layer 16 broadcast | 9.91 with `jtw`, **1.74 without it** | 0.98 to 1.06, except `jtw` at 0.66 |
| common_claim, post-norm | 1.000 | 1.000 exactly, all directions |
| common_claim, pre-norm | 1.25 | 0.985 to 1.000 |
| common_claim, layer 16 broadcast | 7.90 with `jtw`, **1.42 without it** | 0.97 to 1.02, except `jtw` at 0.85 |

**Reading, in two parts, because the claim bundles two different assertions.**

*Scale dependence, the part that would break the program, is absent.* Stated per dataset,
because the two do not share a tolerance: on cities `kappa` is 1.0000 exactly at the
post-norm site, within **0.3%** of 1 pre-norm, and within **6.1%** at layer 16; on
common_claim it is 1.0000 exactly post-norm, within **1.5%** pre-norm, and within **3.3%**
at layer 16. Those layer-16 figures exclude the one direction that does move, the
pulled-back token direction, whose gain falls 34% (cities) and 15% (common_claim) from the
smallest budget to the largest. A first-order certificate is valid across the whole range
we sweep.

*Anisotropy is present at depth, and it is structured rather than chaotic.* At layer 16
exactly one direction stands out, and it is the pulled-back token direction: its gain is
**8.7x** the median random direction on cities and **7.7x** on common_claim, while
everything else, including both truth directions, sits inside a 1.74x band that also
contains all eight random directions. That is signal, not turbulence, and
`math_map.tex` already prices it: `eps(u) = eps*/alpha` is exactly the change of units
that repairs anisotropy. An
anisotropic but linear site needs different units, not a different theory.

**Does `alpha` actually predict behavior?** A4 tests it against realized hit rate, which
had never been checked. On **common_claim**: Spearman `rho = 0.316`, `p = 5.3e-06`,
n = 200, with quartile hit rates rising monotonically **0.010, 0.075, 0.120, 0.155**
across median alphas 0.0025 to 0.0212. On **cities** the test is **underpowered, not
null**: `alpha_md_full` takes only 35 distinct values across the 200 statements, because
cities draws its targets from only 16 distinct target tokens, so `qcut` collapses to
three unequal bins of 81, 84 and 35 with hit rates 0.0000, 0.0000 and 0.0143. cities reads
`rho = 0.129, p = 0.068`, and that number is evidence for nothing in either direction:
it neither shows that alpha predicts behavior nor that it fails to.

**The verdict, stated plainly.** The objection is right in substance and wrong in
mechanism. A norm budget is a fair currency in the sense that matters, because the site is
linear over the swept range. It is the wrong currency in the sense that what a fixed norm
buys depends entirely on alignment, and the unit that should have been reported all along
is the fraction of the logit margin consumed. §8 reports it.

---

## 7. Claim 5: not answerable from these files

**NOT ANSWERABLE FROM THESE FILES.**

**Why.** `sae_features_cities.csv` decomposes the truth readout `w_mean_diff_tgt` at
**layer 20** and every steering vector (`jtw_mean`, `jtw_stem_mean`,
`jtw_full_matched_mean`, `common_v1`, `V64_common_0..3`) at **layer 11**. For
common_claim the same pairing is **layer 22** against **layer 13**. GemmaScope trains a
separate dictionary per layer, so feature id 1371 at layer 11 and feature id 1371 at layer
20 are unrelated atoms. A feature-id overlap between the truth readout and the steering
vectors is therefore not a quantity these files contain, and **no cross-layer overlap
number is reported here for either dataset**. The pre-warned expectation of an overlap
near 0.05 is not testable on this evidence. Answering claim 5 needs one decomposition
run with both vector families at the same layer, which is a cluster job, not a re-read.

**What the files do support**, within a layer, from 32-feature orthogonal-matching-pursuit
decompositions:

| dataset | readout, cumulative explained by top 10 | actuator (`jtw_mean`), same |
|---|---:|---:|
| cities | 0.348 at L20 | 0.149 at L11 |
| common_claim | 0.487 at L22 | 0.248 at L13 |

The readout is roughly twice as SAE-legible as the actuator on both datasets, which is
the asymmetry the earlier 32-feature figures reported, now reproduced at k = 10 and on the
second dataset.

Within layer 11 on cities, the two Jacobian pullbacks computed at different context
lengths share almost nothing: top-10 Jaccard(`jtw_mean`, `jtw_stem_mean`) = **0.048**,
against **0.833** for `jtw_mean` versus `jtw_full_matched_mean`. On common_claim at layer
13 the same pair reads 0.100 against 0.833. Note this is a *top-10* Jaccard; the 0.049 in
the project record is a *top-32* Jaccard, so the two agree in magnitude but are not the
same statistic and should not be quoted as one number.

---

## 8. Claim 6: the token-space target set is well posed, cheap, and actuates where the probe halfspace does not

**CONFIRMED.** This is the main result.

**Method.** At temperature 0 the decoder is exactly an argmax, so the set of activations
emitting token `j` is a convex polyhedral cone with 255,999 faces. Take
`a = E[j_tgt] - E[j_top]`, put `t = 0`, and the gap `g` becomes the logit margin `M`.
Then `eps*_token = M / ||A_S^T a||` is the same formula, the same single VJP, and the same
code as the probe certificate. What changes is that `a` is read off the unembedding rather
than fitted, so `a . z >= 0` is not a proxy for the behavior but a restatement of it.

**Result 1: the pullback actuates and the probe pullback does not, at matched budgets, in
the same experiment.**

cities (n = 200; A1 rows are label 1, n = 101; layer rows are hit rates on the shared
statements):

| site | direction | `frac` 1.0 | `frac` 2.0 |
|---|---|---:|---:|
| post-norm | `oracle` | **1.000** (101/101) | **1.000** (101/101) |
| layer 16, broadcast | `jtw_token` | **0.340** | **0.675** |
| layer 16, last position | `jtw_token` | 0.135 | 0.285 |
| every arm it ran in, every budget | `jtw_legacy` | **0.000** | **0.000** |

common_claim (n = 200 sampled; `jtw_legacy` covers 90 of 200, 44 at label 1, so the layer
rates are on those 90):

| site | direction | `frac` 1.0 | `frac` 2.0 |
|---|---|---:|---:|
| post-norm | `oracle` | **1.000** (44/44) | **1.000** (44/44) |
| layer 8, broadcast | `jtw_token` | 0.811 | 0.844 |
| layer 8, last position | `jtw_token` | 0.800 | 0.878 |
| every arm it ran in, every budget | `jtw_legacy` | **0.000** | **0.000** |

common_claim's target is the runner-up token, so a generic disruption can land on target
there and its high rates should not be read as a stronger version of the cities result.
cities, whose target is a semantically false country, is the discriminating dataset.

**Result 2: the currency, finally reported.** Median `|frac_margin|`, the fraction of the
logit margin actually consumed, restricted to shared statements:

| dataset, arm | direction | median at `frac` 1.0 | p90 | n |
|---|---|---:|---:|---:|
| cities, post-norm | `oracle` | 1.0010 | 1.0010 | 200 |
| cities, post-norm | `md_full` | 1.0877 | 1.2045 | 200 |
| cities, post-norm | `jtw_legacy` | **0.0174** | 0.0345 | 200 |
| cities, layer 16 broadcast | `jtw_token` | 1.0735 | 1.4301 | 200 |
| cities, layer 16 broadcast | `jtw_legacy` | **0.0082** | 0.0221 | 200 |
| common_claim, post-norm | `jtw_legacy` | **0.0174** | 0.0441 | 90 |
| common_claim, layer 8 broadcast | `jtw_token` | 1.9693 | 2.5908 | 90 |
| common_claim, layer 8 broadcast | `jtw_legacy` | **0.0272** | 0.1046 | 90 |

On cities, at the budget the old certificate calls a full trip into the FALSE halfspace,
the legacy direction consumes **1.7%** of the deciding margin at the output layer and
**0.8%** at layer 16. That single line is the quantitative replacement for the whole
qualitative story about why the old null was uninformative.

**Result 3: it buys a semantic flip without destroying the text.** At cities layer 16
broadcast, `jtw_token` moves the fraction of completions naming the *target* false country
from **0.010** unsteered to **0.075** at `frac = 1.0` and **0.120** at `frac = 2.0`, at a
degenerate fraction of 0.005 and 0.020 respectively. Compare `md_full` at the output
layer: target **0.000** at degeneracy 0.885.

**Result 4, the most heavily caveated number in this document: the old readout says it
crossed, and the token never moves.** On cities at `frac = 1.0`, restricted to label 1
(n = 101), the swept scale is a median **2.42x** the reconstructed `eps*_legacy` (median
**2.554**), and by that reconstruction **88.1%** of statements crossed the probe's own
FALSE threshold while **0.000** flipped the token. On common_claim (n = 44, the
restricted set) the same construction gives a median ratio of 0.172 against an
`eps*_legacy` median of **9.430**, so 6.8% crossed, and again 0.000 flipped. Both caveats
from §2 apply to the crossing figures and to nothing else on the line: the `R^2` licenses
linearity and not gain, and the realized slope was 8x to 35x below prediction, so 88.1%
is an optimistic upper bound rather than a count; and `eps*_legacy` is calibrated at this
site, which is the only reason the number is quotable at all. Read the cities cell as
"the readout very probably crossed on most statements and the token never moved", not as
a crossing count.

**The honest limit on all four results.** The certificate is a **first-token** claim and
the completion recovers. On cities `oracle` flips the first token on **200 of 200**
statements, and yet only **0.085** of those same 200 completions name the target country
while **0.700** still name the correct one. Both fractions are over all 200; quoting the
label-1 subset (101 of 101) beside them would halve the denominator mid-sentence.
Reaching the cone at position `t` does not keep the model there at
position `t+1`. Extending the target set over a horizon is the natural next problem, and
it is exactly what the BRT-Align style recursion in the PI's citation is for.

**Verdict.** The inversion the meeting asked for is well posed, computable with machinery
already written, verified against the full vocabulary, and it actuates. Claim 6 is
confirmed.

---

## 9. Claim 7: every object from input to output is named, in the document the PI reads

**ANSWERED.**

**Method.** Documentation, not measurement. The ask was for one place where every
linear-algebra object between token input and token output is written down and justified
rather than described in words. That place is
[`RESULTS_SINCE_LAST_MEETING_PART3.md`](RESULTS_SINCE_LAST_MEETING_PART3.md) §2, "The
reachability formalism: every object named", §2.1 through §2.7, which is the section of
the results document the PI actually reads. The LaTeX source carrying the proofs, the cone
dual and the non-emptiness argument is [`math_map.tex`](math_map.tex), referenced from
§2's opening paragraph. **Both files are now tracked in git**, so both are part of the
shared record rather than two working files on one machine.

**What §2 names, subsection by subsection.**

| where | objects named and justified |
|---|---|
| §2.1 | the forward map: the tied embedding and unembedding `E` (`V x d`), the residual stream `h^(l)_t`, the final RMSNorm gain `gamma`, the post-norm activation `z`, the logits `u = E z`, the softcap `c = 30.0`, and which two of these steps are linear and why the rest do not break the algebra |
| §2.2 | the two readouts as one algebra with a different `w`: the probe halfspace (`w`, `t02`) and the token cone (`a_i = E[j_tgt] - E[j_top]`, `r_i(z) = a_i . z`, the 255,999 faces, `delta_cone` against `delta_face`) |
| §2.3 | the certificate stated once, with the sign convention fixed once: `g = t - w.z`, `m = ||A_S^T w||`, `eps* = g/m`, the minimiser `d* = (g/m^2) A_S^T w`, and the one-VJP cost argument that makes a per-statement certificate affordable |
| §2.4 | the three injection sites with their `A_S`, their adjoints, and their measured `m` medians: post-norm `I` (2.383 cities, 2.138 common_claim), pre-norm `(sqrt(d)/||h||) diag(1+gamma) P_perp` (0.532, 0.432), layer `l` one VJP (0.607 at L16, 0.676 at L8), plus the hop map `F` and the local-linearity qualification |
| §2.5 | the **broadcast operator**, previously prose and now written: `B delta = 1_T (x) delta`, the map actually inverted is `J . B`, its adjoint sums the position gradients, `(J . B)^T w = sum_t (dr / dh^(l)_t)`, which is literally `g.sum(dim=1)` at `src/token_jac.py:82` and `src/reach_jlens.py:97`, priced at `broadcast_gain` **1.355** (cities L16) and **2.016** (common_claim L8) |
| §2.6 | `alpha(u) = |(A_S^T w) . u| / ||A_S^T w||` with `eps_required(u) = eps*/alpha(u)`, and the **cross-layer `_asis` convention**, previously a footnote: those directions are fitted at the target layer and read in post-norm coordinates with no transport at all (`src/token_geom.py:395-400`) |
| §2.7 | what "reachable" would mean, as four conditions rather than one number |

`math_map.tex` gained the same two operators as `\subsection` `sec:broadcast` and
`sec:asis`, so neither document is now incomplete by omission of the other.

**The residual, and it is a missing measurement rather than a missing object.** §2.6
writes down the basis-correct version of the cross-layer cosine: push the target-layer
direction forward through the remaining blocks and the final RMSNorm, then take the cosine
against `a_i`. That transported `alpha` **has not been computed**, so every `_asis` number
in this document, `alpha_mean_diff_tgt_asis = 0.01325` and
`alpha_probe_grad_tgt_asis = 0.01355` on cities (§10 Result B), is a cosine against the raw
vector that was actually injected, and must keep travelling with the convention paragraph
that says so. That is the operational question and the one the program needs, but it is
not a coordinate-free statement about the concept.
[`TOKEN_SPACE_PROGRAM.md`](TOKEN_SPACE_PROGRAM.md) §2.2 raised this first, as a caveat;
§2.6 is now the place it is stated in full.

**Reading.** The claim asked for naming and justification, and that is done and shared.
Computing the transported `alpha` is a separate, cheap job (one forward pass per direction
per dataset) and it belongs to claim 8's alignment question, not to this one. A quantity
that is defined but unmeasured is not the same failure as an object that was never
written down.

---

## 10. Claim 8: the layer sweep ran, and depth is not the binding constraint

**ANSWERED.**

**Result A: control authority is front-loaded, and 11 and 13 were not the cheapest
layers.** E2 swept all 26 layers, 200 statements each, both injection conventions
(read from `token_jac_<ds>.csv`; the dump carries only the swept-layer rows, through A5).
Median certified budget `eps_all`, the norm needed at that layer to flip the target token
under the broadcast convention:

| dataset | layer 0 | layer 8 | layer 11 | layer 13 | layer 16 | layer 25 |
|---|---:|---:|---:|---:|---:|---:|
| cities | **8.73** | 21.44 | 23.11 | 23.21 | 23.00 | 67.24 |
| common_claim | **2.92** | 4.60 | 5.10 | 5.34 | 6.60 | 19.93 |

The cost rises with depth over the whole sweep, with only small local wiggles. Each
dataset's own source layer is well past the peak: layer 11 costs **2.6x** layer 0 on
cities, and layer 13 costs **1.8x** layer 0 on common_claim. The
layer arms that were actually run, 16 on cities and 8 on common_claim, are also not the
sweep's cheapest layer. This reproduces the audit's D4 finding, controllability is
front-loaded and we steered past the peak, in token space with an independent readout.

**Result B: "mean difference is heuristic" is confirmed, quantitatively.** Median
alignment with the token decision, against a chance floor of
`sqrt(2/(pi d)) = 0.0166` for a random unit direction in d = 2304:

| direction | cities | common_claim |
|---|---:|---:|
| `alpha_md_full` | **0.00398** | 0.0094 |
| `alpha_mean_diff_tgt_asis` | 0.01325 | 0.0156 |
| `alpha_probe_grad_tgt_asis` | 0.01355 | 0.01471 |

All six sit at or below chance. The behavioral consequence is the 0.000 hit rate for
`jtw_legacy` in every arm at every budget, and 1 flip in 101 for `md_full` on cities.

**Result C: a correction to the project record.** `broadcast_gain` at cities layer 16 is
**1.355**, read from `token_jac_cities.csv`. Earlier project notes carry a different,
larger figure for this quantity; that figure is wrong and should not be quoted again. At
common_claim layer 8 the artifact reads **2.016**, which also corrects the value carried
in the implementation plan. The corresponding `m_all / m_last` ratios are 1.380 and
2.143.

**Reading.** Sweeping layers is the right methodological instinct, and it does not rescue
this direction. The mean-difference direction fails at the output layer, before the norm,
and at the swept layer alike, at every budget in the sweep. Depth is not the binding
constraint; alignment is. Depth *does* matter for the direction that works: `jtw_token`
was steered only at one layer per dataset, and choosing a cheaper one is the obvious free
improvement (§13).

---

## 11. The quantified ledger, wins stated as numbers

1. The token-space target set actuates at the output layer: **101/101** flips on cities
   and **44/44** on common_claim at label 1, at `frac = 1.0`, verified in closed form
   against all 256,000 vocabulary entries.
2. Pulled back through the Jacobian to layer 16, it actuates at depth: **0.340** hit rate
   at the certified budget on cities and **0.675** at twice it, n = 200; common_claim at
   layer 8 reads 0.811 and 0.844 on its 90 shared statements.
3. The probe-halfspace direction actuates **nowhere**: hit rate **0.000** in every one of
   the 8 arms it was run in, at every budget, on both datasets.
4. The reason, in the right unit, on cities: at `frac = 1.0` the legacy direction consumes
   a median **1.7%** of the logit margin at the output layer (n = 200) and **0.8%** at
   layer 16, against `oracle`'s 1.0010 and `jtw_token`'s 1.0735 on the same dataset.
5. Consuming the margin is necessary and not sufficient, on cities at the post-norm site:
   `md_full` consumes **1.0877** of it and flips **1 of 101** at label 1, while `oracle`
   consumes **1.0010** and flips **101 of 101** at label 1, 200 of 200 over the full set.
6. Every candidate truth direction is at or below the chance alignment floor of **0.0166**
   in d = 2304: 0.00398 to 0.01355 on cities, 0.0094 to 0.0156 on common_claim, n = 200
   each.
7. `alpha` predicts realized behavior where the test has power: Spearman **0.316**,
   p = 5.3e-06 on common_claim (n = 200), quartile hit rates 0.010 to 0.155.
8. The first-order certificate is valid over the whole swept budget range, per dataset:
   `kappa` is 1.0000 exactly at the post-norm site on both, and within **0.3%** of 1
   pre-norm and **6.1%** at layer 16 on cities, within **1.5%** pre-norm and **3.3%** at
   layer 16 on common_claim, for every direction but one (40 statements, 4 budgets, 13
   directions at the post-norm and pre-norm sites, 14 at layer 16).
9. The harness is verified twice at the output layer: the oracle flip is a theorem and it
   fires, and the realized gain is exactly **1.0000** for all 13 directions at all 4
   budgets as arithmetic requires.
10. The decoding confound is bounded: repetition penalty 1.3 changes cities `md_full`
    degeneracy from 0.885 to 0.000 and empty completions from 740 to 20, and changes the
    A1 flip counts **not at all**.
11. The broadcast confound is measured: **2.5x** realized hit rate at cities layer 16
    (0.340 against 0.135), against a first-order `broadcast_gain` of 1.355.
12. The layer sweep prices our layer choice: cities' layer 11 costs **2.6x** its layer 0
    and common_claim's layer 13 costs **1.8x** its layer 0, in certified budget, over all
    26 layers, n = 200 per layer.
13. The pre-norm null is arithmetic: `rmsnorm_penalty` medians **4.380** and **5.177**
    against a sweep that stops at `frac = 2.0`.
14. The failure modes are separated per dataset (A7): cities is **inert** (103 of 199
    statements byte-identical at every scale; of 28 movers, 24 with the direction and 4
    against, p = 1.8e-4), common_claim is partly the Tan anti-steerability regime (8 inert
    of 197, 56 movers, 21 of them against, p = 0.081).
15. The geometry, restated: a flip costs a median **3.4%** of the activation norm on
    cities (`delta_cone` 6.369 against `z_norm` 189.17, margin 13.76) and **1.1%** on
    common_claim (1.738 against 168.75, margin 3.60).

---

## 12. What is not established, the honest column

- **Everything here is first-token.** The certificate makes no claim about the rest of the
  completion, and the completions demonstrably recover: on cities `oracle` flips the first
  token on 200 of 200 statements while only 0.085 of those 200 completions name the target
  country and 0.700 still name the correct one. All three figures are over the same 200.
- **n = 200 per dataset**, subsampled from 1,496 and 4,450. One model
  (`google/gemma-2-2b`, fp32), two datasets, both English, both short declarative
  statements.
- **`jtw_legacy` covers 90 of 200 statements on common_claim, 44 at label 1**, because
  `reach_margins` capped that dataset at 2,000 of 4,450 rows. Every common_claim
  cross-direction number is on that restricted set.
- **A1's readout axis is reconstructed, not measured**, and carries both caveats in §2:
  the R^2 licenses linearity and not gain (realized slope was 8x to 35x below prediction),
  and `eps*_legacy` is calibrated only at the post-norm site.
- **Temperature 0 only.** Nothing here transfers to sampling without re-derivation.
- **The `_asis` alphas are cross-layer carryovers**, fitted at the target layer and read
  in post-norm coordinates. They answer the question we need and are not a claim that the
  two live in the same basis.
- **common_claim's target is the runner-up token**, so generic disruption can land on
  target there. cities is the discriminating dataset and no conclusion rests on
  common_claim alone.
- **No oracle assertion exists at the layer sites.** The layer arms carry only
  `jtw_legacy` and `jtw_token`, so the harness is verified at the post-norm site only and
  the layer-site hit rates have no in-arm ceiling to be read against.
- **E6 is 40 statements per site at 4 budgets**, and on common_claim the sensitivity file
  is at layer 16 while its steering arm is at layer 8, so those two are not matched on
  that dataset.
- **Claim 5 is unanswered, not answered negatively.** It needs one decomposition run with
  both vector families at a single layer.
- **No judge ran in this program.** Nothing here is a truth verdict; `hit_target` is an
  argmax check and the coherence and country readings are string-level.
- **The refusal positive control still has not run**, so "the instrument works and truth
  is not actuatable" and "the instrument does not work" remain unseparated for the
  *probe-halfspace* certificate. Note that the token-space certificate does not need that
  gate in the same way: its oracle arm is a positive control by construction, and it
  passes.

---

## 13. Where it goes next

**First, the cheapest falsifiable thing on the list: sweep the pre-norm arm to `frac` 6.**
The pre-norm null is currently explained as arithmetic. A displacement certified after the
norm must survive it, at a median cost of **4.380** on cities and **5.177** on
common_claim, while the sweep stopped at `frac = 2.0`, and every pre-norm hit rate on the
shared statements is 0.000 (the one unrestricted exception, `md_full` at 0.005 on
common_claim, is §3). Candidate gains at that site cluster
on the random baseline (cities: candidates 0.229 to 0.300, eight random directions 0.2438
to 0.2477, `1/penalty` 0.2283), which says pre-norm gain is a property of the norm layer
and not of the direction. **Prediction, stated here and not run: sweeping the pre-norm
oracle arm to `frac = 6` should recover its hit rate to near 1.0 on both datasets** (about
4.4 is needed on cities, about 5.2 on common_claim). If it does not, the explanation is
wrong and the RMSNorm Jacobian in the certificate needs re-deriving. One job, no new code.

**Second, steer where the budget is cheap.** The sweep says layer 0 is 2.6x cheaper than
layer 11 on cities and 1.8x cheaper than layer 13 on common_claim, and the arms that ran
are at 16 and 8. Re-run the `jtw_token` arm in the layer 0 to 8 window, which is also the
window D4 identified as best-conditioned for the behavioral verdict readout. This is the
maximum-contrast configuration available without new machinery.

**Third, add an oracle arm at the layer sites.** Every layer-site number currently has no
in-arm ceiling. The cone QP already produces the displacement; pulling it back through the
same VJP costs nothing extra and turns 0.340 from a bare number into a fraction of what
was provably available.

**Fourth, the horizon problem.** The certificate reaches the cone at one position and the
model leaves it at the next. That is the honest boundary of the current result and it is
the point where the PI's reachability citations stop being an analogy: a target set
defined over a horizon, with a value recursion or a receding-horizon controller that
re-plans each token, is the structurally correct object. It is also the natural handoff
into A-LQR.

**Fifth, claim 5, and the one measurement claim 7 leaves behind.** Claim 5 needs one SAE
decomposition with the readout and the steering vectors at the same layer, which is a
cluster job and not a re-read. Claim 7's object list is written and tracked (§9), and what
it leaves open is a number rather than a deliverable: the transported cross-layer `alpha`,
which pushes each `_asis` direction forward through the remaining blocks and the final
RMSNorm before taking the cosine against `a_i`. Until it exists, the `_asis` alphas in §10
Result B are cosines against the raw injected vector and carry §2's convention caveat.
Neither of these is a research risk.
