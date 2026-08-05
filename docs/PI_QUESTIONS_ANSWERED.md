# Every Question You Have Asked, Answered

*One document, question by question, from the founding brief through the 2026-07-23 reframe.
Each entry states the question as you put it, gives a one-word verdict, gives the number that
earns the verdict, and names the caveat that travels with it. Where a question is not answered
it says NOT ANSWERED and why, rather than dressing up a partial result.*

**Rule for this document:** every quantity below was read off a committed findings doc or a
cluster artifact on disk. Nothing is from memory. Where two documents disagree, the artifact
wins and the disagreement is flagged.

**Companions.** [`EXPLAINER_FOR_THE_PI.md`](EXPLAINER_FOR_THE_PI.md) is the chronological
walkthrough with the full derivations. [`MEETING_20MIN.md`](MEETING_20MIN.md) is the talk
track. This document is the reference you open when you ask "what happened to the thing I
asked about in June."

---

## 0. The verdict table

### 0.1 The founding questions (the pre-reframe project)

| # | Your question | Verdict | The number |
|---|---|---|---|
| F1 | Is truth linearly encoded in the residual stream? | **YES on clean data** | 0.990 linear probe; XGBoost adds +0.003 |
| F2 | Do the two truth-direction estimates agree, and is that the test? | **YES, and NO** | cos 0.413 at the truth peak; agreement is not the test |
| F3 | Is that readable direction one the model is actually driven by? | **NO** | max cos to 512 causal levers = 1.1 to 1.2x random |
| F4 | Is the null a hand-read artifact? | **NO** | judge at 0.970; lie asymmetry Δ = −0.010, TOST equivalent to 0 |
| F5 | Is the null an artifact of DCT's objective? | **NO** | a second unrelated miner (MAG) lands in the same place |
| F6 | Does seeding the search *at* truth rescue it? | **NO** | 11 panels, both datasets, FALSE pinned at 0.03 to 0.09 |
| F7 | Is there a non-linear escape hatch? | **NO** | XGBoost's +0.057 collapses to −0.01 inside the causal subspace |
| F8 | Which concepts next: toxicity, sycophancy? | **PARTLY** | refusal chosen as the control, built, not yet run |
| F9 | Can we get A-LQR access for behavioral evaluation? | **OPEN** | this is still an ask, see §5 |

### 0.2 The eight notes from 2026-07-23

| # | Your note, compressed | Verdict | The number |
|---|---|---|---|
| N1 | Naively steer on the output; expect the same problem, mostly incoherence | **CONFIRMED** | `md_full` flips 1/101 at degeneracy 0.885; 740 of 1,800 completions empty |
| N2 | If incoherent, the issue is the linear feature in the final layer | **REFUTED** | `oracle` flips 101/101 at the same site, same budget, degeneracy 0.035 |
| N3 | Check the steering vector is what you want it to be | **ANSWERED, with a defect report** | oracle 200/200; realized gain 1.0000; position set was worth 2.5x |
| N4 | Chaotic latent space, so a norm budget is the wrong currency | **HALF RIGHT** | kappa = 1.0000 (not chaotic); gain spread 8.7x on one direction (anisotropic) |
| N5 | Use SAEs to classify the target set | **NOT ANSWERABLE FROM THESE FILES** | the decompositions are at mismatched layers; needs one new run |
| N6 | Temperature 0, deterministic, invert the token mapping | **CONFIRMED. The main result** | 200/200 at the output; 0.340 and 0.675 at layer 16; legacy 0.000 everywhere |
| N7 | Write out all the math from input to output and justify | **DONE** | `math_map.tex` plus `EXPLAINER_FOR_THE_PI.md` §3, 14 subsections |
| N8 | ActAdd sweeps all layers; mean difference is heuristic | **ANSWERED** | controllability peaks L0 to L8; 65 to 87% lost by the layers we used |

---

## 1. The founding questions

### F1. Is truth linearly encoded in gemma-2-2b's activations?

**Yes on clean data, and the answer degrades in a way that is itself the finding.**

A logistic probe on last-token residual activations reads truth at **0.990** on `cities`, and
XGBoost on the same features adds **+0.003**. That is the Marks and Tegmark result reproduced.
On heterogeneous world claims the linear story drops to **0.706** with **+0.082** of non-linear
headroom. The gap grows monotonically across the four datasets: **0.003, 0.028, 0.038, 0.082**,
in exactly the clean-to-messy order we picked in advance.

**The caveat.** Two conclusion-inverting bugs were caught in this chapter before it was
reported: left-padding (the "last token" was a pad token for most rows) and max-gap versus
best-layer gap. Both are documented in [`PROJECT_PROGRESS_TO_DATE.md`](PROJECT_PROGRESS_TO_DATE.md)
§3. The numbers above are post-fix.

### F2. Do the mean-difference and gradient directions agree, and is agreement the test?

**They agree where truth is cleanly defined, and agreement is not the test.**

At the truth-peak layer the two supervised estimates agree at cosine **0.413** on cities,
against **0.161** at a control layer. So the axis is sharpest exactly where we looked, which
matters because it removes the "you probed the wrong layer" escape from F3.

But the non-identifiability literature says different directions can steer equivalently, so a
high cosine is neither necessary nor sufficient for a direction to matter. We flagged that as a
caveat at the first meeting and the reframe is what turned it into a measurement: the right
question is not "do two readouts point the same way" but "does pushing along either one reach a
set the model's behavior lives in." That is §3 N6.

### F3. Is the readable truth direction one of the model's dominant causal levers?

**No. This is the founding null.**

DCT finds the 512 directions the model is most causally sensitive to, unsupervised and
label-free. The supervised truth axis is not one of them and is not in their span:

| Test | cities | common_claim | Chance reference |
|---|---:|---:|---|
| max cos to any of 512 DCT vectors | 0.074 (**1.2x** random) | 0.094 (**1.1x**) | random unit vector |
| fraction inside span(512) | 0.288 (**1.3x**) | 0.316 (**1.4x**) | 512/2304 ≈ 0.22 |

Tested at the layer where the two supervised readouts agree best. **A direction can be read at
99% accuracy and sit essentially orthogonal to the space the model is most sensitive to.**

**What this is not.** Not "DCT is broken" (its vectors do change behavior, strongly) and not
"truth is not represented" (the probe reads it at 0.990). It is a bounded claim about the
*relationship* between readability and causal salience.

### F4. Is that null a hand-read artifact?

**No, and the instrumentation found a bug worth more than the result.**

The Chapter-2 story rested on eyeballed completions. We built an open judge (OLMo-3-7B-Instruct,
validated at **0.970** against gold labels, gate was 0.85) and re-ran both readings as counts.

- **Interpretation:** **0 of 10** top DCT vectors manipulate truthfulness on each dataset,
  0/20 pooled, 95% upper bound about 0.14. The machine count reproduced the hand read exactly.
- **Steering the truth axis:** no lie asymmetry at all. Pooled Δ = **−0.010**, 95% CI
  **[−0.042, +0.023]**, TOST-equivalent to zero on every slice. What steering *does* produce is
  **symmetric degradation**: Cochran-Armitage z = **−5.06**, p < 1e-6 for magnitude, and
  χ² = 4.49, p = 0.34 for direction. Push either way and the model breaks the same amount.

**The bug.** The first cluster pass reported 3/10 vectors manipulating truth. A hand drill-in on
the *zero-steering* completions found the judge was scoring gemma's rambling 24-token tails and
failing correct answers on them. Fixing the measurement moved the unsteered TRUE baseline from
**0.50 to 0.81** and erased all three. That bug was caught before reporting, not after.

**The refined statement:** the truth axis is a **degradation lever, not a truth lever.**

### F5. Is the null an artifact of DCT's objective?

**No. A second miner built on an unrelated principle reaches the same place.**

MAG never steers. It prepends the model's own question ("Is the following statement true?") and
measures the activation shift. Five probes, eight operators. Results:

- Truth is readable from every view, including the question-shift (cities Direct **0.993**,
  InputDelta **0.993**).
- The question-shift is a crisp, real linear direction: `v_Q` explains the shift at cosine
  **0.970** on cities.
- **And it is orthogonal to truth.** The supervised truth axis and DCT's top lever both sit
  inside a `&#124;cos&#124; < 0.1` band against `v_Q`, with `eps_Q` at **1.28 to 1.40**, meaning they
  explain the prefix-induced shift *worse* than predicting no shift at all. That is the figure
  you asked to be redrawn: [`plot_mag_linearity_v2.png`](../plot_mag_linearity_v2.png).
- The sharpest MAG side-finding: **the model never says "false."** Across four datasets the
  answer counts are 0/354, 0/1496, 0/1200 and 1/4450 "no" answers, and the activations at the
  answer position read truth at chance (0.48 to 0.62). The model **contains** truth and does not
  **express** it when asked.

### F6. Does seeding the search at truth itself rescue it?

**No, and this is the strongest form of the founding null.**

We added a soft anchor of strength λ tying DCT's causal search to the supervised truth seed and
swept λ ∈ {0, 0.3, 1, 3}, two seeds, two datasets, 16 warm fits. The control passed first:
cos(warm direction, seed) climbs 0.03, 0.28, 0.72, 0.95 on cities and near-identically on
common_claim, so we really did sweep from free search to essentially the supervised axis.

**In all 11 behavioral panels on both datasets the FALSE fraction stays pinned at 0.03 to 0.09.**
The path runs from **inert** (high λ) to **mild degrader** (low λ) and never passes through
**truth lever**. Anchoring harder toward truth makes the direction *more* inert, which is the
opposite of unlocking causality.

**So the null is not "the supervised axis is inert." It is "no point on the entire geometric
path from the supervised axis to free causal search is a truth lever."**

### F7. Is there a non-linear escape hatch?

**No.** XGBoost's **+0.057** advantage over the linear probe on the full residual stream
collapses to **−0.01** when both are restricted to DCT's causal subspace. The information the
non-linear probe was using is not in the part of the space the model is sensitive to.

### F8. Which concepts should we target next?

**Partly answered, and the answer changed.** You suggested toxicity and sycophancy. We picked
**refusal** instead, for a reason that has nothing to do with interest: refusal is the concept
with the strongest published evidence of being a real, actuatable direction, so it functions as
a **positive control**. Every result in this project is a null, and until an instrument
demonstrates it can detect actuation where actuation exists, "truth is not actuatable" and "our
instrument does not work" are not separated.

**Status:** fully implemented, 976 + 64 rows of prompts prepared, four SLURM jobs written and
gated, **never run on the cluster.** This is the single biggest open item in the project and it
is correctly gated rather than quietly skipped. Toxicity and sycophancy come after refusal, not
before.

### F9. Can we get A-LQR access?

**Still open.** See §5. In the meantime everything here is geometric and behavioral in *our*
harness, not in yours, and the certificate we built is the closest thing to a behavioral test we
could construct without it.

---

## 2. The null question, in full

You have asked some version of "so is the null real" at every meeting, so it is worth putting the
whole thing in one place. The null has had four distinct forms and they did not all survive.

**Form 1: truth is not among the model's causally salient directions.** (F3, F5, F6.)
**STANDS, unchanged.** Two unrelated miners plus an anchored sweep across the whole path between
them. Nothing since has touched this.

**Form 2: steering the truth axis does not make the model lie.** (F4.)
**STANDS as a bounded null.** Δ = −0.010 with a TOST-passing confidence interval is a
measurement that the effect is *near zero*, not a failure to find an effect. The one real effect
is symmetric degradation.

**Form 3: the FALSE halfspace is formally reachable and behaviorally inert.** (The reachability
audit.) **STANDS, and it is the paper.** The certificate is arithmetically exact at its own
linearization point, calibration factor **0.9997** (IQR 0.9910 to 1.0117, n = 199), R² = 0.999
for readout-versus-push. And the lie rate at maximum push is **3.5%** against a **3%** unsteered
baseline, with ε\* comfortably **inside** the trust radius (0.94x). Reachable, certified, inert.

**Form 4: "linear steering at the output layer does not work."** **THIS ONE IS NOW RETIRED, and I
retired it myself.** The token-space work shows that at the budget the old certificate calls a
full trip into the FALSE halfspace, the legacy pullback direction consumes **1.7%** of the logit
margin that actually decides the token at the output layer, and **0.8%** at layer 16. A different
direction, at the same site, same hook, same budget unit, flips **200 of 200**. So the old
experiment was not measuring the weakness of linear steering. It was measuring a direction
pointed almost nowhere useful, and it never had the budget to matter.

**The current one-line statement of the null, which is what I would put in an abstract:**

> The truth probe's FALSE halfspace is a certified backward-reachable target set that is
> behaviorally inert, and the reason is not curvature, not the linearization, and not the
> strength of the push. It is that the probe readout and the token decision are close to
> unrelated: the readout direction's alignment with the deciding margin is **0.00398**, below
> the **0.0166** a random unit vector scores in 2304 dimensions.

That last number is why the null is a mechanism result now instead of a failure report. **We were
not pushing weakly on the right thing. We were pushing on something statistically
indistinguishable from noise.**

**The three controls that make the null a diagnosis and not an excuse:**

1. **The oracle arm.** A direction exists that flips every statement, in the same code path, at
   the same budget scale. So the hook, the budget unit and the decoder are not the problem.
2. **Newton re-steering, a control built to falsify the diagnosis.** If curvature caused the gain
   collapse, recomputing `Jᵀw` at the steered state should fix it. Three iterations on 32
   statements: median `&#124;g&#124;` goes 20.77, 3.30, 0.071, 0.0, ending at **4.65e-05**, with **0 of 32**
   hitting the step cap and the final direction at cosine **0.971** to where we started.
   Curvature is not the obstacle, and behavior is still inert.
3. **The non-linear refit.** On generation prefixes the probe has balanced accuracy exactly
   **0.500**, and refitting with XGBoost recovers **+0.013 / +0.000**. This is an absence of
   signal, not a threshold shift and not a nonlinear boundary.

---

## 3. The eight notes from 2026-07-23

### N1. "Naively steer on the output; if it causes the same problem, mostly making it incoherent." **CONFIRMED**

You predicted a no-Jacobian steer at the output would reproduce the failure and mostly produce
incoherence. It did, literally.

cities, post-norm site, repetition penalty 1.0, at `frac = 1.0` (flips are label-1 rows, n = 101;
the other columns cover all 200):

| direction | flips | median `&#124;frac_margin&#124;` | degenerate | edit ratio |
|---|---:|---:|---:|---:|
| `oracle` | **101 / 101** | 1.0010 | 0.035 | 0.568 |
| `md_full` | 1 / 101 | 1.0877 | **0.885** | 0.862 |
| `jtw_legacy` | 0 / 101 | 0.0174 | 0.000 | 0.057 |

Country outcome against an unsteered baseline of 0.515 correct and 0.010 target: `md_full` at
`frac = 1.0` gives **0.015 correct, 0.985 none, 0.000 target**. At `frac = −2.0` its degenerate
fraction reaches 0.980 and **740 of 1,800 completions in that arm are empty**. The model emits a
newline and stops. "Mostly making it incoherent" turned out to mean the text stops existing.

**And that is exactly the isolation you said this control would provide**, because `oracle` at the
same site, same hook, same budget flips everything at a degeneracy of 0.035. **The incoherence is
a property of the direction, not of steering at the last layer.**

**One arm is uninformative rather than negative, and I want that on the record.** The pre-norm arm
reads 0.000 for every direction including `oracle`. A displacement certified *after* the norm has
to survive the norm, which costs a median penalty of **4.380** on cities and **5.177** on
common_claim, and the sweep stopped at `frac = 2.0`. It was never given the budget. That is a
cheap falsifiable prediction sitting in the queue.

### N2. "If incoherent, the issue is with the linear feature in the final layer." **REFUTED**

The expectation of a refutation was written down in the raw extraction dump *before* the
systematic measurement ran, and the measurement agreed. Recording that the pre-registration came
first.

The decisive comparison needs no new arm. At the post-norm site the map is exactly
`u(z + d) = Ez + Ed` up to a monotone softcap that preserves the argmax. **On cities at
`frac = 1.0`, `md_full` consumes a median 1.0877 of the deciding logit margin, more than
`oracle`'s 1.0010, and flips 1 of 101 against `oracle`'s 101.**

Consuming the margin along the target face is necessary and not sufficient. `md_full` has to
spend roughly `1/alpha` of the optimal norm to buy that margin, and a displacement that large
moves the other 255,999 logits too, so a newline wins the argmax instead of the target country.

If incoherence indicted the linear feature at the final layer, no linear intervention there
should produce a clean targeted flip. One does, on every statement, in the same arm. **The
incoherence indicts the direction, not the layer.** The final layer's linear structure is the one
part of the pipeline that is provably exactly linear.

### N3. "Check that the steering vector is what you want it to be." **ANSWERED, with a defect report**

Three parts, and one of them is a confession.

**Check 1, the oracle assertion.** At the post-norm site the flip is a theorem, solved as a QP and
verified in closed form against the full 256,000-token vocabulary. If the running model does not
emit the target, the injection code is wrong. It emits it on **200/200** on cities and **200/200**
on common_claim, in every post-norm arm.

**Check 2, the arithmetic assertion.** At the post-norm site `A_S = I`, so the realized gain must
be exactly 1 for every direction at every budget. Measured across 13 directions at 4 budgets
spanning `frac` 0.01 to 4.0 on 40 statements: **1.0000 everywhere, both datasets.**

**Check 3, two defects, quantified rather than argued.**

- **`repetition_penalty = 1.3` was on**, in a truth experiment, at 8 tokens. cities post-norm
  `md_full` at `frac = 1.0`: degenerate **0.885** at rp 1.0 against **0.000** at rp 1.3, and empty
  completions **740** against **20** across the arm. But the **flip counts are identical cell by
  cell** between the two arms. The old runs' incoherence readings were inflated by the decoder;
  their flip counts were not.
- **Broadcasting the perturbation to every position, including BOS, was worth about 2.5x.** At
  cities layer 16, `jtw_token` hits **0.340** broadcast against **0.135** at the last position
  only at `frac = 1.0`, and **0.675** against 0.285 at `frac = 2.0`. On common_claim at layer 8
  the same comparison is flat (0.811 against 0.800), and mean stem length explains the difference:
  6.32 words on cities against 9.74 on common_claim.

**The confession, and the general point.** The vector is what we think it is at the post-norm
site, which is the only site carrying an arm that can assert it. There is no oracle assertion at
cities layer 16 or common_claim layer 8, so this is not a general verification at the layer
sites. And what was under-specified was never the vector: it was the **position set** it is added
at, which changes the realized flip rate by 2.5x while the first-order prediction says only 1.355.
**A steering specification is a direction and a position set, and only the first half was ever
written down.** That is a gap in the field's conventions, not only in ours.

### N4. "Small perturbations in different directions have vastly different chains, the latent space is chaotic." **HALF RIGHT, and the halves get opposite answers**

The claim bundles two assertions and they separate cleanly. Both statistics used here have no
Jacobian anywhere in them, so they are an independent check on the framework: realized gain
`G(u, eps) = ||z(eps u) − z(0)|| / eps`, and `kappa(u)`, the ratio of `G` at the largest budget to
`G` at the smallest, which is exactly 1 for a linear site.

| site | spread of median gain across directions | `kappa` |
|---|---:|---|
| cities, post-norm | 1.000 | 1.000 exactly, all directions |
| cities, pre-norm | 1.31 | 0.997 to 1.001 |
| cities, layer 16 broadcast | 9.91 with `jtw`, **1.74 without it** | 0.98 to 1.06, except `jtw` at 0.66 |
| common_claim, layer 16 broadcast | 7.90 with `jtw`, **1.42 without it** | 0.97 to 1.02, except `jtw` at 0.85 |

**The scale-dependence half is absent**, and this is the half that would have destroyed the whole
programme. `kappa` is 1.0000 exactly at the post-norm site on both datasets, within 0.3% of 1
pre-norm, within 6.1% at layer 16 on cities. A first-order certificate is valid across the entire
swept range. If this had come back at 2x, the certificate would have been meaningless and I would
be reporting that instead.

**The anisotropy half is present at depth, and it is structured rather than turbulent.** At layer
16 exactly one direction stands out, and it is the pulled-back token direction: its gain is
**8.7x** the median random direction on cities and 7.7x on common_claim, while everything else,
*including both truth directions*, sits inside a 1.74x band that also contains all eight random
directions. That is signal, not chaos.

**Does alignment actually predict behavior?** This is the test that could have embarrassed the
framework and it had never been run. On **common_claim**: Spearman `rho = 0.316`, p = **5.3e-06**,
n = 200, with quartile hit rates rising monotonically **0.010, 0.075, 0.120, 0.155**. On
**cities** the test is **underpowered, not null**: cities draws targets from only 16 distinct
tokens, so the alignment takes 35 distinct values and the quartile split collapses to three
unequal bins. It reads `rho = 0.129, p = 0.068`, which is evidence for nothing in either
direction.

**The verdict: you were right in substance and wrong in mechanism.** A norm budget is a fair
currency in the sense that the site is linear over the swept range. It is the wrong currency in
the sense that what a fixed norm *buys* depends entirely on alignment. The unit that should have
been reported all along is the fraction of the deciding logit margin consumed, and N6 finally
reports it. The algebra already prices the anisotropy: `eps(u) = eps*/alpha` is exactly the change
of units that repairs it. An anisotropic but linear site needs different **units**, not a
different theory.

### N5. "SAEs are a useful alternative for classifying the target set." **NOT ANSWERABLE FROM THESE FILES**

A null answer, stated rather than papered over. `sae_features_cities.csv` decomposes the truth
readout at **layer 20** and every steering vector at **layer 11** (common_claim: 22 against 13).
GemmaScope trains a **separate dictionary per layer**, so feature 1371 at layer 11 and feature
1371 at layer 20 are unrelated atoms. A feature-id overlap between readout and actuator is
therefore not a quantity these files contain, and no cross-layer overlap number is reported
anywhere in my documents. Answering the note needs one decomposition run with both vector
families at the *same* layer. That is a cluster job, not a re-read.

**What the SAE work does support, and it is worth having anyway.** Method: GemmaScope 16k JumpReLU
SAEs, 32 decoder atoms per direction via orthogonal matching pursuit with a least-squares refit on
the support. We deliberately do **not** run the SAE encoder on these vectors, because
arXiv:2411.08790 shows that is misleading (the encoder is trained on activations, and a steering
direction is not an activation). OMP against the decoder is the correct operation.

- **The readout is legible and the actuator is not.** Cumulative explained variance at 32 atoms:
  the truth readout reaches **0.504**, while every source-side direction we could actually inject
  sits at **0.228 to 0.348**. The thing we want to control is describable in feature terms; the
  thing we have to push on is not.
- **The context shift swaps the feature set outright.** Pairwise Jaccard of the top-32 supports:

| pair | Jaccard | shared features |
|---|---:|---:|
| same quantity, different statement subsets (**the control**) | **0.684** | 26 of 32 |
| full-context versus stem-context (**the real comparison**) | **0.049** | **3 of 32** |
| two random 32-subsets of 16,384 (chance floor) | ~0.001 | |

**The 0.684 control is what makes 0.049 mean anything, and it was not in the original design.**
Measuring the same quantity on a different sample of statements reproduces 68% of the support.
Removing **one word** from the context reproduces 5%, which is 14x below that noise ceiling.
Deleting the final word of the statement nearly completely replaces the set of features the
actuator is built from. That is the mechanism behind the 0.500 balanced accuracy, in a far more
legible form than any cosine. Figures: [`plot_sae_overlap_cities.png`](../plot_sae_overlap_cities.png)
and [`plot_sae_explained_cities.png`](../plot_sae_explained_cities.png).

### N6. "Temperature 0, deterministic, then invert that mapping of the tokens." **CONFIRMED. This is the main result**

**The method.** At temperature 0 the decoder is exactly an argmax, so the set of activations
emitting token `j` is a convex polyhedral cone with 255,999 faces. Take
`a = E[j_tgt] − E[j_top]`, set the threshold to 0, and the gap `g` becomes the logit margin `M`.
Then `eps* = M / ||A_Sᵀ a||` is the **same formula, the same single VJP, the same code** as the
probe certificate. What changes is that `a` is read off the unembedding rather than fitted, so
`a·z >= 0` is not a proxy for the behavior. It is a restatement of it.

**The geometry, which reframes everything.** On cities: median activation norm 189.17, median
logit margin to the cheapest false country 13.76, median least-norm cone displacement 6.369. **A
flip costs a median 3.4% of the activation norm.** On common_claim the same numbers are 168.75,
3.60, 1.738, so **1.1%**. The cone budget sits 8.8% over the single-face bound, meaning 2 to 3
faces bind rather than 1.

**Result 1: the token pullback actuates and the probe pullback does not, at matched budgets, in
the same experiment.**

cities (n = 200; post-norm rows are label 1, n = 101):

| site | direction | `frac` 1.0 | `frac` 2.0 |
|---|---|---:|---:|
| post-norm | `oracle` | **1.000** (101/101) | **1.000** (101/101) |
| layer 16, broadcast | `jtw_token` | **0.340** | **0.675** |
| layer 16, last position only | `jtw_token` | 0.135 | 0.285 |
| every arm, every budget | `jtw_legacy` | **0.000** | **0.000** |

common_claim, layer 8 broadcast: `jtw_token` **0.811** and **0.844**, `jtw_legacy` **0.000**
everywhere. **common_claim's target is the runner-up token**, so generic disruption can land on
target there and its high rates should not be read as a stronger version of the cities result.
cities is the discriminating dataset.

**Result 2: the currency, finally reported.** Median fraction of the deciding logit margin
consumed at `frac = 1.0`:

| dataset, arm | direction | median | n |
|---|---|---:|---:|
| cities, post-norm | `oracle` | 1.0010 | 200 |
| cities, post-norm | `md_full` | 1.0877 | 200 |
| cities, post-norm | `jtw_legacy` | **0.0174** | 200 |
| cities, layer 16 broadcast | `jtw_token` | 1.0735 | 200 |
| cities, layer 16 broadcast | `jtw_legacy` | **0.0082** | 200 |
| common_claim, layer 8 broadcast | `jtw_legacy` | **0.0272** | 90 |

**At the budget the old certificate calls a full trip into the FALSE halfspace, the legacy
direction consumes 1.7% of the deciding margin at the output layer and 0.8% at layer 16.** That
single line is the quantitative replacement for the entire qualitative story about why the old
null was uninformative.

**Result 3: it buys a semantic flip without destroying the text.** At cities layer 16 broadcast,
`jtw_token` moves the fraction of completions naming the *target false country* from **0.010**
unsteered to **0.075** at `frac = 1.0` and **0.120** at `frac = 2.0`, at a degenerate fraction of
0.005 and 0.020. Compare `md_full` at the output layer: target **0.000** at degeneracy 0.885.
**The token direction produces a lie; the truth direction produces silence.**

**Result 4, the most heavily caveated number here, flagged rather than led with.** On cities at
`frac = 1.0` the swept scale is a median 2.42x the reconstructed legacy budget, and by that
reconstruction **88.1%** of statements crossed the probe's own FALSE threshold while **0.000**
flipped the token. Two caveats travel with this line and nothing else. The R² that licenses the
reconstruction (0.9991) licenses **linearity, not gain**, and the realized slope was 8x to 35x
below prediction, so 88.1% is an optimistic upper bound rather than a count. And the legacy budget
is calibrated only at this site. Read it as "the readout very probably crossed on most statements
and the token never moved," not as a crossing count.

**The honest limit on all four results, said out loud rather than waited for.** The certificate is
a **first-token** claim and the completion recovers. On cities `oracle` flips the first token on
**200 of 200** statements, and yet only **0.085** of those same completions name the target
country while **0.700** still name the correct one. Reaching the cone at position `t` does not
keep the model there at `t+1`. Extending the target set over a horizon is the natural next
problem, and it is exactly what the BRT-style recursion in your citation is for.

### N7. "Write out all the math from input to output and justify it." **DONE**

[`math_map.tex`](math_map.tex) carries the derivations with proofs, the cone dual and the
non-emptiness argument. [`EXPLAINER_FOR_THE_PI.md`](EXPLAINER_FOR_THE_PI.md) §3 is the same
material in 14 prose subsections: the forward map as numbered equations, the convention warning
(folding the RMSNorm gain into `z` versus into `E` gives identical logits and **different
distances**, and every budget is a distance), the three facts that make the output layer
tractable, the certificate derived in three steps, and the VJP cost argument.

Two objects that had lived only as prose are now written down as operators: the **broadcast
operator** `B δ = 1_T ⊗ δ`, whose adjoint is exactly the position-sum already in the code, and
the `_asis` convention, which is the honest name for the fact that our cross-layer alignments are
read with **no transport at all**. The basis-correct transported form is written down and is
**not yet computed**. That is the one number outstanding on this note, and it costs one JVP per
direction per dataset.

### N8. "ActAdd sweeps all layers; over some layers the perturbation does nothing; mean difference is heuristic." **ANSWERED**

Both halves confirmed, and the layer half is the more actionable.

**Controllability is front-loaded.** `||J_lᵀ w||` computed for every layer 0 to 25 peaks at
**layers 0 to 8** and has lost **65 to 87%** of its magnitude by the layers we had been using. The
best window is **L5 to L9**, which is **1.5x to 3x** better conditioned than layers 11 and 13. Our
layer choice came from DCT's own configuration and had never been swept. Figure:
[`plot_reach_jlens_cities.png`](../plot_reach_jlens_cities.png).

**And a second layer fact that matters more:** the behavioral verdict readout has **2x to 7x less
margin** than the truth probe at every layer. The probe is easy to move and the behavior is not,
everywhere, not just where we looked.

**"Mean difference is heuristic" is confirmed the hard way.** `md_full`'s alignment with the token
decision is **0.00398**, against a chance floor of **0.0166** for a random unit vector in 2304
dimensions. The heuristic direction is not merely suboptimal; on this measurement it is below what
a random draw scores.

---

## 4. Questions still open, honestly labelled

| Question | Status | What it would cost |
|---|---|---|
| Does the instrument detect actuation on **any** concept? | **Built, never run.** The refusal positive control is the publication gate | about 4 GPU-hr, four gated SLURM jobs |
| Do the SAE features of readout and actuator overlap at a **matched** layer? | **Not answerable from current files** (N5) | one decomposition run |
| Is the pre-norm null real or just an under-budgeted sweep? | **Uninformative arm**, prediction registered | one sweep extended past `frac = 5` |
| What is the **transported** cross-layer alignment? | Formula written, not computed (N7) | one JVP per direction per dataset |
| Does the flip survive past the first token? | **No, and we measured it.** 200/200 first-token, 0.085 completion | a horizon formulation, which is real new work |
| Does any of this hold beyond gemma-2-2b? | **Untested.** Single model, two hops, one probe family | gemma-2-9b invalidates every cached artifact |
| Is arXiv:1910.13272 the reference you meant? | **Unconfirmed.** It is feedback linearization, not reachability | one sentence from you |

---

## 5. What I need from you

1. **The refusal control: run it or replace it.** It is the gate on every claim in this document.
   If you think a different positive control is stronger, I would rather switch now than spend the
   GPU hours twice.
2. **A-LQR access.** Everything here is our own harness. A behavioral evaluation in yours is the
   check I cannot run myself, and the token-space result is the first thing in this project worth
   handing over.
3. **Confirm arXiv:1910.13272.** The paper at that ID is feedback linearization, not reachability.
   If you meant a set-propagation survey I will read the right one.

---

## 6. Where each answer lives

| Question block | Primary source | Figures |
|---|---|---|
| F1, F2 | `MEETING_SUMMARY.md`, `PROJECT_PROGRESS_TO_DATE.md` §3 | `results/plots/plot_summary_maxgap.png` |
| F3, F7 | `DCT_VS_TRUTH_FINDINGS.md`, `FUNNEL_RESULTS.md` | `plot_funnel_*.png` |
| F4 | `INVESTIGATION_steering_validity.md` §6 | `plot_judge_steering_*.png` |
| F5 | `DCT_VS_MAG_ON_TRUTH.md`, `MAG_VS_DCT_CONCEPTUAL.md` | `plot_mag_linearity_v2.png` |
| F6 | `WARM_DCT_RESULTS.md` | `plot_dct_warm_*.png` |
| The null, forms 1 to 3 | `REACH_AUDIT_FINDINGS.md` | `plot_reach_audit_dissociation.png` |
| N1 to N8 | `TOKEN_SPACE_FINDINGS.md` | `plot_token_*.png`, `plot_sae_*.png` |
| The math (N7) | `math_map.tex`, `EXPLAINER_FOR_THE_PI.md` §3 | |
| Cluster provenance | `RESULTS_SINCE_LAST_MEETING_PART3.md` §8 | |
