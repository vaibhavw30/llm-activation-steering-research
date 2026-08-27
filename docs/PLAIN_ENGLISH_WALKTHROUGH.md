# The project in plain English: eight experiments, in the order they happened

*Written to be read cold by someone who has not seen the code, and to be explained out loud
from. Each step exists because the step before it produced a result that could not be
interpreted without it. Every number is read off a named artifact or a findings doc; the source
documents are listed at the bottom.*

**Scope:** `google/gemma-2-2b` (and `gemma-2-2b-it` for the refusal control), 2 core datasets,
26 layers, June to August 2026, DeltaAI GH200.

Web version of this same document:
<https://claude.ai/code/artifact/4ecb92cb-47ea-4bd8-805a-07dcfbe654f7>

---

## 0. The whole thing in one paragraph

A linear probe can read "is this statement true?" off the model's internal activations at about
99% accuracy. We then asked the obvious next question: if we can *read* truth from a direction
in activation space, can we *push* the model along that direction and make it lie? The answer is
no, and we spent four months establishing that the "no" is real rather than a bug in our own
measurement. Pushing the model moves the probe's readout exactly as predicted, in the right
direction, with R-squared = 0.999, and the model goes right on telling the truth. We then proved
the pushing machinery itself works by using it to induce refusal in a chat model, where it moves
behavior decisively. So the null is a fact about **truth**, not about our instrument.

---

## 1. The one distinction everything hangs on

If Julian takes away one thing, make it this: **a readout and a behavior are different objects,
and moving one does not move the other.**

A *probe* is a small classifier trained on the model's internal activations. Show it the
activation vector from the last token of "The city of Busan is in South Korea" and it says TRUE.
It is right 99% of the time on clean data. That accuracy is what most of the interpretability
literature reports, and it is genuinely impressive.

But the probe is an observer we bolted on. It is not the thing the model uses to pick its next
word. Our whole project is the discovery that you can grab the model's activations, shove them
until the probe flips from TRUE to FALSE, and the model will then emit a completely true
sentence anyway. The probe's opinion changed. The model's behavior did not.

> The analogy that lands: you have found the thermometer, not the thermostat. You can hold a
> lighter under the thermometer and watch the number go up. The room stays cold.

---

## 2. The chain

### Step 01 -- Is truth encoded linearly?
*June 2026, CPU only*

**What we did.** Ran about 7,500 true/false statements through gemma-2-2b, saved the activation
vector at the last token of every statement at all 27 layers, and at each layer trained two
classifiers: logistic regression (linear) and XGBoost (non-linear). If truth is a straight line
in activation space the two should tie. If it is a curved surface, XGBoost should win.

**What came out.**

| Dataset | Character | Linear | XGBoost | Gap |
|---|---|---:|---:|---:|
| cities | clean, curated | 0.990 | 0.993 | +0.003 |
| sp_en_trans | translations | 0.972 | 1.000 | +0.028 |
| companies_true_false | messier | 0.917 | 0.954 | +0.038 |
| common_claim_true_false | messiest | 0.706 | 0.788 | +0.082 |

Clean truth is linear, replicating Marks and Tegmark. Messy claims have real non-linear
headroom, and the gap grows monotonically with how messy the dataset is.

**The finding nobody expects.** We also computed two different "truth directions" the standard
ways: the contrastive mean difference (average true activation minus average false one) and the
logistic regression's own gradient. On cities they agree at cosine **0.413**. On common_claim,
**0.085**. These are supposed to be two estimates of the same vector. They are correlated but
genuinely different directions, on data where the probe is at 99%.

> **Which forced the next question:** if two "truth directions" that both read truth perfectly
> point in different directions, then reading accuracy cannot tell you which one is *the* truth
> direction. You need a behavioral test.

---

### Step 02 -- Is the truth direction causally special?
*July 2026*

**What we did.** DCT (Deep Causal Transcoding) is an unsupervised method that searches for the
directions which, when you push along them, cause the *largest* downstream change in the model.
It never sees labels. If truth is an important causal variable in this model, the supervised
truth direction should show up somewhere in DCT's list of most causally salient directions.

**What came out.** Verdict: **null, twice over.** It does not show up. And when we checked
whether the *non-linear* truth structure (the +0.082 XGBoost headroom) might be hiding inside
DCT's causal subspace instead, that was absent too, no better than projecting onto random
directions of the same dimension.

**What it means.** "Linearly readable" and "causally important" are separate properties, and
truth has the first without the second. This is the first appearance of the readout-versus-
behavior gap, though at this stage we could not tell whether it was a fact about the model or a
limitation of DCT.

> **Which forced the next question:** at the 23 July meeting the PI redirected the whole project
> away from "which vector is truth" and toward a control-theory question that can be answered
> with a certificate rather than a correlation.

---

### Step 03 -- Reframing it as backward reachability
*23 July 2026, the pivot*

**The new question.** Standing at layer 11 with some activation, and knowing that the truth probe
sitting at layer 20 carves activation space into a TRUE half and a FALSE half: **what is the
smallest nudge I can apply at layer 11 that lands me in the FALSE half at layer 20?**

That is backward reachability, straight out of control theory. It is a genuinely better question
than "which vector is truth" because it comes with a number attached.

**Why this is an upgrade.** The answer is a **certificate**: a per-statement budget we call
`eps*`, the exact size of nudge the linear model says is required. Layers 11 and 20 are connected
by nine transformer blocks of dense non-linear computation, so we linearize that stretch with its
Jacobian and pull the target set backwards through it. That gives a falsifiable prediction rather
than a similarity score: push by `eps*`, and you should land in the FALSE set.

> **Which forced the next step:** so we built it, certified the budget, and then did the thing
> the literature mostly skips, which is actually generate text and read it.

---

### Step 04 -- The reachability audit: certified, and inert
*24 July 2026, the central result*

**What we did.** Five phases on two datasets (cities 11 -> 20, common_claim 13 -> 22). Certify
the budget (P1), check whether the Jacobian's dominant directions have anything to do with
concepts (P2), *steer and generate* (P3), profile the probe by layer (P4), and check whether the
certified budget is even inside the region where linearizing is legitimate (P5).

**What came out.**

| quantity | value |
|---|---|
| R-squared of readout against push size, per statement | 0.999 (cities), 0.990 (common_claim) |
| statements that moved in the wrong direction | 0% in both datasets |
| lie rate at maximum push | 3.5% cities, 5.5% common_claim (baseline about 3%) |
| realized gain against predicted | 8x to 35x short |

The control works beautifully as control. Push harder, the readout moves further, almost
perfectly linearly, always in the direction we asked for. And the model keeps telling the truth:
at the strongest push it still completes with "The city of Busan is in South Korea". Fluent, on
topic, and correct.

**Two different failure modes, and only one of them is interesting.**

- **common_claim is a broken certificate.** Its `eps*` of 10.69 is 2.2x the trust radius of 4.87,
  meaning the linear approximation was being evaluated far outside where it is valid. The
  certificate was extrapolation, and the readout never actually crossed. This one is our own
  methodological error.
- **cities is the real finding.** Its `eps*` of 2.80 sits *inside* its trust radius of 2.99, the
  readout genuinely does cross into the FALSE half, and behavior still does not move. That cannot
  be blamed on linearization error.

**Why, mechanically.** P2 had already warned us: the top-16 singular subspace of the pullback
overlaps DCT's concept subspace at only **0.062** (cities) and **0.047** (common_claim), against
a stop-and-diagnose threshold of 0.3, and the Jacobian is nearly isotropic at effective rank
about 500 of 2304. The high-gain channel through those layers is *concept-blind*. We were pushing
hard through a pipe that carries no semantics.

**Why this is publishable.** A literature search found **no published work reporting a
certified-reachable-but-behaviorally-inert dissociation with a mechanism attached.** The nearest
neighbours are an existence proof with no certificate machinery (arXiv:2604.09839), a
correlational version done by prompting (arXiv:2507.22149), and a base-rate reliability study
with no mechanism (arXiv:2407.12404). This is the contribution.

> **Which forced the next step:** the PI's response was the sharpest intervention in the project.
> Stop defining the target set with a fitted probe. Define it in *token space*, where being in
> the set *is* the behavior, so the readout-behavior gap cannot exist by construction.

---

### Step 05 -- Token space: the machinery is fine, the target was wrong
*4 to 5 August 2026*

**What we did.** Instead of "get into the halfspace where a fitted probe says FALSE", the target
becomes "make the model's argmax next token be *this specific token*". At the very last layer the
map from activation to logits is exactly linear (a matrix multiply by the unembedding), so the
minimum displacement into a target token's cone can be solved in closed form and verified against
all 256,000 vocabulary entries. Then pull that target back through the Jacobian to an earlier
layer, exactly as before.

**What came out.**

| Direction | Where | Flip rate |
|---|---|---|
| closed-form token target (`oracle`) | output layer | 200 / 200, both datasets |
| token target pulled back (`jtw_token`) | cities layer 16 | 34.0% at budget, 67.5% at 2x |
| old probe-halfspace direction (`jtw_legacy`) | every layer, every budget | 0.000 |

The same certificate machinery, the same Jacobian pullback, the same model. Change what you aim
at and it goes from flipping nothing to flipping a third of statements at budget.

**The number that explains four months of nulls.** We measured how well each candidate truth
direction aligns with the direction that actually decides which token gets emitted. For a random
unit vector in 2304 dimensions the chance alignment floor is `sqrt(2/(pi*d))` = **0.0166**. Every
truth direction we have sits at or below it, and the final-layer mean-difference direction scores
**0.00398**, four times *worse* than random.

Put in budget terms: at the push our old certificate calls a complete trip into the FALSE
halfspace, it consumes a median **1.7%** of the logit margin that decides the token. Meanwhile an
actual token flip costs only **3.4%** of the activation's norm on cities. The flip was never
expensive. We were spending the money in the wrong direction.

**What it means.** The PI predicted that a naive output-layer steer would reproduce the failure
and mostly produce gibberish, and that was **confirmed**. He then flagged that if so, the problem
must be the linear feature in the final layer, and that was **refuted** by the same experiment:
at the output layer the map is exactly linear (up to a monotone softcap that preserves the
argmax), and a linear intervention flips the token cleanly when it is the right one. *What fails
is alignment, not linearity.*

> **Which forced the next step:** at this point we had a strong negative result about truth.
> Before spending more GPU time we turned the instruments on our own conclusions and tried to
> break them.

---

### Step 06 -- Auditing our own negative results
*26 August 2026, four experiments, about a minute of compute each*

**What we did.** Wrote down four assumptions our own conclusions silently depended on, registered
a prediction for each before running anything, and tested them on already-committed artifacts.
Nothing was regenerated, no GPU was spent.

**All four assumptions were wrong.**

| id | Assumption | Verdict | Headline |
|---|---|---|---|
| S1 | flips are directional, not just magnitude | refuted as a blanket claim | of 9 directions: 3 directional, 3 symmetric, 3 inert |
| S2 | truth is one linear object at every depth | refuted | 19 layers read truth at 0.973 or better; their pairwise cosine has median 0.365, minimum 0.014 |
| S3 | our two steering arms are comparable | refuted | disjoint by 6.09x on cities; the band where behavior moves (0.29 to 0.78) was never sampled |
| S4 | hitting the certified target means something | refuted, decisively | 200/200 flips succeed; at most 5.5% made the claim false |

**S2 and S4 are the two to explain.**

*S2 says there is no good layer to steer from.* Truth is readable at 0.99 from layer 8 through
layer 26, but those layers read it along directions that rotate almost to orthogonality. Worse,
controllability peaks at layer 0, where the readout is exactly chance, and falls 7.35x by layer
25, where the readout is at ceiling. The layer where you can best *see* truth is the layer where
you can least *touch* it.

*S4 says our target was semantically empty.* All 200 certified argmax flips succeed. But 85% of
the cities targets are the tokens " North" and " South", so flipping "South Korea" to "North
Korea" is a token flip that leaves the sentence's truth value alone. The vacuity rate is 0.855 to
0.945. At most **5.5%** of our perfect successes actually made the claim false.

**The part to be proud of.** The audit corrected three claims in its *own design document*, and
two of the corrections made the method we were arguing against look *better* than we had said.
That is the direction of correction that is hard to fake and worth mentioning out loud.

**What it means.** This is the most useful result in the project, because it reframes the whole
thing. The control machinery is sound and the objective was wrong. That is a fixable problem, not
a dead end.

> **Which forced the next step:** one loophole survived. Maybe behavior never moved because we
> always pushed too hard and only ever saw breakdown. That needed a dose-response curve.

---

### Step 07 -- The dose-response kill test
*27 August 2026, pre-registered*

**What we did.** Swept 12 magnitudes from 0.01 to 1.00 on a single common axis, for six
directions including a norm-matched random control, and asked whether any direction beats the
random control at the same dose. Sections 1 to 3 of the write-up, including the exact prediction,
were committed to git *before any data existed*.

**What came out.** Verdict: **null on both registered branches.** Zero clean windows out of 120
cells, on both datasets. And the sharpest result is uncomfortable: `sup_grad`, the direction whose
flip curve this whole audit was built to explain, induces *fewer* new false statements than a
random vector of the same size (odds ratio 0.76 on cities and 0.53 on common_claim).

What false statements do track is dose, with Spearman correlation 0.986 (cities) and 0.981
(common_claim). High-dose falsity reads as the model breaking down, not as directed lying.

**A measurement artifact we caught.** The "flip" metric we had been reporting turned out to be
measuring whether the output was parseable at all, at 100% incidence. Worth flagging to Julian as
an example of why the pre-registration and the audit were worth the time.

> **Which forced the last step:** which leaves the one interpretation that would sink everything.
> What if our pipeline simply cannot move *any* behavior, and every null is just a broken harness?

---

### Step 08 -- The refusal positive control
*27 August 2026, the publication gate*

**What we did.** Took the identical pipeline (same certificate machinery, same layers-apart hop,
same steering code) and pointed it at a behavior we know a chat model definitely has: refusal.
Using `gemma-2-2b-it`, layers 5 to 14, we took *harmless* instructions and steered them across
the certified boundary toward the harmful side. If the pipeline works, the model should start
refusing to answer harmless questions.

**What came out.** Verdict: **actuatable.**

| quantity | value |
|---|---|
| statements that flipped into refusal, versus out | 14 against 0 (McNemar p = 1.2e-04) |
| sign control, same statements, opposite direction | 19/64 against 1/64, OR 26.60, p = 9.66e-06 |
| Cochran-Armitage trend | z = -6.86, p = 6.73e-12 |
| Mantel-Haenszel OR for *crossing*, push size held fixed | 24.2 |

Every comparison is within the same 64 statements measured at every dose, so nothing is
confounded by which statements happened to be observed where. The sign control is the key design:
push the same statements the same distance in the *opposite* direction and refusal drops to 1 in
64. A pipeline that merely perturbs activations cannot produce that asymmetry.

The Mantel-Haenszel figure is the one to quote at Julian. It holds the size of the push fixed and
asks whether *crossing the certified boundary* still predicts refusal. It does, at odds 24 to 1.
That is the certificate's boundary doing work, not just a big shove.

The completions confirm it is not breakdown: at the deciding dose the model turns a working HTML
page, a correct factual answer about IBM's CEO, and a complete product pitch into refusals, and
`empty_completion_rows = 0`.

**The judge is validated.** The substring refusal judge was scored against prompt-derived gold
labels on 64 balanced rows: accuracy **0.969**, which is exactly the ceiling the proxy allows,
with a positive rate of 0.500 matching the gold base rate. The OLMo reference judge scored 0.734
with recall 1.000 and a positive rate of 0.766, the textbook always-positive failure, so the
earlier low kappa was prevalence mismatch rather than a problem with our judge.

**What it means.** The instrument can actuate. Therefore the truth nulls are a fact about truth,
not about the harness. This was pre-committed as the publication gate before the job ran, and it
opened.

**The honest caveats, which you should volunteer.**

- The arm that worked is **in-sample**: its 200 statements come from the set the direction was
  fitted on. The held-out arm never crossed the boundary at all, because a safety clamp capped it
  at 1.61x the budget. So this establishes that the instrument can actuate, which is what a
  positive control is for. It is *not* a generalization claim.
- The 64 statements that reached full dose are the ones that started closest to the boundary
  (median `eps*` 13.18 against 27.58 for the rest), so the 27-point effect would be smaller on a
  random draw.
- This is `gemma-2-2b-it`, not the base model where truth was found inert, so it shows the method
  can actuate rather than that it can actuate in that exact model.
- Crossing took 2x `eps*`, not the 1x the certificate promises. Treat `eps*` as a lower bound by
  roughly a factor of two.
- Layer 5 was not chosen by the data. It is the `MIN_SOURCE_LAYER` floor and it happened to work.

---

## 3. The PI's last meeting, item by item

At the last meeting the PI raised **ten items**. Nine of them reduce to one challenge, and it is
worth saying out loud that it was the right challenge: *the null may be an artifact of how we
steered, not a fact about the model.* The audit of 26 to 27 August exists because of that
challenge, and the honest summary is that **the PI was right on the central point and anticipated
a second problem we had not measured.**

The formal mapping lives in section 13 of
[`docs/superpowers/specs/2026-08-26-steering-validity-audit-design.md`](superpowers/specs/2026-08-26-steering-validity-audit-design.md).
This is that table with the outcomes filled in.

| # | The PI's item | Where it was answered | Outcome |
|---|---|---|---|
| 1 | Final layer is sensitive, try smaller magnitudes | D1 | **Done, null.** Not oversteering |
| 2 | Super fine grained sweep in case of oversteering | D1 | **Done, null.** 12 magnitudes, both signs, zero clean windows of 120 cells |
| 3 | Compare naive and reachability on matched parameters | S3, then D1 | **Done. The PI was right, they were never comparable** |
| 4 | The halfspace target set is not constructed properly | S4 | **Done, confirmed decisively.** The strongest result of the audit |
| 5 | Sweep input layers, same mapping from earlier layers | S2 | **Done, and it refuted our assumption** |
| 6 | Check the same linear relationship holds at each layer | S2, `token_jac` | **Done.** The relationship holds; the gain does not |
| 7 | Pick a temperature and commit to it | settled | **Done.** T = 0 everywhere, deterministic, stated once |
| 8 | Token space, the cone, vocabulary geometry | token-space program | **Done, and it worked** |
| 9 | Map relevant words back into activation space | T1 | **Open.** Now the highest-value remaining item |
| 10 | Backward reachability, system-level synthesis | deferred | **Open by the PI's own call** ("might be overkill"), revisit now |

### The seven that closed, with the number that closed them

**1 and 2, oversteering.** This was the loophole that could have voided everything. D1 swept
`rel` = 0.01, 0.02, 0.04, 0.08, 0.15, 0.22, 0.30, 0.40, 0.52, 0.66, 0.82, 1.00, both signs, on a
single common axis, against a norm-matched random control at every dose, with the prediction
committed to git before any data existed. Result: **zero clean windows out of 120 cells on both
datasets**, on both registered branches of the outcome. `sup_grad` induces *fewer* new false
statements than a random vector of the same size. The null is not an artifact of pushing too
hard.

**3, comparability.** S3 confirmed the suspicion exactly. The two arms were disjoint by
**6.09x** on cities, so "naive steering fails" and "reachability steering is inert" were two
conclusions about two non-overlapping experiments. Worse, the behavioral transition sits between
relative magnitude **0.29 and 0.78**, and neither arm had ever measured there. D1's revised grid
was then built specifically to put four points inside that band. One honest correction: this does
*not* generalise. On common_claim the gap is 1.66x and the ranges overlap once the per-statement
arm is included, so our original 6.1x claim was generalising from the mean arm on one dataset.

**4, the halfspace.** This is where the PI's instinct paid off most. The target set is not merely
imperfect, it is **semantically vacuous by construction**. All 200 certified argmax flips
succeed, which means the control machinery does exactly what it promises. But 85% of the cities
targets are the tokens " North" and " South", so a perfect flip turns "South Korea" into "North
Korea" and leaves the sentence's truth value untouched. Vacuity 0.855 to 0.945; **at most 5.5% of
our successes actually made the claim false.** The two datasets are vacuous for different reasons
(`countries` on cities, `runnerup` on common_claim), which is itself a correction to what we had
recorded.

**5 and 6, the layer sweep.** The PI asked whether the same linear relationship holds mapping
from different source layers. It does, and that is not the interesting part. Truth is readable at
0.973 or better across 19 cities layers, but those layers read it along directions whose pairwise
cosine has median **0.365** and minimum **0.014**. Truth is not one linear object that we happened
to sample at layer 11; it is a family of near-orthogonal objects. And the gain moves the opposite
way from the readout: controllability peaks at layer 0, where the readout is exactly chance, and
falls **7.35x** by layer 25, where the readout is at ceiling. **There is no good layer to steer
from.** We steered past the controllability peak, and the token-space sweep reproduces this
independently (layer 11 costs 2.6x layer 0 on cities).

**7, temperature.** T = 0, deterministic, every arm, no sampling anywhere in the token-space
program. Settled and stated.

**8, token space.** This was the PI's own proposal and it is the one item that turned a null into
a positive. Defining the target by which token comes out, rather than by what a fitted probe says,
takes the flip rate from **0.000 at every budget** to 200/200 at the output layer and 34.0% at
budget (67.5% at twice budget) pulled back to layer 16. Two of the PI's specific predictions
resolved in opposite directions, and both are worth reporting: the naive output-layer steer *did*
mostly produce incoherence, as predicted (**confirmed**); but the inference he flagged next, that
the problem must therefore be the linear feature in the final layer, is **refuted**, because at
the output layer the map is exactly linear and a linear intervention flips the token cleanly when
it is the right one. What fails is alignment, not linearity.

### The two that are still open, and what they now need

**9, mapping meaningful words back into activation space (T1).** Still not started. S4 changed
its status from "one experiment among several" to the highest-value item on the list, because S4
proved the current target is empty and T1 is the experiment that builds a non-empty one. S4 also
handed it a calibrated success criterion it did not have before: **T1 has to beat 5.5%.**

**10, system-level synthesis.** Deferred at the last meeting on the PI's own "might be overkill",
with the agreement to revisit after D1. D1 is now done and null, so this is a live decision rather
than a parked one.

### The item the PI did not raise, which turned out to gate everything

None of the ten items covers the possibility that the pipeline cannot move *any* behavior, in
which case every null above would be a statement about our harness rather than about truth. We
added the refusal positive control as a pre-committed publication gate for exactly this, and it
**passed** (step 08). That is what licenses reporting items 1 through 6 as findings rather than as
failures to measure.

### What to say about the previous meeting's eight claims

The meeting before this one produced eight numbered claims, all of which were answered on
2026-08-05 in [`TOKEN_SPACE_FINDINGS.md`](TOKEN_SPACE_FINDINGS.md). If continuity comes up:

| # | Claim, compressed | Verdict |
|---|---|---|
| 1 | Naive output-layer steer will reproduce the failure, mostly incoherence | **Confirmed** |
| 2 | If incoherent, the issue is the linear feature in the final layer | **Refuted** |
| 3 | Check the steering vector is what you think it is | **Answered**, verified at the post-norm site where an oracle exists |
| 4 | Small perturbations in different directions have vastly different chains | **Answered, and split.** Anisotropy holds (9.91x spread); chaos does not (kappa within 6.1% of 1 across a 400x budget range) |
| 5 | SAEs are a useful alternative for classifying the target set | **Not answerable from these files** |
| 6 | Temperature 0, deterministic, invert the token mapping | **Confirmed** |
| 7 | Write out all the math from input to output and justify it | **Answered**, one tracked document plus `math_map.tex` |
| 8 | ActAdd sweeps all layers, some do nothing, mean difference is heuristic | **Answered**, and quantified: mean difference sits *below* the chance alignment floor |

Claim 5 is the only one still owed. It is not answerable from the artifacts we have, and it needs
a decision about whether to spend on it.

---

## 4. Where it lands: the 2x2

The publishable object is not either result alone. It is the contrast, produced by one pipeline
holding everything constant except the concept being steered.

| Cell | Meaning | Observed in |
|---|---|---|
| **actuatable** | readout crosses the boundary and behavior follows | refusal (per-statement arm) |
| **readout-only** | readout crosses and behavior does not | truth (cities) |
| **no-crossing** | the push never reaches the boundary | refusal (held-out mean arm) |
| **inert** | neither moves | none so far |

Same certificate machinery, same layers-apart hop, same steering code, opposite outcomes. That is
much stronger than a bare negative result, because a bare negative result always invites the
reply "your setup was broken".

---

## 5. What is specifically for Julian

His paper is arXiv:2604.19018, *Local Linearity of LLMs Enables Activation Steering via
Model-Based Linear Optimal Control* (A-LQR). Three things connect directly.

### 5.1 We corroborate his core assumption, strongly

A-LQR assumes the model is locally linear enough for control theory to apply. Our per-statement
fits of readout against push size give median R-squared = 0.999 on cities and 0.990 on
common_claim, with 0% of statements moving in the wrong direction. That is about as clean a
corroboration of local linearity as you could ask for, and it is measured on a nine-layer hop,
not a single layer.

### 5.2 Our failure modes are exactly what his closed loop would fix

We ran **open-loop** control: compute the required push once, apply it, hope. Two things go wrong
that a closed loop implicitly answers.

- **Gain collapses 8 to 35x across a one-word context shift.** The certificate is computed at one
  point and applied at a slightly different one, and the realized effect is an order of magnitude
  below the prediction. Closed-loop re-planning would measure and correct this.
- **The probe boundary does not transfer from full statements to generation prefixes.** We fit
  the probe on "The city of Busan is in South Korea" and then read it on "The city of Busan is in
  South...". The baseline readout signs differ (cities prefixes read about -52, common_claim
  statements read +81). This finding appears to be **unclaimed in the literature** and is
  probably the most directly useful thing we have for him.

### 5.3 Two asks

- Access to the A-LQR code, so the behavioral evaluation runs against a closed-loop controller
  rather than our open-loop one.
- Which concept to target next. Truth is now well characterized as a negative; refusal is
  characterized as a positive. The interesting middle is something like sycophancy or toxicity,
  where nobody knows in advance which side of the 2x2 it lands on.

### 5.4 The framing sentence to open with

> "We set out to steer truth, failed, and spent two months proving the failure was real rather
> than incompetent. The proof is a positive control on refusal using the identical pipeline. What
> we have is a certified-reachable-but-behaviorally-inert dissociation with a mechanism, which as
> far as we can find is unreported."

---

## 6. Vocabulary, in the order you will need it

**activation / residual stream.** The vector the model is carrying at a given token and a given
layer. For gemma-2-2b it is 2304 numbers. Every layer reads it, adds something, and passes it on.
This is the thing we read from and push on.

**probe.** A small classifier trained on activations to predict a property, here true versus
false. It is an observer we attach, not a part of the model.

**readout.** What the probe currently says. The central distinction of this project is readout
versus behavior: what the probe reports against what the model actually writes.

**direction.** A unit vector in activation space that supposedly encodes a concept. Steering means
adding a multiple of it to the activation.

**halfspace.** A linear probe splits the 2304-dimensional space with a flat boundary. Everything
on one side is TRUE, everything on the other is FALSE. The FALSE side is a halfspace, and for a
long time it was our target set.

**backward reachability.** Given a target set at a later layer, find the set of earlier-layer
states that can reach it within a budget. Standard control theory, applied here across transformer
layers.

**Jacobian.** The matrix of partial derivatives that linearly approximates what nine transformer
layers do to a small perturbation. It is what lets us pull a target at layer 20 back to layer 11.

**eps\* (the certificate).** The per-statement budget: the smallest push the linear model says
will land you in the target set. This is the falsifiable prediction. Our finding is that it is
directionally right and about 2x optimistic.

**trust radius.** How far you can move before the linear approximation stops being valid. A
certificate larger than the trust radius is extrapolation, which is what happened on common_claim.

**token space.** Defining the target by which word comes out rather than by what a fitted probe
says. Membership in the set *is* the behavior, so the readout-behavior gap cannot open.

**sign control.** Push the same statements the same distance in the opposite direction. If the
effect is real and signed, it reverses. If the effect is just "we perturbed the activation", it
does not. This is what makes the refusal result credible.

**McNemar test.** The right test for paired before-and-after binary outcomes. It counts only the
statements that *changed*, so a statement that refused both before and after contributes nothing.
That is why 14 against 0 is strong evidence at n = 64.

**Mantel-Haenszel odds ratio.** An odds ratio computed within strata and then pooled. Here the
strata are push sizes, which lets us ask whether *crossing the boundary* predicts refusal once we
hold the size of the push fixed. Answer: 24 to 1.

---

## Source documents

[REACH_AUDIT_FINDINGS.md](REACH_AUDIT_FINDINGS.md) ·
[TOKEN_SPACE_FINDINGS.md](TOKEN_SPACE_FINDINGS.md) ·
[AUDIT_SUMMARY.md](AUDIT_SUMMARY.md) ·
[S1_ASYMMETRY.md](S1_ASYMMETRY.md) ·
[S2_LAYER_SWEEP.md](S2_LAYER_SWEEP.md) ·
[S3_COMMON_AXIS.md](S3_COMMON_AXIS.md) ·
[S4_TARGET_CENSUS.md](S4_TARGET_CENSUS.md) ·
[D1_DOSE_RESPONSE.md](D1_DOSE_RESPONSE.md) ·
[REFUSAL_POSITIVE_CONTROL.md](REFUSAL_POSITIVE_CONTROL.md) ·
[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) ·
[MEETING_SUMMARY.md](MEETING_SUMMARY.md)
