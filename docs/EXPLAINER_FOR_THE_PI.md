# The whole project, explained from scratch

*A standalone walkthrough of everything built and everything measured, written so that
someone who has never seen this repo can read it start to finish and understand both the
ideas and the numbers. Every result below comes from a run that actually happened, and
every figure named is a PNG sitting in the repo root. Written 2026-08-05.*

**How to read this.** Sections 1 through 3 are the setup and the plain-English version of
the ideas. Sections 4 through 8 are the experiments, in the order they were run, each one
stating what we asked, what the machine did, what came back, and which figure shows it.
Section 9 is a figure-by-figure guide. Section 10 is what we have *not* shown. Section 11
is what runs next. Section 12 is a glossary of every symbol.

If you have three minutes, read section 0 and then jump to section 9 and look at the
pictures.

---

## 0. The sixty-second version

We asked whether the "truth direction" that a linear probe reads out of a language model
at 99% accuracy is a direction the model can actually be *driven* along. The answer is no,
and we now know why with numbers rather than with a story.

Three findings carry the project:

1. **The math is exact and the behavior does not move.** We compute, for each individual
   statement, the smallest nudge that provably pushes the model's internal truth readout
   across into "FALSE". At the exact point where that nudge is defined, it does what it
   promises to four significant figures (calibration factor **0.9997**). Apply it during
   actual text generation and the model's lie rate goes from about 3% to **3.5%**. It
   keeps telling the truth.

2. **The probe carries no information about what the model is about to say.** At the token
   where generation actually begins, the truth probe's balanced accuracy for predicting
   whether the model is about to state something true is **0.500**, which is coin-flip.
   Refitting it does not help. XGBoost does not help. There is no signal there to find.

3. **The fix works, which proves the machinery was never broken.** When we stop defining
   the target with a fitted probe and start defining it in *token* space, where membership
   in the target set literally *is* the behavior, the same code with the same budget
   arithmetic flips the model's emitted token on **200 of 200** statements at the output
   layer and on **34%** of statements when pulled back nine layers deep, while the old
   probe direction flips **0.000** of them in every arm at every budget ever run.

The reason the old direction fails is now a single number: at the budget the old
certificate calls a complete trip into the FALSE halfspace, it consumes a median **1.7%**
of the logit margin that actually decides which token comes out.

---

## 1. What the objects are, in plain English

### 1.1 The model

`google/gemma-2-2b`. A 2-billion-parameter open language model, chosen deliberately
because it is the model in Julian's A-LQR paper, so anything we find is directly
comparable to that work.

It has **26 decoder blocks** stacked on top of each other. Text enters at the bottom, and
after every block there is a vector of **2,304 numbers** for each token position. That
vector is the **residual stream**, written `h`. It is the model's working memory at that
depth for that token. At the top, the last vector gets normalized (an operation called
RMSNorm) into a vector called `z`, and then multiplied by a big matrix `E` to produce one
score per vocabulary word. There are **256,000** vocabulary words. The highest-scoring word
is the one that comes out. We always run at temperature 0, so there is no sampling: the
output is exactly the argmax, and it is deterministic.

Two facts about this stack matter for everything below:

- The step from `z` to the word scores is a **plain matrix multiply**. It is exactly
  linear. There is one wrinkle, a "softcap" that squashes the scores through a tanh, but
  the squash is strictly increasing, so it never changes *which* word wins. It changes the
  size of the gaps and not their sign.
- Everything below that (attention, MLPs) is nonlinear. But it is smooth, so a first-order
  approximation of it is a *testable* claim rather than an assumption. We tested it, twice,
  and it holds far better than expected.

### 1.2 The data

Four datasets of true/false statements, deliberately arranged on a clean-to-messy gradient,
because the whole point was that the answer might depend on what kind of truth you test.

| Dataset | Statements | What they look like |
|---|---:|---|
| `cities` | 1,496 | "The city of Busan is in South Korea." Rigid template. |
| `sp_en_trans` | 354 | Spanish/English translation judgments. |
| `companies_true_false` | 1,199 | Company business descriptions. |
| `common_claim_true_false` | 4,450 | Heterogeneous claims about the world. |

All are 50/50 true and false. Most of the deep work uses **`cities`** and
**`common_claim_true_false`** as the two poles: the cleanest and the messiest.

### 1.3 A "direction" and what steering means

A direction is just a unit vector in that 2,304-dimensional space. Steering means: run the
model, and at some chosen layer, *add* a multiple of that vector to the residual stream,
then let the rest of the model run. If the direction really is "the truth axis" and you
push hard in the negative direction, the model should start asserting falsehoods.

That is the experiment the entire project is about, and it does not work. The rest of this
document is the increasingly careful account of why.

### 1.4 The cluster

Every expensive run happens on **NCSA DeltaAI**, GH200 nodes, partition `ghx4`, account
`bhhv-dtai-gh`. The scale factor is real: a 64-factor DCT fit takes about **90 minutes on a
laptop CPU** and about **4 seconds on the GH200**, roughly 800x. Roughly **475 GPU-hours**
remain of the allocation. The three big cluster rounds in this document ran on **2026-07-24**
(the reachability audit), **2026-07-29** (the Horizon 0 validations), and **2026-08-04**
(the token-space program).

---

## 2. The question chain

Every negative result here sharpened the next question rather than closing the project.
This table is the actual spine.

| # | Question | Answer |
|---|---|---|
| Q1 | Is truth encoded linearly in the activations? | Yes on clean data (0.990 probe accuracy), partly nonlinear on messy data (+0.082 gap) |
| Q2 | Is that readable direction one of the model's dominant *causal* levers? | **No.** About 1.2x random alignment |
| Q3 | Is that null just me reading completions by hand and seeing what I expect? | No. Instrumented with a validated judge, and bounded to a 0.010 effect |
| Q4 | Is it an artifact of one particular search objective? | No. A second, unrelated miner lands in the same place |
| Q5 | Does *starting* the search at the truth direction rescue it? | No. No point on the whole path is a truth lever |
| Q6 | Reframed as control theory: is the FALSE **set** reachable at all? | Yes, formally. And the certificate is behaviorally inert |
| Q7 | Is that inertness a math error or a real dissociation? | Real. The certificate is exact at its own point (0.9997) |
| Q8 | Was the target set itself the problem? | **Yes.** Defined in token space instead, the same machinery actuates |

---

## 3. The control-theory idea, explained without jargon

This is the reframe the PI proposed at the 2026-07-23 meeting, and it is the reason the
project has a shape rather than a pile of nulls.

### 3.1 The old way of asking, and why it was weak

Before the reframe, every experiment was of the form "here is a direction, does pushing
along it change behavior?" The answer was always no. But "this one arrow does not work" is
a weak claim. There are infinitely many arrows.

### 3.2 The new way

Stop looking for arrows. Define a **target set**: a region of activation space you want to
land in, for example "all activations whose truth readout says FALSE". Then ask for the
**backward-reachable set**: the set of *nudges at an earlier layer* that land you in the
target. If that set is empty within a reasonable budget, the null becomes a theorem rather
than an absence of evidence. If it is not empty, you get the steering set handed to you
directly, and you learn why every single-direction attempt missed it.

### 3.3 Why this is easy linear algebra rather than hard control theory

The target set we care about is a **halfspace**: everything on one side of a flat boundary.
Formally, `{z : w·z >= t}`, where `w` is the normal to the boundary and `t` is where it
sits.

Now here is the fact that makes everything cheap: **the preimage of a halfspace under a
linear map is a halfspace.** If the map from your nudge to the readout is (locally) linear,
then the set of nudges that land in the target is itself just "everything on one side of a
flat boundary", and you can write it down in closed form.

Three numbers fall out, and honestly the whole audit is these three numbers:

- **`g`, the gap.** How far this statement currently sits from the boundary. Positive,
  because it starts on the wrong side. "How much do I need to move the readout?"
- **`m`, the controllability margin.** How much the readout moves per unit of nudge, if you
  nudge in the single best possible direction. "What is my exchange rate?"
- **`eps* = g / m`, the budget.** The smallest nudge that reaches the target. Gap divided by
  exchange rate. And the *best* nudge, the one that achieves it, is a specific vector called
  the **pullback**, `A^T w`, which is just the target's normal vector dragged backward
  through the map.

Analogy: `g` is how far the goalpost is, `m` is how fast your car goes in top gear, `eps*`
is how long the drive takes. The audit is the discovery that the drive is short, the car
works exactly as the spec says, and the goalpost is not where the game is being played.

### 3.4 The one computational trick

Computing `m` looks like it needs the full 2,304 x 2,304 Jacobian of nine transformer
blocks, per statement. It does not. You only ever need `A^T w`, which is one
**vector-Jacobian product**: one backward pass. That is why a per-statement certificate over
hundreds of statements is affordable at all, and it is why this became a real experiment
rather than a thought experiment.

### 3.5 Where you inject matters enormously

The same target, reached from three different places, costs wildly different amounts:

| Where you add the nudge | Exchange rate `m`, cities | `m`, common_claim |
|---|---:|---:|
| After the final norm (into `z`) | 2.383 | 2.138 |
| Before the final norm (into `h`) | 0.532 | 0.432 |
| At a middle layer (11, 13, 16, or 8) | 0.607 | 0.676 |

The pre-norm row is 4 to 5 times worse, and the reason is geometric and worth knowing: the
RMSNorm's derivative contains a projection that **deletes any component of your nudge that
points along the activation itself**, and shrinks what is left. So a steering vector aimed
"outward" from the activation is simply erased by the normalization. We measure that cost
directly and call it the `rmsnorm_penalty`: median **4.38** on cities and **5.18** on
common_claim.

---

## 4. Part I: the four chapters before the reframe

These ran earlier and are summarized briefly, because their job now is to establish that
the null is robust and not an artifact of any one method.

### 4.1 Truth is linearly readable, and messier truth is less linear

Extract the last-token activation at all 27 hidden states, train a linear probe and
XGBoost at each layer, compare.

| Dataset | Best layer | Linear | XGBoost | Nonlinear gap |
|---|---:|---:|---:|---:|
| cities | 11 | **0.990** | 0.993 | **+0.003** |
| sp_en_trans | 7 | 0.972 | 1.000 | +0.028 |
| companies | 14 | 0.917 | 0.954 | +0.038 |
| common_claim | 13 | **0.706** | 0.788 | **+0.082** |

The gap grows monotonically with messiness while readability falls. cities replicates Marks
and Tegmark cleanly.

**Two bugs caught here, both of which would have inverted the published conclusion.** Gemma's
tokenizer left-pads, and the first extraction script assumed right-padding, so it read the
wrong token and cities scored 0.45, which is chance. And the summary script took the max gap
across layers instead of the best-layer gap, which had labeled cities the *most* nonlinear
dataset, the exact inverse of the truth. The lesson that shaped every later smoke test:
**check a number, not just a tensor shape.**

### 4.2 Decodable is not causal

Deep Causal Transcoding (DCT) searches, without labels, for the 512 perturbations at one
layer that produce the largest change at a later layer. Those are, by construction, the
directions the model is most causally sensitive to.

The truth direction is not among them. Alignment with the best of 512 is **1.2x random**.
Even as a *combination* of all 512 it only reaches 1.3x chance. And this is not a layer
artifact, because at exactly that layer the two independent supervised estimates of "the
truth direction" agree with each other better than anywhere else.

**The claim, stated precisely: a direction can be readable at 99% accuracy and sit
essentially orthogonal to the space of directions the model is most sensitive to.**

### 4.3 The judge, and the most valuable bug in the project

The first two chapters were partly read by hand. Chapter 3 turned that into counts using
OLMo-3-7B-Instruct as an automatic judge, gated at 0.85 agreement with gold labels and
passing at **0.970**.

Then a hand drill-in on the *unsteered* completions found this:

> Prompt "Two plus two equals" gives "**four.** That's the message from a new study that
> found people who eat..." judged **FALSE**, reason "completion is unrelated to the stem."

The answer was correct. gemma-2-2b is a *base* model: it does not stop, it rambles for the
full generation length, and the judge was scoring the whole paragraph. Correct answers were
failing on their tails.

This is a measurement-validity bug: the numbers were real and were measuring the wrong
thing. Crucially, **the 0.970 validation gate did not catch it**, because the gate tested
the judge on clean single-claim statements, not on long rambling generations. A passing gate
on clean inputs does not certify labels on messy ones. That lesson is load-bearing in every
plan since.

After the fix (shorter generations, answer-only rubric, 32 prompts per strength), the
unsteered TRUE rate rose from 0.50 to **0.81**, and the corrected result was:

- **0 of 20 top causal vectors flip an established fact.** By the rule of three, a 95% upper
  bound of about 14% on the fraction that could.
- **No lie asymmetry, and now bounded**: pooled effect **-0.010**, 95% CI **[-0.042, +0.023]**.
  The previous run could only assert "about zero"; this one proves no asymmetry larger than
  about 0.04 exists.
- **The one real effect is symmetric degradation.** The TRUE rate falls at *both* extremes of
  the sweep (trend test z = -5.06, p < 1e-6) while the direction-by-sign interaction is not
  significant. Only magnitude matters, not sign. That is the fingerprint of a direction that
  **breaks** the model rather than **flipping its truth value**.

### 4.4 Two more attempts to break the null, both failed

**A second, unrelated miner (MAG).** Instead of steering, it prepends the model's own
question ("Is the following statement true?") and measures the resulting activation shift.
Same conclusion. Two findings are worth keeping: the model answers "yes" to essentially
everything (0, 0, 0 and 1 "no" answers across the four datasets), so it *contains* truth but
does not *express* it when asked; and there is a crisp linear "I am being asked whether this
is true" direction that explains the shift at cosine 0.97, and it is **orthogonal** to the
truth direction. Two different things.

**Warm-starting the search at truth itself.** Anchor the causal search back toward the
supervised truth seed with strength lambda, and sweep lambda from 0 to 3. The control
passes: cosine to the seed climbs 0.03, 0.28, 0.72, 0.95, so we really did sweep from "free
search" to "essentially the supervised axis". The result: **no truth lever at any lambda.**
The path runs from inert to mild degrader and never passes through truth lever. And the
harder you anchor toward truth, the *more* inert the direction becomes.

---

## 5. Part II: the reachability audit (cluster, 2026-07-24)

Five phases. Runbook `deltaai/REACH_RUN.md`, findings `docs/REACH_AUDIT_FINDINGS.md`.

### Phase 1: is the FALSE set reachable on paper?

**What it does.** For every statement, compute the pullback `J^T w` and from it the gap, the
margin, and the budget `eps*`. The battery of readout directions is deliberately padded with
controls: the two truth readouts, 8 directions spanning the truth subspace, DCT's top 4
known-causal directions, and **64 random unit directions as the null**.

**What came back:**

| | cities | common_claim |
|---|---:|---:|
| median margin for the truth readout | **9.07** | **7.00** |
| median margin for a random direction | 1.10 | 1.34 |
| ratio truth to null | **8.3x** | **5.2x** |
| median budget `eps*` | **2.80** | **10.69** |
| fraction reachable within the activation's own norm | 1.00 | 1.00 |
| verdict | reachable-candidate | reachable-candidate |

Read that as: the truth readout is roughly an order of magnitude more controllable than a
random direction, and the nudge needed to flip it is a small fraction of the activation's
own size (2.80 against an activation-norm yardstick of 47.7, so about 6%). **On paper,
reachable.**

**Figures.** `plot_reach_margins_<ds>.png` is a violin plot of the four groups with a dashed
line at the random median. *How to read it:* the truth violins sit visibly above the gray
random violin, and that separation is the entire "reachable-candidate" verdict. If they had
overlapped, the programme would have stopped here.
`plot_reach_curves_<ds>.png` is the money plot: budget on x, fraction of statements reachable
within that budget on y, with the activation-norm yardstick as a dashed line and a gray band
marking "beyond where the linear model is valid". cities saturates well left of both. 
common_claim saturates **inside** the gray band, which is the first warning sign.

### Phase 2: is the model's high-gain channel the concept channel?

**What it does.** Compute the *full* 2,304 x 2,304 Jacobian for 32 statements per dataset,
take its SVD, and ask whether the directions the map amplifies have anything to do with any
concept direction we can name.

**What came back.** The map is structured, and it is concept-blind on the input side.

| | cities | common_claim |
|---|---|---|
| effective rank (of 2,304) | 396 to 551 (17 to 24%) | 447 to 616 (19 to 27%) |
| overlap of the top input directions with DCT's concept subspace | **0.053 to 0.076** | **0.039 to 0.059** |

The pre-registered stop-and-diagnose threshold on that overlap was 0.3. Both datasets fire it
at about 0.05, which is essentially random.

**The asymmetry is the finding.** Measured against the *output* side, the truth readout is
extremely concentrated: **74%** of its energy on cities sits in the top 64 output directions,
which is 17x to 125x what a random direction gets. Measured against the *input* side, the
source-layer truth direction barely beats random. So the truth readout is
**downstream-visible but not upstream-addressable**. Figure: `plot_reach_svd_<ds>.png`.

### Phase 3: does any of it move behavior?

**What it does.** Actually steer and actually generate. Two arms: a shared mean direction on
32 held-out prompts, and the primary per-statement arm where each of 200 true statements is
steered along *its own* certified direction at multiples of *its own* budget. Then the OLMo
judge scores each continuation as TRUE, FALSE, or INCOHERENT.

**What came back: nothing moved.**

| | cities | common_claim |
|---|---:|---:|
| lie rate at the strongest push toward FALSE | **3.5%** | **5.5%** |
| unsteered baseline lie rate | about 3% | about 3% |

Even at maximum push the completions stay coherent and *true*: "The city of Busan is in South
Korea", "Abu Dhabi is in the United Emirates". Figures `plot_judge_reach_<ds>.png` and
`plot_judge_reach_stmt_<ds>.png` are flat lines, and that flatness is the headline null.

### Phase 4: were we even steering at the right depth?

**What it does.** Sweep all 26 layers and compute the controllability margin at each, for
three different readouts, under two input formats.

**What came back.** Controllability is **front-loaded**. It peaks at layer 0 and decays
monotonically with no late resurgence. By the layers we actually used, **42% to 73% of the
available controllability is already gone**.

Worse: the **verdict** readout, the one corresponding to what the model will actually *say*
when asked, has a margin **2.1x to 5.7x smaller** than the truth probe at the same layer. We
were pushing the biggest lever we could measure and it was not the lever attached to
behavior. Figure: `plot_reach_jlens_<ds>.png`, two panels, log scale, curves falling left to
right, with the red verdict curve below the others everywhere.

### Phase 5: was the certificate inside its own validity region?

**What it does.** Push along a direction and measure how far the true model output diverges
from the first-order prediction. Report `eps20`, the norm at which the error reaches 20%.
This is the honest trust radius of the linearization.

| | trust radius | our budget `eps*` | ratio |
|---|---:|---:|---|
| cities | 2.99 | 2.80 | **0.94x, inside** |
| common_claim | 4.87 | 10.69 | **2.20x, outside** |

So the two datasets fail differently, and this matters:

- **common_claim: the certificate is extrapolation.** Its claim is evaluated far outside where
  the first-order model is valid, and empirically 0% of statements actually cross.
- **cities: a genuine readout-versus-behavior dissociation.** The certificate is inside its own
  validity radius, the readout does move onto the FALSE side, and behavior still does not
  change. This one cannot be blamed on linearization error, and it is the scientifically
  interesting case.

There is a second, quieter finding in that table: the pullback direction has by far the
*smallest* trust radius of anything tested, 4 to 5x smaller than a random direction. The
direction the certificate points along is the most curvature-sensitive direction in the space.

### The four mechanisms, D1 through D4

After Phase 3 came back null, four parallel deep-dives asked why. They sharpened, and in two
places corrected, the initial read.

**D1: actuation is linear, correctly signed, and 10 to 35x too weak.** Per-statement fits of
readout against push give **R-squared = 0.999** on cities, with **0% wrong-sign**. Steering
works exactly as the linear model says, just far too weakly: the push actually needed is about
**97**, against a predicted 2.9. Since 97 is 1 to 2x the activation's own norm, a real crossing
needs a nudge the size of the activation itself. And on cities the predicted margins do not even
**rank-order** which statements actuate best (Spearman 0.002), so the certificate loses its
*ordering* content, not just its calibration.

**D2: behavior is inert, strictly.** The earlier "mild coherence degradation" read was too
generous. Incoherence is not dose-dependent in 3 of 4 arms. The FALSE verdicts concentrate on a
handful of problem prompts that recur identically **at zero steering**, including two cases
where the judge itself is factually wrong. The one statistically significant trend runs
**backwards**. Where incoherence does track scale it is sign-symmetric topic derailment into
generic filler, not negation. Verdict: inert. Not even an incoherence knob.

**D3: the map is structured but concept-blind on the input side.** Covered in Phase 2. This
corrected an earlier "near-isotropic" claim, which was wrong: the spectrum is not flat.

**D4: controllability is front-loaded and the behavioral lever is the weakest one.** Covered in
Phase 4.

**The refined headline.** The certificate fails behaviorally *not* because linearization breaks
(R-squared stays at 0.99) but because of **three transfer failures**:

1. **Context transfer.** The gain collapses across a one-word context shift while sign and
   linearity survive.
2. **Population transfer.** The probe's threshold does not transfer from statement tokens to
   generation prefixes. The probe dissociates from behavior *at baseline*, before any steering.
3. **Input-side concept misalignment.** The high-gain channel is not addressable from any
   source-layer concept direction.

And behavioral validation *inherently forces* transfers 1 and 2, because behavior only exists
during generation, which is never the point where the certificate is computed.

---

## 6. Part III: Horizon 0, closing the audit's own loopholes (cluster, 2026-07-29)

Seven validations, each closing a specific hole a reviewer would poke. All seven complete.

### 6.1 The decisive one: was the certificate ever right, at its own point?

The 10-to-35x shortfall had two possible explanations. Either our margins are simply
miscalibrated, or they are exact where they are defined and the loss happens when we move to a
different context. To settle it, do a teacher-forced forward pass on the **full statement** with
the injection active, and read the result at **exactly** the point where the budget was computed.

| | cities (n=199) | common_claim (n=197) |
|---|---|---|
| **calibration factor** at its own point | **0.9997**, IQR [0.9910, 1.0117] | 0.7025 |
| **context factor** (stem slope over same-point slope) | **0.0729** | 0.1944 |
| implied collapse across one word of context | **13.7x** | 5.1x |

On cities the certificate is **exact at its own point, to four significant figures**. The entire
loss is the context shift.

This converts the result from "our math might be wrong" into "the math is exact and the transfer
fails", which is a completely different and much stronger claim. On common_claim the calibration
is 0.70, so there is a real 30% local miscalibration there on top of the context loss, which is
why cities carries the clean argument.

### 6.2 Is the probe failure a threshold shift or an absence of signal?

Take the activation at each generation prefix, label it for free using the judge verdicts we
already have (TRUE means the model completed that prefix truthfully), and try four ways to read
truth off it.

| | cities | common_claim |
|---|---:|---:|
| labeled prefixes | 158 | 178 |
| old-threshold accuracy | 0.057 | 0.944 |
| old-threshold **balanced** accuracy | **0.500** | **0.500** |
| median gap for TRUE prefixes | -52.6 | +82.0 |
| median gap for FALSE prefixes | -52.0 | +79.3 |
| recalibrated threshold | 0.943 (= base rate) | 0.944 (= base rate) |
| refit probe, 5-fold CV | 0.930 | 0.944 |
| **XGBoost**, 5-fold CV | 0.943 (gap **+0.013**) | 0.944 (gap **+0.000**) |

Read the two accuracy rows together. On cities the old threshold calls almost everything FALSE;
on common_claim it calls almost everything TRUE. They look opposite, and both have balanced
accuracy **exactly 0.500**, which is chance. The TRUE and FALSE medians sit within 1 to 3 units
of each other on a scale where the threshold is 100+ units away. Recalibrating recovers nothing.
Refitting recovers nothing. **XGBoost recovers nothing**, which closes the obvious "the boundary
is just nonlinear here" objection.

**So it is not a threshold shift, it is an absence of signal.** At the token where generation
actually begins, the truth probe has no information about whether the model is about to say
something true. This finding appears unclaimed in the literature and it is the strongest
supporting result the project has.

**The caveat, stated plainly:** balanced accuracy at a 0.94 base rate rests on only 9 and 10
FALSE examples. The direction of the result is unambiguous; its precision is not. Enlarging the
labeled set is cheap and should happen.

### 6.3 Does the null survive removing the bad prompts?

D2 identified about 15 prompts that fail at baseline before any steering. Drop them and
recompute. The null gets **cleaner**: the baseline goes to a clean 100% TRUE (FALSE rate 0.000,
incoherent 0.000) and the strongest push barely dents it. The apparent "mild degradation" in the
raw numbers was almost entirely the bad prompts.

### 6.4 Does *any* input direction survive the context shift?

Recompute the pullback at the generation-stem context and compare it statement by statement to
the one computed on the full statement.

| | cities | common_claim |
|---|---:|---:|
| gain ratio, stem over full | **0.241** | 0.385 |
| **cosine between the two pullbacks** | **0.305** | 0.392 |
| median honest robust budget | **67.0** | 155.3 |
| robust over nominal | **22x** | 13x |

One word of context knocks the gain down 3 to 4x *and* rotates the direction to a cosine of about
0.3. It is not the same direction weakened, it is a **different direction**. And the honest
budget, once you account for the measured context sensitivity, is 13 to 22 times the nominal one,
which puts it above the activation's own norm.

Stated as a certificate rather than a null: **once you account for measured context sensitivity,
the FALSE set is not reachable within the activation's own norm.**

### 6.5 The negative control that behaved as predicted

If the gain collapse were caused by curvature rather than context, then recomputing the pullback
at the steered state and iterating (a Newton step) should fix it. The D1 diagnosis predicts it
should *not* help. Result: the residual gap drops **six orders of magnitude in three steps**
(median 20.77 to 4.65e-05), nothing hits the step cap, and on cities the final direction is a
**0.971 cosine** to where we started. Curvature is not the obstacle. A control designed to
falsify the diagnosis confirmed it instead.

---

## 7. Part IV: what the directions actually *are* (SAE forensics)

Everything above is geometry: this direction is misaligned with that one. Can we say what these
directions *mean*?

**What an SAE is, briefly.** A sparse autoencoder is trained to re-express a model's activations
as a sparse combination of thousands of learned "features", many of which turn out to be
human-interpretable. GemmaScope is a public set of them for this exact model. We decompose each
of our directions into 32 of those feature atoms using orthogonal matching pursuit.

One methodological point, because it is easy to get wrong: we deliberately do **not** run the SAE
*encoder* on these vectors. The encoder is trained on activations, and a steering direction is not
an activation. Decomposing against the *decoder* is the correct operation.

**Result 1: the readout is legible, the input side is not.** Variance explained at 32 atoms:

| direction | where | explained |
|---|---|---:|
| the truth readout | target layer | **0.504** |
| every steering direction we have | source layer | 0.228 to 0.348 |

The thing we are trying to *control* is describable in feature terms. The thing we have to *push
on* is about half as describable. This is an independent restatement of D3 in interpretable units.

**Result 2: the context shift swaps the feature set outright.** Pairwise overlap (Jaccard) of the
top-32 feature supports:

| pair | Jaccard | shared |
|---|---:|---|
| a sanity control pair | **0.684** | 26 of 32 |
| **full-context versus stem-context pullback** | **0.049** | **3 of 32** |

The control at 0.684 proves the measurement is not degenerate. Against that, the two directions
separated by one word of context share **3 features out of 32**. That is a feature-level account
of D1 and D2: the context shift does not weaken a direction, it **substitutes a different set of
features**. Figures: `plot_sae_explained_<ds>.png` and `plot_sae_overlap_<ds>.png`, where the red
bar sits near the floor while the control bars are far above it.

---

## 8. Part V: the token-space program, and the thing that finally worked (cluster, 2026-08-04)

### 8.1 The idea

Every failure above shares one feature: the target set was defined by a **fitted probe**. A probe
is a proxy for behavior, and Horizon 0.2 showed that at generation time it is not even a
correlated one.

So define the target set where membership *is* the behavior.

At temperature 0 the model emits `argmax_j (E z)_j`. So the set of activations that emit token `j`
is exactly `{z : (E[j] - E[k])·z >= 0 for every other token k}`. That is a **convex cone with
255,999 flat faces**. It is not a proxy for "the model says X". It is a restatement of it.

Then take `a = E[target token] - E[current token]`. Set the threshold to zero. The gap `g` becomes
**the logit margin**: literally how far ahead the incumbent word is. And now `eps* = margin /
||A^T a||` is the *same formula*, the *same single backward pass*, and the *same code* as before.
The only thing that changed is that `a` is read off the model's own unembedding matrix instead of
being fitted from labels.

### 8.2 The four directions raced against each other

| name | what it is |
|---|---|
| `oracle` | the least-norm displacement into the target token's cone, solved as a quadratic program and verified in closed form against all 256,000 vocabulary entries. Its flip is a theorem, so it is the harness assertion and not a competitor |
| `md_full` | the final-layer mean-difference truth direction (the classic) |
| `jtw_legacy` | the old probe-halfspace pullback |
| `jtw_token` | the new token-space pullback |

### 8.3 The main result

Hit rate is the fraction of statements where the emitted token actually became the target token.
`frac = 1.0` means "spend exactly the certified budget".

**cities** (n = 200; the discriminating dataset, because its target is a semantically false
country):

| site | direction | at budget | at twice budget |
|---|---|---:|---:|
| output layer | `oracle` | **1.000** (200/200) | **1.000** |
| layer 16, all positions | `jtw_token` | **0.340** | **0.675** |
| layer 16, last position only | `jtw_token` | 0.135 | 0.285 |
| every arm, every budget | `jtw_legacy` | **0.000** | **0.000** |

**common_claim** at layer 8 reaches 0.811 and 0.844 on its shared statements, but its target is
the *runner-up* token, so generic disruption can land on target there and it should not be read as
a stronger version of the cities result.

### 8.4 The reason the old direction fails, in one number

The right unit was never "norm of the nudge". It is **fraction of the deciding logit margin
consumed**. Median values at the certified budget:

| dataset, site | direction | margin consumed |
|---|---|---:|
| cities, output layer | `oracle` | 1.0010 |
| cities, output layer | `md_full` | 1.0877 |
| cities, output layer | `jtw_legacy` | **0.0174** |
| cities, layer 16 | `jtw_token` | 1.0735 |
| cities, layer 16 | `jtw_legacy` | **0.0082** |

At the budget the old certificate calls a complete trip into the FALSE halfspace, the legacy
direction consumes **1.7%** of the margin that decides the token at the output layer, and **0.8%**
at layer 16. That single line is the quantitative replacement for the entire qualitative story
about why the old null was uninformative.

### 8.5 The subtlest result: consuming the margin is necessary and not sufficient

Look at `md_full` in that table. It consumes **1.0877** of the margin, which is *more* than
`oracle`'s 1.0010. And it flips **1 statement out of 101** against `oracle`'s 101 out of 101.

Why: `md_full` is so poorly aligned with the deciding direction that it has to spend a huge norm to
buy that one margin, and a displacement that large moves the *other 255,999 logits* too. A newline
wins the argmax instead of the target country. Its degenerate-output fraction is **0.885**, and
**740 of its 1,800 completions in that arm are empty**: the model emits a newline and stops.

This refutes a specific inference the PI flagged as the likely next step. If incoherence indicted
"the linear feature in the final layer", then *no* linear intervention at that layer should produce
a clean targeted flip. One does, on every statement, in the same arm, at the same budget scale.
**What fails is alignment, not linearity.**

### 8.6 Alignment, measured against the right null

Alignment `alpha` is the cosine between a steering direction and the direction that actually decides
the token. For two random unit vectors in 2,304 dimensions the expected absolute cosine is
**0.0166**. That is the bar.

| direction | cities | vs chance | common_claim | vs chance |
|---|---:|---|---:|---|
| `md_full` | 0.00398 | **0.24x** | 0.00940 | 0.57x |
| `mean_diff_tgt` (as injected) | 0.01325 | 0.80x | 0.01560 | 0.94x |
| `probe_grad_tgt` (as injected) | 0.01355 | 0.82x | 0.01471 | 0.88x |

**Every truth direction sits at or below chance alignment with the direction that decides the next
token.** The best of them buys 0.94 of what a random vector would. That is the diagnosis in one
table, and it is a statement about *alignment*, not about the budget: the certificate was never
wrong about how far to push, we were pushing almost perpendicular to the thing that moves the
token. Read through `eps(u) = eps*/alpha`, this is a **74x to 79x** multiplier on the honest budget
for cities. Figure: `plot_token_alpha_<ds>.png`, which draws the chance floor as a line.

### 8.7 Two more things the token program established

**The site is anisotropic, not chaotic.** The PI raised the worry that small perturbations in
different directions might have wildly different downstream chains, which would make a norm budget
a meaningless currency. Half of that is right and half is not. Measured with no Jacobian anywhere
in the measurement: the ratio of realized gain at the largest budget to the smallest is
**1.0000 exactly** at the output site, within **0.3%** of 1 pre-norm, and within **6.1%** at layer
16. So the site is *linear over the whole range we sweep*. But at layer 16 exactly one direction
stands out, the pulled-back token direction, at **8.7x** the median random direction, while
everything else including both truth directions sits inside a 1.74x band that also contains all
eight random controls. That is **signal, not turbulence**, and `eps(u) = eps*/alpha` is exactly the
change of units that prices it. An anisotropic but linear site needs different units, not a
different theory.

**Depth is not the binding constraint.** The layer sweep prices every layer. Median certified
budget to flip the token:

| dataset | layer 0 | layer 8 | layer 11 | layer 13 | layer 16 | layer 25 |
|---|---:|---:|---:|---:|---:|---:|
| cities | **8.73** | 21.44 | 23.11 | 23.21 | 23.00 | 67.24 |
| common_claim | **2.92** | 4.60 | 5.10 | 5.34 | 6.60 | 19.93 |

Cost rises with depth across the whole sweep. Our source layers cost **2.6x** and **1.8x** what
layer 0 costs. This independently reproduces the audit's D4 finding in token space with a
completely different readout. But sweeping layers does not rescue the mean-difference direction: it
fails at the output layer, before the norm, and at depth alike, at every budget. **Depth is not the
binding constraint. Alignment is.**

### 8.8 The honest boundary of this result

The certificate is a **first-token** claim, and the completion recovers. On cities, `oracle` flips
the first token on **200 of 200** statements, and yet only **0.085** of those completions go on to
name the target country while **0.700** still name the correct one. Reaching the cone at one
position does not keep the model there at the next position. Extending the target set over a
*horizon* is the natural next problem, and it is exactly what the backward-reachable-tube recursion
in the PI's citation is for.

---

## 9. The figure guide

Every figure below exists as a PNG in the repo root.

### The reachability audit

| Figure | What it shows | How to read it |
|---|---|---|
| `plot_reach_margins_<ds>.png` | Violin plot of controllability margins for truth, truth-subspace, DCT and random groups | The truth violins sit above the gray random violin. That separation *is* the "reachable on paper" verdict |
| `plot_reach_curves_<ds>.png` | Fraction of statements reachable within a given budget | cities saturates left of both the activation-norm line and the gray invalid band. common_claim saturates *inside* the band, so its certificate is extrapolation |
| `plot_reach_geometry_<ds>.png` | Where the optimal input direction actually points | It is not the source-layer truth direction. This histogram is the seed of the D3 mechanism |
| `plot_reach_svd_<ds>.png` | Energy of four vectors inside the top-k singular subspace | The blue and green output-side curves climb fast; the orange input-side curve barely beats random. That asymmetry is "downstream-visible, not upstream-addressable" |
| `plot_reach_jlens_<ds>.png` | Controllability by layer, two input modes, three readouts | Curves fall left to right, so the place to intervene is the left edge and we intervened in the middle. The red verdict curve is below the others everywhere |
| `plot_judge_reach_<ds>.png`, `plot_judge_reach_stmt_<ds>.png` | Judge verdict fractions against steering strength | Flat lines. That flatness is the headline null |
| `plot_reach_audit_dissociation.png` | Summary panel: readout moves, behavior does not | The two-panel version of the whole audit |
| `plot_reach_audit_certificate.png` | Summary panel: budget against trust radius | cities inside, common_claim outside |

*Reproducibility note on the last two: they were produced ad hoc on 2026-07-24 and no committed
script regenerates them. If they go in a paper they need a script.*

### The token-space program

| Figure | What it shows | How to read it |
|---|---|---|
| `plot_token_steer_<ds>.png` | Hit rate against budget, one line per direction | The `oracle` line is pinned at 1.0, `jtw_token` climbs, `jtw_legacy` is flat on the floor at exactly zero. This is the main result in one picture |
| `plot_token_budget_<ds>.png` | The cost of a flip | What a flip actually costs in activation-norm terms: a median 3.4% of the norm on cities |
| `plot_token_alpha_<ds>.png` | Alignment of each direction, with the chance floor drawn | Every truth direction sits at or below the chance line. That is the diagnosis |
| `plot_token_layers_<ds>.png` | Certified budget at each of the 26 layers | Rises with depth. Layer 0 is the cheapest place to intervene and we used 11, 13, 16 and 8 |
| `plot_signed_steer_<ds>.png` | The per-statement split of who moves and in which direction | cities is inert: 103 of 199 statements are byte-identical at every scale |
| `plot_token_sens_<ds>_<site>.png` | Realized gain against budget, per direction, no Jacobian involved | Flat lines mean the site is linear over the swept range. The one line that bends is the pulled-back token direction |

### The SAE forensics

| Figure | What it shows | How to read it |
|---|---|---|
| `plot_sae_explained_<ds>.png` | Cumulative variance explained against decomposition rank | How sparse each direction is in feature space. The readout line is on top; every steering direction clusters low |
| `plot_sae_overlap_<ds>.png` | Pairwise feature overlap, sorted, full-vs-stem highlighted | The red bar sits near the floor while the control bars are far above it. That contrast is the figure's whole point |

### The earlier chapters

| Figure | What it shows |
|---|---|
| `plot_mag_linearity_v2.png` | The question-framing direction explains the activation shift at cosine 0.84 to 0.98 while truth and DCT's top lever sit inside an orthogonal band |
| `plot_dct_warm_verdict_<ds>.png` | The FALSE fraction pinned to the floor at every anchor strength |
| `plot_judge_steering_<ds>.png` | The corrected judge sweep: symmetric degradation, no lie asymmetry |

---

## 10. What is *not* established

This is the column to read out loud rather than wait to be asked about.

- **Everything is one model.** `google/gemma-2-2b`, fp32, one hop pair per dataset, one probe
  family. Nothing here is known to generalize.
- **The publication gate has not run.** The refusal positive control is fully implemented, four
  SLURM scripts, and **has never been submitted**. Until it runs, "the instrument works and truth
  is not actuatable" and "the instrument does not work" remain unseparated *for the probe-halfspace
  certificate*. (The token-space certificate does not need that gate in the same way: its oracle arm
  is a positive control by construction, and it passes on 200 of 200.)
- **The 0.500 balanced-accuracy result rests on 9 and 10 FALSE examples** at a 0.94 base rate. The
  direction is unambiguous, the precision is not.
- **common_claim's same-point calibration is 0.70, not 1.0**, so there is real local miscalibration
  there and cities carries the clean argument.
- **The token-space results are first-token only.** The completions demonstrably recover.
- **Temperature 0 only.** Nothing transfers to sampling without re-derivation.
- **n = 200 per dataset** in the token program, subsampled from 1,496 and 4,450. On common_claim the
  legacy direction covers only 90 of those 200 (44 at label 1), because the margin extraction capped
  that dataset at 2,000 of its 4,450 rows.
- **The SAE work is thin**: cities-first, one dictionary width, one layer pair, no robustness pass.
- **One claim is unanswerable from the files we have.** Whether SAE features give a better handle on
  the target set cannot be tested from the current decompositions, because the readout was decomposed
  at one layer and the steering vectors at another, and the dictionaries are per-layer and unrelated.
  It needs one more job, not a re-read.
- **No judge ran in the token program.** The hit rate is a mechanical argmax check and the coherence
  readings are string-level.
- **One quantity is defined and not computed**: the basis-correct transported version of the
  cross-layer alignment. The alignments above are cosines against the raw vector we actually
  injected, which is the operational question, but not a coordinate-free statement about the concept.
- **A reference from the meeting notes (arXiv:1910.13272) does not match the paper it was described
  as**, and still needs confirming.

---

## 11. What runs next, and what each outcome means

Ordered by information per GPU-hour.

**1. The refusal positive control (40 GPU-minutes for the gate).** Refusal is a concept with a
*known* behavioral handle in the literature, with public code, validated down to small Gemma models.
Run the **unmodified** pipeline on it: same hook, same budget arithmetic, same verdict logic. Two
outcomes, both pre-registered:

- **actuatable**: the instrument is validated, the truth dissociation becomes a fact about truth
  rather than a fact about our method, and the paper upgrades from a workshop note to a main-venue
  candidate.
- **readout-only**: refusal fails the same way truth did, which means the instrument cannot
  demonstrate actuation even where the literature says it exists. **Stop and take it to Julian.** The
  two prime suspects are named in advance: the all-positions hook convention, and the choice of the
  pullback as the input direction.

One design detail makes this a clean test: for refusal, the instruction *is* the generation prompt,
so the context-shift confound that explains the entire truth failure **cannot arise by construction**.

**2. The cheapest falsifiable prediction on the list.** The pre-norm arm is currently a null, and the
explanation is arithmetic: a displacement certified *after* the norm has to survive the norm, at a
cost of 4.38 and 5.18, while the sweep stopped at twice the budget. **Prediction, recorded here and
not yet run: sweeping the pre-norm oracle arm to six times the budget should recover its hit rate to
near 1.0 on both datasets.** If it does not, the explanation is wrong and the RMSNorm derivative in
the certificate needs re-deriving. One job, no new code.

**3. Steer where the budget is cheap.** Layer 0 is 2.6x cheaper than layer 11 on cities, and the
layers 0 to 8 window is also the one Phase 4 identified as best-conditioned for the *behavioral*
readout. This is the maximum-contrast configuration available with no new machinery.

**4. Add an oracle arm at the layer sites.** Every layer-site number currently has no in-arm ceiling.
The cone solution already exists; pulling it back through the same backward pass costs nothing extra
and turns "0.340" from a bare number into a fraction of what was provably available.

**5. The horizon problem, which is the real handoff to A-LQR.** The certificate reaches the target at
one position and the model leaves it at the next. A target set defined over a *horizon*, with a value
recursion or a receding-horizon controller that re-plans each token, is the structurally correct
object. That is also the direct fix for the context-transfer failure that explains everything in
section 6.

**6. Free CPU work with no gate:** state the truth result in the same pre-registered verdict
vocabulary the refusal control will use, so the two are like-for-like; extend the SAE forensics to a
second dataset and width; enlarge the labeled prefix set so the 0.500 result does not rest on 9 and 10
examples; and write generating scripts for the two ad-hoc summary figures.

---

## 12. Glossary

| Symbol | Plain English |
|---|---|
| `d = 2304` | width of the residual stream, the number of numbers per token per layer |
| `L = 26` | number of decoder blocks |
| `V = 256000` | vocabulary size |
| `h^(l)_t` | the residual stream at layer `l`, position `t`. The model's working memory there |
| `z` | the activation after the final normalization, the last thing before the vocabulary matrix |
| `E` | the embedding matrix, tied, so the same matrix also converts `z` into per-word scores |
| `gamma` | the gain of the final normalization |
| `w` | the normal vector of the target halfspace. "Which way is FALSE" |
| `t` | where the boundary sits along `w` |
| `a` | the token-space version of `w`: the target word's row of `E` minus the current word's row |
| `g` | the **gap**. How far this statement currently sits from the boundary. In token space it is literally the logit margin |
| `A_S` | the linear map from a nudge at injection site `S` to the readout coordinates |
| `A_S^T w` | the **pullback**. The target's normal dragged backward through the map. The single best direction to push in, and the only object that needs computing |
| `m` | the **controllability margin**, the length of the pullback. Readout movement per unit nudge |
| `eps*` | the **budget**, `g / m`. The smallest nudge that reaches the target |
| `J` | the Jacobian of the hop from the source layer to the target layer |
| `alpha` | **alignment**. The cosine between the pullback and whatever direction you actually want to steer with. Chance is 0.0166 |
| `eps(u) = eps*/alpha` | the honest budget once you insist on steering with a named direction rather than the pullback |
| `B` | the **broadcast operator**. Every certificate on disk adds the nudge at *every* position, not one. Its adjoint sums the position gradients, and it is worth a factor of 1.355 to 2.016 |
| `frac` | the fraction of the certified budget actually spent in a steering run |
| `frac_margin` | the fraction of the deciding logit margin actually consumed. The unit that should have been reported from the start |
| `kappa` | realized gain at the largest budget divided by realized gain at the smallest. Exactly 1 at a linear site |
| `rmsnorm_penalty` | how much more perturbation the same effect costs one step before the final norm. 4.38 and 5.18 |
| `input_scale` | the activation's own norm yardstick. Every budget should be read against it: 47.7 on cities |
| **cone** | the set of activations that emit a given token. Convex, with 255,999 flat faces. Membership *is* the behavior |
| **oracle** | the least-norm displacement into the target token's cone. Its flip is a theorem, so it is the harness assertion, not a competitor |
