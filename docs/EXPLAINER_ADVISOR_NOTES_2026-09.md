# Julian's three notes, worked through end to end

*Window: 2026-09-03 to 2026-09-04. 16 commits, 30 files, +4393 / -69. Six SLURM jobs on
DeltaAI. Test suite 551 passed, 1 skipped. Written to be read cold, in order, by someone
who gave me the three notes and has not looked at the repo since.*

**How this document is organised.** Section 0 is the two-minute version. Section 1 restates
the three notes and says how I read each one. Sections 2, 3 and 4 take them one at a time:
what I built, what ran on the cluster, what came back, and what it means. Section 5 is the
infrastructure, section 6 the run log including the job that failed, section 7 the synthesis,
section 8 the honest column, section 9 what I want to do next.

Every number below is read off a named artifact on disk. Where a number appears twice it is
the same artifact both times. Nothing is quoted from a session transcript.

---

## 0. The two-minute version

**The first note worked, and it worked in the direction you predicted.** Moving to TruthfulQA
gave the truth track behavioral headroom it never had: gemma-2-2b answers 64 held-out
TruthfulQA questions truthfully-and-informatively **26.6%** of the time unsteered, against
**94.3%** on `cities`. Steering along the pulled-back mean-difference direction at 2 eps\*
raises that to **50.0%**, paired, **16 gained against 1 lost, exact McNemar p = 2.75e-4**,
monotone in dose and asymmetric in sign. Three norm-matched random directions at the same dose
sit at 0.281, 0.297 and 0.250, and the target beats each of them on the same 64 questions at
p = 0.0013, 0.0024 and 0.00015. Informativeness does not collapse. **This is the first
behavioral steering result the truth track has ever produced.**

**The second note is where it gets interesting, because the certificate did not do the work.**
eps\* on this dataset points at `+frac`, which is exactly the half of the sweep where nothing
happens, and it is about **3.5 times too small** to reach the halfspace boundary anyway:
`crossed = 0` at every dose. So the steering result above is a result about a *direction* at a
*magnitude*, and the reachability certificate that is supposed to license both was wrong about
both. `reach_control.py` names the cell `inert`, correctly. The direction works and the
machinery that was meant to explain why does not.

**That makes the second note load-bearing rather than housekeeping,** and our own artifacts
already localize the fault. At the point the Jacobian is linearized about, predicted and
realized gain agree at **calibration 0.9997** on cities; the pullback arithmetic is exact. The
entire shortfall is the move from that point to the generation stem, **13.7x on its own**. And
on the generation population the probe readout is at chance: **balanced accuracy 0.500, AUC
0.510**, never changing sign. The pipeline is arithmetically correct and pointed at the wrong
population with the wrong readout.

**The third note is delivered.** `docs/DERIVATION_SHORT.md`, 226 lines, six sections: the
forward map with the two linear stages marked, the certificate stated once and instantiated
twice, a table of where every measured number plugs in, and one line saying what the halfspace
target set actually was.

**What is not done.** V1, the closed check that would test the pullback end to end at the
output layer, is written and green but has not been run on a GPU. The per-statement TruthfulQA
arm is still void: its rerun (job 3084471) died in ten seconds on a guard, and that is the one
thing in this window that did not land.

---

## 1. The three notes, and how I read them

The notes as I have them:

> **1.** Switch to the dataset Julian used in his paper, TruthfulQA, which has been able to be
> steered before. A dataset of misleading prompts, steering them to become more truthful. Two
> affiliated LLMs as a judge already, plus a coherence measure, slightly different from my
> pipeline. Good to normalize on things that should be steerable.
>
> **2.** Observed different feature directions at different layers, could use another sanity
> check. Forget about DCT, do this linear mapping from input to output layer with contrastive
> vectors to validate that the Jacobian halfspace scoring pipeline is effective. Find where the
> source of the problem is. Change the feature vector construction to something that works.
>
> **3.** Write out the formulation of the halfspace and the linear mapping easier to follow than
> the current one.

**How I read them, in one sentence each.**

Note 1 is not really about a dataset. It is about **controlling for the concept**. Every null
this project has produced on truth is confounded by the possibility that truth is simply not a
steerable concept in gemma-2-2b, and there is no way to tell that apart from a broken instrument
without a concept that is known to be steerable. Refusal was a positive control on the
instrument. TruthfulQA is a positive control on the concept. That is what *normalize on things
that should be steerable* means, and it is the axis this project has never controlled for.

Note 2 is a request for a **decisive** experiment where the current evidence is merely
suggestive. The failure mode it targets is that we have been reading a fitted probe as if it
were the model's decision, and a fitted probe can be wrong in a way that is invisible from
inside the pipeline. Putting the readout at the output layer with a contrastive vector removes
that possibility by construction, because at the output layer the readout **is** the decision.
"Find where the source of the problem is" is the operative clause, and it is answerable from
artifacts we already have before any new GPU time is spent.

Note 3 is a writing task with a technical constraint: one thread from input layer to output
layer, the certificate stated once rather than re-derived per experiment.

I did note 3 first because it is cheap and it forced me to state the certificate precisely
enough that notes 1 and 2 could be checked against it. Then note 2's free diagnostic. Then
note 1, which is where the GPU time went.

---

## 2. Note 1: TruthfulQA

### 2.1 Why the old datasets could not have shown anything

Before the switch, the case for making it. Recomputed read-only from
`reach_stemacts_<ds>.npz`, which had been sitting on disk since July with its fit statistics
never saved:

| dataset | n | base rate P(TRUE) | balanced acc of the readout | AUC | LR refit, 5-fold CV |
|---|---:|---:|---:|---:|---:|
| cities | 158 | 0.943 | **0.500** | **0.510** | 0.930 |
| common_claim | 178 | 0.944 | **0.500** | **0.568** | 0.944 |

Three readings, and the third is the one that decides the dataset question.

- **Balanced accuracy is exactly 0.500 on both.** The boundary we were steering across carries
  no information about whether the model is about to complete the stem truthfully.
- **The readout never changes sign.** Every cities stem reads between -83.5 and -43.1, so every
  one is scored "deep FALSE" while 94% of them are completed truthfully. It is dominated by a
  prompt-format offset that swamps the truth signal. This is stronger than "the boundary
  shifted": there is no threshold on this direction that separates the classes.
- **There was almost no behavior to move.** Only **9 of 158** cities stems and **10 of 178**
  common_claim stems are completed falsely. The model spontaneously lies about **5.7%** of the
  time on our data. Even a perfectly working actuator would have had almost nothing to show,
  and any effect we did measure would have rested on nine examples.

That last row is the quantitative case for note 1, and it is why the note is not a matter of
taste. Written up in [`D2_PREFIX_TRANSFER.md`](D2_PREFIX_TRANSFER.md), artifacts
`reach_d2_summary_<ds>.csv`.

### 2.2 Q0: the dataset build

`src/prep_truthfulqa.py`, modeled line for line on `prep_refusal.py`, which is the one build
that has ever produced a working positive control. The discipline that matters: **the chat
template is applied once, at dataset-build time**, so extraction, margin computation and
steering all tokenize the identical string. The truth track's old failure was fitting on full
declarative statements and reading on generation prefixes, and section 2.1 prices that mismatch
at 13.7x.

1488 rows from 744 questions, exactly balanced, plus **64 held-out generation prompts that
never enter direction fitting**. Nine questions dropped for having no informative correct
answer. The contrastive pair is (question + correct answer) against (question + incorrect
answer), so the direction is fit in the generation format.

**One decision that inverts everything downstream: `label 1 = UNTRUTHFUL`.** The mean-difference
direction therefore points *toward* untruthful, and the truthful direction is `-v`. This is
deliberate and it is the reason the certificate's sign comes out wrong later. Written up in
[`Q0_TRUTHFULQA_DATASET.md`](Q0_TRUTHFULQA_DATASET.md).

Also deliberate: **"I have no comment" answers are excluded.** A truth direction carrying a
decline-to-answer component would have contaminated the 2x2 against the refusal control, which
is the comparison the whole project rests on.

### 2.3 Q1: the baseline, which was the gate

The pre-registered rule was: measure the unsteered rate first, and **if it is near ceiling as
it is on cities, stop and reconsider the track.** Job 3081925.

| | rate | Wilson 95% |
|---|---:|---|
| truthful | **0.281** | [0.186, 0.401] |
| informative | **0.984** | [0.917, 0.997] |
| truthful **and** informative | **0.266** | [0.173, 0.385] |

Against a cities truthful rate of 0.943 and a TruthfulQA human ceiling around 0.90.

**The combination is what matters, not either number.** Untruthful on 72% while informative on
98% means the model is *lying*, not *dodging*. That is the one regime where a behavioral effect
could be visible at all: there is something to move, and moving it cannot be faked by the model
falling silent.

**The judges are the "two affiliated LLMs" from the note**, already wrapped in
`src/judges/local_hf.py`: `allenai/truthfulqa-truth-judge-llama2-7B` and
`allenai/truthfulqa-info-judge-llama2-7B`. Validated at **1.000 on 128 gold answers, and 1.000
on each side separately**, which is the check that matters (a judge answering TRUE to everything
scores 1.000 on the positive side alone). The informativeness judge is the coherence measure the
note asks for: it is what catches the failure mode where truthfulness rises because the model
was broken.

Two things Q1 measured that Q2 then had to obey. **`n_truncated` was 64 of 64**: every single
generation runs past its own answer into a fabricated next turn, so judging the raw decode would
have scored text the model never meant as its answer on 100% of rows. And **Q2 had to be
registered as paired**: at n = 64 an unpaired test needs the steered rate to reach 0.51 before
clearing significance, while McNemar on the same 64 questions needs only 6 one-directional flips
for p = 0.031.

Written up in [`Q1_TRUTHFULQA_BASELINE.md`](Q1_TRUTHFULQA_BASELINE.md).

### 2.4 Q2: the steer, the judge, and the control

Three arms, all on the same 64 held-out questions, the same greedy decoder, the same 48-token
budget Q1 used, so Q1's 0.266 is a valid frac-0 control for all of them. Jobs 3082192 (steer),
3082210 (judge), 3083390 + 3084307 (control).

**The result, `jtw_mean_diff_tgt`, from `tqa_q2_summary_truthfulqa.csv`:**

| frac | rate | Wilson 95% | mean words | gained | lost | McNemar p |
|---:|---:|---|---:|---:|---:|---:|
| -2.0 | **0.500** | [0.381, 0.619] | 18.58 | 16 | 1 | **2.75e-4** |
| -1.5 | 0.359 | [0.253, 0.482] | 11.39 | 9 | 3 | 0.146 |
| -1.0 | 0.328 | [0.226, 0.450] | 7.23 | 6 | 2 | 0.289 |
| -0.5 | 0.266 | [0.173, 0.385] | 5.22 | 2 | 2 | 1.000 |
| 0.0 | 0.266 | [0.173, 0.385] | 4.69 | 0 | 0 | 1.000 |
| +1.0 | 0.266 | [0.173, 0.385] | 3.20 | 4 | 4 | 1.000 |
| +2.0 | 0.266 | [0.173, 0.385] | 2.53 | 6 | 6 | 1.000 |

Four properties, each ruling out a different artifact:

- **The frac-0 row reproduces Q1 to four decimals** (truthful 0.2812, informative 0.9844,
  both 0.2656). The steering harness with the scale set to zero is byte-identical to the
  standalone baseline script. That is the wiring check.
- **Monotone in dose.** The discordant split runs 2v2, 6v2, 9v3, 16v1 as the dose deepens. A
  judge fluke or a decoder accident does not order itself by dose.
- **Sign-asymmetric.** Every positive frac is symmetric noise, gains matching losses, p = 1.0.
  The effect lives on one side of zero only.
- **Informativeness holds at 0.984** at the strongest dose, the same value Q1 measured
  unsteered. The model did not get truthful by getting broken.

**The mechanism is legible in the completions.** At frac -2 the pattern is confident memorized
falsehood becoming hedged, qualified answer: "Marie Antoinette" becomes "Marie Antoinette is
often credited with..."; a flat "Donald Trump" for a 2032 election becomes an explanation that
the race is not decided; a cat-lives question answered "7" becomes the actual lifespan of a cat.

**The registered bar was a norm-matched random control**, by analogy with how D1 was calibrated,
and it was registered *before* the control ran. Three random unit directions on the mean arm's
exact scale grid. The pre-registered reading table, reproduced unchanged from the draft that
predates the control job:

| `randctrl` at frac -2 | Reading |
|---|---|
| rate near 0.266, ~4 mean words | The direction carries something specific. The result stands. |
| rate near 0.500, ~18 mean words | Any perturbation of this norm makes the model discursive, and discursive scores truthful. The result is about magnitude, not truth. |

**The control came back flat**, which picks the first row. Across all 27 (direction, dose) cells
the rate stays inside [0.250, 0.3125] and mean length inside [3.95, 5.88] words:

| at frac -2 | rate | mean words | gained | lost | McNemar vs frac 0 |
|---|---:|---:|---:|---:|---:|
| `rand_ctrl_0` | 0.281 | 4.34 | 2 | 1 | 1.000 |
| `rand_ctrl_1` | 0.297 | 5.88 | 4 | 2 | 0.688 |
| `rand_ctrl_2` | 0.250 | 5.62 | 4 | 5 | 1.000 |
| **`jtw_mean_diff_tgt`** | **0.500** | **18.58** | 16 | 1 | **2.75e-4** |

And the registered test itself, which is not "the control is flat" but "the direction beats the
control at the same dose", paired on the same questions: **16v2 (p = 0.00131), 15v2
(p = 0.00235), 17v1 (p = 0.000145)**, all three clearing Bonferroni at 0.0167.

Written up in [`Q2_TRUTHFULQA_STEERING.md`](Q2_TRUTHFULQA_STEERING.md).

### 2.5 The three qualifications, which are not optional

**(a) The effect runs through answer length.** Mean length runs 2.53 words at +2 eps\* to 18.58
at -2, monotone across the entire sweep including the positive half where truthfulness does not
move. Adding `-frac` to a logistic model that already carries word count gives **chi2 = 0.146,
p = 0.702**.

The correct reading of that null is narrow, and it is easy to overstate in both directions.
Length is a **mediator**, not a confounder: steer produces longer answer produces judged-truthful.
Adjusting for a mediator can never show the intervention had no effect, and the paired McNemar
already establishes that it did. What the null shows is that the direction is **not specifically
about truth**; whatever it does, it does through length. The control then settles the remaining
question: perturbing by this norm at this layer buys about **1.2 words**, and the other **12.7**
belong to this direction specifically. So the discursiveness is this direction's doing. What is
still not separated is "a representation of truth" from "a representation of hedge, qualify,
decline to assert", and TruthfulQA rewards the second. **That is a separate experiment, not a
reinterpretation of this one.**

**(b) The certificate itself was never tested**, and this is the part that matters most for
note 2.

*Sign.* eps\* is **+14.7065**, so the certificate points at `+frac`, which is precisely where
nothing happens. All the movement is at `-frac`. Traced to Q0's polarity choice: label 1 =
untruthful, so `mean_diff` points toward untruthful and the truthful direction is `-v`. The
readout and the behavior agree with each other; both disagree with the certificate's sign.

*Magnitude.* The readout `mean_g_read` never reaches zero anywhere on the sweep:

| frac | -2 | -1 | 0 | +1 | +2 |
|---|---:|---:|---:|---:|---:|
| `mean_g_read` | 15.05 | 26.45 | 37.88 | 48.13 | 56.39 |
| `crossed` | 0 | 0 | 0 | 0 | 0 |

The slope is about 10.84 per frac unit off a baseline of 37.88, so crossing g = 0 needs roughly
**-3.5 eps\***, not -1. The sweep was never long enough to test the certificate's actual claim.

**(c) The 2x2 cell is `inert`, and `inert` is not a null.** The four cells are: `actuatable` =
crossing the certified boundary changes behavior; `readout-only` = crossing moves the readout but
not behavior; `inert` = **behavior moves without a readout crossing**, an off-target effect;
`no-crossing` = neither. `crossed = 0` at every dose with `delta_vs_baseline = 0.234` at frac -2
is exactly the third cell. The behavior moved and the certificate had nothing to do with it.

### 2.6 So did note 1 do what it was supposed to?

Yes, and more sharply than expected, because it separated two things that had been fused.

The note's purpose was to control for the concept. It did: on a dataset with real headroom, a
truth direction moves behavior, paired, dose-ordered, and beating norm-matched controls. **Truth
is steerable in gemma-2-2b.** Four months of nulls on `cities` and `common_claim` were not
telling us that truth is unsteerable.

But it also demonstrated that **the reachability certificate is not what made it work**. Both
halves of that are results. The first retires a live alternative explanation for every previous
null. The second says the machinery note 2 asks about is not merely unvalidated, it is
measurably wrong on this dataset in both sign and magnitude, and it needs the closed check.

---

## 3. Note 2: validate the pipeline, forget DCT

### 3.1 The source of the problem, from artifacts we already had

The note's operative clause is *find where the source of the problem is*. It is answerable
without new GPU time, from `reach_samepoint_summary_<ds>.csv`, which measures two ratios per
statement: `calibration` is realized over predicted gain **at the point the Jacobian was
linearized about**, and `context_factor` is the same ratio **at the generation stem**.

| dataset | calibration (median) | p10 to p90 | context_factor | shortfall from context alone |
|---|---:|---|---:|---:|
| cities | **0.9997** | 0.982 to 1.023 | **0.0729** | **13.7x** |
| common_claim | 0.7025 | 0.536 to 0.913 | 0.1944 | 5.1x |

**On cities the pullback is essentially exact.** Whatever is wrong, it is not the linear algebra.
The entire 8-to-35x shortfall the audit reported is the move from the linearization point to the
generation stem. On common_claim calibration is 0.70 even at its own point, which its trust
radius already predicted: eps\* of 10.69 against a trust radius of 4.87 is extrapolation, so
that dataset fails for a second and separate reason.

**This is the answer to the note, and it is dataset-dependent.** Combined with 2.1, the diagnosis
is: the pipeline is arithmetically correct and is being pointed at the wrong population with the
wrong readout.

That reframes an old claim in our favour. "Realized gain falls 8 to 35x short" becomes "the
pullback is exact at its own point, calibration 0.9997, and the loss is entirely the context
shift, 13.7x." That is a better result than the one it replaces, and a cleaner one to hand over.

### 3.2 The closed check the note actually asks for (V1)

The diagnosis above is inference from two ratios. The note asks for something decisive, and
decisive means **closing the loop**: predict, intervene, measure the same quantity, compare.

Every reachability number this project has produced is a **prediction**. eps\* = g / ||J^T w||
says how far to push. Nothing has ever measured the realized crossing against it. `reach_control`
gets closest and its `crossed` column is 0 at every dose, so it never got the chance.

**The design, exactly as the note specifies it.** Put the target at the **output layer** with a
**contrastive** readout,

    a = unit( mean W_U[yes first-tokens] - mean W_U[no first-tokens] )

which `reach_jlens.py` already builds as `verdict`. This removes the readout as a confound by
construction: at the output layer the readout *is* the decision, so a readout-versus-behavior
dissociation cannot occur. Then sweep the source layer, and at each layer steer by the
minimum-norm certificate and compare realized against predicted change in `a . z`.

`src/reach_validate.py` is written and its 15 tests pass. It measures four ratios, not the
plan's two:

| ratio | certificate from | evaluated at | what it isolates | expected |
|---|---|---|---|---|
| `ratio_A` | declarative | declarative | the pullback arithmetic alone | near 1 |
| `ratio_B` | declarative | generation stem | the context shift, note 2's point B | near 0.07 |
| `ratio_B_own` | stem | stem | whether linearization is broken *at* the stem | unknown |
| `ratio_Q_own` | question prompt | question prompt | the decision context, with a behavioral closure | unknown |

**The last two are additions to the plan and I want to flag them as such.**

`ratio_B_own` exists because `ratio_B` alone cannot distinguish "the certificate does not
transport across contexts" from "linearization is simply broken at the stem". Those are
different bugs with different fixes and the two-point design conflates them.

`ratio_Q_own` exists because of a gap I found in the plan while implementing it. The claim that
justifies the whole design, that the readout cannot dissociate from behavior, **only holds at the
question prompt**, where the next token really is the verdict. At a bare declarative statement
the model is not being asked to emit yes or no, so `a . z` there is a probe like any other. The
plan's two points never visit the context where its own central claim is true. So V1 adds that
context and closes it behaviorally: a 2x2 of whether `a . z` crossed zero against whether the
argmax over the yes/no token sets actually flipped. If the readout really is the decision, the
off-diagonal cells are empty, and that is a measurement rather than an assumption.

**V1 is also much cheaper than the plan estimated** ("a few GPU-hours"). One backward pass per
batch yields `J_l^T a` at every layer at once, so the pullback is not per-layer at all. The only
per-layer cost is the steered forward passes.

`ratio_A` is a **gate**, not a number to report. If it is off 1 by more than 10% at any layer,
the pullback and the injection are not the same operator and every other column is measuring
that disagreement rather than measuring the model. The analyze stage refuses to pass on a NaN
rather than falling through to the reassuring branch.

**Status: written, tested, uncommitted, not yet run on a GPU.** This is the main thing this
window did not finish.

### 3.3 Note 2's last clause: "change the feature vector construction to something that works"

Three candidates, and the evidence has already eliminated one.

1. ~~**Fit the direction on the generation population.**~~ **Ruled out** by section 2.1 on our
   old data: at 9 and 10 false completions there is no stem-fitted direction to find, linear or
   nonlinear. The XGBoost arm makes this concrete: the +0.062 nonlinear gap common_claim shows
   on declarative statements at the target layer is **+0.000** on generation stems. The question
   is worth re-asking on TruthfulQA, where the negatives are not nine.
2. **The contrastive verdict readout.** A token-space object, so it cannot dissociate. Already
   implemented, and it is what V1 uses.
3. **A-LQR's adaptive semantic setpoint.** Their code is public; see section 9.

**DCT and MAG are dormant, as the note directs.** Nothing in either track needs them. Existing
DCT results stand as recorded and no new DCT work is planned.

---

## 4. Note 3: the readable formulation

**Delivered: [`DERIVATION_SHORT.md`](DERIVATION_SHORT.md), 226 lines, six sections.**
`math_map.tex` stays as the reference and is unchanged; this is the front-end that points into
it. It runs input layer to output layer in a single thread and does four things:

1. **The forward map with the linear stages marked.** Of five stages, exactly two are linear
   (embed, unembed). The final RMSNorm is nonlinear but degree-zero homogeneous, so its Jacobian
   is closed form. Softcapping is monotone coordinatewise, so it is **argmax-irrelevant**: every
   claim about *which token is emitted* can be made about raw logits, and no claim about
   *probabilities* can.
2. **The certificate stated once, then instantiated twice.** The probe halfspace and the argmax
   cone are shown to be the same formula with a different readout vector. This was the actual
   ask, and it is the thing that had been re-derived ad hoc per experiment.
3. **A table of where each measured number plugs in:** eps\*, the trust radius, calibration,
   context_factor.
4. **One line on what the halfspace target set actually is**, and why S4 found it vacuous.

It adds one thing the note did not ask for, because writing the two side by side made the
distinction unavoidable: **the two emptiness results are different results** and had been
described with one sentence. The probe halfspace is not semantically empty so much as
**unlocated**, because its readout is at balanced accuracy 0.500 on the population we steer. The
token-space target set is the one S4 measured as semantically **vacuous**, and it is vacuous for
two different reasons on the two datasets: minimum norm collapsing onto directional modifiers on
`cities`, and `target_mode=runnerup` never aiming at falsity at all on `common_claim`.

---

## 5. The infrastructure that had to exist first

None of section 2 was runnable when the notes arrived. What was built:

| file | what it does |
|---|---|
| `src/prep_truthfulqa.py` | the Q0 dataset build, chat template applied once at build time |
| `src/tqa_baseline.py` | unsteered baseline, `first_answer` truncation, gold-answer judge validation |
| `src/truthfulqa_judge.py` | the two-judge scorer, arm registry, continuation-arm handling |
| `src/tqa_q2_analyze.py` | recomputes every Q2 number from the CSVs, imports no torch |
| `src/reach_steer.py` | the `rand_ctrl` arm, and the `stem_of` newline fix |
| `deltaai/run_tqa_baseline.slurm` | Q1 |
| `deltaai/run_truthfulqa_reach.slurm`, `run_truthfulqa_judge.slurm` | Q2 mean arm, chained |
| `deltaai/run_truthfulqa_randctrl.slurm` | the registered control |
| `deltaai/run_truthfulqa_stmt.slurm` | the per-statement rerun (has not succeeded) |
| `src/reach_validate.py` | V1, written and tested, not yet run |

Plus 4 new test modules, 954 lines. **Suite: 551 passed, 1 skipped.**

**Three infrastructure decisions that changed results, not just ergonomics.**

*The newline fix.* `dct_steer_utils.generate` flattened newlines out of the completion. Q1
measured `n_truncated = 64/64`, so on **100% of rows** the judged text would have included the
model's fabricated next turn. `reach_steer` now writes a separate `answer` column cut at the
first newline, and `truthfulqa_judge` **refuses to run** on a CSV that lacks it rather than
silently judging the wrong string.

*Numbers come from a script, not a transcript.* `tqa_q2_analyze.py` recomputes every number in
the Q2 doc from the CSVs and writes `tqa_q2_summary_truthfulqa.csv`. Nothing in that document is
quoted from a session log.

*The scale-0 determinism oracle.* Every arm regenerates the unsteered block, and the control job
asserts all three of its scale-0 blocks are byte-identical to each other **and** to the mean
arm's, 64 of 64 on all three comparisons. This is what makes the control share a baseline with
the experiment rather than merely resemble one, and it is also what would catch the steering hook
leaking state between directions.

---

## 6. The cluster runs

| job | what | outcome |
|---|---|---|
| 3081925 | Q1 baseline + gold judge validation | **clean.** 0.266 truthful-and-informative, judges 1.000/1.000 |
| 3082192 | Q2 mean arm generation, 1152 rows | **clean** |
| 3082210 | judge the mean arm, then the stmt arm | mean arm judged; **died** on the stmt arm's malformed prompts |
| 3083390 | randctrl generation, 1728 rows, 40 GPU-min | generation clean; **died** in the judge stage on an argparse bug |
| 3084307 | judge the control that 3083390 already generated | **clean.** Oracle passed, control flat |
| 3084471 | rerun the per-statement arm | **died in 10 seconds** at the guard, see below |

**Two of six jobs died after doing real work, and in both cases the fix was to not redo the
work.** 3083390 had already spent 40 GPU-minutes generating 1728 completions when its judge stage
hit an argparse error. Job 3084307 judged those existing completions instead of regenerating
them, so the control arm reported in section 2.4 is one generation, judged once. The rule the
jobs now follow is that a stage skips when its output already exists and is non-empty.

**Job 3084471, the one that did not land.** The per-statement arm's prompts were malformed by the
`stem_of` newline bug. The rerun job's stage 0 triages by *reading* the file rather than assuming:
a CSV whose prompts carry the newline is a post-fix run and is left alone; a CSV without it is the
malformed one and is archived. It found the malformed file **and** an existing archive of it, and
refused to overwrite the archive:

    [stage 0] reach_steer_stmt_truthfulqa.csv: 960 rows, 0 prompts carry the newline
    !!!! reach_steer_stmt_truthfulqa.malformed-prompts.csv already exists. Refusing to
         overwrite the archive; move or remove it by hand if you really mean to.

The two files are byte-identical (verified by md5 on both machines), so nothing is at risk; the
guard is doing its job, it is simply guarding against a state a previous partial run created. The
arm is **exploratory and no result depends on it**: it draws 200 statements from the same
population the direction was fit on, which is the D2 confound rather than a second confirmation.
`Q2_TRUTHFULQA_STEERING.md` section 9 records it as void, and that stays accurate.

---

## 7. What it all means

**The headline.** Steering gemma-2-2b along `J^T w` from `mean_diff_tgt` at 2 eps\* toward
truthful raises the truthful-and-informative rate on 64 held-out TruthfulQA questions from
**0.266 to 0.500**, paired, dose-ordered, sign-asymmetric, beating three norm-matched random
directions on the same questions, with informativeness intact. The registered bar is met.

**The contrast worth carrying is between datasets, not within this one.** Same pipeline, same
certificate machinery, three concepts, three different failure modes:

| concept | what the certificate did | what the behavior did | cell |
|---|---|---|---|
| `cities` | crossed the boundary, inside its trust radius | did not move | `readout-only` |
| TruthfulQA | never crossed (3.5x short, wrong sign) | **moved, p = 2.75e-4** | `inert` |
| refusal | crossed | **moved** | `actuatable` |

Two of the three truth datasets fail in *opposite* directions, and the one non-truth concept
passes. That pattern is the actual finding of the last two months, and it is more informative
than any of the three results alone. It says the readout and the behavior are coupled by
something the current certificate does not capture, in a way that is not a single consistent bias.

**What this does to existing claims: nothing is retracted, the cause is relocated.** The truth
null on cities stands, and the dissociation stands, because the cities readout genuinely crossed
inside its trust radius. The mechanism claim gets sharper and better. The D2 finding gets much
stronger: it was "the baseline readout signs differ" and it is now "the readout is at chance on
the generation population, AUC 0.510, and never changes sign", which we believe is unclaimed in
the literature.

**One caveat I want to volunteer rather than have found.** The refusal positive control changed
**two** variables relative to the truth run, not one: the concept *and* the linearization point.
`prep_refusal` templates once at build time, so its linearization point is the generation prompt's
last token, which is also where behavior is produced; the truth track fit on declarative
statements and read on generation prefixes. Section 3.1 prices that mismatch alone at 13.7x. So
the matched-comparison claim in the 2x2 is weaker than it reads. V1 is what closes it.

---

## 8. The honest column

What is **not** established by anything above:

- **That the direction represents truth rather than hedging.** Section 2.5(a). The control kills
  "any perturbation makes it verbose"; it does not distinguish "truth" from "qualify, hedge,
  decline to assert", and TruthfulQA rewards the second. This needs its own experiment.
- **Anything about the certificate on TruthfulQA.** The sweep never crossed g = 0. A negative
  result about the certificate's predictive content would need a sweep out to about -3.5 eps\*,
  and this was not that.
- **That the pullback is correct end to end.** V1 is written and has not run. Until `ratio_A`
  comes back near 1, the possibility that the pullback and the injection are different operators
  is open, and it would make every ratio in the project a measurement of that disagreement.
- **That TruthfulQA has a linear truth readout at all.** `reach_margins` gates on
  `MIN_ACC_1D = 0.6`, and **3 of 74 directions clear it** on TruthfulQA. The other 71 cannot read
  TruthfulQA truth at 60% accuracy in one dimension. This is a separate finding, and it is
  unchased.
- **Generality.** One model, one decoder setting, 64 questions, one 48-token budget. The budget is
  part of the finding: at frac -2 answers are 18.58 words, within budget, but a longer budget
  could change what hedging costs.
- **The judges at dose.** They were validated on gold answers, not on steered text, and steered
  answers are further from the gold distribution than unsteered ones.

---

## 9. What I want to do next, in order

**1. Run V1.** It is written, tested, and cheap (one backward per batch gives every layer at
once). It is the decisive version of note 2, it closes the caveat in section 7, and its `ratio_A`
gate can invalidate a lot of existing numbers cheaply, which is a good property for a next
experiment to have. Needs a slurm job and `docs/V1_PIPELINE_VALIDATION.md`.

**2. Run A-LQR's code on gemma-2-2b TruthfulQA.** Public at
`github.com/trustworthyrobotics/lqr-activation-steering`, 94 commits, with task areas covering
toxicity, truthfulness and refusal, and data-collection scripts referencing gemma2b. Roughly a
day of setup. The reason to do it before more of ours: without a known-good number from a
closed-loop controller on the same model and dataset, a null from our open-loop pipeline is
uninterpretable, and that has been the recurring failure mode.

*(A correction to carry into the meeting: `PLAIN_ENGLISH_WALKTHROUGH.md` section 5.3 still lists
"access to the A-LQR code" as an open ask. That ask is void, the code is public on the author's
own GitHub, and asking for it would have been embarrassing.)*

**3. Separate truth from hedging.** The one experiment that would turn section 2.4 from a
steering result into a *truth* steering result. The shape I would use: a task where hedging is
penalized rather than rewarded, or a length-matched comparison where the control is forced to the
same word count. Worth designing before running.

**4. Chase the 3-of-74 readout finding.** If only three directions can read TruthfulQA truth
one-dimensionally, that is either a fact about the dataset or a fact about how we build direction
candidates, and both are interesting.

**5. Extend the Q2 sweep to -3.5 eps\*** so the certificate's actual claim gets tested on the one
dataset where behavior demonstrably moves. Cheap, and it converts "the certificate was not tested"
into a result either way.

**Not planned:** any new DCT work, and any rerun of the per-statement arm unless someone wants it
for its own sake.

---

## 10. Where the detail is

| document | what it holds |
|---|---|
| [`PLAN_ADVISOR_NOTES_2026-09.md`](PLAN_ADVISOR_NOTES_2026-09.md) | the plan this executes, with the V/Q/M track structure and costs |
| [`DERIVATION_SHORT.md`](DERIVATION_SHORT.md) | note 3's deliverable; `math_map.tex` is the reference behind it |
| [`D2_PREFIX_TRANSFER.md`](D2_PREFIX_TRANSFER.md) | the readout-at-chance diagnosis, section 2.1 |
| [`Q0_TRUTHFULQA_DATASET.md`](Q0_TRUTHFULQA_DATASET.md) | the dataset build and the polarity decision |
| [`Q1_TRUTHFULQA_BASELINE.md`](Q1_TRUTHFULQA_BASELINE.md) | the baseline, the gate, and the judge validation |
| [`Q2_TRUTHFULQA_STEERING.md`](Q2_TRUTHFULQA_STEERING.md) | the steering result in full, with the control and all caveats |
| [`REFUSAL_POSITIVE_CONTROL.md`](REFUSAL_POSITIVE_CONTROL.md) | the `actuatable` cell, for the three-dataset contrast |
| [`REACH_AUDIT_FINDINGS.md`](REACH_AUDIT_FINDINGS.md) | the original audit the diagnosis relocates |

Everything in sections 2.4 and 2.5 can be recomputed on LAPTOP, no GPU and no torch:

    PYTHONPATH=src ./.venv/bin/python src/tqa_q2_analyze.py --dataset truthfulqa --arms mean randctrl
