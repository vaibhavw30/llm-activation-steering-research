# Plan from the advisor's notes, September 2026

*Three notes came back from the last review. This maps each to what our own artifacts already
say, then to a concrete experiment. Every number below was recomputed read-only from a committed
artifact on 2026-09-04; nothing was regenerated and no GPU was spent.*

**The notes, compressed.**

1. Switch to the dataset Julian used, TruthfulQA. Misleading prompts steered toward truthful.
   Two affiliated LLM judges plus a coherence measure. *Good to normalize on things that should
   be steerable.*
2. Forget DCT. Validate the Jacobian halfspace scoring pipeline directly, with contrastive
   vectors, mapping input layer to output layer. Find where the source of the problem is. Change
   the feature vector construction to something that works.
3. Write out the halfspace and the linear mapping in a form easier to follow than the current one.

---

## 0. The one-paragraph version

Our own artifacts already contain the diagnosis the second note asks for, and it exonerates the
Jacobian. At the point where the Jacobian is computed, predicted gain and realized gain agree at
**calibration 0.9997** on cities. The entire 8-to-35x shortfall is the move from that point to the
generation stem, worth **13.7x on its own**. And on the generation population the probe readout is
at chance: **AUC 0.510**, balanced accuracy **0.500**, with the readout never even changing sign.
So the pipeline is arithmetically correct and is being pointed at the wrong population with the
wrong readout. That is exactly the problem the first note fixes, because TruthfulQA is a
generation task, so the population we fit on and the population we steer are the same object by
construction. The plan is therefore: finish the free diagnostic (V0), validate the pipeline
end-to-end at the output layer where the readout cannot lie (V1), and rebuild the truth track on
TruthfulQA with the dataset-build discipline that made the refusal control work (Q0 to Q3).

---

## 1. What our own data already says

### 1.1 The Jacobian is not the problem

`reach_samepoint_summary_<ds>.csv` measures two ratios per statement. `calibration` is realized
slope over predicted slope **at the same point the Jacobian was linearized about**.
`context_factor` is the same ratio **at the generation stem**.

| dataset | calibration (median) | p10 to p90 | context_factor (median) | shortfall from context alone |
|---|---:|---|---:|---:|
| cities | **0.9997** | 0.982 to 1.023 | **0.0729** | **13.7x** |
| common_claim | 0.7025 | 0.536 to 0.913 | 0.1944 | 5.1x |

On cities the pullback is essentially exact. Whatever is wrong, it is not the linear algebra.
On common_claim calibration is 0.70 even at its own point, which is the same story its trust
radius already told: `eps*` 10.69 against a trust radius of 4.87 is extrapolation, so that dataset
fails for a second, separate reason.

**This is the answer to "find where the source of the problem is," and it is dataset-dependent.**

### 1.2 The readout is at chance on the population we actually steer

`reach_stemprobe.py` extracted target-layer activations at each generation stem's last token in
July, and the activations are still on disk (`reach_stemacts_<ds>.npz`). The fit stage prints its
statistics and never saved them, so they were lost. Recomputed read-only:

| dataset | n | base rate P(TRUE) | old threshold acc | balanced acc | AUC of the old readout | refit LR, 5-fold CV |
|---|---:|---:|---:|---:|---:|---:|
| cities | 158 | 0.943 | 0.057 | **0.500** | **0.510** | 0.930 |
| common_claim | 178 | 0.944 | 0.944 | **0.500** | **0.568** | 0.944 |

Three things to read off this.

- **Balanced accuracy is exactly 0.500 on both.** The old boundary carries no information about
  whether the model will complete the stem truthfully.
- **The readout never changes sign.** Every cities stem reads between -83.5 and -43.1, so every
  one is "deep FALSE" while 94% of them are completed truthfully. Every common_claim stem reads
  between +48.9 and +118.9. The readout is dominated by a **prompt-format offset** that swamps the
  truth signal. This is stronger than "the boundary shifted": there is no threshold on this
  direction that separates the classes.
- **The refits recover only the base rate.** LR at 0.930 and 0.944 against base rates of 0.943 and
  0.944 is the majority-class predictor. Refitting the same direction on the new population does
  not rescue it.

### 1.3 The base rate is itself a reason to change datasets

The honest caveat on 1.2 is that it is underpowered: only **9 of 158** cities stems and **10 of
178** common_claim stems are completed falsely. But that weakness is the point. **The model
spontaneously lies about 5.7% of the time on our data.** There is almost no behavior to move, so
even a working actuator would have little room to show it, and any effect we did measure would
rest on nine examples.

This is the quantitative case for the first note. We have been trying to induce a rare behavior in
a population where the readout has no signal. TruthfulQA is adversarially constructed so that the
model produces false answers at a high rate, which is the headroom our datasets do not have.

### 1.4 Why the refusal control worked, restated

`prep_refusal.apply_template` renders every instruction through the chat template **once, at
dataset-build time**, so extraction, margins, and steering all tokenize the identical string. The
linearization point is therefore the generation prompt's last token, which is also the point where
behavior is produced. The truth track fits on full declarative statements and reads on generation
prefixes, and section 1.1 says that mismatch alone costs 13.7x.

**So the refusal control changed two variables relative to the truth run, not one: the concept and
the linearization point.** That is a real gap in the matched-comparison claim and it should be
volunteered rather than discovered. It is also the reason to expect a TruthfulQA rebuild to
behave differently even before any claim about truth.

---

## 2. Track V: validate the pipeline (note 2)

### V0. Finish D2 and save it. Free, CPU, today

The activations are cached. `reach_stemprobe.py --fit` prints its numbers and writes only four
columns, so add the statistics to the CSV and write the results doc. Add the XGBoost arm, which
was skipped because xgboost is not installed in the cluster env, to check whether the generation
population has non-linear headroom the linear probe misses. Run it in its own process; xgboost and
torch cannot share one.

**Deliverable:** `docs/D2_PREFIX_TRANSFER.md`, plus the table in 1.2 as a committed artifact.
**Cost:** minutes, LAPTOP.

### V1. The end-to-end validation the note actually asks for

**The design.** Put the target at the **output layer**, where the readout is the decision and
therefore cannot dissociate from behavior, and use a **contrastive** readout vector. Then sweep
the source layer and, at each one, compare what the Jacobian predicts against what actually
happens. The readout is

    a = unit( mean W_U[yes first-tokens] - mean W_U[no first-tokens] )

which `reach_jlens.py` already builds as `verdict`. For each source layer l, the certificate is
the same formula we always use, `eps*(l) = margin / ||J_l^T a||`, and `token_jac.py` already
computes those margins for all 26 layers.

**What is new is the closed check.** For each layer, steer at `eps*(l)` and measure the realized
change in `a . z` against the predicted change, at **two evaluation points**:

| point | what it isolates | expected from 1.1 |
|---|---|---|
| A, the declarative statement (where J was linearized) | the pullback arithmetic | ratio near 1 |
| B, the generation stem | the context shift | ratio near 0.07 |

The per-layer gap between A and B is the diagnosis, and it is also the actionable output: **if any
source layer holds its gain across the context shift, that is the layer to steer from.** S2 says
controllability peaks at layer 0 and falls 7.35x by layer 25, but S2 measured margins, not
context-robustness, so this is not yet known.

**Why this is the right validation.** It removes the readout as a confound by construction, so a
failure at point A would be a genuine bug in the pullback and a failure only at point B localizes
the problem to the linearization point. Either outcome is decisive, which the current evidence is
not.

**Deliverable:** `src/reach_validate.py`, `docs/V1_PIPELINE_VALIDATION.md`.
**Cost:** one vjp pass per layer per statement at two contexts. CLUSTER, a few GPU-hours.

### V2. Feature vector construction

The note's last clause, "change the feature vector construction to something that works," is
downstream of V1. Three candidates, in the order the evidence supports:

1. **Fit on the generation population.** The direction fit on stems rather than statements. V0
   tells us whether that direction exists at all.
2. **The contrastive verdict readout.** A token-space object, so it cannot dissociate. Already
   implemented.
3. **A-LQR's adaptive semantic setpoint.** Their code is public; see Q3.

**DCT and MAG go dormant.** The note says forget DCT, and nothing in tracks V or Q needs it. The
existing DCT results stand as recorded; no new DCT work is planned.

---

## 3. Track Q: TruthfulQA (note 1)

### Why this fixes three problems at once

| our current problem | evidence | what TQA changes |
|---|---|---|
| readout fit on a different population than we steer | AUC 0.510 | Q&A format, so the fit unit and the steer unit are the same string |
| 13.7x gain loss from the context shift | context_factor 0.073 | template once at build time, as `prep_refusal` does |
| no behavioral headroom | 5.7% spontaneous-false | TQA is adversarial by construction |
| no known-good reference | n/a | A-LQR reports TQA steering on gemma2b |

The last row is what the advisor means by *normalize on things that should be steerable*. Refusal
was a positive control on the **instrument**. TruthfulQA would be a positive control on the
**concept**, which is the axis we have never controlled for.

### The experiments

**Q0. Dataset build.** `src/prep_truthfulqa.py`, modeled directly on `prep_refusal.py`. Load
`truthful_qa`, apply the chat template once at build time, write `got_datasets/truthfulqa.csv`
plus a holdout that never enters direction fitting. The contrastive pair for the direction is
(question + correct answer) against (question + incorrect answer), so the direction is fit in the
generation format.

**Q1. Baseline, before any steering.** Measure gemma-2-2b's unsteered truthful rate and
informative rate on TQA. This is the number that decides whether the track is viable at all, and
it must be measured rather than assumed. **If the base rate is near ceiling as it is on cities,
stop and reconsider.**

**Q2. Steer and judge.** Certificate, steer toward truthful, generate at T=0, judge with the
existing `TruthJudge` in `src/judges/local_hf.py`, which already wraps
`allenai/truthfulqa-truth-judge-llama2-7B` and `allenai/truthfulqa-info-judge-llama2-7B`. That is
the two-judge plus informativeness structure the note describes, and the adapter and its tests are
already written. Then `reach_control.py --dataset truthfulqa` names the 2x2 cell.

**The bar,** by analogy with how S4 calibrated T1: beat a norm-matched random control at the same
dose, which is the D1 test, with informativeness not collapsing. Steering that raises truthfulness
by breaking the model is the failure mode D1 caught before and the info-judge is what catches it
here.

**Q3. A-LQR as the reference implementation.** The code is **public** at
`github.com/trustworthyrobotics/lqr-activation-steering`, 94 commits, with task areas covering
toxicity, truthfulness (TQA), concept steering, and refusal, and data-collection scripts
referencing `gemma2b`. `LITERATURE.md` recorded this on 22 August.

> **Correction to carry into the next meeting:** `PLAIN_ENGLISH_WALKTHROUGH.md` section 5.3 still
> lists "access to the A-LQR code" as an open ask. That ask is void. Asking for code that is
> public on the author's own GitHub is the kind of thing to catch before a meeting, not during
> one.

**Recommended order: run their code first.** Establish the reference number on gemma-2-2b TQA with
the closed-loop controller, and only then run ours. Without the known-good number, a null from our
open-loop pipeline is uninterpretable, which is precisely the trap the last four months were spent
climbing out of.

---

## 4. Track M: the readable derivation (note 3)

`docs/math_map.tex` already exists, is 494 lines, and is correct. It is a paper-style document
with full notation, twelve sections, and a diagnosis section. The note is not that it is missing;
it is that it is hard to follow.

**Write a short front-end**, two to three pages, that goes input layer to output layer in one
thread and does only this:

1. the forward map, with what is linear and what is not marked in one place
2. the certificate stated **once**, then instantiated as the probe halfspace and as the argmax
   cone, showing they are the same formula with a different readout vector
3. where each measured number plugs in: `eps*`, the trust radius, calibration, context_factor
4. the one-line statement of what the halfspace target set actually is, and why S4 found it vacuous

Keep `math_map.tex` as the reference; the new note points into it.
**Deliverable:** `docs/DERIVATION_SHORT.md` or a `\section` reorganisation of the .tex.

---

## 5. Sequencing and cost

| id | Experiment | Machine | Cost | Blocks |
|---|---|---|---|---|
| V0 | finish D2, save the numbers | LAPTOP | minutes | nothing |
| Q1 | TQA unsteered baseline | CLUSTER | ~1 GPU-hr | the whole Q track |
| M | the readable derivation | LAPTOP | a day of writing | nothing |
| V1 | pipeline validation, layer sweep, two contexts | CLUSTER | a few GPU-hr | V2 |
| Q0 | TQA dataset build | LAPTOP + 1 GPU pass | ~1 GPU-hr | Q2 |
| Q3 | run A-LQR's code on gemma-2-2b TQA | CLUSTER | ~1 day setup | interpreting Q2 |
| Q2 | our TQA steer and judge | CLUSTER | ~4 GPU-hr | the verdict |
| V2 | feature construction, chosen by V1 | CLUSTER | TBD | the verdict |

**Do V0 and Q1 first.** Both are cheap and either can kill a track. V0 tells us whether a
generation-population direction exists at all; Q1 tells us whether TQA has the headroom the whole
first note assumes. Everything else is conditional on those two.

**T1 is deprioritized but not dropped.** S4 calibrated it at beating 5.5%. If the Q track shows
truth is steerable in the right framing, T1's question changes from "can we build a non-vacuous
target set" to "do we still need one," so it should wait for Q1.

---

## 6. Decisions needed

1. **Run A-LQR's code first, or ours first?** Recommendation: theirs. A null from our pipeline is
   uninterpretable without a known-good reference on the same model and dataset, and that has been
   the recurring failure mode.
2. **TQA generation or multiple-choice?** Their repo's choice should decide it, so this resolves
   when we read their code. Generation is what our judges expect.
3. **Does the 2x2 survive?** It currently changes two variables, concept and linearization point
   (section 1.4). Either add the caveat and keep it, or close it with V1 plus a stem-fit truth
   direction. Recommendation: state the caveat now, close it with V1.
4. **How much of the old truth track gets re-run** under the corrected linearization point, versus
   reported as-is with the diagnosis attached.

---

## 7. What this does to existing claims

Nothing here retracts a result. It relocates the cause.

- **The truth null stands**, and the dissociation stands, because the cities readout genuinely
  crossed inside its trust radius.
- **The mechanism claim gets sharper.** "Realized gain falls 8 to 35x short" becomes "the pullback
  is exact at its own point, calibration 0.9997, and the loss is the context shift, 13.7x." That is
  a better result, not a worse one, and it is a cleaner thing to hand to Julian.
- **The D2 finding gets much stronger.** It was "the baseline readout signs differ." It is now
  "the readout is at chance on the generation population, AUC 0.510, and never changes sign." That
  is the finding we believe is unclaimed in the literature and it is now quantified.
- **The 2x2 acquires a caveat** that we should state ourselves.
