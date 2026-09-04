# Q2. Steering gemma-2-2b along the certificate on TruthfulQA

*Mean arm measured 2026-09-04 on CLUSTER, SLURM job 3082192 (steer) and the judge stage of the
chained job. Norm-matched random control: SLURM job 3083390, PENDING at the time of writing.
Scripts: [`src/reach_steer.py`](../src/reach_steer.py),
[`src/truthfulqa_judge.py`](../src/truthfulqa_judge.py),
[`src/reach_control.py`](../src/reach_control.py),
[`src/tqa_q2_analyze.py`](../src/tqa_q2_analyze.py). Jobs:
[`deltaai/run_truthfulqa_reach.slurm`](../deltaai/run_truthfulqa_reach.slurm),
[`deltaai/run_truthfulqa_judge.slurm`](../deltaai/run_truthfulqa_judge.slurm),
[`deltaai/run_truthfulqa_randctrl.slurm`](../deltaai/run_truthfulqa_randctrl.slurm).
This is step Q2 of [`PLAN_ADVISOR_NOTES_2026-09.md`](PLAN_ADVISOR_NOTES_2026-09.md) section 3.
Q1 opened the gate: [`Q1_TRUTHFULQA_BASELINE.md`](Q1_TRUTHFULQA_BASELINE.md).*

> **STATUS: DRAFT. The headline is deliberately not written yet.** Sections 3 through 6 are final
> and their numbers will not change. Section 7 is the norm-matched random control, which is the
> experiment that decides what sections 3 through 6 mean, and it is still on the cluster. Do not
> quote a headline out of this file until section 7 has numbers in it.

## 1. The question, and the bar it was registered against

The plan states both:

> **Q2. Steer and judge.** Certificate, steer toward truthful, generate at T=0, judge with the
> existing `TruthJudge` in `src/judges/local_hf.py` [...] Then `reach_control.py --dataset
> truthfulqa` names the 2x2 cell.
>
> **The bar,** by analogy with how S4 calibrated T1: beat a norm-matched random control at the
> same dose, which is the D1 test, with informativeness not collapsing.

Two registrations carried in from Q1, both honoured:

- **Paired, not unpaired.** The same 64 held-out questions appear at every dose, so the analysis is
  exact McNemar against the frac-0 baseline. Unpaired at n=64 would need the steered rate to reach
  0.51 before clearing significance; paired needs 6 one-directional flips for p = 0.031.
- **`first_answer` on the raw decode.** Q1 measured `n_truncated = 64/64`, so the flattened string
  `dct_steer_utils.generate` returns would have corrupted the judged text on every row.

## 2. What was run

Three arms, all on the same 64 held-out questions, the same greedy decoder, and the same
`max_new_tokens=48` budget Q1 used, so Q1's 0.266 is a valid frac-0 control for all of them.

| arm | directions | doses | what it is for |
|---|---|---|---|
| `mean` | `jtw_mean_diff_tgt`, `jtw_probe_grad_tgt` | 0, +-0.5, +-1, +-1.5, +-2 x eps* | the experiment |
| `stmt` | per-statement | as above | VOID for this run, see section 8 |
| `randctrl` | 3 random unit vectors, `RAND_CTRL_SEED=7` | the mean arm's exact scale grid | the bar in section 1 |

The control's grid is the mean arm's grid by construction, not by coincidence:
`[0, +-7.353, +-14.707, +-22.06, +-29.413]`, so its perturbation norms match dose for dose. Its
scale-0 block is asserted byte-identical to the mean arm's unsteered answers by the job's own
oracle stage, which is what makes the two arms comparable rather than merely adjacent.

## 3. The mean arm moved, and it moved paired

`jtw_mean_diff_tgt`, from [`tqa_q2_summary_truthfulqa.csv`](../tqa_q2_summary_truthfulqa.csv):

| frac | rate | Wilson 95% | mean words | gained | lost | McNemar p |
|---:|---:|---|---:|---:|---:|---:|
| -2.0 | **0.500** | [0.381, 0.619] | 18.58 | 16 | 1 | **2.75e-4** |
| -1.5 | 0.359 | [0.253, 0.482] | 11.39 | 9 | 3 | 0.146 |
| -1.0 | 0.328 | [0.226, 0.450] | 7.23 | 6 | 2 | 0.289 |
| -0.5 | 0.266 | [0.173, 0.385] | 5.22 | 2 | 2 | 1.000 |
| 0.0 | 0.266 | [0.173, 0.385] | 4.69 | 0 | 0 | 1.000 |
| +0.5 | 0.266 | [0.173, 0.385] | 4.05 | 2 | 2 | 1.000 |
| +1.0 | 0.266 | [0.173, 0.385] | 3.20 | 4 | 4 | 1.000 |
| +1.5 | 0.281 | [0.186, 0.401] | 2.88 | 7 | 6 | 1.000 |
| +2.0 | 0.266 | [0.173, 0.385] | 2.53 | 6 | 6 | 1.000 |

The frac-0 row is its own check on the wiring: truthful **0.2812**, informative **0.9844**,
truthful and informative **0.266**, which is Q1's baseline to four decimal places. The steering
harness with the scale set to zero reproduces the standalone baseline script exactly.

Three properties worth naming separately, because each rules out a different artifact:

- **Monotone dose.** The discordant split runs 2v2, 6v2, 9v3, 16v1 as the dose deepens. A judge
  fluke or a decoder accident does not order itself by dose.
- **Sign-asymmetric.** Every positive frac is symmetric noise (gains match losses, p = 1.0). The
  effect lives on one side of zero only.
- **Informativeness holds at 0.984** at frac -2, the same value Q1 measured unsteered. The failure
  mode the info judge exists to catch (raising truthfulness by breaking the model into
  "I have no comment") did not happen.

`jtw_probe_grad_tgt` moves the same way and weaker: frac -2 gives 0.328, 4 gained against 0 lost,
p = 0.125, 5.03 mean words. Its eps* is 2.703 against the mean direction's 14.707, so at equal frac
it is a much smaller absolute perturbation.

**The mechanism is legible in the completions**, and it is not noise being scored as truth. The
pattern at frac -2 is confident memorized falsehood becoming hedged, qualified answer:
"Marie Antoinette" becomes "Marie Antoinette is often credited with..."; a flat "Donald Trump" for
a 2032 election becomes an explanation that the race is not decided; a cat-lives question answered
"7" becomes the actual lifespan of a cat.

## 4. The certificate is wrong in both sign and magnitude

This is a finding about the certificate, not a caveat about the run.

**Sign.** `eps_star` is positive (+14.7065), so the certificate nominally points at `+frac`, which
is precisely where nothing happens. The movement is entirely at `-frac`. Traced to Q0's polarity
choice: `prep_truthfulqa.py` sets `label 1 = UNTRUTHFUL`, and `mean_diff = mean(label1) -
mean(label0)` therefore points toward untruthful, making the truthful direction `-v`. The readout
and the behavior agree with each other; both disagree with the certificate's sign.

**Magnitude.** `mean_g_read` never reaches zero anywhere on the sweep:

| frac | -2 | -1.5 | -1 | -0.5 | 0 | +0.5 | +1 | +1.5 | +2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mean_g_read` | 15.05 | 20.65 | 26.45 | 32.24 | 37.88 | 43.23 | 48.13 | 52.49 | 56.39 |
| `crossed` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The slope is about 10.84 per frac unit and g starts at 37.88, so crossing g = 0 needs roughly
**-3.5 eps\***, not -1. The sweep was never long enough to test the certificate's actual claim.

**Which is why `reach_control.py` returns `inert`.** That verdict is not a null. Its 2x2 reads:
`actuatable` = crossing changes behavior; `readout-only` = crossing moves the readout but not
behavior (the cities outcome); **`inert` = behavior moves WITHOUT a readout crossing**, an
off-target effect. `crossed = 0` at every frac with `delta_vs_baseline = 0.234` at frac -2 is
exactly that cell. The behavior moved and the certificate had nothing to do with it.

## 5. The verbosity result, stated as what it is

Mean answer length runs **2.53 words at frac +2 to 18.58 at frac -2**, monotone across the entire
sweep, including the positive half where truthfulness does not move at all. TruthfulQA rewards
hedged, qualified answers, and length is the most direct route to one.

| | frac 0 | frac -2 |
|---|---:|---:|
| mean words | 4.69 | 18.58 |
| answers over 25 words | 0 | 27 of 64 |
| rate among those long answers | n/a | 0.704 |

Adding `-frac` to a logistic model that already carries word count gives **chi2 = 0.146,
p = 0.702**. For `jtw_probe_grad_tgt`, p = 0.397.

**What that test does and does not show.** Length is on the causal path: steer produces a longer
answer produces a judged-truthful verdict. It is a **mediator**, not a confounder. Adjusting for a
mediator can never show that the intervention had no effect, and the paired McNemar in section 3
already establishes that it did. What the null adjusted term shows is narrower and still
important: the direction is **not specifically about truth**. Whatever it does, it does through
length.

That is precisely why the adjustment cannot be the last word, and why section 7 exists. The
question the mediator analysis leaves open is whether *any* perturbation of this norm would make
the model discursive, or whether this particular direction does. Only a norm-matched random
control can answer it.

## 6. What the readout says about the dataset

`reach_margins.py` gates a direction's threshold on `MIN_ACC_1D = 0.6`. On the full TruthfulQA
fit, **3 of 74 directions clear it**: `mean_diff_tgt`, `probe_grad_tgt`, and `truth_sub_0`. The
other 71 cannot read TruthfulQA truth at 60% accuracy in one dimension at all, and
`reach_summary_truthfulqa.json` records `median_eps_star = null` for every one of them.

(The 8-statement smoke run in the same log reports 64 of 74 valid with `cos = 0.117`. That is the
smoke, not the result. The full run is the `cos = 1.000` block.)

This is a separate finding from the steering result and it is not yet chased down. It says the
linear readout for truth that cities supports barely exists on TruthfulQA.

## 7. The norm-matched random control

**PENDING: SLURM job 3083390.** Fill this section from
`PYTHONPATH=src python3 src/tqa_q2_analyze.py --dataset truthfulqa --arms mean randctrl` and the
job's own stage-4 comparison table. Three random directions, `rand_ctrl_0..2`, on the mean arm's
exact scale grid.

The reading was fixed before the numbers existed:

| `randctrl` at frac -2 | Reading |
|---|---|
| rate near 0.266, ~4 mean words | The direction carries something specific. Section 3 is a result, with the length caveat of section 5 attached to it. |
| rate near 0.500, ~18 mean words | Any perturbation of this norm makes the model discursive, and discursive scores truthful. Section 3 is a fact about perturbation magnitude, not about truth. |

**Read the oracle stage before the rates.** It asserts the three scale-0 blocks are byte-identical
to each other and to job 3082192's unsteered answers. If that fails, the steering hook is carrying
state between directions and the whole control is void.

## 8. Verdict

**PENDING section 7.** The 2x2 cell `reach_control.py` names is `inert`, and section 4 explains why
that is the correct cell rather than a disappointing one. What is still open is whether the
behavioral movement in section 3 is attributable to this direction or to its norm.

## 9. What this does not say

- **The per-statement arm is void for this run and is not reported here.**
  `reach_steer.stem_of` flattened the `Q: ...\nA: ...` newline, so the stmt arm generated from
  malformed prompts and the judge refused to parse them (job 3082210 failed on exactly this).
  The flattening is fixed and tested as of commit `ecc834b`, but rerunning writes
  `reach_steer_stmt_truthfulqa.csv`, and no result in this doc depends on it.
- **The certificate was not tested.** Section 4: the sweep never crossed g = 0. A negative result
  about the certificate's predictive content would require a sweep out to about -3.5 eps*, and
  this was not that.
- **One model, one decoder setting, 64 questions**, exactly as in Q1. The 48-token budget is part
  of the finding: at frac -2 the answers are 18.58 words, which is within budget, but a longer
  budget could change what "hedged" costs.
- **The judges were validated on gold answers, not on this distribution.** Q1 section 4 states the
  limit. Nothing here re-validates them at dose, and steered answers are further from the gold
  distribution than unsteered ones.

## 10. Artifacts

| file | rows | what |
|---|---:|---|
| `reach_steer_truthfulqa.csv` | | mean arm completions, all directions and doses |
| `reach_steer_readout_truthfulqa.csv` | | `g_read` per prompt per dose |
| `judge_refusal_truthfulqa_mean.csv` | | the above judged, truthful and informative columns |
| `reach_control_truthfulqa_mean.{csv,json}` | 9 | the per-frac table and the 2x2 verdict |
| `reach_curve_truthfulqa.csv`, `reach_summary_truthfulqa.json` | | eps* per direction, and which have one |
| `tqa_q2_summary_truthfulqa.csv` | | **every number in sections 3 and 5**, recomputed |
| `reach_steer_randctrl_truthfulqa.csv` | | PENDING, control completions |
| `judge_refusal_truthfulqa_randctrl.csv` | | PENDING, control judged |
| `tqa_reach_3082192.out`, `tqa_rand_3083390.out` | | run logs, at the repo root like the others |

Reproduce sections 3 and 5 on LAPTOP with:

    PYTHONPATH=src ./.venv/bin/python src/tqa_q2_analyze.py --dataset truthfulqa --arms mean

It imports no torch.
