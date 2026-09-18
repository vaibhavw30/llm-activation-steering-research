# Judge-free TruthfulQA and the learned ceiling (job J-E, round 3)

Approved in chat 2026-09-18. Two experiments that pay off whichever way U1, J-A, J-B/C and
round 2 (J-D1, J-D2) come out. One job, `deltaai/run_tqa_mc.slurm`.

## Why

Every TruthfulQA number so far goes through a judge, and the registered hypothesis of the
dataset card (PLAN_PI_FEEDBACK_2026-09-18.md section 9) is that TQA steers only because its
judged score rewards form. Two things are missing whatever the pending jobs say:

1. **A score with no judge and no generation**, so a gain can be content without the judge
   rewarding length. This also fills the missing half of a dataset x format 2x2.
2. **A ceiling**: how much truthfulness ANY single layer-11 vector of a given norm can buy.
   Without it a null cannot tell "truth is not steerable this way" from "nothing is".

|            | short answer              | long answer                        |
|------------|---------------------------|------------------------------------|
| cities     | J-D1 (next-token country) | **new**: `cities_long`             |
| TruthfulQA | **new**: `mc` (logprob)   | J-D2 (generated, judged)           |

## Shared core: answer log-probability under steering

`answer_logprob(model, tok, prompts, answers, st, vec, scale)` returns, per row, the SUM and
MEAN log-probability of the answer tokens given the prompt, and the answer's token count.

- The prompt is `prompt_of(question)` and the answer `" " + answer`, tokenized SEPARATELY
  and concatenated, so the prompt's ids are exactly generation's and the answer's are what
  the model would emit after them (no merge across the seam). Stage `mc` also checks once
  that batched and one-at-a-time scores agree within 1e-3.
- Left padding; the mask excludes pads. `st` is a `dct_steer_utils.Steerer` at layer 11,
  set with `xc.steer_vec(vec, scale)`, which adds at every position exactly as generation
  does. `vec` may be (d,) or (B, d).
- No grad in scoring. The training path calls the same forward with grad enabled on the
  vector only. Every forward passes `use_cache=False`: transformers 4.51 otherwise builds a
  Gemma2 HybridCache per forward, and training would backprop through its in-place writes.

## Experiment 2: the learned ceiling (stage `train`, runs first)

- **Data.** `got_datasets/truthfulqa.csv`: 744 questions, one true and one false answer
  each (verified: no question overlaps the 64-question holdout). Split by question, seed
  37: 600 train, 144 validation. A question never sits on both sides.
- **Objective.** Maximize mean over pairs of `logsigmoid(mean_lp(true) - mean_lp(false))`,
  per-token means so the vector cannot win by preferring the shorter answer.
- **Norm.** ||v|| fixed at r = frac x median ||h_11|| (holdout prompts, as J-D2 measures
  it), frac in `xc.NORM_FRACS` = (0.125, 0.25, 0.5). Projected back to the sphere after
  every step. 3 random starts (seeds 0, 1, 2) per frac.
- **Optimizer.** Adam on the tangent-projected gradient (its radial part removed before
  each step), lr 3e-2 x r/sqrt(d) with d = hidden size, then projected back to the sphere;
  batch 16 pairs, at most 4 epochs; keep the step with the best validation objective,
  checked every 10 steps. The sqrt(d) because Adam's per-coordinate steps are sign-like,
  so a step's length scales with sqrt(d): the earlier lr 1e-2 x r moved ~0.48 r per step
  at d = 2304 and reached only cos 0.79 to a known optimum in the real step budget.
- **Outputs.** `mc_learned.npz` (unit vectors, frac, seed, val objective, val accuracy of
  mean_lp(true) > mean_lp(false), the unsteered val objective and accuracy, norm_med, and
  the settings LR_PER_R, MAX_EPOCHS, EVAL_EVERY, BATCH_PAIRS, n_train, n_val);
  `mc_learned_cos.csv`: cosine between the 3 seeds per frac, and between each learned
  vector and every direction in the J-D2 list.
- **Directions it adds** to the lists below: `learned_f{frac}_s{seed}`, each steered at
  its own frac only (a vector trained at one norm is not claimed at another).

## Experiment 1a: TruthfulQA multiple choice (stage `mc`)

- **Questions.** The 64 holdout questions, every `correct_answers` and `incorrect_answers`
  entry (median 3 and 4 per question).
- **Directions.** `xc.build_directions(N_RAND=32, RAND_SEED=41, jb_prefix)`: the J-D2 list
  with 32 randoms, so a permutation p can reach 1/33 = 0.03. Plus the learned vectors.
- **Doses.** `xc.dose_grid(norm_med, eps*)` plus the unsteered baseline.
- **Per question metrics.**
  - `margin` (PRIMARY): mean over correct answers of mean_lp, minus the same over incorrect.
  - `mc1`: 1 if the answer with the highest mean_lp is a correct one.
  - `mc2`: sum of exp(sum_lp) over correct, divided by the sum over all (the standard
    sum-logprob MC2; reported, not primary, because it favours short answers).
- **Secondary set.** The 744 training pairs, `margin` = mean_lp(true) - mean_lp(false), at
  the read dose only, flagged `in_sample` for every TQA-sourced direction and every
  learned vector (validation pairs reported separately for the learned ones as
  `val_selection`: not trained on, but they chose the checkpoint).
- **"Moves".** Round 2's rule: Wilcoxon on the per-question margin vs unsteered at
  p <= 0.05, AND `xc.beyond_null` on the permutation p among the 32 randoms at the same
  (unit, frac), effect e = sign(frac) x (mean margin - baseline mean margin), plus e > 0.
  The e > 0 clause is a deliberate tightening: at +0.5, where every random lowers the
  margin, round 2's rule could call a significant decrease that beats the randoms "moves".
- **Output.** `mc_tqa_scores.csv` (one row per direction x dose x question x answer),
  `mc_tqa_summary.csv`.

## Experiment 1b: cities in long form (stage `cities_long`)

- Prompt `Q: Where is the city of {city}?\nA:`, the 177 cities of `xfer_cities.targets`
  with its clean pool, greedy 48 tokens, repetition penalty 1.0, scored with
  `xfer_cities.score_gen` (first country named in the first sentence).
- Directions: the J-D2 list with 8 randoms (seed 29, J-D1's), plus the learned vectors.
  Doses: norm +/-0.25 only (`xc.READ_FRAC`) and the baseline.
- "Moves" on `gen_correct`: exact McNemar vs unsteered, AND beyond_null among 8 randoms,
  AND e > 0 (as in 1a).
  Words and incoherence reported beside it, as in J-D1.
- Output `mc_cities_long.csv`, `mc_cities_long_summary.csv`.

## Experiment 2b: the judged ceiling (stages `gen`, `judge`)

The learned vectors, each at its own frac, generate on the 64 holdout questions with
`xfer_tqa.generate` (batched, rep penalty 1.3, 48 tokens), plus the unsteered baseline,
judged by the v2 allenai judges through `judge_resumable`. `mc_learned_judged.csv`. This is
the ceiling on the same scale as Q2 and J-D2.

## Registered readings

Every primary reading uses round 2's read dose only, norm +0.25 (`xc.READ_UNIT`,
`xc.READ_FRAC`), as J-D1 and J-D2 do. "Truth direction" excludes the randoms, the learned
vectors and the potency-matched DCT controls (`<src>:dct_ctl_<j>`); moving controls print
on their own line, and every other (unit, frac) cell that moves prints on a `[secondary]`
line marked not corrected for multiplicity (~160 cells against a 1/33 null expect chance
hits).

- **Content, not form**: some truth direction moves `margin` beyond null (1a) at the read
  dose.
- **Form, not content**: from round 2's `xfer_cities_outcomes.json` and
  `xfer_truthfulqa_outcomes.json`, one line per truth direction with its J-D1 and J-D2
  outcomes and whether it moves 1a and 1b at the read dose. Verdict: CONTENT if any truth
  direction moves 1a; else FORM NOT CONTENT if some truth direction moves a long-form cell
  (J-D2 outcome "gain", `xfer_tqa.classify`'s moved outcome, or 1b moves); else NEITHER.
  This is the card's hypothesis, confirmed without a judge. Not read, and said so, when
  either JSON is missing.
- **Method limit**: no learned vector at norm 0.25 whose training worked is beyond null
  (e_margin > 0 and `xc.beyond_null` on the permutation p; the Wilcoxon p is printed, not
  required). Training worked when the stored val objective is finite and above the
  unsteered one; a vector where it did not prints TRAINING FAILED and is left out. Not
  read without `mc_learned.npz` or without a trained vector at 0.25. Then every steering
  null in the project is about single-vector steering at layer 11, not about truth.
- **Identifiability**: per frac, the seed-vs-seed cosines (median, min) beside each seed's
  val objective. Cosines near 1 mean one optimal direction; low cosines with equal
  objectives mean many, which is the non-identifiability result in miniature.

## Job and safety

- `run_tqa_mc.slurm`, 3 h wall. Preflight the same files J-D2 checks plus
  `got_datasets/truthfulqa.csv`, `token_acts_cities.npz`. Stage `smoke` first: 4
  questions, 1 training step, 2 cities, prefix `smoke_`, then removed.
- Every stage resumes: `train` skips a (frac, seed) already in `mc_learned.npz`, after
  checking the file's stored settings and norm_med (1e-3 relative) against the current
  ones and aborting on any difference or on a file without them (the old optimizer's); scoring and
  generation use `xc.done_blocks`; judging uses `judge_resumable`. Nothing is overwritten.
- J-B outputs (TQA DCT picks, G0 MAG) are left out LOUDLY if absent, so the job can take
  any free slot; the full list needs J-B. Submitter entry `round3`.
- `summary` runs on the LAPTOP from pulled CSVs.

## Files

- `src/tqa_mc.py`: `answer_logprob`, the metrics, stages mc / cities_long / gen / judge /
  summary, `main`.
- `src/tqa_learned.py`: split, sphere projection, training loop, stage `train`.
- `tests/test_tqa_mc.py`: answer masking against a fake model with known logits (with left
  padding and a seam case), mc1 / mc2 / margin on hand-computed arrays, sphere projection,
  split disjointness, summary classification.
- `deltaai/run_tqa_mc.slurm`, `deltaai/submit_pi_feedback.sh` (round3), README row, plan
  section 8 note.
