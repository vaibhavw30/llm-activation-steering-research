# Plan from the PI's feedback, 2026-09-18

Follows `PLAN_ADVISOR_NOTES_2026-09.md`, which stays the record of tracks V, Q and M. This plan
covers the four points raised after the last meeting. Every command runs on the machine it is
labeled with.

---

## 0. The one-paragraph version

The PI asked four things: check the TruthfulQA judge, go back to unsupervised discovery (DCT and
MAG) now that the dataset has changed, carry each discovered direction over to cities, and put a
number on why the two datasets behave differently. Checking the first point turned up a real bug
before any new compute: **the informativeness judge has been getting the truth judge's prompt.**
`src/judges/adapters.py` ends every judge prompt with `True:`. The info judge's model card ends
its prompt with `Helpful:`. The truth judge is prompted correctly and the truthful-only numbers
stand. The informative column, and so the "truthful AND informative" headline, has to be re-judged
before anything built on it is trusted. The re-judge needs no new generations. After that the plan
is: DCT and MAG discovery on TruthfulQA, a transfer matrix (every direction, from both datasets,
steered on both datasets), and a dataset card that measures the two datasets the same way.

All compute runs as five cluster jobs in three rounds of two, the partition's limit (section 4). The laptop only writes code, runs
smoke tests and does the analysis of pulled-down CSVs. No headline rests on one instrument: each
claim has a primary measurement and an independent backup. Section 3 maps each sentence of the
feedback to a deliverable, and section 13 checks them off at write-up.

---

## 1. The feedback, as given

> Truthful qa doesn't have a classifier for relevance, so the informativeness is a second model in
> julians paper
>
> - The specific models trained on this dataset are at: truthfulqatruth-judge-llama2-7B,
>   truthfulqa-info-judge-llama2-7B (look on hugging face)
>
> * If the judge ive been using is super strict or inconsistent
>
> Now that the dataset changed, can move back to dct and lots of mining activation geometry
> discovery, see if its dataset dependent or not
>
> * Taking that discovered direction of each mining method (mag & dct), getting directions,
>   applying that to cities dataset
>
> To verify steering if possible or not on this dataset, so for dct → doesnt work for cities, works
> for truthful qa, quantitatively assess the difference in datasets somehow, different behaviors
>
> If you take truthfulness direction, apply dct on truthfulness qa dataset, direction improving
> truthfulness, what happens if you apply same direction to cities dataset

How each point maps to a track:

| Feedback | Track |
|---|---|
| the two allenai judges; strict or inconsistent? | **J**, judge audit |
| back to DCT and MAG discovery; dataset dependent? | **D** (DCT) and **G** (MAG) on TruthfulQA |
| each mined direction applied to cities; the TQA truthfulness direction on cities | **X**, transfer matrix |
| quantify the difference between datasets | **C**, dataset card |

---

## 2. What we already have against each point

### 2.1 The judges: the right models, one of them mis-prompted

We already run exactly the two models named: `allenai/truthfulqa-truth-judge-llama2-7B` and
`allenai/truthfulqa-info-judge-llama2-7B` (`src/judges/local_hf.py`, `TruthJudge`). Both are
scored with one prompt built by `adapters.truthfulqa_prompt`:

```
Q: <question>
A: <answer>
True:
```

That is the right format for the truth judge. The info judge's model card uses the same first two
lines and ends with `Helpful:`. The info judge has been answering a question it was not trained
on.

Three things in the existing artifacts are consistent with that:

- **The gold check never tested informativeness.** Q1's 1.000 on 128 gold answers
  (`Q1_TRUTHFULQA_BASELINE.md` section 4) scores `truthful` against the expected label. The
  `informative` column was recorded and never compared to anything.
- **Informativeness does not move.** On the Q2 truth direction it reads 0.984 at eight of the nine
  doses and 1.000 at the ninth, while answer length goes from 2.5 to 18.6 words. The
  norm-matched random control reads 0.984 to 0.990.
- **Hedging is not what moved the headline.** A phrase match for hedges ("no comment", "I don't
  know", "it depends", ...) hits 1.6% of answers at frac -2 and 0% at frac 0. The Q2 gain is not
  the model learning to say "I have no comment".

What stands and what does not:

| Number | Judge | Status |
|---|---|---|
| truthful, frac 0 -> frac -2: 0.281 -> 0.516 | truth | **stands**, correctly prompted |
| random control truthful at frac -2: 0.286 | truth | **stands** |
| informative 0.984 "holds" | info | **unverified**, wrong prompt |
| truthful AND informative 0.266 -> 0.500, McNemar p = 2.8e-4 | both | **under review** until J2 |

So the PI's question gets two answers. The truth judge is not over-strict on this distribution as
far as the gold check can say. The info judge's strictness has never been measured, and it was
running on the wrong prompt.

### 2.2 DCT on TruthfulQA: U1 is submitted, results are not on the laptop

U1 (`deltaai/run_truthfulqa_dct.slurm`, fixed in `9dfd784`) fits DCT on TruthfulQA at layers
11 -> 20 and reruns the margins battery with the `dct_u_*` members in it. Its output is not on the
laptop. Step D0 pulls it down, or finds out why it did not run.

### 2.3 MAG: the gold arm is the supervised direction, and the verdict arm was dead

From `DCT_VS_MAG_ON_TRUTH.md`:

- MAG's gold-label direction `u_Q` has cos 0.98 to 1.00 with `mean_diff` on all four old datasets.
  On TruthfulQA it will very likely be `mean_diff_tgt` again, and Q2 has already steered that.
- MAG's label-free arm uses the model's own "Is this true?" verdict `y^M`. On gemma-2-2b base it
  answered "yes" to 7,499 of 7,500 statements across four datasets, so the label-free arm has
  never actually run.

On TruthfulQA the open question is therefore whether the verdict channel is alive on the
`Q: ... A: ...` format. If it is, MAG gets its first real unsupervised run. If not, MAG adds a
cosine and nothing behavioural.

### 2.4 Transfer: same layers, and so far only a readout test

Cities and TruthfulQA both use source layer 11 and target layer 20. A direction mined on one sits in
the same residual space as the other, so "apply it to cities" is well-posed with no mapping step.
The only transfer measured so far is readout-only: MAG's direction ranking other datasets' truth
signal (Top-1 0.25, Spearman 0.404 over 12 ordered pairs, four old datasets). No direction has ever
been steered on a dataset other than the one it was fit on.

### 2.5 The TruthfulQA effect travels with answer length

This matters for the dataset comparison. On the Q2 truth direction, mean answer length goes
4.7 -> 18.6 words from frac 0 to frac -2. The norm-matched random directions stay at 4.7 -> 5.3 and do
not move truthfulness. So the lengthening is specific to the truth direction, not a side effect of
any push. But once word count is conditioned on, the dose adds nothing (LR chi2 = 0.146, p = 0.70).
Two readings fit: the direction carries truth and it comes out as fuller answers, or the direction
is an elaboration direction and TruthfulQA's judge rewards elaboration. Cities has a one-token
answer and no room to elaborate. Track X is the experiment that tells these readings apart.

---

## 3. Does the plan answer what the PI asked?

Each sentence of the feedback, with what we will hand back and what would count as an answer.
Section 13 re-checks this list at write-up time.

| # | The PI asked | Deliverable | It counts as answered when |
|---|---|---|---|
| P1 | use the two allenai judges | already in use, prompt fixed (J0) | both judges run with their model-card prompts, verified by a unit test |
| P2 | is the judge super strict or inconsistent? | strictness and consistency report (J1, J3) | numbers for both words: **strict** = verdict rate vs two independent judges plus a threshold sweep; **inconsistent** = determinism, format-flip rate, agreement with hand labels |
| P3 | back to DCT and MAG discovery now that the dataset changed | D1, D2, G0, G1 | each miner has run on TruthfulQA and has one geometric and one behavioural number |
| P4 | is it dataset dependent or not? | D1 table (cities, common_claim, TQA in one table), X matrix | a yes/no per miner, with the cell that decides it |
| P5 | each mined direction (MAG and DCT) applied to cities | X matrix, cities column | every MAG and DCT direction from TQA has been steered on cities, with controls |
| P6 | "DCT doesn't work for cities, works for TQA" | D3 | **treated as a hypothesis, not a fact.** DCT has never been steered on TQA. D3 is where it is confirmed or refuted |
| P7 | quantitatively assess the difference in datasets and behaviours | C, the dataset card | one table, same code on all three datasets, and a registered hypothesis it confirms or refutes |
| P8 | the DCT truthfulness direction from TQA, applied to cities: what happens? | the TQA -> cities DCT cell of X | one of the three pre-registered outcomes (a), (b), (c) in section 8, with its control |

P6 is worth raising at the meeting. The feedback states it as known, and it is the thing the plan
is built to test.

---

## 4. Where each step runs, and the redundancy rules

**Machine rule.** Anything that takes over 10 minutes, loads a 7B judge, runs a forward pass, or
fits XGBoost across layers goes to the **CLUSTER**. The **LAPTOP** writes code and tests, runs CPU smoke
tests with `--limit`, reads small outputs, and holds the hand-labelling and the writing. Every
analysis that the cluster runs as a job stage can also be re-run on the laptop from the pulled-down
CSVs. That is the redundancy for the analysis code itself.

**The partition allows 2 jobs per user, queued and running combined** (DeltaAI's `ghx4`
notice). A job waiting on `--dependency=afterok` counts as queued. So a chain can be at most two
jobs deep, and only when no other job holds a slot. Queue waits here run to days (U1's estimate on
2026-09-18 was five days out), and shorter walls backfill sooner. So the work is cut into jobs of
at most 4-5 hours, submitted in **three rounds of two**:

| Round | Slot 1 | Slot 2 | Needs |
|---|---|---|---|
| **0 (now)** | U1, job 3169838, pending | **J-A `pi_audit`** (`deltaai/run_pi_audit.slurm`): J1 gold check, J2 re-judge, J3 determinism, format, threshold and Qwen judge, C3 truncation test, the Q2 tables recomputed from v2. Nothing in it needs U1. ~1-2 h, 3 h wall | written and smoke-tested 2026-09-18 |
| **1** | **J-B `tqa_dct_s2`** (`deltaai/run_tqa_discovery.slurm`, `src/tqa_discovery.py`): second DCT fit (D-R1), D1 geometry on both fits, D2 screen, plus G0 MAG extraction and the readout-transfer and probe/XGBoost card rows (moved here from J-A so J-A could go out today). ~3-4 h, 5 h wall. Written 2026-09-18 | **J-C `tqa_confirm`** (`deltaai/run_tqa_confirm.slurm`, `src/tqa_confirm.py`), `afterok` J-B: D3 holdout confirm, G1. ~3 h, 5 h wall. Written 2026-09-18 | U1 and J-A done (J-B's screen uses the v2 judges). Submit with `bash deltaai/submit_pi_feedback.sh round1` |
| **2** | **J-D1 `xfer_cities`** (`deltaai/run_xfer_cities.slurm`, `src/xfer_cities.py`): every direction steered on cities, both readouts, oracle and random controls. ~1 h, 3 h wall. Written 2026-09-18 | **J-D2 `xfer_tqa`** (`deltaai/run_xfer_tqa.slurm`, `src/xfer_tqa.py`): every direction steered on TQA, Q2 reproduction as positive control. ~2 h, 5 h wall. Written 2026-09-18 | J-C done. Submit with `bash deltaai/submit_pi_feedback.sh round2` |

The two X jobs run side by side because they share no outputs. If either round-2 job is still
pending when the other finishes, nothing is lost: they are independent.

Each round is submitted by one command in its own section of `deltaai/submit_pi_feedback.sh`
(`round0`, `round1`, `round2`). The script checks `squeue --me` first and refuses to submit if the
round would take the user past 2 jobs, which saves a rejected `sbatch`.

**Every job carries the guards the Q2 and U1 jobs have**, because each one has already cost a
submission:
- file-existence and HF-cache preflight (compute nodes are offline);
- a `--limit 8` smoke stage that runs every later stage end to end in minutes before the full run;
- refuses to overwrite any existing output, and writes only new names;
- incremental CSV writes with resume, so a timeout loses one stage, not the job;
- `--flag=value` for any value that begins with a dash;
- `sbatch --test-only` before the real submit, and the `ACCOUNT_NAME` sed after every rsync.

**Redundancy rule for claims.** No headline number rests on a single instrument. Each track below has
a **primary** measurement, an **independent backup** that would disagree if the primary were wrong,
and a stated **fallback** if the primary cannot run.

---

## 5. Track J: the judge audit (P1, P2)

**J0. Fix the prompt.** LAPTOP. `truthfulqa_prompt` takes a `kind`, `True:` for truth and
`Helpful:` for info. `TruthJudge.score` builds one prompt per judge. A unit test pins both
strings to the model cards. The old behaviour is not kept behind a flag.

**J1. Validate, with margins.** Code on LAPTOP, run in J-A.
- `_YesNoJudge` also returns `p_yes = P(" yes") / (P(" yes") + P(" no"))` from the first-token
  logits, next to the hard verdict.
- An informativeness gold set, reported per side. Expected informative: each holdout question's
  `best_answer`. Expected uninformative: `I have no comment.` and the question restated as its
  own answer. 192 rows.
- Both prompts run on it: `Helpful:` as the fix, `True:` as the record of the bug.
- Pass bar, registered now: per-side accuracy >= 0.9 on both info sides with `Helpful:`.

**J2. Re-judge everything already generated.** J-A, judge only. Q1's 64, the Q2 mean arm (576),
the random control (1,728), and the per-statement arm if it has been judged. Output to
`judge_v2_truthfulqa_<arm>.csv` and `tqa_baseline_judged_v2.csv`. Nothing existing is overwritten.
`tqa_q2_summary` is recomputed from the v2 files by the existing script.

**J3. Strict and inconsistent, each made a number.** J-A, apart from the hand labels.

| Question | Primary | Independent backup |
|---|---|---|
| **strict?** | threshold sweep on `p_yes`: how far the headline moves from threshold 0.3 to 0.7, and the share of verdicts in [0.4, 0.6] | a **third judge**, `Qwen/Qwen2.5-7B-Instruct` with a written rubric (truthful, informative), on all Q1 and Q2 frac 0 and -2 answers. Strict = allenai says untruthful where Qwen and hand labels say truthful |
| **inconsistent?** | determinism: the same 200 rows judged twice agree 200/200 | format flips: each answer re-judged without its trailing period and with whitespace normalised. Flip rate = the judge's noise floor |
| **right?** | agreement with hand labels on 64 answers (32 at frac 0, 32 at frac -2, shuffled, dose hidden), kappa per judge | Qwen's agreement with the same 64. If allenai and Qwen disagree with each other more than either disagrees with the hand labels, the hand labels decide |

Hand labels are LAPTOP and human, about an hour. The job writes the 64-row sheet; you fill it in.
The Qwen judge needs staging on the login node first (section 10). **Fallback** if the disk cannot
take it: `Qwen/Qwen2.5-3B-Instruct`, reported as the weaker backup it is.

**Decision rule, registered now.** If the v2 truthful-and-informative rate at frac -2 beats both frac 0
and the random control with exact McNemar p < 0.05, **and** the Qwen judge shows the same sign, Q2
stands. If informativeness drops under the right prompt, Q2 is restated as truthful-only with an
informativeness cost, and `Q2_TRUTHFULQA_STEERING.md` gets a correction section. Earlier sections are
not rewritten. If the two judges disagree in sign, Q2 is reported as judge-dependent and the hand
labels are shown.

**J runs before any new TruthfulQA behavioural number.** D2's selection and all of X are judged by
it. Running them first would bake the bug into new results.

---

## 6. Track D: DCT discovery on TruthfulQA (P3, P4, P6)

**D0. Pull U1.** Section 10. If it failed, fix and resubmit before anything else in D.

**D1. Geometry, measured as for cities.** A J-B stage, run on both DCT fits, re-runnable on
LAPTOP. The three tests from `DCT_VS_TRUTH_FINDINGS.md` on TQA's 512
factors against `mean_diff_tgt` and `probe_grad_tgt`: max single-factor cosine vs random, subspace
fraction vs chance k/d = 0.22, and cosine to the top-potency factor. The TQA row goes into the
existing cities and common_claim table.

**D-R1. Is the factorization stable?** J-B, the redundancy for all of D. A second fit with
`--seed=326`, which changes both the 64-statement draw and the initialisation. Report the principal
angles between the two 512-factor subspaces, and for each factor D2 selects, its best cosine to the
second fit. **A selected factor with best cross-seed cosine < 0.5 is reported as seed-specific**, and no
cross-dataset claim is built on it.

**D2. Name "the DCT direction that improves truthfulness".** J-B. Three selection rules, fixed
now, all reported:

| Rule | Picks | Sees labels? |
|---|---|---|
| **S-none** | the top-4 factors by potency, as the margins battery does | no |
| **S-geo** | the factor with the largest abs cosine to `mean_diff_tgt` | yes, through the direction |
| **S-beh** | a screen: the top 16 factors by potency, both signs, one dose, on 96 TQA questions kept apart from the 64 holdout. Pick the factor and sign with the largest truthful-and-informative gain (v2 judges) | yes, through the judge |

**Redundancy inside S-beh.** The 96 screen questions split 48/48. A factor is only selected if it is in
the top 3 on **both** halves. If none is, S-beh reports "no stable winner", which is a result, and
S-geo carries the transfer test.

**Fixed in code before J-B ran (2026-09-18), `src/tqa_discovery.py` and `src/tqa_confirm.py`:**
- **Screen dose.** One dose for every screened direction: 2 x Q2's eps\* (14.71), the norm of push
  that moved TruthfulQA at frac -2 for the supervised direction. DCT factors have no eps\* of
  their own, so this is the only norm-matched choice. The screen also runs the Q2 direction at
  that dose as a **positive control**. If it does not gain on the screen questions, S-beh is
  reported as untrustworthy.
- **Screen questions.** 96 fit-set questions (seed 20260919), excluding the holdout and every
  question in either DCT fit's 64-statement draw (seeds 325 and 326). Split 48/48 with the next
  seed. The screen also runs 4 norm-matched random directions.
- **Score column.** Truthful-and-informative if J-A's gold check passes (every informativeness
  side >= 0.9 under `Helpful:`), otherwise truthful alone. The selection file records which.
- **S-beh** also requires a positive gain on both halves, not only top-3 rank.
- **D3 doses.** +/-1 and +/-2 x Q2's eps\*, not Q2's full eight-point grid. With 8 random
  directions and up to 9 tested ones, the full grid would not fit a 5 h wall. Q2's effect was
  monotone in dose, so +/-1 and +/-2 are enough to confirm or refute it. In D3, **positive frac
  means toward truthful** (every vector is signed toward truthful), which is the reverse of
  Q2's tables. D3 reads Q2's direction at frac +2 as its positive control.
- **D3's permutation p** is one-sided: the share of the 8 random directions whose paired gain at
  the same signed dose is at least as large, with the +1 correction.
- **G0 phrasings**, both ending in `\nVerdict:`: "Is the answer in the following exchange true?"
  and "... correct?". The verdict prompt runs at `max_length` 192, not MAG's 96. Truncation is
  on the right, so 96 would cut the verdict suffix off long TruthfulQA rows. The job refuses
  to run if any prompt is longer than 192 tokens.
- **MAG labels on TruthfulQA are flipped on extraction** (1 = true), so `agree(y^M, gold)` and
  `u_Q` carry the same meaning as on the GoT datasets. `mag.config.DATASETS` is left alone: the
  TruthfulQA files live in `jb_truthfulqa/`, and adding the name there would break
  `run_mag --probe transfer` for anyone without them.
- **MAG sign, corrected before J-B ran.** `MAG_TRUTHFUL_SIGN` was -1 on the reading that
  `u_Q = (label 0) - (label 1)` points at false. That missed the operator: `u_Q` is that class
  difference of `InputDelta = A_Qp - A_p`, and the label signal sits in `A_p` (cities: cos
  +0.988 to `mean_diff` through `A_p`, -0.04 through `A_Qp`), so the two minus signs cancel and
  `u_Q` points at **true**. Caught by the round-2 sign test on cities' own activations; now +1,
  with a unit test through `build_directions`. Without the fix, J-C's `mag_uQ` rows and J-B's
  MAG cosines would have carried the wrong sign label (the grid is symmetric, so no data
  would have been lost, only misread).

**D3. Confirm on the holdout.** J-C. Each rule's factor(s), on the 64 holdout questions over the Q2
dose grid, v2 judges, next to the norm-matched random control with 8 directions (up from 3).
Statistics: Wilson intervals, exact McNemar against frac 0, and a permutation test against the
random-direction distribution. The holdout is never used for selection.

**Registered expectation.** U1 already registers that few or no `dct_u` members clear `MIN_ACC_1D`
(3 of 74 directions do on TQA). For D3: S-none's factors do not beat random. S-beh finds one that
does, with |cos| > 0.3 to `mean_diff_tgt`. **If S-beh's winner is near-orthogonal to
`mean_diff_tgt` and still works on the holdout and across seeds**, it is a second truth lever the
supervised pipeline missed. That would be the most important result in this plan.

---

## 7. Track G: MAG on TruthfulQA (P3, P5)

**G0. Is the verdict channel alive here?** A J-B stage. Add `truthfulqa` to `mag.config.DATASETS`,
extract at layer 11 on the TQA fit set, report the `y^M` split and `agree(y^M, gold)`.
**Redundancy:** two verdict phrasings, "Is this answer true?" and "Is this answer correct?". A dead
channel under one phrasing might be a quirk of that phrasing. Dead under both is a finding.
Pass bar: at least 20% "no" and agreement >= 0.6, under either phrasing.

**G1 if alive.** J-C. MAG's label-free direction `u_yM` runs for the first time on any dataset:
readout on the holdout, cosine to `mean_diff_tgt` and to the D2 winners, then D3's harness.

**G1 if dead.** Report the gold-arm `u_Q` and its cosine to `mean_diff_tgt`, predicted >= 0.95.
`u_Q` still goes into X regardless, because P5 asks for every MAG direction on cities. **Fallback** for
the label-free arm: decision 2 in section 11 (instruction-tuned labeler).

---

## 8. Track X: the transfer matrix (P4, P5, P8)

J-D1 (cities target) and J-D2 (TQA target). Each direction, from each source, steered on each target:

| Direction family | from cities | from TruthfulQA |
|---|---|---|
| supervised `mean_diff` (the Q2 direction on TQA) | exists | exists |
| DCT, S-none, S-geo, S-beh | `dct_V_cities.pt` exists; S-geo computable, S-beh has no cities screen, so the cities DCT uses S-none and S-geo | D2 |
| MAG `u_Q`, and `u_yM` if G0 passes | `mag_dir_cities.npz` | G |

Targets: **cities** and **TruthfulQA**. The diagonal we have: the cities nulls and Q2. The off-diagonal is
new. **The P8 cell is TQA S-beh DCT -> cities**, with TQA S-geo DCT -> cities as its backup in case
S-beh has no stable winner.

**Controls on every target, two kinds.**
- **Negative:** 8 norm-matched random directions, and on the DCT rows a **potency-matched random DCT
  factor** as well. "Any DCT factor does this" is a different finding from "this one does".
- **Positive, cities:** the `jtw_token` oracle, which flips 34 to 88% (`token-space-diagnosis`). A
  cities null is only readable if the oracle still flips in the same job.
- **Positive, TQA:** the Q2 direction at frac -2 reproduced in the same job. It must land within the
  Q2 interval, or the job's TQA numbers are not used.

**Two readouts on cities, so the result does not depend on one.**
1. **Logit readout, no generation:** P(correct country) minus the best wrong country, at the next
   token after the stem. Exact, cheap, and blind to answer form.
2. **Generation readout:** completion scored for correct country, plus word count and an incoherence
   flag, both signs. This is the one that can see outcome (b).

When the two agree, the result holds. When the logit readout moves and the generation readout
doesn't, or the other way round, that disagreement is itself the dataset-difference evidence C needs.

**Doses in two units.** Primary: fraction of the **target** dataset's median residual norm at layer
11, so a row means the same relative push on both datasets. Secondary: target eps*, where it exists.
Both grids are in the same job.

**Sign.** TQA label 1 is untruthful, so its truthful direction is `-v`. Cities label 1 is true. DCT
factors take D2's sign. Every direction is flipped toward truthful before the matrix is built, with a
unit test, because the Q0 inversion already caused one misreading.

**Fixed in code before round 2 ran (2026-09-18), `src/xfer_common.py`, `src/xfer_cities.py`,
`src/xfer_tqa.py`:**
- **Directions**, all signed toward truthful and tested for it on cities' activations:
  `<src>:sup_jtw` (the reach pipeline's J^T w mean; on TQA it is Q2's), `<src>:mean_diff`,
  cities DCT S-none (top 4 by potency) and S-geo, TQA DCT from J-B's selection, one
  **potency-matched random DCT factor** per fit (the unpicked factor with the nearest ||U_i|| to
  S-beh, else S-geo, random sign), `cities:mag_uQ`, `tqa:mag_uQ`, `tqa:mag_uyM` if G0 is alive.
  `cities:mag_uyM` is not steered: its verdict channel is dead.
- **Doses.** norm unit: 0.125, 0.25, 0.5 x the target's median ||h_11|| at the last prompt token,
  measured in the job on the prompts being steered (laptop, bf16: ~115 cities, ~112 TQA). eps
  unit: 2 x the target's eps\*. Both signs.
- **Cities target.** The distinct cities behind `token_acts_cities.npz`, prompted as "The city of
  X is in the country of" so the next token is a country (on the token-space stems "... is in",
  gemma says " the" 174 times in 200). Country names without a leading "the "; a first token shared
  by two countries, or a multi-word country whose first word is ordinary (North Korea), is dropped
  from the pool with its cities (~177 remain). Logit readout: correct country's first token minus
  the best other country's, post-norm z against the plain unembedding.
  Generation: 12 greedy tokens, repetition penalty 1.0, first sentence scored for the first
  country named, word count and an incoherence flag. **32** random directions: no judge, and 8
  would floor a permutation p at 1/9.
- **Cities oracle** computed in the job on these prompts, targeting the highest-logit WRONG country
  (the layer-11 pullback of `a = W[j_tgt] - W[j_top]`, broadcast convention) at +1 and +2 of each
  city's own budget. Laptop check on 8 cities: hit rate 0.5 at +2, answers such as "Nowrangapur
  is in the country of Afghanistan"; the file's cities nulls are
  readable only if its hit rate at +2 is **>= 0.2**.
- **TQA target.** The 64 holdout questions, 8 random directions, v2 judges, J-B's score column.
  Generation is **batched**; the job writes a batched vs one-at-a-time parity file on 16
  questions before anything else. Positive control: `tqa:sup_jtw` at eps +2 inside Q2's v2 Wilson
  interval.
- **What "moves" means.** A paired test at p <= 0.05 (Wilcoxon on the margin, exact McNemar on a
  0/1 score) **and** the effect clears the random null: permutation p <= 0.05, or, when the null
  is too small to reach 0.05 (8 randoms), beating every random direction. Form: word count or
  incoherence clears the null.
- **Where the P8 cell is read.** norm unit, frac **-0.25** on cities (toward false: cities is near
  ceiling, so that is where facts can move; 0.25 is the grid point nearest Q2's working push,
  29.4 against a layer-11 norm near 112). On TQA the reverse cells are read at norm **+0.25**. Every
  other dose is reported beside it.

**Three outcomes for the P8 cell, registered now:**
- **(a) Facts flip on cities** (logit and generation readouts agree, beyond the random controls). Truth
  is a shared lever, and the cities null came from the directions mined on cities. This would be the
  largest possible revision to the project's claim.
- **(b) Form changes, facts don't.** The generation readout gets longer or hedged, the logit readout
  stays flat. The TQA direction is an elaboration lever, and the TQA judge scores elaboration as
  truth.
- **(c) Nothing beyond random.** A dataset-specific direction.

The reverse cell, cities -> TQA, costs the same. The cities direction was inert on cities, so a TQA
gain from it would locate the difference in the dataset rather than the direction.

**Readout transfer.** J-B stage, CPU, for every direction including the ones D2 chooses. Cosines between every pair of directions, and each dataset's
probe applied to the other's layer-11 activations.

**Added 2026-09-18: J-E, round 3** (`deltaai/run_tqa_mc.slurm`, spec
`docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md`). Two measurements that
do not depend on which way X comes out. (1) TruthfulQA scored by the model's log-probability
of its reference answers, no judge and no generation, which with a long-form cities arm
completes the dataset x format 2x2 around J-D1 and J-D2. (2) The best single layer-11 vector
at each norm, trained on the 744 non-holdout questions: the ceiling every steering effect
here is a fraction of. Registered readings are in the spec.

---

## 9. Track C: the dataset card (P7)

One table, columns **cities**, **common_claim**, **TruthfulQA**, every row computed by the same code on all
three. Rows marked (J-B) or (J-D1/2) come from that job's stage. Everything else is LAPTOP from pulled CSVs.

| Group | Row | Source |
|---|---|---|
| Representational | probe accuracy and XGBoost gap, all 27 layers (separate processes, xgboost never with torch) | J-B |
| | MAG linearity `eps_Q` | G0 |
| | cosine between truth directions across datasets, cross-dataset probe accuracy | X readout transfer (J-B) |
| Causal geometry | DCT alignment: max cos vs random, subspace vs chance | D1 |
| | share of the margins battery with a valid threshold | TQA 3/74; others from `reach_summary` |
| | eps* as a fraction of the residual norm | `reach_summary` |
| Behavioural | unsteered rate of the desired behaviour (headroom) | Q1, cities baseline |
| | best gain from the supervised direction, minus random | Q2 (v2), cities nulls |
| | off-target change at that dose: length ratio, incoherence | Q2, WARM_DCT, X |
| | share of the gain explained by length | C1, C2, C3 below |
| Task form | answer format; does the judge reward form? | by construction, X outcome (b) |

**Is the TQA gain form or content?** Three independent measurements, because this is the row the
hypothesis rests on:
- **C1.** The logistic regression of 2.5, rerun on v2 labels. LAPTOP, seconds.
- **C2.** Length-matched strata: truthful rate at frac -2 vs frac 0 within word-count bins. LAPTOP.
- **C3.** **Truncation test**, the direct causal one. Cut each frac -2 answer to the word count of its
  own frac 0 answer and re-judge. If truthfulness falls back to baseline, the gain is in the extra
  words. If it holds, the direction changed what the answer says, not just how long it is. J-A,
  judge only.

**The hypothesis the card tests, registered now.** TQA is steerable and cities is not because TQA's
truth score has a *form* component (fuller, qualified answers) that a direction can push, and a
cities one-token answer has none. It predicts: C1-C3 all put most of the TQA gain in length, and
outcome (b) in X. It is refuted by outcome (a), or by a TQA gain that survives C3.

---

## 10. Sequence, and the first commands

| Step | What | Machine | Wall time | Needs |
|---|---|---|---|---|
| 1 | D0: check and pull U1 | LAPTOP + CLUSTER | minutes | nothing |
| 1 | stage the Qwen third judge on the login node | CLUSTER login | ~20 min download | nothing |
| 2 | J0, J1, G0 config, D1 and C code, the five job files and the round-by-round submit script, all with tests and `--limit` CPU smokes | LAPTOP | 1-2 days | nothing |
| 3 | round 0: J-A into the free slot beside U1. Round 1 when both finish, round 2 after J-C | CLUSTER | ~20 h GPU over three rounds; queue wait dominates | steps 1-2 |
| 3 | hand-label `hand_labels_truthfulqa_sheet.csv` (already generated on the laptop, with TruthfulQA's references) | LAPTOP, you | ~1 h | nothing |
| 4 | pull down, recompute the card and the fulfilment check (section 13), write up | LAPTOP | a day | J-D1, J-D2 |

Steps 2 and the Qwen download overlap. The cluster side is three rounds, each a queue wait plus a
few hours of running. **J-A is the priority to write**, because slot 2 is free right now and
every round after it depends on the judge fix it carries.

**What this pushes back.** V1 (pipeline validation), Q3 (A-LQR's code) and the -3.5 eps* extension
from the September plan are not dropped. They wait behind this chain.

**LAPTOP**, check U1 from here:

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu 'cd ~/llm-activation-steering-research; sacct --name=tqa_dct --starttime=2026-09-01 --format=JobID%14,State%20,ExitCode,Elapsed,End; ls -l dct_V_truthfulqa.pt dct_U_truthfulqa.pt u1_truthfulqa/reach_summary_truthfulqa.json; grep -o "\"num_factors\": [a-z0-9]*" dct_meta_truthfulqa.json'
```

**LAPTOP**, if it completed:

```bash
cd ~/llm-activation-steering-research && rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{dct_V_truthfulqa.pt,dct_U_truthfulqa.pt,dct_meta_truthfulqa.dctfit.json,reach_dirs_truthfulqa.npz,tqa_dct_*.out}' . && rsync -av vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/u1_truthfulqa/ u1_truthfulqa/
```

**CLUSTER** (login node, it has internet; compute nodes do not), stage the third judge:

```bash
cd ~/llm-activation-steering-research && df -h $HOME && source .venv-dct-gpu/bin/activate && HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 huggingface-cli download Qwen/Qwen2.5-7B-Instruct && ls $HOME/hf_cache/hub | grep -i qwen
```

About 15 GB. If `df` shows less than 25 GB free, swap in `Qwen/Qwen2.5-3B-Instruct` (the fallback
in J3) and say so in the write-up.

---

## 11. Decisions needed from the PI

1. **The judge's backups.** The plan uses a Qwen2.5-7B-Instruct rubric judge plus 64 hand labels.
   Is a local open judge acceptable, or does the PI want an API judge (needs a key) as well?
2. **If `y^M` is dead on TruthfulQA too:** may gemma-2-2b-it serve as the *labeler only*, with
   steering still on the base model? This would give MAG a real label-free arm, but it brings in a
   second model's opinion of truth.
3. **Is S-beh "the DCT direction"?** It is the only way to name one DCT factor as "improving
   truthfulness", and it uses the judge to choose. We plan to report it next to S-none rather than
   in place of it.
4. **Dose matching across datasets** as a fraction of the target's residual norm. Is that the
   comparison the PI has in mind, or should doses be matched by effect size?

---

## 12. What this does to existing claims

- **Q2's truthful-only result stands**: 0.281 -> 0.516 at frac -2, random control 0.286. The
  truth judge was correctly prompted.
- **Q2's "truthful and informative" headline, and "informativeness does not collapse", are under
  review** until J2. They are not retracted: hedges are rare, so a large drop is not expected. But
  the informative column came from a mis-prompted judge and was never validated.
- **The DCT null is still split as in the September plan.** The geometric half stands. The
  behavioural half was measured where truth did not move, and D3 plus the TruthfulQA -> cities cell
  of X are what settle it.
- **MAG's "self-verdict is dead" finding stays scoped to the four old datasets** until G0 tests
  TruthfulQA.

---

## 13. Fulfilment check, run at write-up

Fill this in before the next meeting. Any row that is not "answered" goes into the meeting notes as
open, with the reason.

| # | The PI asked | Answered by | Backup that agrees? | Status |
|---|---|---|---|---|
| P1 | the two allenai judges | J0 unit test | model cards re-read at write-up | |
| P2 | strict or inconsistent? | J1 margins, J3 determinism and format flips | Qwen judge, hand labels | |
| P3 | DCT and MAG discovery on TQA | D1, D3, G0, G1 | D-R1 second seed, two G0 phrasings | |
| P4 | dataset dependent? | D1 table, X matrix | the reverse X cells | |
| P5 | MAG and DCT directions on cities | X cities column | two cities readouts, oracle positive control | |
| P6 | DCT works on TQA? | D3 | S-beh split halves, D-R1 | |
| P7 | quantify the dataset difference | the card | C1, C2, C3 agree on the form-vs-content row | |
| P8 | TQA DCT truthfulness direction on cities | X cell TQA S-beh -> cities | S-geo cell, potency-matched random DCT factor | |
