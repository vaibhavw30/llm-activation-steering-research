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

## 3. Track J: the judge audit (feedback point 1)

**J0. Fix the prompt.** `truthfulqa_prompt` takes a `kind` argument, `True:` for truth and
`Helpful:` for info, and `TruthJudge.score` builds one prompt per judge. A unit test pins both
strings against the model cards. The old behaviour is not kept behind a flag.

**J1. Validate the info judge, and record margins.** Two changes to `_YesNoJudge`:

- Return `p_yes = P(" yes") / (P(" yes") + P(" no"))` from the first-token logits, as well as the
  hard verdict. That gives strictness a number: how many verdicts sit near 0.5, and how the
  headline moves if the threshold moves.
- Add an **informativeness gold set** to the gold check, reported per side as the truth gold check
  is. Expected informative: each holdout question's `best_answer`. Expected uninformative:
  `I have no comment.` and the question restated as its own answer. 64 x 3 = 192 rows.

Pass bar, registered now: per-side accuracy >= 0.9 on both info sides with the `Helpful:` prompt.
The `True:` prompt runs on the same set as a record of the bug.

**J2. Re-judge everything already generated.** Judge only, no generation: Q1's 64 baseline
answers, the Q2 mean arm (576), the random control (1,728), and the per-statement arm if it has been judged.
Output goes to new files (`judge_v2_truthfulqa_<arm>.csv`, `tqa_baseline_judged_v2.csv`). No
existing judge file is overwritten. `tqa_q2_summary` is recomputed from the v2 files with the
existing script.

**J3. Consistency.** Three cheap checks on the J2 output:

1. **Determinism.** The same 200 rows judged twice agree 200/200. Greedy fp16 should, but the claim
   has never been checked.
2. **Format invariance.** Each answer judged with and without its trailing period, and with
   whitespace normalised. The flip rate is the judge's noise floor for these edits.
3. **A second opinion.** 64 answers (32 at frac 0, 32 at frac -2, shuffled, dose hidden) labeled by
   hand for truthful and informative. Report agreement and kappa per judge. An LLM judge could
   stand in, but it needs `ANTHROPIC_API_KEY`, which has been blocked before, so hand labels are
   the default.

**Decision rule, registered now.** If the v2 truthful-and-informative rate at frac -2 still beats
frac 0 and the random control with exact McNemar p < 0.05, Q2 stands as written. If
informativeness drops at frac -2 under the correct prompt, Q2 is restated as a truthful-only result
with an informativeness cost, and `Q2_TRUTHFULQA_STEERING.md` gets a correction section. Sections
before it are not rewritten.

Cost: CLUSTER, 1 to 2 GPU-hours, all judge time. **J runs first**, because every later TruthfulQA
behavioural number goes through this judge.

---

## 4. Track D: DCT discovery on TruthfulQA (feedback point 2)

**D0. Pull U1.** See section 8 for commands. If it failed, fix and resubmit before anything below.

**D1. The geometric comparison, measured the same way as cities.** Rerun the three tests from
`DCT_VS_TRUTH_FINDINGS.md` on TruthfulQA's 512 factors, against `mean_diff_tgt` and
`probe_grad_tgt`: max single-factor cosine vs random, subspace fraction vs chance k/d = 0.22, and
the orthogonality to DCT's top-potency factor. The TruthfulQA row goes straight into the existing
cities/common_claim table. LAPTOP, once `dct_V_truthfulqa.pt` is down.

**D2. Name "the DCT direction that improves truthfulness".** The feedback's last point needs a
single DCT direction, and DCT gives 512 unlabeled factors. Choosing one means deciding what is
allowed to see labels. Three selection rules, all pre-registered and all reported:

| Rule | Picks | Sees labels? |
|---|---|---|
| **S-none** | the top-4 factors by potency, as the margins battery already does | no |
| **S-geo** | the factor with the largest abs cosine to `mean_diff_tgt` | yes, via the direction |
| **S-beh** | a behavioural screen: the top 16 factors by potency, both signs, one dose, on a **screen split** of 96 TruthfulQA questions disjoint from the 64-question holdout. Pick the factor and sign with the largest gain in truthful-and-informative (v2 judges) | yes, via the judge |

S-beh is what "direction improving truthfulness" most literally means. S-none is what "unsupervised"
means. Report both, and do not mix them up in the write-up.

**D3. Confirm on the holdout.** The selected factor(s) from each rule, steered on the 64 holdout
questions over the Q2 dose grid, judged with v2, next to the existing norm-matched random control.
Selection never touches the holdout, so the S-beh number is not inflated by choosing the winner.

**Registered expectation.** U1's own report already registers that few or no `dct_u` members will
clear `MIN_ACC_1D` (3 of 74 directions do on TruthfulQA). For D3: S-none's factors do not move
truthfulness beyond the random control. S-beh finds one that does, and that factor has |cos| > 0.3 with
`mean_diff_tgt`. If S-beh's winner is near-orthogonal to `mean_diff_tgt` and still works, that is
the most interesting outcome on the page: a second truth lever the supervised pipeline missed.

Cost: CLUSTER. Screen ~3,100 generations plus judging, holdout ~1,700 plus judging. About 4 to 6
GPU-hours as one job.

---

## 5. Track G: MAG on TruthfulQA (feedback point 2)

**G0. Is the verdict channel alive here?** Add `truthfulqa` to `mag.config.DATASETS`, run MAG
extraction at layer 11 on the TruthfulQA fit set, and report the `y^M` split and `agree(y^M, gold)`.
Pass bar: at least 20% "no" and agreement >= 0.6. CLUSTER, under an hour.

**G1, if alive.** MAG's label-free direction `u_yM` runs for the first time on any dataset:
readout on the holdout, cosine to `mean_diff_tgt` and to the D2 winners, then steered through D3's
harness. This would be the project's first fully label-free truth direction that did not come
from DCT.

**G1, if dead.** Report `u_Q` (gold arm) and its cosine to `mean_diff_tgt`, predicted >= 0.95. If
that holds, MAG's behavioural result on TruthfulQA is Q2's result and is not re-steered. The only
way forward for the label-free arm is an instruction-tuned labeler (decision 2, section 9).

---

## 6. Track X: the transfer matrix (feedback points 2 and 4)

Each direction, from each source dataset, steered on each target dataset:

| Direction family | from cities | from TruthfulQA |
|---|---|---|
| supervised `mean_diff` (the Q2 direction on TQA) | exists | exists |
| DCT, S-none and S-beh | exists (`dct_V_cities.pt`) | D2 |
| MAG `u_Q`, and `u_yM` if G0 passes | exists (`mag_dir_cities.npz`) | G |

Targets: **cities** and **TruthfulQA**. The diagonal is what we already have (cities nulls, Q2). The
off-diagonal is new. The cell the PI asked for is **TruthfulQA's S-beh DCT factor steered on
cities**.

**Controls on every target.** A norm-matched random direction, as Q2's control. On cities also the
`jtw_token` oracle, which flips the answer 34 to 88% of the time (`token-space-diagnosis`), so a null on
cities cannot be blamed on a broken harness.

**Dose units across datasets.** A dose defined by one dataset's eps* is meaningless on another. The
grid is set as a fraction of the **target** dataset's median residual norm at layer 11, so the same
row means the same relative push on both datasets. eps* units are reported alongside where they
exist.

**Sign.** TruthfulQA label 1 is untruthful, so its supervised truthful direction is `-v`. Cities
label 1 is true. DCT factors take the sign D2 picked. Every direction is flipped to point toward
truthful before the matrix is built, and this step has a unit test, because the Q0 inversion has
already caused one wrong reading.

**Cities readout.** The readout the cities nulls were measured with: the statement stem's
completion, correct country vs not, both signs. The model already gets most true stems right, so
"toward truth" has little room. The informative test is two-sided: can the direction make it
**wrong** on true stems (-), and can it make it **right** where it was wrong (+).

**Three outcomes, registered now, for the TruthfulQA -> cities cell:**

- **(a) Facts flip on cities.** Truth is a shared lever, and the cities null was a property of the
  directions mined on cities. The largest possible revision to the project's claim.
- **(b) Cities completions change form, facts do not.** Longer, hedged, or off-format completions at
  the same rate of correct countries. The TruthfulQA direction is an elaboration lever, and
  TruthfulQA's judge scores elaboration as truth. Reading 2.5's second option wins.
- **(c) Nothing moves beyond the random control.** Dataset-specific direction, no finding either
  way.

The reverse cell, cities -> TruthfulQA, costs the same and is in the matrix. The cities direction was
behaviourally inert on cities, so a TruthfulQA gain from it would locate the difference in the
dataset rather than in the direction.

**Readout transfer, cheap.** Cosines between the datasets' directions, and each dataset's probe
applied to the other's activations at layer 11. LAPTOP, from existing activations plus
`reach_dirs_truthfulqa.npz` once it is pulled down.

Cost: CLUSTER, one job. About 6 directions x 2 targets x 7 doses, plus controls. Cities generations
are short. Roughly 4 to 6 GPU-hours with judging.

---

## 7. Track C: the dataset card (feedback point 3)

One table with the columns **cities**, **common_claim**, **TruthfulQA**, and every row computed by the
same code on all three. Most rows exist for the old datasets already and only need the
TruthfulQA entry.

| Group | Row | Where it comes from |
|---|---|---|
| Representational | probe accuracy and XGBoost gap at layer 11 | `analyze.py`, TQA activations |
| | MAG linearity `eps_Q` | G0 |
| | cosine between truth directions across datasets | X, readout transfer |
| Causal geometry | DCT alignment: max cos vs random, subspace vs chance | D1 |
| | share of the margins battery with a valid threshold (`MIN_ACC_1D`) | TQA: 3 of 74; cities, common_claim from existing `reach_summary` |
| | eps* as a fraction of the residual norm | `reach_summary` |
| Behavioural | unsteered rate of the desired behaviour (headroom) | Q1, cities baseline |
| | best gain from the supervised direction, minus the random control | Q2 (v2), cities nulls |
| | off-target change at that dose: length ratio, incoherence | Q2, WARM_DCT |
| | share of the gain explained by answer length | the LR test in 2.5, run the same way on cities |
| Task form | answer format (one token vs free text); does the judge reward form? | by construction, and X outcome (b) |

**The hypothesis the card tests, registered now:** TruthfulQA is steerable and cities is not
because TruthfulQA's truth score has a *form* component (fuller, qualified answers) that a
direction can push, and cities' single-token factual answer has none. It predicts: a large length
share on TruthfulQA, near zero on cities, and outcome (b) in track X. It is refuted by outcome (a),
or by a TruthfulQA gain that survives conditioning on length once the judge is fixed.

Cost: LAPTOP, a day once J2, D1, G0 and X are in.

---

## 8. Sequencing and cost

| id | What | Machine | Cost | Needs | State |
|---|---|---|---|---|---|
| D0 | pull U1, or find why it did not run | LAPTOP + CLUSTER | minutes | nothing | **first** |
| J0-J1 | fix the info prompt, add margins, info gold set | LAPTOP | half a day | nothing | **first** |
| J2-J3 | re-judge everything, consistency checks | CLUSTER | 1-2 GPU-hr | J1 | next |
| D1 | DCT geometry on TQA, cities-comparable | LAPTOP | an hour | D0 | next |
| G0 | MAG extraction on TQA, is `y^M` alive | CLUSTER | < 1 GPU-hr | nothing | can ride with J2 |
| D2-D3 | select DCT truth factor(s), confirm on holdout | CLUSTER | 4-6 GPU-hr | J2, D0 | after J |
| G1 | MAG label-free arm, if G0 passes | CLUSTER | 1-2 GPU-hr | G0, D3 harness | conditional |
| X | transfer matrix, both directions | CLUSTER | 4-6 GPU-hr | D2, G0 | after D |
| C | dataset card | LAPTOP | a day | J2, D1, G0, X | last |

**J before any new TruthfulQA behavioural number.** D2's selection and everything in X use the info
judge, so running them first would bake the bug into new results.

**What this pushes back.** V1 (pipeline validation), Q3 (A-LQR's code) and the -3.5 eps* extension
from the September plan are not dropped. They wait behind J and D.

First commands, for D0.

**LAPTOP**, to check the cluster from here:

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu 'cd ~/llm-activation-steering-research; sacct --name=tqa_dct --starttime=2026-09-01 --format=JobID%14,State%20,ExitCode,Elapsed,End; ls -l dct_V_truthfulqa.pt dct_U_truthfulqa.pt u1_truthfulqa/reach_summary_truthfulqa.json; grep -o "\"num_factors\": [a-z0-9]*" dct_meta_truthfulqa.json'
```

**LAPTOP**, if it completed:

```bash
cd ~/llm-activation-steering-research && rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{dct_V_truthfulqa.pt,dct_U_truthfulqa.pt,dct_meta_truthfulqa.dctfit.json,reach_dirs_truthfulqa.npz,tqa_dct_*.out}' . && rsync -av vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/u1_truthfulqa/ u1_truthfulqa/
```

---

## 9. Decisions needed from the PI

1. **Second opinion on the judge.** Hand-label 64 answers (default), or an LLM judge, which needs
   an API key?
2. **If `y^M` is dead on TruthfulQA too:** may gemma-2-2b-it serve as the *labeler only*, with
   steering still on the base model? This would give MAG a real label-free arm, but it brings in a
   second model's opinion of truth.
3. **Is S-beh "the DCT direction"?** It is the only way to name one DCT factor as "improving
   truthfulness", and it uses the judge to choose. We plan to report it next to S-none rather than
   in place of it.
4. **Dose matching across datasets** as a fraction of the target's residual norm. Is that the
   comparison the PI has in mind, or should doses be matched by effect size?

---

## 10. What this does to existing claims

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
