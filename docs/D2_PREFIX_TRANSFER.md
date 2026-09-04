# D2: the truth readout does not transfer to the generation population

**Run:** LAPTOP, 2026-09-04, `src/reach_stemprobe.py --summary`. Inputs are all July cluster
output: `reach_stemacts_<ds>.npz` (target-layer activations at each generation stem's last
token, extracted on DeltaAI 2026-07-29), `reach_dirs_<ds>.npz` (the `mean_diff_tgt` direction
and its statement-fitted threshold), `judge_reach_steer_stmt_<ds>.csv` (scale-0 verdicts,
which are the labels). Artifacts: `reach_d2_summary_<ds>.csv`.

This is Horizon-0 item 0.2 and Track V item V0. The `--fit` path computed these numbers in July
but only printed them and wrote four columns, so they were lost. Nothing was recomputed on the
model: the fit path is torch-free, and running it now reproduces `reach_stemprobe_<ds>.csv`
byte for byte.

## Headline

**On the population we actually steer, the truth readout carries no usable signal.** Balanced
accuracy is exactly 0.500 on both datasets, AUC is 0.510 and 0.568, the readout never changes
sign, and every refit, linear or nonlinear, collapses to the base rate.

D2 was previously stated as "the probe boundary does not transfer to generation prefixes,"
which reads as a threshold problem. It is not a threshold problem. There is no threshold on
this direction that separates the classes.

## The numbers

Source layer 11, target layer 20, gemma-2-2b. Labels: `TRUE` = the model completed the stem
truthfully at scale 0.

| | cities | common_claim_true_false |
|---|---:|---:|
| labeled stems, n | 158 | 178 |
| TRUE / FALSE | 149 / 9 | 168 / 10 |
| base rate P(TRUE) | 0.943 | 0.944 |
| `g_old` range | -83.5 to -43.1 | +48.9 to +118.9 |
| readout ever changes sign | **no** | **no** |
| accuracy at the statement-fitted threshold | 0.057 | 0.944 |
| **balanced accuracy** | **0.500** | **0.500** |
| **AUC of `g_old`** | **0.510** | **0.568** |
| recalibrated 1-D threshold on the same `w` | 0.943 | 0.944 |
| refit LR probe, 5-fold CV | 0.930 | 0.944 |
| **XGBoost, 5-fold CV** | **0.943** | **0.944** |
| nonlinear gap (XGB minus LR) | +0.013 | +0.000 |

Every entry in the bottom four rows equals the base rate to three decimals (149/158 = 0.943,
168/178 = 0.944). The cities LR refit is *below* its base rate. The recalibrated threshold
lands at `t02_stem` = -251.3, far outside the observed range [-83.5, -43.1], which is what
"predict TRUE for everything" looks like when you write it as a threshold.

The two datasets fail in mirrored ways, which is worth noting because it rules out a sign
convention bug. Every cities stem reads deep FALSE and 94% are completed truthfully, so the
statement-fitted threshold scores 0.057. Every common_claim stem reads TRUE, so the same
threshold scores 0.944 and looks like it works. Both are the same fact: a constant prediction.

## The pre-registered question, answered

Roadmap item 0.2 asked whether prefix-transfer failure is **a threshold shift (recalibrable) or
absence of signal**, and predicted the XGBoost arm would find nonlinear headroom on
common_claim, since Project 1 measured a gap of +0.082 at layer 13 on declarative statements.

The answer is absence of signal, and the headroom does not survive the move to stems. At the
target layer where the stem probe reads:

| layer 20, common_claim | linear | XGBoost | gap |
|---|---:|---:|---:|
| declarative statements (P1) | 0.706 | 0.767 | **+0.062** |
| generation stems (D2) | 0.944 | 0.944 | **+0.000** |

| layer 20, cities | linear | XGBoost | gap |
|---|---:|---:|---:|
| declarative statements (P1) | 0.993 | 0.993 | +0.000 |
| generation stems (D2) | 0.930 | 0.943 | +0.013 (both at or below base rate) |

Same model, same layer, same probe family. A linear probe reads truth on cities statements at
0.993 and at the base rate on cities stems.

## What this is not

**It is not a claim that truth is absent from stem activations.** The test is underpowered in
one specific direction: only 9 and 10 stems are labeled FALSE, so a 5-fold CV has roughly two
negatives per fold. What the result licenses is an upper bound. Any signal is small enough to
be invisible at this n, and it is certainly not the large, sign-separated signal the statement
population shows.

The underpowering is itself the argument for changing datasets. The model spontaneously
completes falsely 5.7% of the time here, so there is close to no behavior to move and any
steering effect would rest on nine examples. See `PLAN_ADVISOR_NOTES_2026-09.md` §1.3 and
Track Q.

## What it changes

**Nothing already published is retracted.** The audit's `readout-only` verdict on truth stands.
What changes is where the cause sits. Read with §1.1 of the plan doc, which shows the Jacobian
pullback is exact at its own linearization point (cities calibration 0.9997) and loses 13.7x at
the generation stem, the picture is consistent: **the readout and the pullback are both fine
where they were built, and both are evaluated somewhere else.**

For Track V this closes one of the three V2 candidates in advance. "Fit the direction on the
generation population" was listed first because V0 would tell us whether such a direction
exists. At this n it does not. V2 should go to the contrastive verdict readout, which is a
token-space object and cannot dissociate from behavior, and the question of whether a
stem-fitted direction exists should be re-asked on TruthfulQA where the negatives are not nine.

## Reproducing

**LAPTOP**, seconds, no GPU:

```bash
PYTHONPATH=src ./.venv/bin/python src/reach_stemprobe.py --dataset cities --summary
```

`--summary` writes only `reach_d2_summary_<ds>.csv`. Use it rather than `--fit`, which also
rewrites `reach_stemprobe_<ds>.csv`; that file is July cluster output and there is no reason
to touch it. The path never imports torch, so the XGBoost arm is safe to run here (torch and
xgboost each load their own libomp and segfault macOS ARM when they share a process).
