# Steering Validity Audit and Semantic Target Set: Design

**Date:** 2026-08-26
**Status:** Gate 1 (S1 to S4) complete and committed; see `docs/AUDIT_SUMMARY.md`. Gate 2 (D1)
is implemented and pre-registered in `docs/D1_DOSE_RESPONSE.md`, awaiting cluster time. Findings
1 and 3 below have been corrected by the experiments this document specifies; the correction
notes are inline and dated.
**Branch:** `feat/mag-e4-steering`
**Scope agreed with researcher:** Tier 0 (S1 to S4) + Tier 1 (D1) + T1. De-inerting specified but deferred.

---

## 1. Why this exists

At the last meeting the PI raised ten items. Most reduce to one challenge: *the null may be an
artifact of how we steered, not a fact about the model.* Specifically they asked whether the
final layer is too sensitive and we oversteered, whether the halfspace target set was built
correctly, whether the same linear relationship holds if we map from different source layers,
and whether naive and reachability steering were ever compared on matched parameters.

An audit of committed artifacts on 2026-08-26 found the PI is right on the central point, and
found a second problem they anticipated but that we had not measured. Both are recorded below.
Every number here was read from files already on disk. Nothing was re-run.

### Finding 1: the two steering arms never overlapped in perturbation size

> **Corrected by S3 on 2026-08-26.** The 6.1x figure holds on cities on the shared prompt set
> and was confirmed exactly (6.09x). It does not generalise: on common_claim the shared-prompt
> gap is 1.66x, and once the per-statement reachability arm is included the two ranges
> **overlap**. This paragraph reached its number from the mean arm alone. See
> `docs/S3_COMMON_AXIS.md` section 6.

Both arms inject at the same layer, so they are directly comparable.

| Dataset | Inject layer | Reachability arm swept | Naive arm swept |
|---|---:|---|---|
| `cities` | 11 | 1.40 to 5.60 (1.2% to 4.8% of h) | 34.14 to 113.80 (28.9% to 96.4%) |
| `common_claim_true_false` | 13 | per-statement budgets | 35.45 to 118.17 (23.4% to 78.1%) |

On `cities` the naive arm's *smallest* nonzero probe is **6.1 times larger than the
reachability arm's largest**. The band between them was never sampled by either arm. The
certified requirement (`delta_rel_z`, median 3.4% on cities and 1.1% on common_claim) sits
inside the reachability range and roughly an order of magnitude below the finest naive probe.

Consequence: "naive steering fails" was measured only at 30% to 100% perturbations, and
"reachability steering is behaviorally inert" only at 1% to 5%. The head to head comparison
the PI asked for has never been performed.

### Finding 2: the naive flip curve is non-monotone, and the transition is unsampled

From `mag_verdict_flips_cities.csv`, 8 of 9 directions flip 0.0% at absolute tau 0.3, then 50%
to 96% at absolute tau 1.0. `mag_u_gold` runs 83.3% at tau -1.0, 0% at -0.3, 29.2% at +0.3, and
back to **0% at +1.0**. A dose-response that returns to zero at the largest magnitude is a
measurement crossing a regime change, not a dose-response.

### Finding 3: the large-magnitude flips co-occur with degradation

> **Refuted by S3 on 2026-08-26.** Pooled by magnitude with an exact test on the judged
> free-form completions, the incoherence rise is not significant (p = 0.22 on cities, p = 0.16
> on common_claim) while the FALSE rise is (p < 5e-7 on both). The 15.6 against 21.2 comparison
> below is a single signed cell, not a magnitude effect. See `docs/S3_COMMON_AXIS.md`
> section 4.3.

Pooling `judge_mag_steer_cities.csv`, INCOHERENT runs 15.6% at tau 0 and 21.2% at tau -1.0
while TRUE falls 81.2% to 60.8%. On common_claim INCOHERENT reaches 24.3%. Some of what was
counted as a flip is the model breaking. Separately, `mag_resid_pc1` on cities flips 95.8% at
tau -1.0 and 95.8% at tau +1.0. An effect identical under negation is not carrying a signed
concept. There is no norm-matched random control anywhere in the behavioral arm, so we cannot
currently separate steering from a perturbation of that size.

### Finding 4: the certified target set is semantically vacuous by construction

`target_mode` in `token_geom_cities.npz` is `countries`. What the solver actually selected
across 200 statements is 16 distinct tokens, of which " North" (87) and " South" (83) are
**85% of all targets**. At the certified budget the oracle direction flips the argmax on
200 of 200 and the text stays fluent, but the claim's truth value does not move:

| stem | unsteered | steered |
|---|---|---|
| The city of Weifang is in | the north of Shandong Province, Ch | **North** East China, and is the capit |
| The city of Nowrangapur is in | the district of Srikakulam in | **South** Odisha, India. It is located |
| The city of Warri is in | Delta State, Nigeria. It is the | **the** Delta State of Nigeria. It is |

Weifang is in China, Nowrangapur is in India, Warri is in Nigeria's Delta State. Each row is a
certified success and a semantic no-op, while `readout_delta` is 13.86, so the probe moved
enormously and the fact did not move at all.

The mechanism: minimum norm is adversarial to semantics. The cheapest face of the argmax cone
is by definition the token the model was already nearly going to emit, so a min-norm objective
systematically finds the vacuous flip. " North" is cheap precisely because "North East China"
was already plausible.

This is the PI's "halfspace isn't constructed properly", now with a measured mechanism. It
reframes the project: the control theory works, the certificate is tight, the perturbation is
benign to fluency, and the defect is in how the target region was specified.

---

## 2. What this design does and does not cover

**In scope:** four CPU-only audits on committed artifacts (S1 to S4), one GPU dose-response
sweep that fills the unsampled band (D1), and one restricted-target-set re-solve (T1).

**Deferred with reasons:**

| Item | Reason |
|---|---|
| System level synthesis, backward reachability for linear time-varying systems | The PI called it possible overkill. Unjustified until D1 and T1 report |
| Temperature sweep | Settled at T = 0. Gets one stated paragraph and a single robustness point, not a sweep |
| De-inerting the datasets | Designed in section 9, held until Tier 0 and Tier 1 report. Runs as a separate arm so nothing committed is invalidated |

**Hard constraint on every task below:** no committed artifact is re-run, regenerated, or
overwritten. Every experiment writes to a new filename. The full protected list is section 10.

---

## 3. S1: direction asymmetry audit

**Assumption under test.** Behavioral flips attributed to steering are directional effects of a
truth-carrying vector, not magnitude effects of an arbitrary perturbation.

**Prediction registered before running.** If flips are directional, flip rate at +tau and -tau
differ substantially for the supervised directions (`sup_grad`, `sup_mean_diff`, `mag_u_gold`).
If they are magnitude effects, the rates match within sampling noise. `mag_resid_pc1` is
predicted symmetric, since 95.8 against 95.8 is already visible in the committed file.

**Inputs (read only).** `mag_verdict_flips_{ds}.csv`, `judge_mag_steer_{ds}.csv`.

**Method.** For each direction compute an asymmetry index at each magnitude,
`asym = abs(flip(+tau) - flip(-tau))`, and a two-sided exact binomial test on the 2x2 table of
flip counts at +tau against -tau. Separately compute incoherence rate against absolute tau per
direction. Classify each direction as directional, symmetric, or inert.

**Outputs.** `src/audit_asymmetry.py`, `audit_asymmetry_{ds}.csv`,
`plot_s1_asymmetry_{ds}.png`, `docs/S1_ASYMMETRY.md`.

**Cost.** CPU, seconds. Only pandas, numpy, scipy. No torch, no xgboost.

**How D1 depends on it.** D1 carries only the directions S1 classifies as directional, plus the
norm-matched random control. Spending GPU time sweeping a symmetric direction is spending it on
a perturbation, not a steering vector.

---

## 4. S2: layer sweep of readout against controllability

**Assumption under test.** This is the PI's exact question: does the same linear relationship
hold if we map from different earlier layers, and does perturbation affect the halfspace metric
the same way at each?

**Prediction registered before running.** Readout accuracy rises with depth and saturates in
the middle layers. The controllability margin behaves oppositely, since `token_jac_{ds}.csv`
already shows `m_all_median` falling from 1.528 at layer 0 to lower values with depth. The two
peaks are predicted to be in different places, which would mean the layer that decodes truth
best is not the layer from which truth is cheapest to actuate.

**Inputs (read only).** `mag_acts_{ds}.npz` (holds `A_p` at shape 27 by N by 2304, so all 27
layers are already cached and no GPU is needed), `token_jac_{ds}.csv` (already holds
`m_all_median`, `eps_all_median`, `broadcast_gain` for 26 layers).

**Method.** Load `A_p` once into memory, never index the npz inside a loop. For each of the 27
layers: 80/20 stratified split at `random_state=42`, `StandardScaler` fit on train,
`LogisticRegression(max_iter=2000)`, record test accuracy and ROC AUC. Compute the layer's
mean-difference direction and its cosine against the layer-11 (cities) or layer-13
(common_claim) direction actually used for steering, which measures whether the concept is the
same object at every depth or drifts. Overlay against `m_all_median` and `eps_all_median` from
`token_jac_{ds}.csv` on a twin axis.

**Outputs.** `src/audit_layer_sweep.py`, `audit_layers_{ds}.csv`,
`plot_s2_layers_{ds}.png`, `docs/S2_LAYER_SWEEP.md`.

**Cost.** CPU. 27 fits of 1496 by 2304 on cities, larger on common_claim. Minutes.

**Risk.** `mag_acts_common_claim_true_false.npz` is large. If memory is tight, process one
layer at a time by slicing on load with a single read per layer, still never re-indexing the
npz object inside the loop.

---

## 5. S3: put both arms on one axis

**Assumption under test.** The naive and reachability results are comparable as reported.

**Prediction registered before running.** They are not. The two swept ranges are disjoint by
roughly a factor of 6 on cities, with the certified budget lying inside the reachability range
and far below every naive point.

**Inputs (read only).** `mag_dir_{ds}.npz` (`A_prefix_norm`, `layer`), `reach_acts_{ds}.npz`
(`h_src`, `src_layer`), `mag_verdict_flips_{ds}.csv`, `judge_reach_steer_{ds}.csv`,
`judge_reach_steer_stmt_{ds}.csv`, `token_geom_{ds}.csv` (`delta_rel_z`).

**Method.** Adopt one canonical denominator, the median of the norm of `h_src` at the injection
layer, and express every swept point of both arms as relative perturbation size. Record that
the normalizers differ (on cities `A_prefix_norm` 113.80 against median h norm 118.06, a 4%
difference; on common_claim 118.17 against 151.28, a 28% difference), and state which is used.
Plot flip rate for both arms on the shared axis, shade the unsampled band, and draw a vertical
line at the median certified budget.

**Outputs.** `src/audit_common_axis.py`, `audit_common_axis_{ds}.csv`,
`plot_s3_common_axis_{ds}.png`, `docs/S3_COMMON_AXIS.md`.

**Cost.** CPU, seconds.

**Note.** Confirm that both arms use the same injection operator (all positions against last
token only) before claiming strict comparability. If the operators differ, report the
comparison as approximate and state the difference. Do not assert a match that has not been
checked in the code.

---

## 6. S4: target set census and semantic vacuity rate

**Assumption under test.** The certified target set represents the behavior we care about,
namely the model asserting something false.

**Prediction registered before running.** It does not. Finding 4 already shows 85% of targets
are " North" or " South". The vacuity rate, meaning the fraction of certified argmax flips that
leave the claim's truth value unchanged, is predicted above 0.8 on cities.

**Inputs (read only).** `token_geom_{ds}.csv`, `token_steer_{ds}_postnorm_all_rp1.csv`
(read through `token_conclusions.load_arm`, never a bare `pd.read_csv`).

**Method.** Census the target token distribution. Then take the oracle arm at `frac = 1.0`,
where `hit_target` is 1.000, and judge each steered completion against its unsteered
counterpart for whether the claim's truth value changed, using the existing judge in
`src/judge_results.py` with the Haiku 4.5 backend. Report the vacuity rate with a Wilson
interval. Cross-tabulate vacuity against `readout_delta` to show the probe moving while the
claim does not.

**Outputs.** `src/audit_target_census.py`, `audit_target_census_{ds}.csv`,
`judge_vacuity_{ds}.csv`, `plot_s4_target_census_{ds}.png`, `docs/S4_TARGET_CENSUS.md`.

**Cost.** CPU plus roughly 400 short judge calls per dataset, cents at Haiku pricing.

---

## 7. D1: the dose-response kill test

**Assumption under test.** The behavioral null is a fact about the model, not an artifact of
sampling only oversteered magnitudes. This is simultaneously the PI's oversteering check and
their falsification condition: if naive steering succeeds cleanly at a small magnitude, the
reachability framing is unnecessary.

**Prediction registered before running.** State both branches in advance and commit to them.

- *Null is real:* flip rate stays at the unsteered baseline across the whole band until
  incoherence begins rising, and the two curves then rise together. No magnitude produces
  elevated flips at flat incoherence. The random control matches the truth directions
  everywhere, showing the effect is norm-driven.
- *PI is right:* there exists a window, expected near the certified budget of 1% to 5%, where
  flip rate is elevated while incoherence is at baseline, and the truth directions separate
  from the random control inside that window.

**Inputs.** Directions surviving S1 as directional, at most three per dataset, plus a
norm-matched random unit vector, plus the reachability direction `jtw_mean_diff_tgt` so both
arms land on one grid. The 32 prompts already used by `mag_steer_{ds}.csv`.

**Method.** Inject at layer 11 (cities) and layer 13 (common_claim), the same sites the
committed arms used. Sweep relative magnitude on a log-spaced grid covering the unsampled band
and both endpoints:

```
0.005, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 1.00
```

> **Superseded by S3 on 2026-08-26.** This grid puts 8 of 12 points below 0.20, but S3 brackets
> the behavioral transition between 0.29 and 0.78 and shows the naive directions are already at
> baseline at 0.23 to 0.29. The revised grid is
> `0.01, 0.02, 0.04, 0.08, 0.15, 0.22, 0.30, 0.40, 0.52, 0.66, 0.82, 1.00`. Rationale in
> `docs/S3_COMMON_AXIS.md` section 7.

twelve magnitudes, both signs, plus zero, so 25 points per direction. Note that 0.035 is the
median certified budget on cities and is deliberately a grid point. Generate at temperature 0,
matching the committed convention. Judge every completion for flip and for coherence as two
separate outcomes, never collapsing them into one rate.

Total generations: 5 directions by 25 magnitudes by 32 prompts = 4,000 per dataset, 8,000
across both.

**Outputs.** `src/dose_response.py`, `dose_{ds}.csv`, `judge_dose_{ds}.csv`,
`plot_d1_dose_{ds}.png` (flip rate and incoherence rate on one axis against relative magnitude,
log x, with the certified budget marked and the previously unsampled band shaded),
`docs/D1_DOSE_RESPONSE.md`.

**Cost.** A few GPU hours on DeltaAI plus roughly 8,000 short judge calls. Use a tight SLURM
`--time` per the cluster notes.

**Artifact safety.** Writes `dose_{ds}.csv`. Must not write to `mag_steer_{ds}.csv` or any
existing name.

---

## 8. T1: restricted, semantically meaningful target set

**Assumption under test.** A target set that actually encodes falsity is reachable at a budget
comparable to the vacuous one. If it is, the certificate framework survives Finding 4 intact
and only the target specification was wrong. If the budget explodes, the framework's practical
reach is much narrower than claimed.

**Prediction registered before running.** The restricted budget will exceed the unrestricted
one, since we are removing the cheapest faces by construction. The open quantity is the ratio.
Predict a median ratio between 2 and 10. A ratio above 50 would mean semantically meaningful
targets are effectively unreachable at benign perturbation sizes, which is itself a strong and
reportable result.

**Method, four steps.**

1. **Re-extract the postnorm state.** `h_final` in `reach_acts_{ds}.npz` is a different site
   (median norm 229.2 against `z_norm` 189.2), so it cannot be reused. Run 200 forward passes
   for the rows in `token_geom_{ds}.npz`'s `row_index`, capturing the postnorm residual state.
   Write `token_z_{ds}.npz`. Verify the recovered norms match `z_norm` in `token_geom_{ds}.csv`
   before proceeding. If they do not match, stop and diagnose rather than continuing.
2. **Build the allowed target set per statement.** For cities the claim is "The city of X is in
   Y." The allowed set is country-name tokens excluding the true country. Explicitly exclude a
   stoplist of directional and function tokens that produce vacuous flips: North, South, East,
   West, the, a, an, and their leading-space variants. Record the stoplist in the output so the
   choice is auditable.
3. **Re-solve the cone problem restricted to that set.** Reuse `_min_norm_ineq` and
   `cone_budget` from `src/token_geom.py` unchanged. For each allowed target token solve the
   single-target cone problem and take the minimum over allowed targets. Keep the existing
   full-vocabulary argmax recheck so a claimed solution is verified to actually win the argmax.
4. **Verify behaviorally.** Steer at the restricted budget and judge whether the claim's truth
   value now changes, which is the metric S4 shows the current target set fails.

**Outputs.** `src/token_target_semantic.py`, `token_z_{ds}.npz`,
`token_geom_sem_{ds}.csv`, `judge_semantic_{ds}.csv`,
`plot_t1_semantic_budget_{ds}.png` (restricted against unrestricted budget, paired per
statement), `docs/T1_SEMANTIC_TARGET.md`.

**Cost.** Extraction is 200 forward passes, minutes. Solving is CPU over the unembedding.
Behavioral verification is 200 generations plus 400 judge calls.

**Risk.** The allowed set for `common_claim_true_false` is much harder to define than for
cities, because the claims are not templated. Start with cities only. If cities works, design
the common_claim variant as a follow-on rather than guessing at it now.

---

## 9. Deferred: the de-inerting arm

**Problem.** 103 of 199 cities statements are byte-identical at every steering scale, so half
the dataset contributes nothing to any behavioral measurement and depresses the statistical
power of every result above.

**Design, held until Tier 0 and Tier 1 report.** Vary the elicitation prompt so the model must
commit to an answer, rather than being free to continue with unrelated text. Candidate
operators already exist in `mag_readability_{ds}.csv`, which shows readout accuracy varying by
operator from 0.625 (Verdict) to 0.993 (Direct), so the operators are known to behave
differently. Metric is the inert fraction, meaning statements byte-identical at every scale.
Runs as a separate arm writing `deinert_{ds}.csv`, invalidating nothing committed, per the
researcher's decision on 2026-08-26.

---

## 10. Artifact safety

**Protected, read only, never regenerated.** All `token_steer_*.csv`, `token_geom_*`,
`token_jac_*`, `token_sens_*`, `token_acts_*`, `sae_features_*`, `signed_steer_summary_*`,
`reach_*.npz`, `reach_*.json`, `mag_*.csv`, `mag_*.npz`, and every `judge_*.csv` currently at
repo root. These are finished cluster output and are irreplaceable.

**Every new file introduced by this design uses a new name.** `audit_*`, `dose_*`,
`token_z_*`, `token_geom_sem_*`, `judge_vacuity_*`, `judge_dose_*`, `judge_semantic_*`,
`deinert_*`, `plot_s1_*` through `plot_t1_*`.

**Standing engineering rules.** Run everything with `./.venv/bin/python`. Never import
`xgboost` in the same process as `torch`. Never index into a loaded npz inside a loop. Read
every `token_steer_*.csv` through `token_conclusions.load_arm`. Never pool the two datasets.
Stage named files only, never `git add -A`. No em dashes in any generated prose.

---

## 11. Documentation framework

One document per experiment under `docs/`, each with the same five headings, in this order:

1. **Assumption** being tested, stated as a falsifiable proposition
2. **Prediction** registered before the run, with both branches named
3. **Method**, including the exact command that produced the result
4. **Result**, with the figure and the numbers
5. **Verdict** on the assumption: upheld, refuted, or underpowered

The prediction-before-running discipline is the point. It is what makes the set presentable,
and it is the cleanest thing to hand a new collaborator, because every claim in the project
then carries a falsifiable test and a pre-registered expectation next to it.

A single `docs/AUDIT_SUMMARY.md` collects the verdict line from each experiment into one table.

---

## 12. Sequencing and gates

```
S1 ─┬─> D1 (S1 selects which directions D1 carries)
S2 ─┤
S3 ─┘
S4 ───> T1 (S4 quantifies the defect T1 fixes)

D1 and T1 are independent of each other and can run concurrently.
```

**Gate 1.** S1 to S4 all complete before any GPU time is spent. They are free, they are fast,
and S1 determines D1's direction list while S4 determines whether T1 is worth building.

**Gate 2.** D1's verdict determines what happens to the deferred items. If D1 finds a clean
steering window, the project pivots toward characterizing that window and backward reachability
stays shelved. If D1 confirms the null across the full band, the null hardens substantially,
because it would then be established that the certified budget is reachable, benign to fluency,
and behaviorally silent, and that only destructive magnitudes move behavior.

**Gate 3.** T1's budget ratio determines whether the semantic target set is a fix or a finding.
A modest ratio makes it a fix and the certificate framework proceeds with a corrected target. A
large ratio makes it a finding about the limits of minimum-norm actuation, which is reportable
in its own right.

---

## 13. Mapping to the PI's ten items

| PI item | Addressed by | Status entering this work |
|---|---|---|
| Final layer sensitive, try smaller magnitudes | D1 | Open, the main gap |
| Super fine grained sweep in case of oversteering | D1 | Open, 5 grid points with the transition unsampled |
| Compare naive and reachability parameters | S3, D1 | Open, now possible: same layer, common axis established |
| Halfspace not constructed properly | S4, T1 | Confirmed by Finding 4, with a mechanism |
| Sweep input layers, same mapping from earlier layers | S2 | Free, all 27 layers already cached |
| Same linear relationship at each layer | S2 | Partly done, `token_jac` already covers 26 layers of margin |
| Pick a temperature and commit | Section 2 | Settled at T = 0, needs stating once |
| Token space, cone, vocabulary geometry | Prior work | Largely done, 200 cone solutions over 255,999 faces |
| Map relevant words back into activation space | T1 | Open, and Finding 4 makes it the highest value item |
| Backward reachability, system level synthesis | Deferred | The PI's own "might be overkill", revisit after D1 |
