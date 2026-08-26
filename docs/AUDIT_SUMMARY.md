# Steering Validity Audit: Verdict Summary

One line per experiment. Full method, pre-registered prediction, and numbers in the linked
document. Design: `docs/superpowers/specs/2026-08-26-steering-validity-audit-design.md`.

**Gate 1 (S1 to S4) complete, 2026-08-26.** All four ran on committed artifacts, CPU only, in
under a minute of compute each. No GPU time has been spent. Nothing on disk was regenerated.

| id | assumption tested | verdict | headline number |
|---|---|---|---|
| [S1](S1_ASYMMETRY.md) | flips are directional, not magnitude effects | **refuted as a blanket claim** | of 9 cities directions: 3 directional, 3 symmetric, 3 inert; only `sup_grad` is significant at the smaller magnitude |
| [S2](S2_LAYER_SWEEP.md) | truth is the same linear object at every depth | **refuted** | 19 cities layers all read truth at 0.973 or better; their pairwise direction cosine has median 0.365, minimum 0.014 |
| [S3](S3_COMMON_AXIS.md) | the two steering arms are comparable as reported | **refuted** | disjoint by 6.09x on cities, 1.66x on common_claim; the behavioral transition is bracketed between relative magnitude 0.29 and 0.78, where nothing has ever been measured |
| [S4](S4_TARGET_CENSUS.md) | the certified target set represents the behavior we care about | **refuted, decisively** | 200 of 200 certified argmax flips succeed; vacuity 0.855 to 0.945; at most 5.5% made the claim false |
| D1 | the behavioral null is a fact, not an artifact of oversteering | not started | Gate 2 |
| T1 | a semantically meaningful target set is reachable at a comparable budget | not started | Gate 3 |

## What Gate 1 changed

**Three findings in the audit's own design spec were corrected by the experiments it
specified.** These are recorded inline in the spec with dates, and in section 6 of the S3 doc
and section 4.1 of the S4 doc.

1. *Finding 1, the 6.1x disjointness.* Correct on cities on the shared prompt set, confirmed at
   6.09x. Does not generalise: 1.66x on common_claim, and including the per-statement arm the
   ranges overlap there. The spec generalised from the mean arm on one dataset.
2. *Finding 3, "large-magnitude flips co-occur with degradation".* Not supported. The FALSE
   rate more than quadruples with p below 5e-7 on both datasets while the incoherence rise is
   not significant (p = 0.22 and p = 0.16). The spec's claim rested on a single signed cell.
3. *`target_mode`.* It is `countries` on cities and `runnerup` on common_claim, not `countries`
   on both. The two datasets' target sets are vacuous for different reasons.

Corrections 1 and 2 both make the naive arm look **better** than we had claimed, which is the
direction that argues against our own prior conclusion.

## The four results as one argument

The project's negative results were reported from a position that Gate 1 shows was not
measured. Specifically:

- **The readout is not the problem.** Truth is linearly decodable at 0.99 on cities from layer 8
  through layer 26 (S2).
- **The direction is not a single object.** Those layers read it along directions that rotate
  smoothly to near-orthogonality, and only one of nine steering directions behaves like a signed
  concept at a non-destructive magnitude (S2, S1).
- **There is no good layer to steer from.** Controllability peaks at layer 0, where readout is
  exactly chance, and falls 7.35x by layer 25, where readout is at ceiling (S2).
- **The magnitude band where behavior actually moves was never sampled** by either arm on a
  shared prompt set (S3).
- **And the certified target, when it was hit perfectly, meant almost nothing** (S4).

The last is the most useful. It says the control machinery is sound and the objective was
wrong, which is a fixable problem rather than a dead end.

## Gates

**Gate 1: passed.** S1 to S4 complete, so GPU spend is now justified. Two things changed in
what comes next:

- **D1's direction list** comes from S1: `sup_grad`, `sup_mean_diff`, `mag_u_gold`, a
  norm-matched random control, `jtw_mean_diff_tgt`, and `mag_resid_pc1` added as the sharpest
  available test of whether any window found is a truth effect or a norm effect.
- **D1's magnitude grid is superseded** by S3. The spec's grid put 8 of 12 points below 0.20,
  which is where the null is already well established. Revised grid:
  `0.01, 0.02, 0.04, 0.08, 0.15, 0.22, 0.30, 0.40, 0.52, 0.66, 0.82, 1.00`.
- **Sample size is the binding constraint.** Every behavioral rate in this project rests on 32
  prompts per cell, where one completion is 3.1 percentage points, and every flip rate on 24
  yes/no statements, where the smallest detectable asymmetry over 18 tests is 9 of 24. D1
  cannot resolve effects smaller than roughly 10 points without more prompts.

**Gate 2 (D1): implemented, pre-registered, not yet run.** `src/dose_response.py`,
`src/dose_analyze.py`, `deltaai/run_dose.slurm`, and `docs/D1_DOSE_RESPONSE.md` sections 1 to 3
are committed before any D1 data exists. Three things beyond the spec, all traceable to Gate 1:

- **One magnitude axis.** Dose is `rel * median ||h_src||`, the S3 canonical axis, so the naive
  and reachability directions are on the same ruler by construction rather than by conversion.
- **A continuous outcome, added because of S1.** Every yes/no cell also records the paired shift
  in the MAG first-token margin `p(yes) - p(no)`. Flip rate over 24 statements cannot resolve an
  effect below 0.375; a within-statement paired difference resolves far smaller shifts. A
  direction that moves the verdict readout monotonically without ever crossing the argmax is a
  real result, and no measurement in this project so far could have seen it. Registered as
  branch C in `docs/D1_DOSE_RESPONSE.md` section 2.
- **Significance is against the norm-matched random control at the same dose**, not against
  zero, because S1's lesson is that a bare rate at n = 24 cannot separate "the direction did
  something" from "a vector of that size did something".

**Gate 3 (T1)** is unchanged in structure. Its success criterion is now calibrated by S4: it has
to beat 5.5%.

## Outstanding

- The judge-based vacuity rate for `common_claim_true_false` (S4 section 4.4) needs
  `ANTHROPIC_API_KEY`, which is not set in this environment. It is the only part of Gate 1 that
  did not complete.
- S2 raises the priority of the transported cross-layer alignment `alpha^tr`, already on the
  project's open list. If transported directions agree where raw ones do not, the rotation is
  bookkeeping; if they still disagree, the concept genuinely moves with depth.
- S2 suggests a cheap experiment that did not exist before: inject at layer 4 on cities, which
  has readout 0.897 at margin 0.934, a 1.5x controllability gain over layer 11 for 0.09 of
  accuracy.
