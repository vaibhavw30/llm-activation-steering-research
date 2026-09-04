# Documentation index

44 documents accumulated over four months. This page says which one to open, and flags the ones
that are point-in-time snapshots rather than current statements.

**If you read one thing, read [`PLAIN_ENGLISH_WALKTHROUGH.md`](PLAIN_ENGLISH_WALKTHROUGH.md).**
It supersedes every status document below and is the only doc written to be read cold.

---

## 1. Orientation

| Doc | What it is |
|---|---|
| [`PLAIN_ENGLISH_WALKTHROUGH.md`](PLAIN_ENGLISH_WALKTHROUGH.md) | The whole project: eight experiments in the order they happened, the PI's ten items mapped to outcomes, a vocabulary section. Start here. |
| [`MEETING_20MIN.md`](MEETING_20MIN.md) | The spoken version, timed for a 20-minute slot. |
| [`PLAN_ADVISOR_NOTES_2026-09.md`](PLAN_ADVISOR_NOTES_2026-09.md) | **The current plan.** The advisor's three September notes, mapped to what our artifacts already say, then to experiments with costs and a sequence. |
| [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) | What is genuinely unresolved, each question placed against the literature. |
| [`LITERATURE.md`](LITERATURE.md) | The literature compilation on automated feature discovery, with what each paper does and does not establish. |
| [`math_map.tex`](math_map.tex) | Every object in the project written down formally: the certificate, the pullback, the target sets. |

---

## 2. Findings, by experiment

These are the current statements of record. Each one owns its result.

### The probe (June)

| Doc | Result |
|---|---|
| [`EXPLAINER.md`](EXPLAINER.md) | Linear vs XGBoost probes at every layer, four datasets. Clean truth is linear; messy truth has non-linear headroom that grows with messiness. |
| [`DIRECTIONS_FINDINGS.md`](DIRECTIONS_FINDINGS.md) | The two standard "truth directions" (contrastive mean difference, logistic gradient) agree at cosine 0.413 on cities and 0.085 on common_claim. |
| [`MEETING_SUMMARY.md`](MEETING_SUMMARY.md) | The one-page version of the above. |

### Unsupervised direction discovery (July)

| Doc | Result |
|---|---|
| [`DCT_METHODOLOGY.md`](DCT_METHODOLOGY.md) | How Deep Causal Transcoding works and how it was set up here. Read before any DCT result. |
| [`DCT_VS_TRUTH_FINDINGS.md`](DCT_VS_TRUTH_FINDINGS.md) | DCT never surfaces the supervised truth direction. Null. |
| [`DCT_VS_XGBOOST_FINDINGS.md`](DCT_VS_XGBOOST_FINDINGS.md) | The non-linear truth headroom is not hiding in DCT's subspace either. Null, twice over. |
| [`WARM_DCT_RESULTS.md`](WARM_DCT_RESULTS.md) | Warm-starting DCT from the truth direction, to rule out an optimisation failure. |
| [`MAG_VS_DCT_CONCEPTUAL.md`](MAG_VS_DCT_CONCEPTUAL.md) | The concepts behind the two miners, before any numbers. Read this before the next one. |
| [`DCT_VS_MAG_ON_TRUTH.md`](DCT_VS_MAG_ON_TRUTH.md) | Both miners put to the same test on truth. |
| [`FUNNEL_RESULTS.md`](FUNNEL_RESULTS.md) | The DCT interpretation funnel: from vectors to interpreted text to judged steering. |

### Backward reachability (late July, the central result)

| Doc | Result |
|---|---|
| [`REACHABILITY_RUNBOOK.md`](REACHABILITY_RUNBOOK.md) | What the certificate `eps*` is, how the pullback works, how to run the five phases. |
| [`REACH_AUDIT_FINDINGS.md`](REACH_AUDIT_FINDINGS.md) | **The dissociation.** On cities the readout crosses inside the trust radius and the lie rate does not move. Jacobian overlap with concept space is 0.062. |
| [`INVESTIGATION_steering_validity.md`](INVESTIGATION_steering_validity.md) | An earlier investigation into whether a judged steering result was real. Method is still worth reading; conclusions are superseded by the audit below. |
| [`CONDITIONAL_UANCHOR_RUNBOOK.md`](CONDITIONAL_UANCHOR_RUNBOOK.md) | Conditional steering and U-anchored DCT. A branch that did not become load-bearing. |

### Token space (August)

| Doc | Result |
|---|---|
| [`TOKEN_SPACE_PROGRAM.md`](TOKEN_SPACE_PROGRAM.md) | Why the target set moved to token space and what the program is. Read first. |
| [`TOKEN_SPACE_FINDINGS.md`](TOKEN_SPACE_FINDINGS.md) | The PI's eight claims, each answered. The closed-form token target flips 200/200; the old probe-halfspace direction flips 0.000 at every layer and budget. |
| [`TOKEN_SPACE_RAW_FINDINGS.md`](TOKEN_SPACE_RAW_FINDINGS.md) | Generated pass-1 dump behind the above. Regenerable; do not hand-edit. |

### Auditing our own results (26 to 27 August)

| Doc | Result |
|---|---|
| [`AUDIT_SUMMARY.md`](AUDIT_SUMMARY.md) | Verdict summary for S1 to S4. All four registered assumptions were refuted. |
| [`S1_ASYMMETRY.md`](S1_ASYMMETRY.md) | Of nine directions: 3 directional, 3 symmetric, 3 inert. Flips are not directional as a blanket claim. |
| [`S2_LAYER_SWEEP.md`](S2_LAYER_SWEEP.md) | 19 layers read truth at 0.973 or better along directions whose pairwise cosine has median 0.365. There is no single truth object across depth. |
| [`S3_COMMON_AXIS.md`](S3_COMMON_AXIS.md) | The two steering arms were disjoint by 6.09x; the band where behavior moves was never sampled. |
| [`S4_TARGET_CENSUS.md`](S4_TARGET_CENSUS.md) | 200/200 certified flips succeed, and at most 5.5% made the claim false. The target set was semantically empty. |
| [`D1_DOSE_RESPONSE.md`](D1_DOSE_RESPONSE.md) | Pre-registered dose sweep. 0 clean windows out of 120 cells. Sections 1 to 3 were committed before any data existed and must not be edited. |
| [`D2_PREFIX_TRANSFER.md`](D2_PREFIX_TRANSFER.md) | The truth readout is at chance on the population we steer: balanced accuracy 0.500, AUC 0.510 and 0.568, never changes sign, and every refit including XGBoost collapses to the base rate. |
| [`Q0_TRUTHFULQA_DATASET.md`](Q0_TRUTHFULQA_DATASET.md) | The TruthfulQA build for track Q: 1488 contrastive rows off 744 questions plus a 64-question holdout, polarity inverted so the certificate steers toward truthful, and the fit-point caveat Q1 has to measure. |
| [`REFUSAL_POSITIVE_CONTROL.md`](REFUSAL_POSITIVE_CONTROL.md) | **The publication gate.** The identical pipeline induces refusal: 14 flips against 0, Mantel-Haenszel OR 24.2 for crossing. Includes the judge's gold-label validation at 0.969. |

---

## 3. Point-in-time snapshots

These were written for a specific meeting or handoff and are **not** maintained. They are kept
because they record what was believed and when, which matters for reading the audit. Where they
disagree with the walkthrough, the walkthrough wins.

| Doc | Written | Superseded by |
|---|---|---|
| [`PROJECT_CONTEXT_AND_ROADMAP.md`](PROJECT_CONTEXT_AND_ROADMAP.md) | 7 Jul | the walkthrough |
| [`PI_MEETING_RESULTS.md`](PI_MEETING_RESULTS.md) | 15 Jul | the walkthrough |
| [`STATUS_SINCE_LAST_MEETING.md`](STATUS_SINCE_LAST_MEETING.md) | 15 Jul | the walkthrough |
| [`PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md`](PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md) | 23 Jul | the walkthrough |
| [`RESULTS_SINCE_LAST_MEETING_PART2.md`](RESULTS_SINCE_LAST_MEETING_PART2.md) | 29 Jul | the walkthrough |
| [`PROJECT_PROGRESS_TO_DATE.md`](PROJECT_PROGRESS_TO_DATE.md) | 30 Jul | the walkthrough, but this is the most complete retrospective of the first 5.5 weeks |
| [`EXPLAINER_FOR_THE_PI.md`](EXPLAINER_FOR_THE_PI.md) | 5 Aug | the walkthrough |
| [`PI_QUESTIONS_ANSWERED.md`](PI_QUESTIONS_ANSWERED.md) | 5 Aug | walkthrough section 3 |
| [`RESULTS_SINCE_LAST_MEETING_PART3.md`](RESULTS_SINCE_LAST_MEETING_PART3.md) | 5 Aug | the walkthrough |

---

## 4. Plans and prompts

| Doc | What it is |
|---|---|
| [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) | The horizons plan written after the reachability audit. Horizon-0 validations, Horizon-1 refusal, the A-LQR handoffs. |
| [`NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md`](NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md) | The brief that produced the token-space pivot. |
| [`OLMO_JUDGE_IMPLEMENTATION_PLAN.md`](OLMO_JUDGE_IMPLEMENTATION_PLAN.md) | How the local OLMo-3 judge backend was built, and why a local judge was needed. |
| [`superpowers/specs/`](superpowers/specs/) | Design specs, one per program. The 2026-08-26 steering-validity spec holds the PI's ten items table. |
| [`superpowers/plans/`](superpowers/plans/) | Implementation plans, one per program. |

---

## Conventions

- Every finding is reported **per dataset**. `cities` and `common_claim_true_false` are never
  pooled; they fail in different ways and the difference is part of the result.
- A doc that states a number names the artifact the number came from.
- `D1_DOSE_RESPONSE.md` sections 1 to 3 are a pre-registration. They are frozen.
- `TOKEN_SPACE_RAW_FINDINGS.md` is generated. Regenerate it rather than editing it.
