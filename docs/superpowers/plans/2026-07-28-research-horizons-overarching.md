# Research Horizons — Overarching Plan (post-reachability-audit)

> **For agentic workers:** This is the milestone-level master plan. Horizon 0 is planned in
> detail in [2026-07-28-horizon0-validations.md](2026-07-28-horizon0-validations.md) — execute
> that plan. Horizons 1–2 get their own detailed plans when their gate opens; do NOT start them
> from this document alone.

**Goal:** Turn the audit's diagnosed negative result (certified-reachable readout, behaviorally
inert) into a publishable, instrument-validated dissociation paper, with concrete handoffs into
Julian's A-LQR line (arXiv:2604.19018).

**Rationale and evidence base:** [../RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md) (decision
document, four-scout synthesis) and [../REACH_AUDIT_FINDINGS.md](../REACH_AUDIT_FINDINGS.md)
(D1–D4 findings). This plan only sequences the work.

**Budget frame:** ~477 GPU-hr remain on `bhhv-dtai-gh`; the full three-horizon plan uses
~50–80 GPU-hr (naive per-token MPC excluded).

---

## Horizon 0 — close the audit's own loopholes (week 1; ≈1–2 GPU-hr + CPU)

Detailed plan: [2026-07-28-horizon0-validations.md](2026-07-28-horizon0-validations.md).
Runbook: [../../deltaai/REACH_H0_RUN.md](../../deltaai/REACH_H0_RUN.md).

| # | Deliverable | Script | Where |
|---|---|---|---|
| 0.1 | Same-point actuation control — calibration vs context factor | `src/reach_samepoint.py` (done) | GPU |
| 0.2 | Stem-population probe refit; labels = scale-0 judge verdicts; + XGBoost arm | `src/reach_stemprobe.py` | GPU extract, CPU fit |
| 0.3 | Judge hardening — drop baseline-failing prompts, recompute fractions | `src/reach_judge_harden.py` | CPU, free |
| 0.4 | Stem-context Jᵀw extraction + full-vs-stem geometry (common direction, principal angles) | `src/reach_stemjac.py` | GPU compute, CPU analyze |
| 0.5 | First-order robust margin ε\*_robust from 0.4's measured ΔJᵀw | inside `reach_stemjac.py --analyze` | CPU, free |
| 0.6 | Newton iterated re-steering (negative control: context, not curvature) | `src/reach_newton.py` | GPU |
| 0.7 | Recover U64/V64 singular vectors — **rsync-back glob fix only** (vectors already on cluster in `reach_svd_<ds>/`); GPU re-run only if scratch was purged | runbook change | free |

**Exit gate:** the D1 decomposition is quantified (calibration × context factors), D2 is
quantified (stem-probe accuracy vs old threshold), the robust certificate exists, and the SVD
vectors are local. All feed directly into the paper's mechanism section.

## Horizon 1 — instrument validation + feature-level mechanism (weeks 2–3)

Gate to open: Horizon 0 complete. Plan to write: `2026-08-XX-horizon1-refusal-sae.md`.

- **1.1 Refusal positive control (the publication gate; ~19–30 GPU-hr).** Unmodified audit
  pipeline on the Arditi refusal direction (arXiv:2406.11717). New pipeline pass:
  `src/prep_refusal.py` never run; needs AdvBench+Alpaca fetch, refusal DCT factors, and
  truth_dir-equivalents before any reach script. Decision tree: passes → main-venue
  dissociation-with-instrument paper; also fails → STOP, recalibrate with Julian before
  spending more.
- **1.2 SAE feature forensics (laptop-parallel, GemmaScope).** Gradient-pursuit decomposition
  (arXiv:2411.08790 — never naive encoding) of w, Jᵀw, and the V64 high-gain input subspace;
  statement-final vs generation-prefix feature comparison (D2 mechanism). Do NOT pivot to SAE
  steering (AxBench 2501.17148 negative prior on Gemma-2).

**Exit gate:** refusal verdict in hand → paper framing decision with Julian.

## Horizon 2 — A-LQR handoffs and the paper (weeks 4–6, with Julian)

Gate to open: Horizon 1 verdict. Plan to write after the Week-4 decision meeting.

- **2.1 Verdict-cost closed-loop steering** — receding-horizon re-planning and/or LQR with
  verdict-logit-margin cost; maximum-contrast configuration = source layers 5–8 × verdict
  direction × closed-loop (D4's best-conditioned window).
- **2.2 Balanced-truncation workspace subspace** — empirical Gramians from the 32 recovered
  full-Jacobian SVDs (needs 0.7; CPU). Present per-statement, n=32 caveat.
- **2.3 In-channel steering** along top right-singular vectors (needs 0.7) — does pushing the
  channel J amplifies restore cross-context gain / move behavior?
- **2.4 Paper** — dissociation with mechanism; refusal contrast as instrument validation; D2
  prefix-vs-statement finding (unclaimed in literature) as supporting figure.

## Explicitly deprioritized

HJ/NN-verification reachability at d=2304 · crosscoder training · gemma-2-9b ·
all-position probing (24 GB trap). Reasons in RESEARCH_ROADMAP.md.

## Sequencing

```
Week 1:    Horizon 0 (one cluster batch: samepoint + h0 jobs; laptop analyses).
Weeks 2–3: 1.1 refusal (cluster) ∥ 1.2 SAE forensics (laptop).
Week 4:    Decision point with Julian → write Horizon-2 plan.
Weeks 5–6: 2.1 run + 2.2/2.3 analyses; paper draft.
```
