# Research Roadmap — Paths Forward from the Reachability Audit

*2026-07-28. Synthesis of four parallel investigations: feature discovery (SAEs), control
theory, backtracking options (artifact-verified), and framings/positive controls. Grading
accounts for the ~477 GPU-hr remaining allocation; nothing recommended below threatens it.*

## Where we stand

The audit produced a diagnosed negative result: per-statement linear reachability certificates
(margins ‖Jᵀw‖, budgets ε\*) certify the truth readout as reachable; steering moves the readout
with R²=0.999 but behavior never moves. Three mechanisms: (D1) gain collapses 8–35× across a
one-word context shift; (D2) the probe boundary doesn't transfer to generation prefixes; (D3)
the Jacobian's high-gain channel is concept-blind on its input side. Controllability peaks at
layers 5–9 (we steered later); the verdict readout has 2–7× less margin everywhere (D4).

**Literature position (verified by search):** no published work reports a
certified-reachable-but-behaviorally-inert dissociation with mechanism. Closest neighbors:
"Steered LLM Activations are Non-Surjective" (arXiv:2604.09839 — existence proof, no
certificate machinery), "When Truthful Representations Flip" (arXiv:2507.22149 — correlational
version via prompting), Tan et al. steering-reliability (arXiv:2407.12404 — base rate, no
mechanism). Julian's A-LQR paper is arXiv:2604.19018 ("Local Linearity of LLMs Enables
Activation Steering via Model-Based Linear Optimal Control") — our R²=0.999 corroborates its
core assumption; our D1/D2 are the open-loop failure modes its closed-loop design implicitly
answers. The prefix-vs-statement probe finding (D2) appears **unclaimed** in the literature.

## The decision structure

Three horizons, ordered by dependency, not just cost:

### Horizon 0 — close the audit's own loopholes (this week; ≈1–2 GPU-hr + CPU)

Every item reuses cached artifacts (verified by schema inspection); each closes a specific
hole a reviewer would poke. **Implementation plan:**
[superpowers/plans/2026-07-28-horizon0-validations.md](superpowers/plans/2026-07-28-horizon0-validations.md)
· **milestone view:**
[superpowers/plans/2026-07-28-research-horizons-overarching.md](superpowers/plans/2026-07-28-research-horizons-overarching.md)
· **cluster runbook:** [../deltaai/REACH_H0_RUN.md](../deltaai/REACH_H0_RUN.md).

| # | Action | Cost | Closes |
|---|--------|------|--------|
| 0.1 | **Run `reach_samepoint.py`** (drafted, tested) | GPU-minutes | Decomposes 8–35× into calibration vs context factor — "was the certificate ever right at its own point?" |
| 0.2 | **Refit truth probe on generation-stem activations**, labels = existing scale-0 judge verdicts in `judge_reach_steer_stmt_*.csv` (200/ds, already judged) | CPU + one small forward pass | D2 quantified: is prefix-transfer failure a threshold shift (recalibrable) or absence of signal? Add XGBoost arm for free — original results show common_claim has nonlinear headroom (gap 0.082) at exactly this hop's source layer |
| 0.3 | **Judge hardening**: drop the ~15 identified problem prompts, recompute fractions | Free (local CSVs) | Cleans the null's noise floor |
| 0.4 | **Stacked-Jᵀw common-direction extraction**: one new vjp pass at the *stem* context, then joint SVD / principal angles of [jtw_full; jtw_stem] (full-statement rows already cached in `reach_margins_*.npz['jtw']`) | <10 GPU-hr, mostly CPU | D1 operationalized: does ANY input direction's gain survive the one-word shift? Distinguish from "Power Steering" (LW 2026) which aggregated across topics, not local context shifts |
| 0.5 | **First-order robust margin**: ε\*_robust = g/(‖Jᵀw‖+‖ΔJ‖) using 0.4's measured spread | Free | Honest robustified certificate (explicitly not a certified tube) |
| 0.6 | **Iterative re-steering (Newton, 2–3 steps)**: `vjp_rows` already accepts nonzero delta0; short driver script | GPU-minutes | Cheap negative control — prediction: does NOT fix D1 (context, not curvature); confirms the diagnosis either way |
| 0.7 | **Recover the U64/V64 singular vectors** — they ARE saved per-statement in `reach_svd_<ds>/` on the cluster (verified in `reach_svd.py:compute`); the audit's rsync-back glob just excluded the dirs. Fix = pull ~19 MB/dataset in the H0 rsync; GPU re-run only if scratch was purged | Free (rsync) | Unblocks 1.2, 2.2, 2.3 below — shared prerequisite |

### Horizon 1 — the gating experiment + feature-level mechanism (2–4 weeks)

**1.1 Refusal positive control — the publication gate.** Run the *unmodified* audit pipeline on
the Arditi refusal direction (arXiv:2406.11717; public code, validated down to small Gemma
models, known steering layer). NOTE (artifact-verified): this is a *new pipeline pass*, not a
backtrack — `src/prep_refusal.py` exists but was never run (needs AdvBench+Alpaca fetch), and
refusal needs its own DCT factors + truth_dir-equivalents before any reach script starts.
Budget ~19–30 GPU-hr end-to-end. Decision tree:
- Refusal **passes** (gain transfers, behavior moves) + truth fails → instrument-validated
  contrast; the paper upgrades from workshop to main-venue candidate.
- Refusal **also fails** → STOP, take to Julian before spending more: either our thresholds
  are miscalibrated (check against Arditi's effect sizes) or open-loop steering is broken more
  generally (a bigger, more contestable claim; echoes arXiv:2407.12404).

**1.2 SAE feature forensics (CPU-parallel, GemmaScope).** gemma-2-2b is the *best-covered*
model in GemmaScope (JumpReLU SAEs at every layer/site, 16k width CPU-feasible; pretrained
transcoders too). Two plays, both requiring **gradient pursuit against the decoder** (naive
encoding of steering/gradient vectors is known-misleading, arXiv:2411.08790):
- Decompose the high-gain input subspace (V64 from 0.7) + our w and Jᵀw directions → name what
  the channel transports (positional/syntactic features? entangled topics?). Either answer
  upgrades D3 from "random-level overlap" to a legible mechanism.
- Statement-final vs generation-prefix feature comparison (models the honesty-subspace-flip
  methodology of arXiv:2507.22149, run on our 2b instead of their 9B) → feature-level account
  of D2.
Robustness duties: check across SAE widths (absorption/non-canonicity, arXiv:2409.14507,
2502.04878); check residual "dark matter" fraction (arXiv:2410.14670). Deprioritize
SAE-feature *steering* (AxBench, arXiv:2501.17148: SAEs underperform on Gemma-2 specifically —
contested by later work, but not where our leverage is).

### Horizon 2 — A-LQR handoffs and the paper (with Julian)

**2.1 Verdict-cost closed-loop steering** — the two natural handoffs into arXiv:2604.19018:
(a) receding-horizon/MPC re-planning at each generated token (the true fix for D1's
context-transfer failure; 100–300 GPU-hr done naively, 5–10× cheaper re-planning every K
tokens); (b) LQR with verdict-logit-margin cost (w_verdict already built free in
`reach_jlens.py`; needs battery addition + vjp rerun, ~hours). Combine with **source at layers
5–8** (D4's best-conditioned window; needs new full-sequence acts, ~6 h cap) for the
maximum-contrast configuration: early window × verdict direction × closed-loop.

**2.2 Balanced-truncation "workspace" subspace** — empirical controllability Gramian E[JJᵀ] and
observability Gramian E[Jᵀwwᵀ J] from the 32 recomputed full Jacobians (0.7), Hankel-SV
balancing (method template: arXiv:2607.05457; framing: arXiv:2511.12852). CPU after 0.7.
Caveat: n=32 → present per-statement, not as a population claim.

**2.3 In-channel steering** along top right-singular vectors (needs 0.7) — does pushing the
directions J actually amplifies restore gain across context / move anything behaviorally?

**2.4 The paper.** Primary framing (scout-recommended, gated on 1.1): *dissociation with
mechanism* — quantitative per-statement certificates + behavioral falsification + D1–D4
decomposition, with the refusal contrast as instrument validation and the D2
prefix-vs-statement finding as a supporting figure (currently unclaimed in the literature).
Concept-choice discussion: truth is plausibly a structurally hard target — 2-D truth subspace
(arXiv:2407.12831), task-orthogonal truth geometries (OpenReview), output-accessibility
framework (arXiv:2604.15557), Anthropic honesty-elicitation precedent.

### Explicitly deprioritized (with reasons)

- **Hamilton-Jacobi / NN-verification reachability at d=2304**: infeasible at our scale
  (searched; DeepReach-class needs weeks of training, bound propagation has no precedent
  through a 9-block 2304-wide attention stack). Scenario/sampling reachability within trust
  radii is the honest middle ground if needed.
- **Crosscoder training**: no pretrained checkpoint for 2b; 1–2+ week training project; only
  if transcoder analysis leaves a cross-layer-smearing signature.
- **gemma-2-9b**: invalidates every cached artifact; most expensive item; generality check,
  not falsification of any diagnosed mechanism.
- **Token-position-resolved probing (all positions)**: looks cheap, is not — no per-token acts
  cached anywhere; ~24 GB unsampled. Subsample only if the D2 story needs a position axis.

## Recommended sequencing (one student, ~6 weeks)

```
Week 1:   Horizon 0 complete (0.1–0.7; one cluster batch + laptop).
Weeks 2-3: 1.1 refusal positive control (cluster) ∥ 1.2 SAE forensics (laptop).
Week 4:   Decision point with Julian: refusal outcome → paper framing;
          choose 2.1 configuration (early-window × verdict × closed-loop).
Weeks 5-6: 2.1 run + 2.2/2.3 analyses; paper draft with figures.
```

Total new GPU: ~50–80 hr of the 477 available (naive MPC variant excluded; add ~100–300 if
per-token re-planning is chosen — still within budget but decide with Julian first).

## Key citations gathered

Steering-as-control: A-LQR 2604.19018 · feedback steering 2510.04309 · steering vector fields
2602.01654 · non-identifiability 2602.06801 · CBF alignment 2511.03121 · BRT-Align 2509.21528.
Dissociation neighbors: non-surjective 2604.09839 · truthful-flip 2507.22149 · steering
reliability 2407.12404 · where-steering-succeeds 2604.15557 · LiSeCo 2405.15454.
Positive control: refusal single direction 2406.11717 · refusal geometry 2502.17420.
SAEs: GemmaScope 2408.05147 · SAE-steering-vector caveat 2411.08790 · AxBench 2501.17148 ·
transcoders 2406.11944 · absorption 2409.14507 · non-canonical 2502.04878 · dark matter
2410.14670 · attribution patching 2310.10348.
Truth geometry: truth-is-universal 2407.12831 · logical-transformations 2506.00823 ·
Anthropic honesty-elicitation (alignment.anthropic.com/2025/honesty-elicitation).
