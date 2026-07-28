# Backward-Reachability Audit of the Truth Set — Design

**Date:** 2026-07-24
**Origin:** PI's 2026-07-23 reframe ("define a target set in J-space, use backward reachability;
the formalization is control theory, the actual tools are just linear algebra") + the deep-research
report (docs/DEEP_RESEARCH_PROMPT_REACHABILITY.md answered sections A–E).
**Model:** google/gemma-2-2b base, d=2304. **Datasets:** cities (hop 11→20), common_claim_true_false
(hop 13→22). **Compute:** local laptop (.venv, MPS/CPU) for analysis; DeltaAI GH200 SLURM for
extraction/SVD/generation.

## 1. Question

Does the FALSE-verdict region at the target layer — the halfspace where the linear truth probe
reads FALSE, intersected with an on-distribution constraint — have a non-empty backward-reachable
preimage at the source layer within the DCT-calibrated steering budget? If not, why not,
mechanistically? This upgrades the project's direction-level nulls ("this vector failed") to a
set-level claim ("no coherent-lie state is first-order reachable within budget"), or discovers the
off-axis lever every 1-D probe missed.

## 2. Core definitions and conventions

- **Control convention:** a vector Δ ∈ R^2304 added to the residual stream at **every token
  position** at the source layer — identical to how all existing steering in this repo injects.
  This makes reachability verdicts transfer directly to real interventions.
- **Hop map:** F: Δ ↦ h_tgt(last non-pad token) through the 9 blocks (11→20 / 13→22), holding the
  input statement fixed. J = ∂F/∂Δ at Δ=0, a 2304×2304 per-statement matrix. Never materialized in
  Phase 1: **Jᵀw is one vjp** (backward pass with cotangent w); **Jv is one jvp**.
- **Target set:** T = {h : w·h ≤ c − δ}, w and threshold c from a truth readout fit at the target
  layer, δ = a confidence margin (default: δ such that the probe's predicted P(true) ≤ 0.2).
  On-distribution constraint handled by Phase 5's validity region rather than a second halfspace
  in v0 (YAGNI: a coherence probe is future work; Phase 3's judge is the behavioral coherence check).
- **Reachability test (linear, closed form):** statement i, direction w, budget ε:
  reachable ⟺ g_i(w) ≤ ε · m_i(w), where g_i(w) = w·h_tgt,i − (c − δ) and
  m_i(w) = ‖J_iᵀw‖ (the **controllability margin**).
- **Budget:** ε swept 0 → 1.5 × input_scale (input_scale from dct_meta_<ds>.json: cities ≈ 47.7,
  common_claim ≈ 86.7).
- Sign convention throughout matches the repo: +τ→TRUE, −τ→FALSE.

## 3. Direction battery (target-layer readouts w)

| Key | Definition | Role |
|---|---|---|
| `mean_diff_tgt` | class-mean difference at target layer (exists: truth_dir_tgt_<ds>.npz) | primary truth readout |
| `probe_grad_tgt` | logistic-probe gradient fit at target layer on this run's own activations | second truth readout (non-identifiability) |
| `truth_sub_k` | k=8 orthonormal basis of the truth subspace at target: Gram–Schmidt over [mean_diff_tgt, probe_grad_tgt, gradients of 6 logistic probes each fit on a random 50% subset (seed 0–5)] — spans the non-identifiability equivalence class of truth readouts | best-case margin over the subspace |
| `dct_u_top` | DCT's top effect directions U at the target layer | positive control — must show large margins |
| `rand_64` | 64 iid random unit vectors | null distribution for margins |

For each Jᵀw we also record cosines to source-layer landmarks: mean_diff@source
(truth_dir_<ds>.npz), v_Q (mag_dir_<ds>.npz), DCT top V. Free at extraction time.

## 4. Phases

Dependency-ordered; each phase independently reportable. Phases 1, 2, 3 are per-dataset; 4 and 5
are cross-cutting.

### Phase 1 — Margins + reachability curves (the PI answer)

Extraction (`reach_margins.py`, GH200 job; staged so the probe can be fit before vjps):
1. `--stage acts`: batched no-grad forwards; save h_src, h_tgt (last non-pad token) per statement.
2. `--stage dirs`: fit `probe_grad_tgt`, `truth_sub_k`, threshold c from the stage-1 activations;
   assemble the full battery into `reach_dirs_<ds>.npz`. (Runs anywhere; seconds.)
3. `--stage vjp`: per statement × direction, one vjp → margins m_i(w), vectors J_iᵀw (unit),
   cosines to source landmarks. Checkpoint every 100 statements; `--resume`.

Sample: all 1,496 cities statements; 2,000 stratified common_claim statements.

Analysis (`reach_analyze.py` + `viz_reach.py`, local):
- **Margin distribution figure:** m(w) for truth readouts vs rand_64 null vs dct_u_top control;
  report the truth readouts' percentile within the random null.
- **Reachability curves:** fraction of statements reachable vs ε, one curve per direction,
  vertical line at input_scale; `reach_curve_<ds>.csv` (columns: direction, eps, frac_reachable).
- **Geometry:** distributions of cos(Jᵀw, mean_diff@src) and cos(Jᵀw, v_Q).

Decision criterion: truth-subspace **best-case** margin below the random-null median ⇒ the
unreachability finding (proceed to Phase 2 for mechanism). Any truth-subspace w with margin
comfortably above null ⇒ Phase 3 is the interesting phase.

### Phase 2 — Full-Jacobian SVD subsample (the mechanism)

`reach_svd.py`, GH200: full J for 32 statements/dataset (16 true / 16 false) via batched jvp
(2304 tangents; batch tangents to fill the GPU). One output file per statement
(`reach_svd_<ds>/stmt_<i>.npz`: singular values, top-64 U and V vectors) so partial completion is
usable. Analysis:
- singular spectrum / effective rank per statement (the 2605.14258 low-rank funnel, on our hop);
- energy of each truth w in the top-k **left** singular subspace vs k;
- energy of mean_diff@source in the top-k **right** singular subspace vs k;
- overlap of top right-singular vectors with DCT's V (consistency check on the whole linearized
  picture — a mismatch here is a stop-and-diagnose event).

Turns "margin is small" into "truth lies in the singular tail of the trained hop dynamics."

### Phase 3 — Steer along Jᵀw + behavioral judge (readout vs behavior)

`reach_steer.py`, GH200: steer along (a) the dataset-mean unit Jᵀw for w = mean_diff_tgt and the
best truth-subspace w, (b) per-statement Jᵀw for a 200-statement subset. Strengths bracket the
Phase-1 reachability boundary ε*(w): {0.5, 1.0, 1.5, 2.0} × ε*, both signs, capped at
1.5×input_scale. Generate free-form continuations with the existing generation harness; judge
TRUE/FALSE/INCOHERENT with the existing validated judge; log the target-layer probe readout per
generation. Deliverable: the 2×2 — readout moved × behavior moved — with the existing mean_diff
steering results as the comparison row. "Readout flips, model still won't lie" (the LiSeCo
activation→behavior gap) is a first-class outcome, not a failure.

### Phase 4 — J-lens margins (the workspace headline)

`reach_jlens.py`. Fit the Jacobian lens on gemma-2-2b with `anthropics/jacobian-lens`
(verified real, Apache-2.0, fits open-weight HF decoders). Isolated behind try/import; fallback =
our own vjp estimate of J_ℓᵀw via reach_hop (which we compute for 3 layers regardless, as a
spot-check of their fit). Then, per source layer ℓ, margins ‖J_ℓᵀw‖ for:
- **verdict direction:** unembed("yes") − unembed("no") (final-basis; the lens maps it back);
- **truth-content:** the target/final-layer truth probe;
- **question-mode:** v_Q.

Run on both declarative statements and question-prefixed inputs. Workspace-selectivity
predictions: v_Q margin ≫ truth-content margin on declaratives; truth-content margin **rises**
under the question prefix. That interaction is the headline figure if it shows.

### Phase 5 — Linearization error (the reviewer armor)

`reach_linerr.py`, GH200 (cheap): for Δ ∈ {unit Jᵀw, mean_diff@src, top DCT V, random}, relative
error ‖F(εΔ) − F(0) − εJΔ‖ / ‖εJΔ‖ on 64 statements/dataset, ε log-swept to 1.5×input_scale.
Deliverable: empirical validity radius per direction (ε at 20% relative error), overlaid as a
shaded trust region on every Phase-1 reachability curve. All Phase 1–3 claims are stated
inside-vs-outside this region. (Expectation: input_scale is beyond the linear radius — the claim
must be phrased at the ε where linearity holds, with nonlinear spot-checks beyond.)

### Cross-phase decision logic

- P1 unreachable + P2 tail + P5 small-ε confirmation ⇒ the bounded first-order unreachability
  claim, with mandatory hedges (first-order only; probe-halfspace ≠ behavior; single hop; truth
  subspace as tested).
- P1 reachable + P3 no-lie ⇒ the readout/behavior dissociation claim.
- P3 produces judged lies ⇒ lever found; pivot to characterizing it.

## 5. Files

| File | Responsibility |
|---|---|
| `src/reach_hop.py` | Shared lib: block-slice module, broadcast-injection F(Δ), vjp_w / batched jvp; borrows DCT's eager-attention setup |
| `src/reach_margins.py` | Phase 1 extraction (staged: acts / dirs / vjp), checkpoint+resume |
| `src/reach_analyze.py` | Phase 1 analysis → reach_curve_<ds>.csv |
| `src/reach_svd.py` | Phase 2 full-J subsample + SVD |
| `src/reach_steer.py` | Phase 3 steering + generation logging |
| `src/reach_jlens.py` | Phase 4 J-lens margins (with vjp fallback) |
| `src/reach_linerr.py` | Phase 5 relative-error sweep |
| `src/viz_reach.py` | All figures |
| `deltaai/run_reach_margins.slurm`, `run_reach_svd.slurm`, `run_reach_steer.slurm`, `run_reach_linerr.slurm` | GH200 jobs, tight --time, per-dataset failure guards |
| `docs/REACHABILITY_RUNBOOK.md` | Phase-by-phase run commands, local↔cluster rsync batches |
| `tests/test_reach_hop.py`, `tests/test_reach_analyze.py`, `tests/test_reach_linerr.py` | see §7 |

Artifacts: `reach_margins_<ds>.npz` (margins, unit Jᵀw vectors, cosines, h_src/h_tgt, probe
readings, battery metadata), `reach_dirs_<ds>.npz`, `reach_curve_<ds>.csv`,
`reach_svd_<ds>/stmt_<i>.npz`, `reach_steer_<ds>.csv`, `judge_reach_steer_<ds>.csv`,
`reach_jlens_<ds>.csv`, `reach_linerr_<ds>.csv`, `plot_reach_*.png`.

## 6. Error handling / resilience

- Fail-fast startup validation: required inputs exist and have expected keys
  (truth_dir_tgt_<ds>.npz, dct npz with U/V/input_scale or dct_meta_<ds>.json, mag_dir_<ds>.npz);
  a config error dies in seconds, not after an hour of forwards.
- Checkpoint per 100-statement chunk + `--resume` in reach_margins; per-statement files in
  reach_svd; multi-dataset drivers continue past per-dataset failure (guard pattern from 45314ef).
- Tight SLURM `--time` on every job.
- Phase 4's external dependency isolated so it can never block Phases 1–3.

## 7. Testing (pytest, tiny stand-in model fixture as in existing DCT tests)

- `test_reach_hop.py`: (a) vjp vs finite differences per coordinate; (b) the transposition
  identity w·(Jv) [jvp] == (Jᵀw)·v [vjp] — guards Phases 1 and 2 simultaneously; (c) broadcast
  convention: Δ perturbs all positions at source layer; F(0) equals the clean forward.
- `test_reach_analyze.py`: reachability verdicts on hand-computed 2-D cases; frac-reachable
  monotone in ε.
- `test_reach_linerr.py`: relative error ≈ 0 for a purely linear stand-in map at all ε.
- Phases 3–4 reuse tested generation/judge code; new glue gets smoke tests only.

## 8. Parallel track (no new code — launch the built round)

The conditional-steering / U-anchor round (Arms A0/A1/A2 + Arm B) is built, reviewed, and unrun.
Launch per `docs/CONDITIONAL_UANCHOR_RUNBOOK.md`: local Phase 1 (export_target_dir, steer_verdict,
steer_conditional on MPS) → one rsync batch up → `sbatch deltaai/run_mag_cond_judge.slurm` +
`sbatch deltaai/run_dct_uwarm.slurm` (independent; submit together) → rsync back → local figures.
Requires one NCSA password + Duo push per rsync batch (user-performed). Its results feed directly
into Phase 3/4 interpretation (composed steering is the workspace theory's key behavioral
prediction).

## 9. Out of scope (v0)

- Coherence-probe halfspace in the target set (judge covers behavior; revisit if P1 says reachable).
- INVPROP/PREMAP/GenBaB certified preimages through a block (hardening; only after P1–P5).
- Multi-hop composed linearizations with accumulated bounds (Phase 5's empirical radius suffices
  for v0).
- Any layer-pair sweep beyond the two established hops.

## 10. Key references

- Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models"
  (transformer-circuits, 2026-07); code: github.com/anthropics/jacobian-lens (verified 2026-07-24).
- Fernando & Guitchounts, arXiv:2605.14258 — per-layer Jacobian spectra, low-rank perturbation
  funnel (verified 2026-07-24).
- Karnik & Bansal, arXiv:2509.21528 (BRT-Align) — latent-space backward reachability, token axis.
- Cheng et al., arXiv:2405.15454 (LiSeCo) — probe-halfspace control on gemma-2-2b; the
  activation→behavior gap.
- Skifstad et al., arXiv:2604.19018 (A-LQR) — local linearity, per-hop error bounds.
- Deep-research report answers A–E: pasted in session 2026-07-24 (summarized in
  docs/RESULTS_SINCE_LAST_MEETING_PART2.md PI-feedback section).
