# Horizon-0 Validations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the reachability audit's own loopholes — quantify the D1 gain-collapse
decomposition, the D2 probe-transfer failure, the robust certificate, and the Newton negative
control — reusing cached artifacts; one small cluster batch (≈1–2 GPU-hr) plus local CPU.

**Architecture:** Each item is a standalone script in `src/` following the reach_* conventions:
argparse with `--dataset`, GPU stages separated from CPU `--analyze`/`--fit`/`--summarize`
stages, pure helpers unit-tested without a model, outputs as flat CSVs/NPZs in cwd. GPU stages
mirror `reach_steer.arm_per_stmt`'s exact statement selection so populations are identical
across arms. Two SLURM jobs (`run_reach_samepoint.slurm`, new `run_reach_h0.slurm`) run the GPU
stages; everything else runs on the laptop.

**Tech Stack:** torch (+torch.func vjp via `reach_hop`), numpy, sklearn (fit only), optional
xgboost (fit only), csv/json. No new dependencies on the cluster's `.venv-dct-gpu`.

## Global Constraints

- Statement selection everywhere: `picks = np.random.default_rng(SEED=42).permutation(idx1)[:N_PER_STMT=200]`, then skip `stem_of(stmt) is None` — byte-identical to `reach_steer.arm_per_stmt`.
- The probe direction is always `mean_diff_tgt` from `reach_dirs_<ds>.npz` (`W[k]`, threshold `thresh02[k]`); its unit Jᵀw rows come from `reach_margins_<ds>.npz["jtw"][:, ks, :]` with `ks = store_names.index("mean_diff_tgt")`.
- Steering-norm cap: `1.5 * input_scale` (`dct_meta_<ds>.json`), matching `reach_steer.scale_grid`.
- Datasets: `cities`, `common_claim_true_false`. Never touch the other-track uncommitted files (spectrum/viz_spectrum/mag_linearity/RESULTS_PART2/DEEP_RESEARCH_PROMPT).
- Tests are pure-python (no model download), run under `-W error`, live in `tests/test_<module>.py`.
- SLURM: account `bhhv-dtai-gh` filled by sed from `ACCOUNT_NAME` placeholder; `--partition=ghx4`, dct env, `TRANSFORMERS_OFFLINE=1`, per-dataset `|| echo "!!!! … FAILED — continuing"` guards; never remove `--time` caps.

---

### Task 0 (DONE): `src/reach_samepoint.py` + tests + `deltaai/run_reach_samepoint.slurm`

Same-point actuation control. Committed artifacts: script, `tests/test_reach_samepoint.py`
(3 tests), 1-hour SLURM job. Outputs `reach_samepoint_<ds>.csv`,
`reach_samepoint_summary_<ds>.csv`; summary prints the calibration factor (slope/m_pred),
context factor (stem/samepoint), baseline offset.

### Task 1: `src/reach_judge_harden.py` — judge hardening (item 0.3, CPU, free)

**Files:**
- Create: `src/reach_judge_harden.py`
- Test: `tests/test_reach_judge_harden.py`

**Interfaces:**
- Consumes: `judge_reach_steer_<ds>.csv`, `judge_reach_steer_stmt_<ds>.csv` (columns
  `direction,scale,prompt,completion,verdict,reason`).
- Produces: `judge_hardened_fracs_<ds>.csv` (columns
  `arm,direction,scale,n,frac_true,frac_false,frac_incoherent,hardened`) with raw (hardened=0)
  and hardened (hardened=1) rows; printed before/after per-direction Spearman of
  frac_false vs |scale|.
- Pure helpers: `problem_prompts(rows) -> set[str]` (prompts with any verdict != "TRUE" at
  scale 0 — the audit's "baseline-failing prompt" criterion), `fractions(rows, exclude) ->
  dict[(direction, scale) -> (n, fT, fF, fI)]`, `spearman(x, y) -> float` (numpy ranks, no
  scipy).

- [x] Write failing tests: synthetic 2-direction CSV where one prompt fails at scale 0 —
  assert it is identified, dropped rows change frac_false as hand-computed, spearman on a
  monotone sequence = 1.0.
- [x] Implement; run `pytest tests/test_reach_judge_harden.py -q` → pass.
- [x] Commit.

### Task 2: `src/reach_stemprobe.py` — stem-population probe refit (item 0.2)

**Files:**
- Create: `src/reach_stemprobe.py`
- Test: `tests/test_reach_stemprobe.py`

**Interfaces:**
- Consumes: `reach_acts_<ds>.npz` (statements, labels), `reach_dirs_<ds>.npz` (w, t02),
  `dct_meta_<ds>.json`, `judge_reach_steer_stmt_<ds>.csv` (scale-0 verdicts as labels),
  `reach_hop.forward_source_batch`, `reach_margins.fit_threshold`.
- Produces: `--extract` (GPU): `reach_stemacts_<ds>.npz` {`h_tgt_stem` (n,d) f32,
  `stmt_index`, `stems` (object)}. `--fit` (CPU): `reach_stemprobe_<ds>.csv` (per-statement
  `stmt_index,stem,label,g_old`) + printed summary: base rate, old-threshold acc/balanced-acc,
  recalibrated-threshold acc + t02_stem, 5-fold CV refit-LR acc, 5-fold CV XGBoost acc
  (skipped with message if xgboost missing).
- Pure helpers: `stem_labels(path) -> dict[str, int]` (scale-0 TRUE→1 / FALSE→0, others
  dropped), `transfer_metrics(g, y) -> (acc, balanced_acc)`.

- [x] Failing tests: stem_labels on synthetic judge CSV (TRUE/FALSE/INCOHERENT rows);
  transfer_metrics on a hand-computed case; cv-fit smoke on separable 2-D synthetic data.
- [x] Implement (`--extract --device cuda|cpu [--limit N]`, `--fit`); tests pass.
- [x] Commit.

### Task 3: `src/reach_stemjac.py` — stem-context Jᵀw + geometry + robust margin (items 0.4+0.5)

**Files:**
- Create: `src/reach_stemjac.py`
- Test: `tests/test_reach_stemjac.py`

**Interfaces:**
- Consumes: picks protocol; `reach_hop` (`load_model_and_slice`, `forward_source_batch`,
  `make_hop`, `vjp_rows`); `reach_margins_<ds>.npz` (full-context unit jtw rows + margins);
  `reach_acts_<ds>.npz` (g at full point); `reach_dirs_<ds>.npz`.
- Produces: `--compute` (GPU): `reach_stemjac_<ds>.npz` {`jtw_stem` unit rows (n,d) f16,
  `m_stem` (n,) f32, `stmt_index`}. `--analyze` (CPU): `reach_stemjac_summary_<ds>.csv`
  (per-statement `stmt_index,m_full,m_stem,ratio,cos_full_stem,delta_norm,eps_star,eps_robust`)
  + printed aggregates: median gain ratio, median cosine, principal angles (q=8) between the
  full-row and stem-row subspaces, top stacked-SVD common direction v1 with its median
  |cos| to each row family, fraction of statements with a finite robust certificate.
- Pure helpers: `pair_metrics(jtw_full, m_full, jtw_stem, m_stem, g) -> dict of arrays`
  (cos, ratio, delta_norm = ‖m_f·u_f − m_s·u_s‖, eps_robust = g/max(m_f − delta_norm, 0) else
  inf), `principal_angles(A, B, q=8) -> degrees array`, `common_direction(A, B) -> (v1,
  cosA, cosB)`.

- [x] Failing tests: pair_metrics on hand-built 2-D vectors (identical rows → ratio 1, cos 1,
  delta 0, eps_robust = eps_star; orthogonal rows → robust margin dead); principal_angles of
  identical/orthogonal spans → 0°/90°; common_direction recovers a planted shared vector.
- [x] Implement (`--compute --device … [--limit N]`, `--analyze`); tests pass.
- [x] Commit.

### Task 4: `src/reach_newton.py` — iterated re-steering negative control (item 0.6)

**Files:**
- Create: `src/reach_newton.py`
- Test: `tests/test_reach_newton.py`

**Interfaces:**
- Consumes: picks[:N_NEWTON=32]; hop machinery with **nonzero delta0** (`vjp_rows` supports
  it); target g = 0 (the reachability boundary); cap ‖δ‖ ≤ 1.5·input_scale.
- Produces: `reach_newton_<ds>.csv` (`stmt_index,step,g,m,delta_norm,cos_jtw0,capped`) for
  steps 0..K_STEPS=3 (state logged before each update; final row = post-update state), +
  printed summary: median |g| per step, fraction converged (|g| < 5% of |g₀|), median
  direction rotation cos(jtwₖ, jtw₀).
- Pure helper: `newton_update(delta, g, jtw, cap) -> (delta_new, capped)` implementing
  δ ← δ − g·Jᵀw/‖Jᵀw‖², norm-capped.

- [x] Failing tests: newton_update on a linear scalar map converges in one step; cap
  triggers and rescales; zero-gradient guard.
- [x] Implement (`--dataset … --device … [--limit N]`); tests pass.
- [x] Commit.

### Task 5: SLURM batch + runbook + svd-vector recovery (item 0.7)

**Files:**
- Create: `deltaai/run_reach_h0.slurm` (stemprobe --extract, stemjac --compute, newton; both
  datasets; `--time=02:00:00`; dct env; per-step guards)
- Create: `deltaai/REACH_H0_RUN.md` (instance-specific runbook: laptop smoke, rsync up,
  account sed, submit samepoint + h0 concurrently, rsync back — **including
  `reach_svd_cities/` and `reach_svd_common_claim_true_false/` dirs**, then the local CPU
  analyses in order)
- Modify: `deltaai/REACH_RUN.md` Phase-3 glob (add the `reach_svd_*/` dirs so future runs
  don't repeat the omission)

- [x] Write both files; `bash -n` the slurm script.
- [x] Full test suite green.
- [x] Commit.

---

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -W error
bash -n deltaai/run_reach_h0.slurm deltaai/run_reach_samepoint.slurm
```

Cluster execution + result interpretation: `deltaai/REACH_H0_RUN.md`.
