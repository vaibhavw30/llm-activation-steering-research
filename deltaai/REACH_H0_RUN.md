# Horizon-0 Validations — DeltaAI Run (this instance)

Copy-paste runbook for the Horizon-0 batch: same-point control (0.1), stem-probe extraction
(0.2), stem-context Jᵀw (0.4/0.5), Newton control (0.6), plus the svd-vector recovery (0.7).
Plan: [../docs/superpowers/plans/2026-07-28-horizon0-validations.md](../docs/superpowers/plans/2026-07-28-horizon0-validations.md).
Coordinates, envs, and gotchas are identical to [REACH_RUN.md](REACH_RUN.md) — account
`bhhv-dtai-gh`, user `vwudaru@dtai-login.delta.ncsa.illinois.edu`, partition `ghx4`, dct env.

**Everything these jobs read is already on the cluster** from the audit run
(`reach_acts/dirs/margins/summary`, `dct_meta`, direction files, model cache) — only code
travels up. Expected GPU spend: samepoint ≤1 h cap, h0 ≤2 h cap; real usage well under.

**The two jobs** (independent — submit both at once):

| Job | `--time` | Runs | Writes |
|---|---|---|---|
| `run_reach_samepoint.slurm` | 01:00 | reach_samepoint + --summarize, both ds | `reach_samepoint_*.csv`, `reach_samepoint_summary_*.csv` |
| `run_reach_h0.slurm` | 02:00 | stemprobe --extract, stemjac --compute, newton, both ds | `reach_stemacts_*.npz`, `reach_stemjac_*.npz`, `reach_newton_*.csv` |

---

## Phase 0 — laptop smoke (💻)

```bash
cd ~/llm-activation-steering-research
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_reach_samepoint.py tests/test_reach_judge_harden.py \
  tests/test_reach_stemprobe.py tests/test_reach_stemjac.py \
  tests/test_reach_newton.py -q
```

Expected: all pass. (0.3 judge hardening is purely local — you can run it right now, no
cluster needed: `PYTHONPATH=src .venv/bin/python src/reach_judge_harden.py --dataset cities`
and same for `common_claim_true_false`.)

## Phase 1 — rsync code up (💻, one batch)

```bash
cd ~/llm-activation-steering-research
rsync -av \
  --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

## Phase 2 — submit both jobs (🖥️)

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research
sed -i "s/ACCOUNT_NAME/bhhv-dtai-gh/" \
  deltaai/run_reach_samepoint.slurm deltaai/run_reach_h0.slurm
grep -H -- --account deltaai/run_reach_samepoint.slurm deltaai/run_reach_h0.slurm
    # must both print bhhv-dtai-gh, not ACCOUNT_NAME

sbatch deltaai/run_reach_samepoint.slurm
sbatch deltaai/run_reach_h0.slurm
squeue -u vwudaru          # wait until EMPTY (poll; each job well under its cap)
grep -a -E "===|wrote|FAILED|Traceback" reach_samepoint_*.out reach_h0_*.out
ls reach_samepoint_*.csv reach_stemacts_*.npz reach_stemjac_*.npz reach_newton_*.csv
    # expect 2 of each (4 samepoint CSVs: per-scale + summary per dataset)
```

## Phase 3 — rsync results back (💻) — includes the 0.7 svd-vector recovery

The `reach_svd_cities/` and `reach_svd_common_claim_true_false/` dirs (32 per-statement
U64/V64 npz files each, ~19 MB/dataset) were computed in the audit run but excluded from the
original rsync-back glob — this pulls them now. **No GPU re-run needed** unless the dirs were
purged (then: `sbatch deltaai/run_reach_svd.slurm`, ~10 min).

```bash
cd ~/llm-activation-steering-research
rsync -av \
  'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{reach_samepoint_*.csv,reach_stemacts_*.npz,reach_stemjac_*.npz,reach_newton_*.csv,reach_svd_cities,reach_svd_common_claim_true_false}' \
  ./
ls reach_svd_cities/ | head -3          # expect stmt_*.npz files
```

## Phase 4 — local CPU analyses (💻, in order)

```bash
cd ~/llm-activation-steering-research
for ds in cities common_claim_true_false; do
  # 0.1 decomposition ladder: m_pred -> same-point slope -> stem slope
  PYTHONPATH=src .venv/bin/python src/reach_samepoint.py --dataset $ds --summarize
  # 0.3 judge hardening (if not already run in Phase 0)
  PYTHONPATH=src .venv/bin/python src/reach_judge_harden.py --dataset $ds
  # 0.2 probe refit on the generation population (LR + XGBoost arms)
  PYTHONPATH=src .venv/bin/python src/reach_stemprobe.py --dataset $ds --fit
  # 0.4 + 0.5 full-vs-stem Jacobian geometry + robust certificate
  PYTHONPATH=src .venv/bin/python src/reach_stemjac.py --dataset $ds --analyze
done
```

## Reading the results (decision gates)

- **0.1 samepoint** — calibration factor ≈ 1 ⇒ vjp margins are right at their own point and
  the whole 8–35× collapse is the context factor (clean D1 decomposition). Calibration ≪ 1 ⇒
  margins locally miscalibrated: cross-check Newton before writing anything.
- **0.6 newton** — expected: converged ≥ 90% in ≤ 2 steps at the same point. If it does NOT
  converge, D1's "context, not curvature" claim is wrong — stop and diagnose with 0.1.
- **0.2 stemprobe** — recalibrated/refit accuracy ≫ old-threshold accuracy ⇒ D2 is a threshold
  shift (recalibrable; steering may have crossed a *meaningless* boundary). Accuracy ≈ base
  rate even refit ⇒ no truth signal at stem tokens at all (stronger claim). XGBoost gap ≥ 0.05
  ⇒ nonlinear structure on the generation population — feeds the SAE forensics (Horizon 1.2).
- **0.4/0.5 stemjac** — median cos(full, stem) and the common-direction alignment answer "does
  ANY input direction survive the shift?"; the finite-fraction of `eps_robust` is the honest
  robust certificate to report instead of eps*.
- **0.7** — with `reach_svd_*/` local, Horizon-2 items 2.2 (balanced truncation) and 2.3
  (in-channel steering) and the SAE decomposition of V64 are unblocked.

## Gotcha: transformers version and the hop map

`dct.SlicedModel` divides gemma-2 inputs by √d expecting the HF model to re-multiply
`inputs_embeds` by its normalizer. **transformers ≥ 5 no longer does** (the local `.venv` is
5.12.1), which silently made the hop map unrelated to the real forward (cos ≈ 0.04–0.23).
`reach_hop.load_model_and_slice` now runs a startup fidelity probe and auto-applies a √d
compensation — every hop job prints `[reach] slice fidelity cos=1.000000 (input compensation
x…)` as its first model line. **If a job instead dies with `SlicedModel unfaithful`, stop:**
the environment can't reproduce the forward and any margins it produced would be Jacobians of
the wrong map. The cluster's `.venv-dct-gpu` (4.51.3) is faithful with factor 1 — the audit's
margins were computed there and stand.
