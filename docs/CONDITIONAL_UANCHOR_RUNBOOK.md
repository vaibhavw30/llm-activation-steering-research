# Conditional Steering + U-Anchored DCT — Runbook

Spec: `docs/superpowers/specs/2026-07-23-conditional-steering-uspace-anchor-design.md`.
Datasets: cities, common_claim_true_false. Sign convention: +τ→TRUE, −τ→FALSE.

## Phase 1 — local (laptop, .venv, MPS)

1. Target-layer seeds (Arm B prerequisite, ~seconds):
   `.venv/bin/python src/export_target_dir.py --dataset cities`
   `.venv/bin/python src/export_target_dir.py --dataset common_claim_true_false`
2. Arm A1 smoke, then full (~1-2 h/dataset on MPS):
   `PYTHONPATH=src .venv/bin/python -m mag.steer_verdict --dataset cities --device mps --limit 2 --only sup_mean_diff`
   `PYTHONPATH=src .venv/bin/python -m mag.steer_verdict --dataset cities --device mps`
   `PYTHONPATH=src .venv/bin/python -m mag.steer_verdict --dataset common_claim_true_false --device mps`
3. A1 figures (immediately — A1 needs no judge):
   `PYTHONPATH=src .venv/bin/python -m mag.viz_verdict --dataset cities`
   `PYTHONPATH=src .venv/bin/python -m mag.viz_verdict --dataset common_claim_true_false`
4. Arm A2 generation (~1 h/dataset):
   `PYTHONPATH=src .venv/bin/python -m mag.steer_conditional --dataset cities --device mps`
   `PYTHONPATH=src .venv/bin/python -m mag.steer_conditional --dataset common_claim_true_false --device mps`

## Phase 2 — rsync up (one batch: one NCSA password + Duo push)

Code + inputs (from repo root; adjust remote path to the existing project dir):

    rsync -av --relative src deltaai tests docs \
      truth_dir_tgt_cities.npz truth_dir_tgt_common_claim_true_false.npz \
      mag_conditional_cities.csv mag_conditional_common_claim_true_false.csv \
      USER@dt-login.delta.ncsa.illinois.edu:~/PROJECT_DIR/

(`mag_steer_*.csv` for A0 are already on the cluster from the MAG run; if not, add them.)

## Phase 3 — cluster jobs (GH200)

    sbatch deltaai/run_mag_cond_judge.slurm     # A0 + A2 judging (needs mag_conditional_*.csv)
    sbatch deltaai/run_dct_uwarm.slurm          # Arm B: 6 fits + directions + steer
    # after run_dct_uwarm completes:
    sbatch deltaai/run_dct_uwarm_judge.slurm

`run_mag_cond_judge` and `run_dct_uwarm` are independent — submit together.

## Phase 4 — rsync back (one batch; excludes .pt/.npz weights, matching the two-rsync pattern)

    rsync -av --exclude '*.pt' --exclude '*.npz' \
      USER@dt-login.delta.ncsa.illinois.edu:~/PROJECT_DIR/'judge_mag_steer_*.csv judge_mag_conditional_*.csv judge_dct_uwarm_steer_*.csv dct_uwarm_geometry_*.csv plot_judge_*.png *.out' \
      ./

## Phase 5 — figures + interpretation (local)

    PYTHONPATH=src .venv/bin/python -m mag.viz_conditional --dataset cities
    PYTHONPATH=src .venv/bin/python -m mag.viz_conditional --dataset common_claim_true_false
    PYTHONPATH=src .venv/bin/python src/viz_dct_uwarm.py --dataset cities
    PYTHONPATH=src .venv/bin/python src/viz_dct_uwarm.py --dataset common_claim_true_false

Read results against the interpretation matrix in the spec (§Interpretation matrix).

## Outputs checklist

| Arm | Data | Figures |
|---|---|---|
| A0 | judge_mag_steer_{ds}.csv | plot_judge_mag_steer_{ds}.png |
| A1 | mag_verdict_logits_{ds}.csv | plot_mag_verdict_margin/acc_{ds}.png |
| A2 | mag_conditional_{ds}.csv, judge_mag_conditional_{ds}.csv | plot_mag_conditional_{ds}.png |
| B | dct_uwarm_geometry_{ds}.csv, judge_dct_uwarm_steer_{ds}.csv | plot_dct_uwarm_curves/geometry_{ds}.png |
