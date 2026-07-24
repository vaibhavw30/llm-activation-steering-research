# Reachability Audit — Runbook

Spec: `docs/superpowers/specs/2026-07-24-reachability-audit-design.md`.
Datasets: cities (hop 11→20), common_claim_true_false (13→22). All cluster jobs
follow the DeltaAI conventions in `deltaai/CLUSTER_OPERATIONS.md` (set `--account`,
one NCSA password + Duo push per ssh/rsync batch).

## Phase 0 — one-time cluster prerequisite (login node)

`.venv-dct-gpu` has no scikit-learn (setup_env.sh installs only
transformers/scipy/tqdm/pandas), and `reach_margins --stage dirs` and
`reach_jlens` need it (both call `fit_probe_dir`/`fit_threshold`):

    module load python/miniforge3_pytorch
    source .venv-dct-gpu/bin/activate
    pip install scikit-learn

## Phase 0b — local smoke (laptop, before any rsync)

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_hop.py tests/test_reach_margins.py tests/test_reach_analyze.py tests/test_reach_linerr.py tests/test_viz_reach.py -q
    PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage acts --limit 8 --device cpu
    PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage dirs
    PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage vjp --limit 8 --device cpu
    PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset cities
    PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset cities

(The `--limit 8` artifacts are throwaway sanity checks — delete
`reach_acts_cities.npz reach_dirs_cities.npz reach_chunks_cities reach_margins_cities.npz reach_curve_cities.csv reach_summary_cities.json plot_reach_*_cities.png`
before rsyncing, so the cluster runs regenerate them at full size.)

## Phase 1 — rsync up (one batch)

    rsync -av --relative src deltaai tests docs got_datasets \
      dct_meta_cities.json dct_meta_common_claim_true_false.json \
      truth_dir_cities.npz truth_dir_common_claim_true_false.npz \
      truth_dir_tgt_cities.npz truth_dir_tgt_common_claim_true_false.npz \
      mag_dir_cities.npz mag_dir_common_claim_true_false.npz \
      USER@dt-login.delta.ncsa.illinois.edu:~/PROJECT_DIR/

(`dct_V_*.pt` / `dct_U_*.pt` are already on the cluster from the DCT runs; if
`ls ~/PROJECT_DIR/dct_V_*.pt` says otherwise, add them to the batch.)

## Phase 2 — cluster jobs (GH200)

Submit order (margins gates everything; svd/linerr only need margins' stage
outputs; steer additionally needs the local Phase-3 analyze step — see below):

    sbatch deltaai/run_reach_margins.slurm          # ~2-4 h: acts+dirs+vjp, both datasets
    # after run_reach_margins completes:
    sbatch deltaai/run_reach_svd.slurm              # independent of analyze
    sbatch deltaai/run_reach_linerr.slurm           # independent of analyze

`reach_steer` needs `reach_summary_<ds>.json` (produced by the LOCAL analyze
step). Two options:
  (a) run analyze ON the cluster (sklearn now installed):
        PYTHONPATH=src python3 src/reach_analyze.py --dataset cities
        PYTHONPATH=src python3 src/reach_analyze.py --dataset common_claim_true_false
      (interactive on the login node, seconds — margins npz stays on scratch), then
        sbatch deltaai/run_reach_steer.slurm
        # after run_reach_steer completes:
        sbatch deltaai/run_reach_judge.slurm
  (b) or rsync margins down, analyze locally, rsync summary up (adds a Duo batch).
Option (a) is the default.

## Phase 3 — rsync back (one batch; excludes weights)

    rsync -av --exclude '*.pt' \
      USER@dt-login.delta.ncsa.illinois.edu:~/PROJECT_DIR/'reach_margins_*.npz reach_acts_*.npz reach_dirs_*.npz reach_svd_*/ reach_svd_*.csv reach_curve_*.csv reach_summary_*.json reach_steer_*.csv reach_jlens_*.csv reach_linerr_*.csv judge_reach_*.csv plot_judge_reach_*.png reach_*_%j.out *.out' \
      ./

(reach_margins npz ≈ 200 MB/dataset — jtw rows are float16 to keep this small.
If bandwidth hurts, exclude `reach_acts_*.npz` and re-derive locally later.)

## Phase 4 — local analysis + figures

    PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset cities
    PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset common_claim_true_false
    PYTHONPATH=src .venv/bin/python src/reach_linerr.py --dataset cities --summarize
    PYTHONPATH=src .venv/bin/python src/reach_linerr.py --dataset common_claim_true_false --summarize
    PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset cities
    PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset common_claim_true_false

## Decision gates between phases (spec §4 cross-phase logic)

- **Gate P1:** read `reach_summary_<ds>.json`. `verdict=unreachable-tail`
  (truth-subspace best-case median margin < random-null median) ⇒ Phase 2 is the
  mechanism story. `reachable-candidate` ⇒ Phase 3 is the headline experiment.
- **Gate P2:** `dctV_overlap_top16` low (< 0.3 mean) in `reach_svd_summary_<ds>.csv`
  is a STOP-AND-DIAGNOSE — the linearized picture disagrees with DCT; do not
  interpret P1 margins until resolved.
- **Gate P3:** the 2×2 from `judge_reach_steer_*.csv` × `reach_steer_readout_*.csv`:
  readout moves & behavior moves ⇒ lever found (pivot to characterizing it);
  readout moves & behavior doesn't ⇒ LiSeCo-style dissociation claim.
- **Gate P5:** all P1–P3 claims must be restated inside/outside the
  `reach_linerr_summary` validity radius; if the radius ≪ input_scale, phrase the
  unreachability claim at the radius, with nonlinear spot-checks beyond.

## Parallel track (unchanged — spec §8)

The conditional-steering / U-anchor round launches independently per
`docs/CONDITIONAL_UANCHOR_RUNBOOK.md`; its judge results feed Phase 3/4
interpretation.
