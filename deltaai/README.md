# DeltaAI (NCSA GH200): what to run, and where it is written down

Every command in this directory runs on **one of two machines**, and the runbooks label which.
Check the label before you run anything. The two-machine workflow has already caused one
misdirected rsync.

- **LAPTOP** (Apple M-series): direction building, analysis, plotting, judging against an API.
- **CLUSTER** (DeltaAI GH200, ARM64 + ~96 GB Hopper): every forward pass through the model.

## Read these two first

| Doc | What it covers |
|---|---|
| [`CLUSTER_OPERATIONS.md`](CLUSTER_OPERATIONS.md) | **The master guide.** Login, the two environments, SLURM, the two-rsync pattern, the gotchas, budget discipline. Parts 1 to 4 are read-once; Part 5 indexes the older experiments. |
| [`GPU_SETUP.md`](GPU_SETUP.md) | Building the environments from scratch on the cluster. |

`CLUSTER_WALKTHROUGH.md` is still accurate for the original DCT funnel but predates everything
after it. `CLUSTER_OPERATIONS.md` is the superset.

## Experiment index

The table below is the current one, including the experiments that postdate Part 5 of
`CLUSTER_OPERATIONS.md`.

| Experiment | Runbook | SLURM scripts |
|---|---|---|
| DCT funnel (fit, interpret, steer) | `CLUSTER_WALKTHROUGH.md` | `run_dct.slurm` `run_interpret.slurm` `run_steer.slurm` |
| Warm-started DCT | [`DCT_WARM_RUN.md`](DCT_WARM_RUN.md) | `run_dct_warm.slurm` `run_dct_warm_judge.slurm` `run_dct_uwarm.slurm` `run_dct_uwarm_judge.slurm` |
| MAG battery | (runbook kept local) | `run_mag_extract.slurm` `run_mag_steer.slurm` `run_mag_judge.slurm` `run_mag_cond_judge.slurm` |
| Length steering | [`LENGTH_STEER_RUN.md`](LENGTH_STEER_RUN.md) | `run_length_steer.slurm` `run_length_judge.slurm` |
| Backward reachability, 5 phases | [`REACH_RUN.md`](REACH_RUN.md) | `run_reach_steer.slurm` `run_reach_margins.slurm` `run_reach_svd.slurm` `run_reach_linerr.slurm` `run_reach_samepoint.slurm` `run_reach_judge.slurm` |
| Reachability Horizon-0 validations | [`REACH_H0_RUN.md`](REACH_H0_RUN.md) | `run_reach_h0.slurm` |
| Token space | [`TOKEN_SPACE_RUN.md`](TOKEN_SPACE_RUN.md), [`TOKEN_SPACE_CLUSTER_STEPS.md`](TOKEN_SPACE_CLUSTER_STEPS.md) | `run_token_geom.slurm` `run_token_jac.slurm` `run_token_sens.slurm` `run_token_steer.slurm` |
| D1 dose-response | [`D1_DOSE_RUN.md`](D1_DOSE_RUN.md) | `run_dose.slurm` `run_dose_judge.slurm` |
| Refusal positive control | [`REFUSAL_RUN.md`](REFUSAL_RUN.md) | `run_refusal_prep.slurm` `run_refusal_reach.slurm` `run_refusal_screen.slurm` `run_refusal_spotcheck.slurm` |
| TruthfulQA, track Q | [`TRUTHFULQA_RUN.md`](TRUTHFULQA_RUN.md) | `run_truthfulqa_prep.slurm` `run_tqa_baseline.slurm` `run_truthfulqa_reach.slurm` `run_truthfulqa_judge.slurm` `run_truthfulqa_randctrl.slurm` `run_truthfulqa_stmt.slurm` `run_truthfulqa_dct.slurm` |
| PI feedback, 2026-09-18 | [`../docs/PLAN_PI_FEEDBACK_2026-09-18.md`](../docs/PLAN_PI_FEEDBACK_2026-09-18.md) section 4 | `run_pi_audit.slurm` (J-A, judge audit, round 0) |
| PI feedback, 2026-09-18 | [`../docs/PLAN_PI_FEEDBACK_2026-09-18.md`](../docs/PLAN_PI_FEEDBACK_2026-09-18.md) section 4 | `run_tqa_discovery.slurm` (J-B) + `run_tqa_confirm.slurm` (J-C, afterok J-B), round 1, via `submit_pi_feedback.sh round1` |
| PI feedback, round 2 | same, sections 4 and 8 | `run_xfer_cities.slurm` (J-D1) + `run_xfer_tqa.slurm` (J-D2), independent, after J-C, via `submit_pi_feedback.sh round2` |
| Judging (local OLMo backend) | [`JUDGE_RUN_STEPS.md`](JUDGE_RUN_STEPS.md) | `run_judge.slurm` |

Some runbooks contain account and login specifics and are kept out of git (`MY_RUN_STEPS.md`,
`MAG_EXTRACT_RUN.md`, `MAG_E4_RUN.md`, `FUNNEL_RUN_STEPS.md`). If you are picking this up on a new
account, `CLUSTER_OPERATIONS.md` Parts 1 to 4 plus `GPU_SETUP.md` have everything those add.

## Environment setup

```bash
bash deltaai/setup_env.sh        # CLUSTER: .venv-dct-gpu, the main compute env
bash deltaai/setup_judge_env.sh  # CLUSTER: .venv-judge-gpu, the local OLMo judge
```

## The habit that saves a job

Dry-run the submission before spending the allocation:

```bash
sbatch --test-only deltaai/run_<job>.slurm
```

Compute nodes have **no internet**. Any model weights must be pre-downloaded to `$HOME/hf_cache`
from the login node first, or the job dies at the first `from_pretrained`.
