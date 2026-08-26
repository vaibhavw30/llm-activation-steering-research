#!/bin/bash
#SBATCH --job-name=dose
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=dose_%j.out
# D1 — the dose-response kill test (design spec section 7, Gate 2).
#
# READ THE LIVENESS LINE FIRST. Every run prints
#     [D1][assert] hook live: rel=5.0 changes the text ...
# before generating anything. At five times the median activation norm the completion cannot
# stay byte-identical. If that assert raises instead, the forward_pre_hook is not firing and
# EVERY null this job produces is an artifact of a dead hook. Fix the harness; read nothing else.
#
# Requires, all already committed: mag_dir_<ds>.npz, truth_dir_<ds>.npz, dct_meta_<ds>.json,
# reach_acts_<ds>.npz, reach_margins_<ds>.npz, reach_summary_<ds>.json.
#
# WRITES ONLY dose_*. It cannot touch mag_steer_<ds>.csv, reach_steer_<ds>.csv or any other
# committed artifact, and it refuses to start if a dose file already exists (pass --resume).
#
# Cost: 6 directions x 25 doses x 56 prompts + baseline = about 8,400 generations per dataset,
# plus one forward per yes/no cell for the verdict margin. Budget roughly 2 hours per dataset
# on a GH200; the 6h wall is deliberate headroom. If it times out anyway, resubmit with
# --resume and it picks up at the first unfinished (direction, dose) cell.
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# Pass RESUME=1 when resubmitting after a wall-clock kill.
EXTRA=""
if [ "${RESUME:-0}" = "1" ]; then EXTRA="--resume"; fi

for ds in cities common_claim_true_false; do
  echo "=== dose_response $ds $(date) ==="
  PYTHONPATH=src python3 src/dose_response.py --dataset "$ds" --device cuda $EXTRA \
    || echo "!!!! $ds FAILED — continuing"
done
echo "=== dose_response done $(date) ==="

# Judging and analysis run locally afterwards, not here (the judge needs ANTHROPIC_API_KEY and
# the cluster nodes have no outbound network):
#
#   ./.venv/bin/python src/judge_results.py --mode steer --dataset cities \
#       --steer-input dose_cities.csv --steer-output judge_dose_cities.csv \
#       --steer-plot plot_judge_dose_cities.png
#   ./.venv/bin/python src/dose_analyze.py --dataset cities
