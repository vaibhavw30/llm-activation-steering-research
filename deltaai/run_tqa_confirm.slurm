#!/bin/bash
#SBATCH --job-name=tqa_confirm
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=05:00:00
#SBATCH --output=tqa_confirm_%j.out

# J-C of docs/PLAN_PI_FEEDBACK_2026-09-18.md (round 1, afterok J-B): D3 and G1. Every
# direction J-B picked, plus the Q2 direction as the positive control, MAG's directions
# and 8 norm-matched random ones, steered on the 64 HOLDOUT questions at +/-1 and +/-2
# x Q2's eps*, judged with the v2 judges. Selection happened in J-B on fit-set questions;
# nothing here chooses anything.
#
# Writes only d3_steer_, d3_judged_, d3_summary_ *_truthfulqa.csv. Resumes per
# (direction, dose) and per 64-row judge block.
#
# READ THIS FIRST WHEN IT FINISHES:
#   [confirm] the positive control's row at frac +2    did Q2's effect reproduce here?
#   then each dct_ / mag_ row: McNemar p and perm p (rank among the random directions)

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

DS=truthfulqa
echo "=== stage 0: preflight ==="
for f in dct_selection_${DS}.json dct_V_${DS}.pt dct_U_${DS}.pt reach_summary_${DS}.json \
         reach_margins_${DS}.npz reach_acts_${DS}.npz reach_dirs_${DS}.npz \
         dct_meta_${DS}.json got_datasets/${DS}_holdout.csv; do
  test -f "$f" || { echo "!!!! $f missing. dct_selection comes from J-B."; exit 1; }
done
test -f jb_truthfulqa/mag_dir_${DS}.npz \
  || echo "[preflight] no MAG directions from J-B; the MAG rows will be absent"
for m in models--google--gemma-2-2b models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done

echo "=== stage 1: smoke, 4 questions, 3 directions, smoke_ outputs ==="
python3 src/tqa_confirm.py --stage all --device cuda --limit 4 --prefix smoke_

echo "=== stage 2: the full run ==="
python3 src/tqa_confirm.py --stage all --device cuda

echo "=== job complete ==="
ls -la d3_*_${DS}.csv
