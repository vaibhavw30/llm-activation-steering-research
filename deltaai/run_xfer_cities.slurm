#!/bin/bash
#SBATCH --job-name=xfer_cities
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=xfer_cities_%j.out

# J-D1 of docs/PLAN_PI_FEEDBACK_2026-09-18.md (round 2, section 8): every truth direction
# from cities AND TruthfulQA, steered on ~177 cities ("The city of X is in the country of") at layer 11, in two dose
# units (x median ||h_11||, x eps*), both signs. Two readouts per steer: the country
# logit margin and a 12-token generation. Controls: the jtw_token oracle (positive, must
# flip), 32 norm-matched random directions and a potency-matched random DCT factor
# (negative). No judge, so no judge models are needed. src/xfer_cities.py.
#
# NEEDS: J-B's dct_selection_truthfulqa.json (and J-C done, per the plan's rounds).
# Writes only xfer_cities_{meta.json,oracle.npz,steer.csv,summary.csv,outcomes.json}.
# Resumes per (direction, unit, dose); the oracle's actuators are cached.
#
# READ THIS FIRST WHEN IT FINISHES, in order:
#   [summary] ORACLE hit rate ... READABLE     if not, no cities null in the file means anything
#   [summary] P8 CELL tqa:dct_S-beh_* -> a/b/c  the PI's question 8
#   [xfer] !!!! left out: ...                  any direction that could not be built

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

echo "=== stage 0: preflight ==="
for ds in cities truthfulqa; do
  for f in reach_summary_${ds}.json reach_margins_${ds}.npz reach_acts_${ds}.npz \
           reach_dirs_${ds}.npz truth_dir_${ds}.npz dct_meta_${ds}.json \
           dct_V_${ds}.pt dct_U_${ds}.pt; do
    test -f "$f" || { echo "!!!! $f missing."; exit 1; }
  done
done
for f in token_acts_cities.npz token_geom_cities.npz got_datasets/cities.csv \
         mag_dir_cities.npz dct_selection_truthfulqa.json; do
  test -f "$f" || { echo "!!!! $f missing. dct_selection comes from J-B."; exit 1; }
done
test -f jb_truthfulqa/mag_dir_truthfulqa.npz \
  || echo "[preflight] no TQA MAG directions from J-B; those rows will be absent"
test -d "$HF_HOME/hub/models--google--gemma-2-2b" \
  || { echo "!!!! gemma-2-2b not in $HF_HOME/hub."; exit 1; }
python3 -c "import scipy.stats" || { echo "!!!! scipy missing"; exit 1; }
# The TQA meta must be the Q2 meta, not a DCT fit's copy (U1's stage-1 swap).
python3 -c "
import json, sys
m = json.load(open('dct_meta_truthfulqa.json'))
sys.exit(0 if m.get('num_factors') is None and m['source_layer'] == 11 else
         '!!!! dct_meta_truthfulqa.json is not the Q2 meta')"

echo "=== stage 1: smoke, 8 cities, 5 directions, smoke_ outputs ==="
python3 src/xfer_cities.py --stage all --device cuda --limit 8 --prefix smoke_

echo "=== stage 2: the full run ==="
python3 src/xfer_cities.py --stage all --device cuda

echo "=== job complete ==="
ls -la xfer_cities_*
