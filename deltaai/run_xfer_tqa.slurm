#!/bin/bash
#SBATCH --job-name=xfer_tqa
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=05:00:00
#SBATCH --output=xfer_tqa_%j.out

# J-D2 of docs/PLAN_PI_FEEDBACK_2026-09-18.md (round 2, section 8): every truth direction
# from cities AND TruthfulQA, steered on the 64 TruthfulQA holdout questions at layer 11,
# same direction list and dose grid as J-D1, judged by the v2 allenai judges.
# Positive control: tqa:sup_jtw at eps +2 is Q2's registered dose and must land inside
# Q2's v2 Wilson interval. 8 norm-matched random directions are the null.
# src/xfer_tqa.py.
#
# NEEDS: J-B's dct_selection_truthfulqa.json, J-A's v2 Q2 summary. Independent of J-D1.
# Writes only xfer_truthfulqa_{meta.json,parity.csv,steer.csv,judged.csv,summary.csv,
# outcomes.json}. Resumes per (direction, unit, dose) and per 64-row judge block.
# Generation is BATCHED here (Q2 and J-C were one prompt at a time); the parity file
# says how often that changes an answer.
#
# READ THIS FIRST WHEN IT FINISHES, in order:
#   [parity] batched == one-at-a-time on ...  how much batching changed answers
#   [summary] POSITIVE CONTROL ... REPRODUCED if not, no TQA number here is used
#   [summary] cities:* rows at norm +0.5      the reverse cell: cities directions on TQA

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
for f in got_datasets/truthfulqa_holdout.csv mag_dir_cities.npz \
         dct_selection_truthfulqa.json; do
  test -f "$f" || { echo "!!!! $f missing. dct_selection comes from J-B."; exit 1; }
done
ls tqa_q2_summary_truthfulqa_judge_v2*.csv \
  || echo "[preflight] !!!! no v2 Q2 summary from J-A: the positive control falls back to v1"
test -f jb_truthfulqa/mag_dir_truthfulqa.npz \
  || echo "[preflight] no TQA MAG directions from J-B; those rows will be absent"
for m in models--google--gemma-2-2b models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done
python3 -c "
import json, sys
m = json.load(open('dct_meta_truthfulqa.json'))
sys.exit(0 if m.get('num_factors') is None and m['source_layer'] == 11 else
         '!!!! dct_meta_truthfulqa.json is not the Q2 meta')"

echo "=== stage 1: smoke, 4 questions, 5 directions, smoke_ outputs ==="
python3 src/xfer_tqa.py --stage all --device cuda --limit 4 --prefix smoke_

echo "=== stage 2: the full run ==="
python3 src/xfer_tqa.py --stage all --device cuda

echo "=== job complete ==="
ls -la xfer_truthfulqa_*
