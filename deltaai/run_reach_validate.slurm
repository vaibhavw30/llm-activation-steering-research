#!/bin/bash
#SBATCH --job-name=reach_validate
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=01:30:00
#SBATCH --output=reach_validate_%j.out

# Round 4, V1 (docs/superpowers/specs/2026-09-30-round4-q2x-v1-design.md;
# PLAN_ADVISOR_NOTES_2026-09.md section 2). The certificate's closed check at the output
# layer, where the readout is the yes-minus-no decision and cannot dissociate from it:
# 64 statements, every source layer, contexts decl / stem / quest, on cities then
# truthfulqa.
#
# Writes only reach_validate_{cities,truthfulqa}.npz and reach_validate_summary_*.csv.
# A resubmit after a timeout resumes per dataset: a finished summary is skipped, and an
# npz without its summary is only analyzed. reach_validate.py itself never overwrites.
#
# READ THIS FIRST WHEN IT FINISHES, per dataset:
#   the gate line: "arithmetic is sound" (short hops >= 20 within 0.10) or !!!!
#   then ratio_B per layer: any layer >= 0.5 holds its gain across the context shift

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
  for f in reach_acts_${ds}.npz dct_meta_${ds}.json; do
    test -f "$f" || { echo "!!!! $f missing."; exit 1; }
  done
done
test -d "$HF_HOME/hub/models--google--gemma-2-2b" \
  || { echo "!!!! gemma-2-2b not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }

echo "=== stage 1: smoke, cities, 4 statements, layers 0 and 25, smoke_ outputs ==="
rm -f smoke_reach_validate_*
python3 src/reach_validate.py --dataset cities --compute --analyze --device cuda \
  --limit 4 --layers 0,25 --prefix smoke_
rm -f smoke_reach_validate_*

n=2
for ds in cities truthfulqa; do
  echo "=== stage $n: $ds ==="
  if [ -e reach_validate_summary_${ds}.csv ]; then
    echo "reach_validate_summary_${ds}.csv exists, skipping $ds"
  elif [ -e reach_validate_${ds}.npz ]; then
    python3 src/reach_validate.py --dataset $ds --analyze
  else
    python3 src/reach_validate.py --dataset $ds --compute --analyze --device cuda
  fi
  n=$((n + 1))
done

echo "=== job complete ==="
ls -la reach_validate_*
