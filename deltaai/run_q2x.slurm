#!/bin/bash
#SBATCH --job-name=tqa_q2x
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=tqa_q2x_%j.out

# Round 4, Q2x (docs/superpowers/specs/2026-09-30-round4-q2x-v1-design.md). Q2's
# direction on the 64 holdout questions at 2, 2.5, 3, 3.5, 4, 5 x eps* toward truthful,
# past the certificate's predicted crossing (3.49), with 8 norm-matched random directions.
# Each answer is read at layer 20 in the same pass, judged with the v2 judges, and judged
# again cut to its question's unsteered length (C3 per dose).
#
# Writes only q2x_{steer,judged,trunc_judged,summary}_truthfulqa.csv and
# q2x_outcome_truthfulqa.json. Resumes per (direction, dose) block and per 64 judged rows.
#
# READ THIS FIRST WHEN IT FINISHES:
#   [q2x] R0 anchor ...      PASS, or nothing else here is read
#   [q2x] R1 crossing ...    realized vs predicted 3.49
#   [q2x] HEADLINE ...       a_content / b_form / c_degeneration at the crossing

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
for f in reach_summary_${DS}.json reach_dirs_${DS}.npz reach_margins_${DS}.npz \
         reach_acts_${DS}.npz dct_meta_${DS}.json got_datasets/${DS}_holdout.csv \
         tqa_q2_summary_truthfulqa_judge_v2_truthful.csv src/tqa_q2x.py; do
  test -f "$f" || { echo "!!!! $f missing. The reach_* files are Q2's; the v2 summary is J-A's."; exit 1; }
done
for m in models--google--gemma-2-2b models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done

echo "=== stage 1: smoke, 2 questions, Q2's direction and one random, smoke_ outputs ==="
rm -f smoke_q2x_*
python3 src/tqa_q2x.py --stage all --device cuda --limit 2 --prefix smoke_
rm -f smoke_q2x_*

echo "=== stage 2: the full run ==="
python3 src/tqa_q2x.py --stage all --device cuda

echo "=== job complete ==="
ls -la q2x_*_${DS}.*
