#!/bin/bash
#SBATCH --job-name=tqa_reach
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=tqa_reach_%j.out

# Q2, the GPU half. Certificate on TruthfulQA, then steer the 64 held-out questions.
#
# THIS JOB DOES NOT JUDGE. A TruthfulQA analogue of src/refusal_judge.py does not exist
# yet, and reach_control.py reads judge_refusal_{ds}_{arm}.csv by hardcoded name. Those
# are a laptop morning of work. Everything expensive is here, so the judging can follow
# in a short second job against the CSVs this one writes.
#
# Q1 (job 3081925) is this job's control: unsteered truthful AND informative = 0.266
# [0.173, 0.385] on the same 64 prompts. See docs/Q1_TRUTHFULQA_BASELINE.md.

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# Fail in the first ten seconds, not after the VJP stage. Every one of these is a Q0
# output that must have been rsynced up, or a login-node staging step.
for f in got_datasets/truthfulqa.csv got_datasets/truthfulqa_holdout.csv \
         truth_dir_tgt_truthfulqa.npz dct_meta_truthfulqa.json; do
  test -f "$f" || { echo "!!!! missing $f. Q0 output: rsync it up."; exit 1; }
done
test -d "$HF_HOME/hub/models--google--gemma-2-2b" \
  || { echo "!!!! gemma-2-2b not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }

# SMOKE: load_model_and_slice and stage_vjp are only ever exercised against a real
# model, never in the torch-free test suite. Eight statements means a broken hop fails
# in minutes rather than hours in. Hard stop: the full run is not worth attempting.
echo "=== SMOKE: tiny --limit reach_margins pass before the full budget ==="
python3 src/reach_margins.py --dataset truthfulqa --stage all --device cuda --limit 8 \
  || { echo "!!!! SMOKE FAILED, aborting, full run not attempted"; exit 1; }
# The smoke's 8-statement chunk must not leak into the full run: stage_vjp's resume
# would treat it as a completed 100-statement chunk and silently misalign the margins.
rm -rf reach_acts_truthfulqa.npz reach_dirs_truthfulqa.npz \
       reach_margins_truthfulqa.npz reach_chunks_truthfulqa || exit 1
echo "=== smoke OK, smoke artifacts cleared ==="

echo "=== P1 margins (acts + dirs + vjp) ==="
python3 src/reach_margins.py --dataset truthfulqa --stage all --device cuda || exit 1

echo "=== P1 analysis: eps*, the trust radius, and the per-direction summary ==="
python3 src/reach_analyze.py --dataset truthfulqa || exit 1

fail=0

# THE ARM THAT MATTERS. Same 64 prompts as Q1, byte for byte: load_prompt_set reads the
# holdout's `statement` column, which is exactly the string tqa_baseline.py generated
# from (verified on the returned CSVs). --max-new-tokens 48 is load-bearing and must
# match Q1 or this is not its control.
#
# BUILT-IN ORACLE: the scale-0 rows of this arm are an unsteered greedy pass at the same
# budget and repetition penalty, so their `answer` column must reproduce
# tqa_baseline_completions.csv byte for byte. If it does not, the harness is wrong and
# no steered row from this job means anything. Check that FIRST when reading the output.
echo "=== P3 mean arm: the 64 TruthfulQA holdout questions, 48 new tokens ==="
python3 src/reach_steer.py --dataset truthfulqa --device cuda --arm mean \
  --prompts truthfulqa_holdout --max-new-tokens 48 \
  || { echo "!!!! mean arm FAILED, continuing"; fail=1; }

# Exploratory, and deliberately second. This is the per-statement arm on the 1488
# contrastive rows, which is the design D2 showed was fitting and steering two different
# populations. It is here because the GPU window is free, not because it is the result.
echo "=== P3 per-statement arm (EXPLORATORY, see the note above) ==="
python3 src/reach_steer.py --dataset truthfulqa --device cuda --arm per_stmt \
  --prompt-mode stem --max-new-tokens 48 \
  || { echo "!!!! stmt arm FAILED, continuing"; fail=1; }

echo "=== what this job did NOT do ==="
echo "No judging. Write the TruthfulQA judge on the laptop and run it against"
echo "reach_steer_truthfulqa.csv, column 'answer' (NOT 'completion')."

if [ "$fail" -ne 0 ]; then
  echo "=== tqa_reach: at least one P3 arm FAILED, exiting non-zero ==="
  exit 1
fi
echo "=== tqa_reach: all stages OK ==="
