#!/bin/bash
#SBATCH --job-name=refusal_reach
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=refusal_reach_%j.out

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# "optional artifacts absent for refusal: dct, mag" is EXPECTED here — the minimal
# control trains no DCT factors. A SlicedModel-unfaithful abort is NOT expected.

# SMOKE: load_model_and_slice + stage_vjp are only ever exercised against a real
# model, never in a test (torch-free suite). A tiny --limit pass here means a
# broken hop (bad slice fidelity, shape mismatch, missing landmark) fails in
# minutes, not 8 hours in. Mirrors the smoke invocation documented in
# reach_margins.py's own module docstring. Hard-stop (no "continuing") on
# failure — the full run is not worth attempting if this doesn't pass.
echo "=== SMOKE: tiny --limit reach_margins pass before the full budget ==="
python3 src/reach_margins.py --dataset refusal --stage all --device cuda --limit 8 \
  || { echo "!!!! SMOKE FAILED — aborting reach job, full run not attempted"; exit 1; }
# The smoke's tiny reach_acts/reach_dirs/reach_margins/reach_chunks files must not
# leak into the full run: stage_vjp's chunk resume would otherwise treat an
# 8-statement chunk_00000.npz as a completed 100-statement chunk and silently
# misalign the merged margins. (merge_chunks itself now also refuses to merge a
# short chunk, but this cleanup keeps the full run from ever hitting that error.)
rm -rf reach_acts_refusal.npz reach_dirs_refusal.npz reach_margins_refusal.npz \
       reach_chunks_refusal || exit 1
echo "=== smoke OK, smoke artifacts cleared ==="

echo "=== P1 margins (acts + dirs + vjp) ==="
python3 src/reach_margins.py --dataset refusal --stage all --device cuda || exit 1

echo "=== P1 analysis ==="
python3 src/reach_analyze.py --dataset refusal || exit 1

echo "=== P3 mean arm: held-out harmless prompts, 32 new tokens ==="
python3 src/reach_steer.py --dataset refusal --device cuda --arm mean \
  --prompts refusal_holdout --max-new-tokens 32 \
  || echo "!!!! mean arm FAILED — continuing"

echo "=== P3 per-statement arm: FULL prompts (linearization point == prompt end) ==="
python3 src/reach_steer.py --dataset refusal --device cuda --arm per_stmt \
  --prompt-mode full --max-new-tokens 32 \
  || echo "!!!! stmt arm FAILED — continuing"

echo "=== substring judging (no OLMo here: that needs the judge spot-check job) ==="
for arm in mean stmt; do
  python3 src/refusal_judge.py --dataset refusal --arm $arm \
    || echo "!!!! judge $arm FAILED — continuing"
done

echo "=== control verdict ==="
for arm in mean stmt; do
  python3 src/reach_control.py --dataset refusal --arm $arm \
    || echo "!!!! control $arm FAILED — continuing"
done
