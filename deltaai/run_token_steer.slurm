#!/bin/bash
#SBATCH --job-name=token_steer
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=token_steer_%j.out
# E1 (naive last-layer steer) + E4 (decoding audit) + E7 (injection ablation).
#
# READ THE ORACLE LINE FIRST. Every run prints
#     [assert] oracle @ +1.00 hit rate ...
# and at the post-norm site that flip is a closed-form theorem, so a hit rate below
# ~0.9 means the injection code is wrong and NOTHING else in the output is meaningful.
# Fix the harness before reading any other row.
#
# Requires token_geom_<ds>.npz (run_token_geom.slurm) first.
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

for ds in cities common_claim_true_false; do
  # --- E1a: post-norm site. The map to logits is exactly linear here, so a failure
  #          indicts the DIRECTION and nothing else. This is the decisive arm.
  echo "=== token_steer $ds postnorm $(date) ==="
  PYTHONPATH=src python3 src/token_steer.py --dataset "$ds" --device cuda \
    --site postnorm --dirs oracle,md_full,jtw_legacy --gen \
    || echo "!!!! $ds postnorm FAILED — continuing"

  # --- E1b: pre-norm site. Same cone, but RMSNorm annihilates the radial component.
  #          Comparing to E1a isolates how much of the loss is the norm layer.
  echo "=== token_steer $ds prenorm $(date) ==="
  PYTHONPATH=src python3 src/token_steer.py --dataset "$ds" --device cuda \
    --site prenorm --dirs oracle,md_full,jtw_legacy --gen \
    || echo "!!!! $ds prenorm FAILED — continuing"

  # --- E7: injection ablation. ActAdd adds at ONE aligned position; every run we
  #         have done broadcast to all positions including BOS, which injects
  #         eps*sqrt(T) of energy and perturbs the attention sink.
  echo "=== token_steer $ds postnorm last-position only $(date) ==="
  PYTHONPATH=src python3 src/token_steer.py --dataset "$ds" --device cuda \
    --site postnorm --positions last --dirs oracle,md_full --gen \
    || echo "!!!! $ds positions=last FAILED — continuing"

  # --- E4: decoding audit. repetition_penalty=1.3 was active in the existing runs,
  #         which makes the decoder NOT an argmax and voids the cone geometry.
  echo "=== token_steer $ds decoding audit rp=1.3 $(date) ==="
  PYTHONPATH=src python3 src/token_steer.py --dataset "$ds" --device cuda \
    --site postnorm --dirs oracle,md_full --rep-penalty 1.3 --gen \
    || echo "!!!! $ds rp=1.3 FAILED — continuing"
done
echo "=== token_steer done $(date) ==="
