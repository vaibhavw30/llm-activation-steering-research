#!/bin/bash
#SBATCH --job-name=token_geom
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=token_geom_%j.out
# E0 — the denominator. Extracts z (post-final-norm) and h (pre-norm) at the STEM's
# last token, then solves the argmax-cone budget at both the post-norm and pre-norm
# sites. Pure linear algebra after the forward pass; the whole job is minutes.
#
# cities gets --target countries (a semantically FALSE target exists). common_claim
# has no such structure, so it gets --target runnerup: the cheapest token flip of any
# kind, which is a hard LOWER bound on the cost of any semantic flip.
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

echo "=== token_geom cities (country targets) $(date) ==="
PYTHONPATH=src python3 src/token_geom.py --dataset cities --device cuda \
  --n 200 --target countries \
  || echo "!!!! cities token_geom FAILED — continuing"

echo "=== token_geom common_claim (runner-up targets) $(date) ==="
PYTHONPATH=src python3 src/token_geom.py --dataset common_claim_true_false \
  --device cuda --n 200 --target runnerup \
  || echo "!!!! common_claim token_geom FAILED — continuing"

# The runner-up variant for cities is a useful lower bound but writes the SAME
# filenames, so it gets its own tag rather than clobbering the country targets:
echo "=== token_geom cities (runner-up lower bound) $(date) ==="
PYTHONPATH=src python3 src/token_geom.py --dataset cities --device cuda \
  --stage geom --target runnerup --tag runnerup \
  || echo "!!!! cities runnerup token_geom FAILED — continuing"

echo "=== token_geom done $(date) ==="
