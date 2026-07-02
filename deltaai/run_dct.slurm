#!/bin/bash
# run_dct.slurm — DCT job for NCSA DeltaAI (1x GH200, partition ghx4).
# Submit from the repo root:   sbatch deltaai/run_dct.slurm
#
# EDIT --account below before submitting (find yours with: accounts  OR  sacctmgr show assoc user=$USER).

#SBATCH --job-name=dct
#SBATCH --account=ACCOUNT_NAME          # <-- EDIT: your DeltaAI allocation
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=dct_%j.out

set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate

# Read the model from a pre-downloaded cache; compute nodes typically have no internet.
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1          # fail fast if the cache is missing, instead of hanging

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# ---- Stage 1: fast GPU verification (should finish in a few minutes on GH200) ----
srun python3 src/run_dct_minimal.py --device cuda --num-factors 64 --max-iters 10

# ---- Stage 3: real run on true/false data (uncomment & tune) --------------------
# srun python3 src/run_dct_data.py --dataset cities --device cuda \
#      --num-factors 512 --num-iters 30 --num-samples 64 --balanced
# srun python3 src/run_dct_data.py --dataset common_claim_true_false --device cuda \
#      --num-factors 512 --num-iters 30 --num-samples 64 --balanced

echo "=== job complete ==="
