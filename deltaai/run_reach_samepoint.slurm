#!/bin/bash
#SBATCH --job-name=reach_samepoint
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=reach_samepoint_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== reach samepoint $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_samepoint.py --dataset "$ds" --device cuda \
    || echo "!!!! $ds reach samepoint FAILED — continuing"
  echo "=== reach samepoint summary $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_samepoint.py --dataset "$ds" --summarize \
    || echo "!!!! $ds reach samepoint summary FAILED — continuing"
done
echo "=== reach samepoint done $(date) ==="
