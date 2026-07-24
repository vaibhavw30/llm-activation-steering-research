#!/bin/bash
#SBATCH --job-name=reach_steer
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=05:00:00
#SBATCH --output=reach_steer_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== reach steer mean $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_steer.py --dataset "$ds" --device cuda --arm mean \
    || echo "!!!! $ds reach steer mean FAILED — continuing"
  echo "=== reach steer per_stmt $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_steer.py --dataset "$ds" --device cuda --arm per_stmt \
    || echo "!!!! $ds reach steer per_stmt FAILED — continuing"
  echo "=== reach jlens $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_jlens.py --dataset "$ds" --device cuda \
    || echo "!!!! $ds reach jlens FAILED — continuing"
done
echo "=== reach steer done $(date) ==="
