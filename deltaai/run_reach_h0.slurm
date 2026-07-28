#!/bin/bash
#SBATCH --job-name=reach_h0
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=reach_h0_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== h0 stemprobe extract $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_stemprobe.py --dataset "$ds" --extract --device cuda \
    || echo "!!!! $ds stemprobe extract FAILED — continuing"
  echo "=== h0 stemjac compute $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_stemjac.py --dataset "$ds" --compute --device cuda \
    || echo "!!!! $ds stemjac compute FAILED — continuing"
  echo "=== h0 newton $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_newton.py --dataset "$ds" --device cuda \
    || echo "!!!! $ds newton FAILED — continuing"
done
echo "=== h0 done $(date) ==="
