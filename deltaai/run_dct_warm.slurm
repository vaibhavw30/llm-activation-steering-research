#!/bin/bash
#SBATCH --job-name=dct_warm
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=dct_warm_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  for seed in mean_diff grad; do
    for lam in 0.0 0.3 1.0 3.0; do
      echo "=== warm train $ds $seed lam=$lam $(date) ==="
      PYTHONPATH=src python3 src/dct_warm.py --dataset "$ds" --seed-name "$seed" \
        --anchor-lambda "$lam" --device cuda || echo "!!!! $ds $seed $lam FAILED — continuing"
    done
  done
  echo "=== assemble directions $ds ==="
  PYTHONPATH=src python3 src/dct_warm_directions.py --dataset "$ds"
  echo "=== warm steer $ds ==="
  PYTHONPATH=src python3 src/dct_warm_steer.py --dataset "$ds" --device cuda \
    || echo "!!!! steer $ds FAILED — continuing"
done
echo "=== dct warm done $(date) ==="
