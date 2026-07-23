#!/bin/bash
#SBATCH --job-name=dct_uwarm
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=dct_uwarm_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  for lam in 0.3 1.0 3.0; do
    echo "=== uwarm train $ds mean_diff lam=$lam $(date) ==="
    PYTHONPATH=src python3 src/dct_warm.py --dataset "$ds" --seed-name mean_diff \
      --anchor-space u --anchor-lambda "$lam" --device cuda \
      || echo "!!!! $ds uwarm $lam FAILED — continuing"
  done
  echo "=== assemble uwarm directions $ds ==="
  PYTHONPATH=src python3 src/dct_uwarm_directions.py --dataset "$ds"
  echo "=== uwarm steer $ds ==="
  PYTHONPATH=src python3 src/dct_warm_steer.py --dataset "$ds" --device cuda \
    --dirs-file "dct_uwarm_dirs_${ds}.npz" --out "dct_uwarm_steer_${ds}.csv" \
    || echo "!!!! uwarm steer $ds FAILED — continuing"
done
echo "=== dct uwarm done $(date) ==="
