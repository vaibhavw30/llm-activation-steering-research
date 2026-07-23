#!/bin/bash
#SBATCH --job-name=dct_uwarm_judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=dct_uwarm_judge_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== uwarm judge $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "dct_uwarm_steer_${ds}.csv" \
    --steer-output "judge_dct_uwarm_steer_${ds}.csv" \
    --steer-plot "plot_judge_dct_uwarm_${ds}.png" \
    || echo "!!!! $ds FAILED — continuing"
done
echo "=== dct uwarm judge done $(date) ==="
