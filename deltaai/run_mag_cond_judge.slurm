#!/bin/bash
#SBATCH --job-name=mag_cond_judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=mag_cond_judge_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== A0 backfill: judge existing MAG E4 free-form $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "mag_steer_${ds}.csv" \
    --steer-output "judge_mag_steer_${ds}.csv" \
    --steer-plot "plot_judge_mag_steer_${ds}.png" \
    || echo "!!!! A0 $ds FAILED — continuing"
  echo "=== A2: judge conditional steering $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "mag_conditional_${ds}.csv" \
    --steer-output "judge_mag_conditional_${ds}.csv" \
    --steer-plot "plot_judge_mag_conditional_${ds}.png" \
    || echo "!!!! A2 $ds FAILED — continuing"
done
echo "=== mag cond judge done $(date) ==="
