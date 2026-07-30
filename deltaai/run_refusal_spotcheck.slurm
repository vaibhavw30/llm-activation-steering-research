#!/bin/bash
#SBATCH --job-name=refusal_spotcheck
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=refusal_spotcheck_%j.out

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# "continuing" past a failed arm must not mean the job's own exit status still
# says success — fail is set inside an explicit { ...; fail=1; } block (not
# relied on via the echo's exit status) so set -e does not abort here.
fail=0
# Re-runs the substring pass (identical output) and adds the OLMo agreement check.
for arm in mean stmt; do
  python3 src/refusal_judge.py --dataset refusal --arm $arm \
    --spot-check 60 --device cuda \
    || { echo "!!!! spot-check $arm FAILED — continuing"; fail=1; }
done

if [ "$fail" -ne 0 ]; then
  echo "=== refusal_spotcheck: at least one arm FAILED (see FAILED lines above) — exiting non-zero ==="
  exit 1
fi
echo "=== refusal_spotcheck: all stages OK ==="
