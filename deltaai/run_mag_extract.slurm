#!/bin/bash
# run_mag_extract.slurm — full MAG activation extraction on NCSA DeltaAI (1x GH200, partition ghx4).
# Submit from the repo root:   sbatch deltaai/run_mag_extract.slurm
#
# EDIT --account below before submitting (find yours with: accounts  OR  sacctmgr show assoc user=$USER).
#
# Runs all 4 datasets sequentially, smallest first. Each dataset = ~6 forward passes
# (y^M verdict + 5 readout sweeps) over gemma-2-2b fp32. Resilient: one dataset
# failing does not stop the rest. Outputs mag_acts_<ds>.npz in the repo root
# (~7 GB total across the four) — rsync them back to the laptop afterwards.

#SBATCH --job-name=mag_extract
#SBATCH --account=ACCOUNT_NAME          # <-- EDIT: your DeltaAI allocation
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=mag_extract_%j.out

set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate

# Read the model from a pre-downloaded cache; compute nodes typically have no internet.
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1          # fail fast if the cache is missing, instead of hanging

export PYTHONPATH=src

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

for ds in sp_en_trans.csv cities.csv companies_true_false.csv common_claim_true_false.csv; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') — extracting $ds ====="
    srun python3 -m mag.extract "$ds" --device cuda || echo "!!!! $ds FAILED — continuing"
done
echo "=== job complete ==="
