#!/bin/bash
# run_interpret.slurm — Funnel Step 1: interpret the top-10 DCT vectors (generation on GH200).
# Edit --account, then: sbatch deltaai/run_interpret.slurm
#SBATCH --job-name=dct-interpret
#SBATCH --account=ACCOUNT_NAME          # <-- EDIT (e.g. bhhv-dtai-gh)
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=00:30:00                 # tight cap — protects the GPU-hour budget
#SBATCH --output=interpret_%j.out

set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

srun python3 src/interpret_top10.py --dataset cities --device cuda --top-k 10
srun python3 src/interpret_top10.py --dataset common_claim_true_false --device cuda --top-k 10
echo "=== interpret done -> interpret_top10_<dataset>.md ==="
