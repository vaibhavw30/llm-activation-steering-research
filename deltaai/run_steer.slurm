#!/bin/bash
# run_steer.slurm — Funnel Step 2: does the supervised truth direction steer? (GH200)
# Needs truth_dir_<ds>.npz on the cluster first (make with export_truth_dir.py on the laptop,
# then rsync over). Edit --account, then: sbatch deltaai/run_steer.slurm
#SBATCH --job-name=dct-steer-sup
#SBATCH --account=ACCOUNT_NAME          # <-- EDIT (e.g. bhhv-dtai-gh)
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=00:30:00                 # tight cap — protects the GPU-hour budget
#SBATCH --output=steer_%j.out

set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# Symmetric ± sweep: the "-" side tests whether steering the truth direction the OTHER way
# pushes the model toward FALSEHOOD (the missing half of the causal test), the "+" side repeats
# the earlier more-factual effect, and 0 is the unsteered baseline.
SCALES="-120,-80,-40,-20,0,20,40,80,120"
# NOTE: use --scales=... (attached with '='): the value starts with '-', which argparse would
# otherwise mistake for another flag ("expected one argument").
# Corrected design (see docs/INVESTIGATION_steering_validity.md): 32 factual prompts (was 8) for
# statistical power, and --max-new-tokens 8 so the judge scores the DIRECT answer, not a rambling
# paragraph (a correct "four..." was getting marked FALSE on its trailing text).
srun python3 src/steer_supervised.py --dataset cities --device cuda --scales="$SCALES" --max-new-tokens 8
srun python3 src/steer_supervised.py --dataset common_claim_true_false --device cuda --scales="$SCALES" --max-new-tokens 8
echo "=== steer done -> steer_supervised_<dataset>.md/.csv ==="
