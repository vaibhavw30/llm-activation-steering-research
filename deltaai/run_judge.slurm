#!/bin/bash
# run_judge.slurm — Task 3: run the OLMo-3 LLM-as-judge to make funnel Tests 1 & 2 quantitative.
#
#   steer mode     -> judge_steer_<ds>.csv + plot_judge_steering_<ds>.png
#                     (FALSE = steering made it LIE = causal; INCOHERENT = it just degraded)
#   interpret mode -> judge_interpret_<ds>.csv  (the 0/10 "manipulates truthfulness" count)
#
# DEPENDS ON (must already be in the repo root on the cluster):
#   steer_supervised_<ds>.csv      <- produced by run_steer.slurm
#   interpret_top10_<ds>.md        <- produced by run_interpret.slurm
# So run this AFTER those two funnel jobs. One-time env: bash deltaai/setup_judge_env.sh
# Edit --account, then: sbatch deltaai/run_judge.slurm
#SBATCH --job-name=olmo-judge
#SBATCH --account=ACCOUNT_NAME          # <-- EDIT (e.g. bhhv-dtai-gh)
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=00:30:00                 # tight cap — protects the GPU-hour budget
#SBATCH --output=judge_%j.out

set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate     # the judge env (transformers 5.x), NOT .venv-dct-gpu
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1           # model was pre-downloaded on the login node
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# --- steer mode: the money plot (closes the Test-2 keyword-heuristic caveat) ---
# Smoke first (24 rows) so a broken run fails in seconds, not after the full sweep.
srun python3 src/judge_results.py --mode steer --backend olmo \
    --dataset cities --device cuda --limit 24
srun python3 src/judge_results.py --mode steer --backend olmo \
    --dataset cities --device cuda
srun python3 src/judge_results.py --mode steer --backend olmo \
    --dataset common_claim_true_false --device cuda

# --- interpret mode: makes Test 1's hand-reading a 0/10 count ---
srun python3 src/judge_results.py --mode interpret --backend olmo \
    --dataset cities --device cuda
srun python3 src/judge_results.py --mode interpret --backend olmo \
    --dataset common_claim_true_false --device cuda

echo "=== judge done -> judge_steer_<ds>.csv, plot_judge_steering_<ds>.png, judge_interpret_<ds>.csv ==="
echo "rsync those back to the laptop for the PI figures."
