#!/bin/bash
# run_mag_judge.slurm — score the MAG E4 completions with the OLMo-3 judge (GH200).
# The --mag flag routes I/O to the MAG artifacts so nothing collides with the supervised judge run:
#   mag_steer_<ds>.csv  ->  judge_mag_steer_<ds>.csv  +  plot_judge_mag_steering_<ds>.png
# Verdict reading: FALSE at tau<0 = a real lie-lever (causal); INCOHERENT = mere degradation.
#
# DEPENDS ON: mag_steer_<ds>.csv (from run_mag_steer.slurm). One-time env: bash deltaai/setup_judge_env.sh
# EDIT --account, then: sbatch deltaai/run_mag_judge.slurm
#SBATCH --job-name=mag-judge
#SBATCH --account=ACCOUNT_NAME          # <-- EDIT (e.g. bhhv-dtai-gh)
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH --output=mag_judge_%j.out

set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate     # the judge env (transformers 5.x), NOT .venv-dct-gpu
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# The OLMo judge was already gold-gated on the supervised run (validate_judge.py). It is the same
# judge and the same TRUE/FALSE/INCOHERENT rubric here, so we re-score directly. A short smoke first
# so a broken path fails in seconds, not after the full sweep.
srun python3 src/judge_results.py --mode steer --backend olmo --mag \
    --dataset cities --device cuda --limit 24

for ds in cities common_claim_true_false; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') — judging MAG E4 $ds ====="
    srun python3 src/judge_results.py --mode steer --backend olmo --mag \
        --dataset "$ds" --device cuda
done
echo "=== mag judge done -> judge_mag_steer_<ds>.csv, plot_judge_mag_steering_<ds>.png ==="
echo "rsync those + mag_verdict_flips_<ds>.csv back to the laptop for the E4 head-to-head figures."
