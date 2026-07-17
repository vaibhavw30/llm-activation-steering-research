#!/bin/bash
# run_mag_steer.slurm — MAG E4 calibrated steering on NCSA DeltaAI (1x GH200, partition ghx4).
# Injects alpha(tau)*u at each candidate direction's layer over gemma-2-2b and generates, for
# BOTH E4 leads:
#   lead #1 — the divergent operators (mag_u_Prefixed/Answered/QuestionDelta/FewShot)
#   lead #3 — mag_resid_pc1 (dominant off-truth-axis shift component; causal probe)
# plus the canonical mag_u_gold / mag_u_yM and the supervised mean_diff/grad baselines.
#
# DEPENDS ON (must be in the repo root on the cluster before submitting):
#   mag_dir_<ds>.npz     <- regenerate on the laptop with the lead keys, then rsync up:
#                           PYTHONPATH=src python src/run_mag.py --dataset <ds> --probe directions
#   truth_dir_<ds>.npz   <- optional; adds the supervised baseline directions if present
#
# Outputs mag_steer_<ds>.csv + mag_verdict_flips_<ds>.csv. Score them next with run_mag_judge.slurm.
# EDIT --account below. Submit from repo root:   sbatch deltaai/run_mag_steer.slurm
#SBATCH --job-name=mag-steer
#SBATCH --account=ACCOUNT_NAME          # <-- EDIT: your DeltaAI allocation (e.g. bhhv-dtai-gh)
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:00:00                 # 9 dirs x 5 taus x (32+24) gens x 2 datasets, gpu-fast
#SBATCH --output=mag_steer_%j.out

set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate       # the gemma/gen env (same as extraction & supervised steer)
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=src
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

for ds in cities common_claim_true_false; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') — E4 steering $ds ====="
    # no srun: single-node/single-task job, and srun hits "Error configuring interconnect" on ghx4
    python3 -m mag.steer --dataset "$ds" --device cuda || echo "!!!! $ds FAILED — continuing"
done
echo "=== mag steer done -> mag_steer_<ds>.csv, mag_verdict_flips_<ds>.csv ==="
echo "next: sbatch deltaai/run_mag_judge.slurm"
