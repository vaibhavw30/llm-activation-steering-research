#!/bin/bash
#SBATCH --job-name=refusal_prep
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=refusal_prep_%j.out

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
# Set by the operator AFTER reading run_refusal_screen's DECISION lines.
MODEL="${REFUSAL_MODEL:-google/gemma-2-2b}"
echo "using MODEL=$MODEL"

echo "=== A3 activations, every layer ==="
python3 src/extract.py refusal.csv --model "$MODEL" || exit 1

# make_reach_meta accepts acts_refusal.npz in ./ or ./activations/, so the sweep
# runs before the move. calibrate_scale.py rebuilds its features from the raw
# statements (load_statements), not from acts_refusal.npz, so its position
# relative to the move is irrelevant; only the two export_*_dir.py calls below
# need the file already in activations/ (funnel_utils.load_acts looks there only).
echo "=== A4 layer sweep + meta ==="
python3 src/make_reach_meta.py --dataset refusal --model "$MODEL" || exit 1
mkdir -p activations && mv acts_refusal.npz activations/ || exit 1

echo "=== A4 calibration (SteeringCalibrator only, no DCT fit) ==="
python3 src/calibrate_scale.py --dataset refusal --device cuda || exit 1

echo "=== direction exports (source and target layer) ==="
python3 src/export_truth_dir.py --dataset refusal || exit 1
python3 src/export_target_dir.py --dataset refusal || exit 1
cat dct_meta_refusal.json
