#!/bin/bash
#SBATCH --job-name=truthfulqa_prep
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=truthfulqa_prep_%j.out

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1
# Compute nodes are OFFLINE (deltaai/CLUSTER_OPERATIONS.md Part 1). Without these two
# the job looks in ~/.cache/huggingface, finds nothing, and dies at from_pretrained.
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# The truth track ran on the base model (dct_meta_cities.json), so that is the default.
# Set TQA_MODEL=google/gemma-2-2b-it only if got_datasets/truthfulqa.csv was rebuilt on
# the laptop with --chat-template; the CSV and the model must match or the rows are
# tokenized in a format the model has never seen.
MODEL="${TQA_MODEL:-google/gemma-2-2b}"
echo "using MODEL=$MODEL"

# got_datasets/truthfulqa.csv is built on the LAPTOP (prep_truthfulqa.py needs the
# network to fetch truthfulqa/truthful_qa) and travels up with the code rsync.
test -f got_datasets/truthfulqa.csv || { echo "MISSING got_datasets/truthfulqa.csv"; exit 1; }

# --max-length 96: these rows are prompt PLUS answer and 4 of 1488 run past extract.py's
# default 64 tokens (max 68). A truncated row has its activation read mid-answer instead
# of at its last token, which is the fit point the whole contrast depends on. The job
# prints a WARNING line if any row still hits the cap; grep for it.
echo "=== Q0 activations, every layer ==="
python3 src/extract.py truthfulqa.csv --model "$MODEL" --max-length 96 || exit 1

# make_reach_meta accepts acts_truthfulqa.npz in ./ or ./activations/, so the layer
# sweep runs before the move. Only the two export_*_dir.py calls need it already moved
# (funnel_utils.load_acts looks in activations/ only).
echo "=== layer sweep + meta ==="
python3 src/make_reach_meta.py --dataset truthfulqa --model "$MODEL" || exit 1
mkdir -p activations && mv acts_truthfulqa.npz activations/ || exit 1

echo "=== calibration (SteeringCalibrator only, no DCT fit) ==="
python3 src/calibrate_scale.py --dataset truthfulqa --device cuda || exit 1

echo "=== direction exports (source and target layer) ==="
python3 src/export_truth_dir.py --dataset truthfulqa || exit 1
python3 src/export_target_dir.py --dataset truthfulqa || exit 1
cat dct_meta_truthfulqa.json
