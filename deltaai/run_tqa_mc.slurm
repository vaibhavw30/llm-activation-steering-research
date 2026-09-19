#!/bin/bash
#SBATCH --job-name=tqa_mc
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=tqa_mc_%j.out

# J-E (round 3): judge-free TruthfulQA and the learned single-vector ceiling.
# docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md; src/tqa_mc.py and
# src/tqa_learned.py.
#   train        the best ||v||-fixed layer-11 vector per norm frac, 3 starts (~15 min)
#   mc           every direction x dose on the 64 holdout questions' reference answers,
#                scored by log-prob, 32 randoms; plus the 744 training pairs at the read dose
#   cities_long  "Q: Where is the city of X?" 48 tokens, norm +/-0.25, 8 randoms
#   gen, judge   the learned vectors generated and judged by the v2 allenai judges
# Full direction list needs J-B (TQA DCT picks, G0 MAG); without it they are left out
# loudly. Writes only mc_*. Every stage resumes after a timeout.
#
# READ THIS FIRST WHEN IT FINISHES:
#   [mc] padding check              must be <= 1e-3 or the stage stopped
#   [train] frac 0.25 seed ...       learned val objective vs unsteered
#   then LAPTOP: python src/tqa_mc.py --stage summary, the [reading] lines

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

echo "=== stage 0: preflight ==="
for ds in cities truthfulqa; do
  for f in reach_summary_${ds}.json reach_margins_${ds}.npz reach_acts_${ds}.npz \
           reach_dirs_${ds}.npz truth_dir_${ds}.npz dct_meta_${ds}.json \
           dct_V_${ds}.pt dct_U_${ds}.pt; do
    test -f "$f" || { echo "!!!! $f missing."; exit 1; }
  done
done
for f in got_datasets/truthfulqa_holdout.csv got_datasets/truthfulqa.csv \
         got_datasets/cities.csv token_acts_cities.npz mag_dir_cities.npz; do
  test -f "$f" || { echo "!!!! $f missing."; exit 1; }
done
test -f dct_selection_truthfulqa.json \
  || echo "[preflight] !!!! no J-B selection: the TQA DCT rows will be absent"
test -f jb_truthfulqa/mag_dir_truthfulqa.npz \
  || echo "[preflight] no TQA MAG directions from J-B; those rows will be absent"
for m in models--google--gemma-2-2b models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done
python3 -c "
import json, sys
m = json.load(open('dct_meta_truthfulqa.json'))
sys.exit(0 if m.get('num_factors') is None and m['source_layer'] == 11 else
         '!!!! dct_meta_truthfulqa.json is not the Q2 meta')"

echo "=== stage 1: smoke, 4 questions, 1 training step, smoke_ outputs ==="
# a smoke_mc_* left by a prior attempt that died mid-smoke (different --steps/limit, or
# an old optimizer) would trip the resume guard in stage train under set -e; clear first.
rm -f smoke_mc_*
python3 src/tqa_mc.py --stage all --device cuda --limit 4 --steps 1 --prefix smoke_
rm -f smoke_mc_*

echo "=== stage 2: the full run ==="
python3 src/tqa_mc.py --stage all --device cuda

echo "=== job complete ==="
ls -la mc_*
