#!/bin/bash
#SBATCH --job-name=refusal_screen
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=refusal_screen_%j.out

set -x
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1

# Screens BOTH candidates so one job produces the whole decision input.
# The -it checkpoint needs its HuggingFace license accepted first.
echo "=== base model, no chat template ==="
python src/refusal_screen.py --model google/gemma-2-2b --device cuda \
  || echo "!!!! base screen FAILED"

echo "=== instruct model, chat template ==="
python src/refusal_screen.py --model google/gemma-2-2b-it --device cuda --chat-template \
  || echo "!!!! it screen FAILED"

echo "=== DECISION: keep google/gemma-2-2b iff its harmful-prompt rate >= 0.10 ==="
grep -a "DECISION INPUT" "refusal_screen_${SLURM_JOB_ID}.out" || true
