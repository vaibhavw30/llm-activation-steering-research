#!/bin/bash
#SBATCH --job-name=token_sens
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=token_sens_%j.out
# E6: the directional sensitivity spectrum. The PI's claim 4, that the latent space is
# chaotic and so a scalar norm budget is the wrong currency for a certificate.
#
# The whole certificate program rests on epsilon being a meaningful budget. If the
# realized gain varies by orders of magnitude across directions, it is not, and the
# right object is a gain-normalized budget. That would be a real result rather than a
# setback, but we cannot claim either way without measuring it, and nobody has.
#
# Three sites, deliberately:
#   postnorm  A = I, so gain must be exactly 1 and slope exactly a.u. This arm is an
#             ASSERTION, not an experiment. If it fails the other two are unreadable.
#   prenorm   RMSNorm only. Isolates the radial annihilation from everything deeper.
#   layer:L   the real question. Set TOKEN_SENS_LAYER, or it defaults to 13.
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# E8, free, and it fails in seconds if the actuator is not injecting where we think.
# Never spend GPU-hours downstream of a broken hook. If pytest is not in this env that
# is a missing tool, not a failed assertion, so say so and carry on rather than letting
# `set -e` kill a two-hour job over it.
if python3 -c "import pytest" 2>/dev/null; then
  python3 -m pytest tests/test_token_geom.py tests/test_token_steer.py \
    tests/test_token_sens.py -q
else
  echo "!!!! pytest not installed in this env — E8 assertions SKIPPED."
  echo "!!!! Run them on the login node before trusting anything below."
fi

LAYER="${TOKEN_SENS_LAYER:-13}"
for ds in cities common_claim_true_false; do
  for site in postnorm prenorm "layer:$LAYER"; do
    echo "=== token_sens $ds $site $(date) ==="
    PYTHONPATH=src python3 src/token_sens.py --dataset "$ds" --device cuda \
      --site "$site" --n-stmt 40 --n-random 8 \
      || echo "!!!! $ds $site token_sens FAILED — continuing"
  done
done

# The broadcast convention is a separate experiment, not a detail: injecting at every
# position spends sqrt(T) times the reported budget and perturbs the BOS attention
# sink. One extra arm makes the two comparable.
echo "=== token_sens positions=last ablation $(date) ==="
for ds in cities common_claim_true_false; do
  PYTHONPATH=src python3 src/token_sens.py --dataset "$ds" --device cuda \
    --site "layer:$LAYER" --positions last --n-stmt 40 --n-random 8 \
    || echo "!!!! $ds positions=last FAILED — continuing"
done
echo "=== token_sens done $(date) ==="
