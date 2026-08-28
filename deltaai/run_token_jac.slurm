#!/bin/bash
#SBATCH --job-name=token_jac
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=token_jac_%j.out
# E2 (layer sweep) + E3 (certified per-layer budget), in one pass.
#
# One backward per batch yields ||J_l^T a|| for ALL 26 layers, under both injection
# conventions. This is the experiment the literature says is decisive: layer choice
# dominates steering success, and the layer where a property is most DECODABLE is not
# the layer where it is most CAUSAL (LEACE section 5.3). We picked layer 11 for cities
# because DCT and the probe both peaked there. Neither is a steerability criterion.
#
# --save-dirs keeps the pullback vectors so token_steer can inject the certified
# actuator at whichever layer turns out cheapest (~50MB per dataset).
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# The sweep is SKIPPED when its output already exists. Two E3 submissions (one per
# dataset, because the two have different cheapest layers) would otherwise both rewrite
# token_jac_<ds>.npz from two nodes at once, while token_sens is reading the same file.
# np.savez_compressed writes straight to the final path, so a concurrent reader lands on
# a truncated archive. Set TOKEN_JAC_FORCE=1 to recompute deliberately.
for ds in cities common_claim_true_false; do
  if [ -s "token_jac_${ds}.npz" ] && [ -z "${TOKEN_JAC_FORCE:-}" ]; then
    echo "=== token_jac $ds SKIPPED, token_jac_${ds}.npz exists $(date) ==="
    continue
  fi
  echo "=== token_jac $ds $(date) ==="
  PYTHONPATH=src python3 src/token_jac.py --dataset "$ds" --device cuda --save-dirs \
    || echo "!!!! $ds token_jac FAILED — continuing"
done

# E3 proper: steer at the cheapest layer the sweep found, along the certified
# actuator unit(J_L^T a), at the certified budget. Set LAYER from the token_jac
# stdout line "cheapest layer (broadcast injection)" before submitting this job,
# or split it into a second submission once the sweep has reported.
# Set TOKEN_STEER_DS to run only the dataset whose cheapest layer this is. The sweep
# gives a DIFFERENT best layer per dataset (16 for cities, 8 for common_claim under
# per-position injection), so running both at one layer certifies one of them at a
# budget that is not its own.
LAYER="${TOKEN_STEER_LAYER:-}"
DSETS="${TOKEN_STEER_DS:-cities common_claim_true_false}"
if [ -n "$LAYER" ]; then
  for ds in $DSETS; do
    # --positions last is the ActAdd convention and the one the cheapest-layer number
    # was read from. token_steer then uses jtw_last and eps_last to match; injecting at
    # one position while certifying with the broadcast budget crosses the two
    # conventions and sweeps a budget the run never spends.
    echo "=== token_steer $ds layer:$LAYER certified, per-position $(date) ==="
    PYTHONPATH=src python3 src/token_steer.py --dataset "$ds" --device cuda \
      --site "layer:$LAYER" --positions last --dirs jtw_token,jtw_legacy --gen \
      || echo "!!!! $ds layer:$LAYER per-position FAILED — continuing"

    # The broadcast arm, kept for continuity with every previous run. Its layer choice
    # is read off the m_all column, where layer 0 wins for a reason that is bookkeeping
    # rather than control: the summed gradient counts energy the budget does not.
    echo "=== token_steer $ds layer:$LAYER certified, broadcast $(date) ==="
    PYTHONPATH=src python3 src/token_steer.py --dataset "$ds" --device cuda \
      --site "layer:$LAYER" --dirs jtw_token,jtw_legacy --gen \
      || echo "!!!! $ds layer:$LAYER broadcast FAILED — continuing"
  done
else
  echo "TOKEN_STEER_LAYER unset — skipping the E3 steering arm. Read the cheapest"
  echo "layer off the sweep above, then resubmit with:"
  echo "  sbatch --export=ALL,TOKEN_STEER_LAYER=<L> deltaai/run_token_jac.slurm"
fi
echo "=== token_jac done $(date) ==="
