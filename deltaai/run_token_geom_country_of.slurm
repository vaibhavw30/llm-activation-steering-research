#!/bin/bash
#SBATCH --job-name=token_geom_cof
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=token_geom_cof_%j.out
# E0 re-run on "The city of X is in the country of". The Sep 4 run stemmed "... is in",
# where gemma's next token is " the" (174/200) and the "false country" target was
# " North"/" South" (170/200); 22/200 stems were also cut mid-country. So its alpha numbers
# measured a word-choice flip, not a lie. This run uses the xfer_cities prompt and clean
# country pool. The same 200-row sample (SEED 42), so md_full is the same fit.
# Outputs are tagged _country_of, so token_acts_cities.npz (read by xfer_cities) and the
# Sep 4 token_geom_cities*.csv are untouched.
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for f in got_datasets/cities.csv reach_dirs_cities.npz src/xfer_cities.py; do
  test -f "$f" || { echo "!!!! missing $f"; exit 1; }
done

TEMPLATE="The city of {city} is in the country of"
echo "=== smoke (16 rows) $(date) ==="
PYTHONPATH=src python3 src/token_geom.py --dataset cities --device cuda --n 16 \
  --target countries_clean --template "$TEMPLATE" \
  --acts-tag smoke_country_of --tag smoke_country_of
rm -f token_acts_cities_smoke_country_of.npz token_geom_cities_smoke_country_of.*

echo "=== token_geom cities, country-of, clean country targets $(date) ==="
PYTHONPATH=src python3 src/token_geom.py --dataset cities --device cuda --n 200 \
  --target countries_clean --template "$TEMPLATE" \
  --acts-tag country_of --tag country_of

echo "=== same stems, runner-up lower bound $(date) ==="
PYTHONPATH=src python3 src/token_geom.py --dataset cities --device cuda \
  --stage geom --target runnerup --acts-tag country_of --tag country_of_runnerup \
  || echo "!!!! runnerup FAILED — the main result above stands"
echo "=== done $(date) ==="
