#!/bin/bash
# Full MAG extraction on all 4 datasets, smallest first. Resilient: one failure doesn't stop the rest.
cd "$(dirname "$0")"
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1

for ds in sp_en_trans.csv cities.csv companies_true_false.csv common_claim_true_false.csv; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') — extracting $ds ====="
    .venv/bin/python -m mag.extract "$ds" --device mps || echo "!!!! $ds FAILED — continuing"
done
echo "===== $(date '+%Y-%m-%d %H:%M:%S') — all done ====="
