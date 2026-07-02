#!/usr/bin/env bash
# run_all_extract.sh — extract activations for all datasets, smallest first.
# Launch with:  nohup bash run_all_extract.sh > extract.log 2>&1 &
# Resilient: if one dataset fails, continues to the next.

set +e  # don't abort on a single failure

DATASETS=(
  "sp_en_trans.csv"            # 354  — fastest, results first
  "cities.csv"                 # 1496 — clean baseline
  "companies_true_false.csv"   # 1199 — messier
  "common_claim_true_false.csv" # 4450 — messiest, run last
)

for ds in "${DATASETS[@]}"; do
  echo "=================================================="
  echo "[$(date '+%H:%M:%S')] Extracting: $ds"
  echo "=================================================="
  HF_HUB_DISABLE_XET=1 .venv/bin/python extract.py "$ds"
  echo "[$(date '+%H:%M:%S')] Done: $ds (exit $?)"
  echo
done

echo "[$(date '+%H:%M:%S')] ALL EXTRACTION COMPLETE"