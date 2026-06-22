#!/usr/bin/env bash
# auto_analyze.sh — runs analyze.py (PROMPT 3 + plots) on each dataset as soon as
# extract.py reports its .npz saved. Polls extract.log; analyzes each dataset once;
# exits after "ALL EXTRACTION COMPLETE".
#
# bash 3.2 compatible (no associative arrays): dedup via existence of results_<name>.csv.
#
# Launch with:  nohup bash auto_analyze.sh > analyze.log 2>&1 &

cd "$(dirname "$0")" || exit 1

while true; do
  # Datasets whose .npz extract.py has finished writing (logged "Saved acts_X.npz")
  for name in $(grep -oE "Saved acts_[^ ]+\.npz" extract.log 2>/dev/null \
                  | sed -E 's/^Saved acts_//; s/\.npz$//' | sort -u); do
    # Skip if no npz yet, or already analyzed (results CSV exists = our done-marker)
    if [ -f "acts_${name}.npz" ] && [ ! -f "results_${name}.csv" ]; then
      echo "================================================================"
      echo "[$(date '+%H:%M:%S')] ANALYZING: $name"
      echo "================================================================"
      .venv/bin/python analyze.py "$name"
      echo "[$(date '+%H:%M:%S')] DONE ANALYZING: $name (exit $?)"
      echo
    fi
  done

  if grep -q "ALL EXTRACTION COMPLETE" extract.log 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] Extraction complete and all datasets analyzed. Exiting."
    break
  fi
  sleep 15
done
