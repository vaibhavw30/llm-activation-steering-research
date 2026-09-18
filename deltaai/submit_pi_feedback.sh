#!/bin/bash
# submit_pi_feedback.sh <round|tokgeom>: submit one round of docs/PLAN_PI_FEEDBACK_2026-09-18.md.
# CLUSTER, from the repo root, after the rsync and the ACCOUNT_NAME sed.
#
#   bash deltaai/submit_pi_feedback.sh round1
#
# ghx4 allows 2 jobs per user, queued plus running, and a job waiting on afterok counts.
# The script refuses to submit a round that would take you past 2, which saves a rejected
# sbatch, and runs sbatch --test-only on each file first.
set -euo pipefail
LIMIT=2

have=$(squeue --me --noheader --partition=ghx4 | wc -l)
case "${1:-}" in
  round1) files=(deltaai/run_tqa_discovery.slurm)
          [ -f deltaai/run_tqa_confirm.slurm ] && files+=(deltaai/run_tqa_confirm.slurm) ;;
  round2) files=(deltaai/run_xfer_cities.slurm deltaai/run_xfer_tqa.slurm) ;;
  # One short job; submit it into whichever slot frees first.
  tokgeom) files=(deltaai/run_token_geom_country_of.slurm) ;;
  *) echo "usage: $0 round1|round2|tokgeom"; exit 2 ;;
esac

if [ $((have + ${#files[@]})) -gt $LIMIT ]; then
  echo "!!!! you have $have job(s) in ghx4 and this round adds ${#files[@]}; the limit is $LIMIT."
  squeue --me
  exit 1
fi
for f in "${files[@]}"; do
  test -f "$f" || { echo "!!!! $f not here: rsync deltaai/ up."; exit 1; }
  if grep -q ACCOUNT_NAME "$f"; then
    echo "!!!! $f still says ACCOUNT_NAME: run the sed first."; exit 1
  fi
  sbatch --test-only "$f"
done

# The second job of a round waits on the first: J-C reads J-B's selection file.
first=$(sbatch --parsable "${files[0]}")
echo "submitted ${files[0]} as $first"
if [ ${#files[@]} -gt 1 ]; then
  if [ "$1" = round1 ]; then
    second=$(sbatch --parsable --dependency=afterok:$first "${files[1]}")
  else
    second=$(sbatch --parsable "${files[1]}")      # round 2's two jobs are independent
  fi
  echo "submitted ${files[1]} as $second"
fi
squeue --me
