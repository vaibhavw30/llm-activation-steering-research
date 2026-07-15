#!/bin/bash
# submit_rerun.sh — one-shot corrected rerun: steer sweep, then the judge auto-starts on success.
#
# Chains  run_steer.slurm  --dependency=afterok-->  run_judge.slurm  so you submit ONCE and walk away.
#   - run_steer regenerates steer_supervised_<ds>.csv with the corrected design (32 factual prompts,
#     --max-new-tokens 8).
#   - run_judge then re-judges those (answer-only STEER_SYS) AND re-judges the existing
#     interpret_top10_<ds>.md with the tightened INTERPRET_SYS. It also runs the validation gate first.
#
# Run on a DeltaAI login node from the repo root:   bash deltaai/submit_rerun.sh
#
# Prereqs (persist on the cluster from earlier runs):
#   - account set in BOTH scripts (this script checks and tells you the sed if not),
#   - .venv-dct-gpu + .venv-judge-gpu built, gemma + OLMo cached in $HF_HOME,
#   - truth_dir_<ds>.npz present (run_steer needs them), interpret_top10_<ds>.md present (judge needs them).
set -e
cd "$(dirname "$0")/.."

# --- guard: account placeholder must be filled in both job scripts ---
for f in deltaai/run_steer.slurm deltaai/run_judge.slurm; do
  if grep -q 'ACCOUNT_NAME' "$f"; then
    echo "ERROR: $f still has the ACCOUNT_NAME placeholder. Fix both, then re-run:"
    echo "  sed -i 's/ACCOUNT_NAME/bhhv-dtai-gh/' deltaai/run_steer.slurm deltaai/run_judge.slurm"
    exit 1
  fi
done

# --- guard: inputs the two jobs read must exist ---
for f in truth_dir_cities.npz truth_dir_common_claim_true_false.npz \
         interpret_top10_cities.md interpret_top10_common_claim_true_false.md; do
  [ -e "$f" ] || { echo "ERROR: missing $f (a job input). See deltaai/CLUSTER_WALKTHROUGH.md §3."; exit 1; }
done

# --- submit steer, capture its job id, then submit judge gated on steer succeeding ---
STEER_ID=$(sbatch --parsable deltaai/run_steer.slurm)
echo "submitted STEER job:  $STEER_ID"
JUDGE_ID=$(sbatch --parsable --dependency=afterok:"$STEER_ID" deltaai/run_judge.slurm)
echo "submitted JUDGE job:  $JUDGE_ID   (afterok:$STEER_ID — runs only if steer succeeds)"
echo
echo "Watch both:   squeue --me"
echo "  the judge shows REASON '(Dependency)' until steer finishes, then PD -> R."
echo "Logs (read with 'grep -a', never 'tail -f'):"
echo "  steer:  grep -a -E 'Saved|Error|Traceback' steer_${STEER_ID}.out"
echo "  judge:  grep -a -E 'PASS|FAIL|accuracy|saved plot_judge|wrote judge|Error' judge_${JUDGE_ID}.out"
echo
echo "If steer FAILS, SLURM auto-cancels the judge (afterok). Diagnose with: sacct -j $STEER_ID"
echo "When done, pull back (from the laptop):"
echo "  rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/judge_steer_*.csv' \\"
echo "            'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/plot_judge_steering_*.png' \\"
echo "            'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/judge_interpret_*.csv' \\"
echo "            /Users/vaibhav.wudaru/llm-activation-steering-research/"
