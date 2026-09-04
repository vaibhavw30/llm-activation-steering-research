#!/bin/bash
#SBATCH --job-name=tqa_judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=tqa_judge_%j.out

# Q2, the judging half. Runs the two allenai judges over what run_truthfulqa_reach.slurm
# generated. Submit it CHAINED so it starts the moment the reach job succeeds:
#
#   sbatch --dependency=afterok:<reach_jobid> deltaai/run_truthfulqa_judge.slurm
#
# afterok, not afterany: if the reach job dies there is nothing to judge and this must
# not burn two 7B model loads discovering that.

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

for m in models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done
test -f reach_steer_truthfulqa.csv \
  || { echo "!!!! reach_steer_truthfulqa.csv missing. Did the reach job finish?"; exit 1; }

# The same sentencepiece/protobuf guard that cost job 3081838 a whole cycle: the allenai
# repos ship tokenizer.model and no tokenizer.json, so transformers must convert slow to
# fast, and without these it raises a tiktoken error that names the wrong package.
python3 - <<'DEPCHECK' || exit 1
import importlib.util as u
import sys

missing = [name for name, spec in (("sentencepiece", "sentencepiece"),
                                   ("protobuf", "google.protobuf"))
           if u.find_spec(spec) is None]
if missing:
    print("!!!! missing python packages: " + " ".join(missing))
    print("!!!!     pip install --no-deps " + " ".join(missing))
    sys.exit(1)
print("[guard] tokenizer conversion deps present")
DEPCHECK

# THE ORACLE, and it runs before anything expensive. The mean arm's scale-0 rows are an
# unsteered greedy pass at the same prompts, budget and repetition penalty as Q1, so
# their `answer` column must reproduce tqa_baseline_completions.csv byte for byte. If it
# does not, the steering harness is wrong and no judged number from this job means
# anything. This project has never carried this assertion; math_map.tex section 4 asks
# for it by name.
echo "=== ORACLE: scale-0 rows must reproduce the Q1 baseline byte for byte ==="
python3 - <<'ORACLE' || { echo "!!!! ORACLE FAILED, not judging"; exit 1; }
import sys

import pandas as pd

s = pd.read_csv("reach_steer_truthfulqa.csv")
q1 = pd.read_csv("tqa_baseline_completions.csv")
base = s[s["scale"].astype(float) == 0.0]
if base.empty:
    sys.exit("!!!! no scale-0 rows in reach_steer_truthfulqa.csv")
# one row per prompt per direction; every direction shares the unsteered pass
got = base.drop_duplicates(subset=["prompt"]).set_index("prompt")["answer"].astype(str)
want = q1.assign(prompt=q1["prompt"].astype(str).str.replace("\\n", "\n", regex=False)) \
         .set_index("prompt")["answer"].astype(str)
shared = want.index.intersection(got.index)
print(f"[oracle] {len(shared)} of {len(want)} Q1 prompts found at scale 0")
if len(shared) < len(want):
    sys.exit(f"!!!! only {len(shared)}/{len(want)} prompts matched")
bad = [p for p in shared if got[p].strip() != want[p].strip()]
print(f"[oracle] {len(shared) - len(bad)}/{len(shared)} answers identical")
for p in bad[:3]:
    print(f"  MISMATCH prompt={p!r}\n    steer={got[p]!r}\n    q1   ={want[p]!r}")
if bad:
    sys.exit(f"!!!! {len(bad)} scale-0 answers differ from the Q1 baseline")
print("[oracle] PASS: the unsteered arm reproduces Q1 exactly")
ORACLE

# SMOKE: 8 rows through both judges. They have been run exactly once in this repo (Q1),
# never against reach_steer's schema, and a two-model load is not the place to discover
# a column name is wrong.
echo "=== SMOKE: 8 rows through both judges ==="
python3 src/truthfulqa_judge.py --dataset truthfulqa --arm mean --device cuda --limit 8 \
  || { echo "!!!! SMOKE FAILED, aborting"; exit 1; }
rm -f judge_refusal_truthfulqa_mean.csv
echo "=== smoke OK, smoke artifact cleared ==="

fail=0

echo "=== judge the mean arm (the arm paired with Q1) ==="
python3 src/truthfulqa_judge.py --dataset truthfulqa --arm mean --device cuda \
  || { echo "!!!! mean judge FAILED, continuing"; fail=1; }

if [ -f reach_steer_stmt_truthfulqa.csv ]; then
  echo "=== judge the per-statement arm (exploratory) ==="
  python3 src/truthfulqa_judge.py --dataset truthfulqa --arm stmt --device cuda \
    || { echo "!!!! stmt judge FAILED, continuing"; fail=1; }
else
  echo "=== no reach_steer_stmt_truthfulqa.csv, skipping the per-statement arm ==="
fi

echo "=== control verdict, per arm ==="
# reach_control reads judge_refusal_{ds}_{arm}.csv and keys on the `refused` column,
# which truthfulqa_judge fills with truthful_and_informative. See its docstring.
for arm in mean stmt; do
  if [ -f "judge_refusal_truthfulqa_${arm}.csv" ]; then
    python3 src/reach_control.py --dataset truthfulqa --arm $arm \
      || { echo "!!!! control $arm FAILED, continuing"; fail=1; }
  fi
done

echo "=== READ THIS FIRST IN THE MORNING ==="
echo "1. the ORACLE line above: did scale 0 reproduce Q1 byte for byte"
echo "2. the mean arm's truthful AND informative rate against Q1's 0.266"
echo "3. the control table's frac_refused column, which here means truthful+informative"

if [ "$fail" -ne 0 ]; then
  echo "=== tqa_judge: at least one stage FAILED ==="
  exit 1
fi
echo "=== tqa_judge: all stages OK ==="
