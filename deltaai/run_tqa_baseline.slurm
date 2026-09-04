#!/bin/bash
#SBATCH --job-name=tqa_baseline
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=tqa_baseline_%j.out

# Q1: gemma-2-2b's UNSTEERED truthful and informative rate on the 64 held-out
# TruthfulQA questions. This is the gate the plan puts in front of the whole Q track:
# if the base rate is near ceiling as it is on cities, there is no headroom and Q2
# would be measuring noise against a ceiling. See deltaai/TRUTHFULQA_RUN.md Phase 5.
#
# Independent of run_truthfulqa_prep.slurm. It reads only got_datasets/
# truthfulqa_holdout.csv, so the two can sit in the queue at the same time.

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1
# Compute nodes are OFFLINE (deltaai/CLUSTER_OPERATIONS.md:65). Every model this job
# touches must already be in $HF_HOME, staged from the LOGIN node.
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

MODEL="${TQA_MODEL:-google/gemma-2-2b}"

# TQA_TRUTH_ONLY=1 loads one 7B judge instead of two. Disk contingency only: it leaves
# Q1 half-answered, because a model that answers "I have no comment" to everything is
# 100% truthful and useless, and the plan's Q2 bar is "informativeness not collapsing".
JUDGE_FLAGS=""
NEEDED="models--allenai--truthfulqa-truth-judge-llama2-7B"
if [ -n "${TQA_TRUTH_ONLY:-}" ]; then
  JUDGE_FLAGS="--truth-only"
else
  NEEDED="$NEEDED models--allenai--truthfulqa-info-judge-llama2-7B"
fi

# Fail in the first 10 seconds rather than after the generation stage. A missing judge
# on an offline compute node is not a retryable error, it is a staging step that was
# skipped on the login node.
test -f got_datasets/truthfulqa_holdout.csv \
  || { echo "!!!! got_datasets/truthfulqa_holdout.csv missing. rsync the repo up."; exit 1; }
for m in $NEEDED; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m is not in $HF_HOME/hub. Compute nodes have no internet: stage it "\
"on the LOGIN node first (TRUTHFULQA_RUN.md Phase 5)."; exit 1; }
done

# The allenai judges ship tokenizer.model (SentencePiece) and NO tokenizer.json, so
# transformers has to convert slow to fast and needs both of these. Without them it
# falls through to its tiktoken branch and raises "`tiktoken` is required to read a
# `tiktoken` file", which names the wrong package and only surfaces after the whole
# generation stage has already run. gemma never exposes this: its repo ships a
# ready-made tokenizer.json and skips the conversion entirely.
python3 - <<'DEPCHECK' || exit 1
import importlib.util as u
import sys

missing = [name for name, spec in (("sentencepiece", "sentencepiece"),
                                   ("protobuf", "google.protobuf"))
           if u.find_spec(spec) is None]
if missing:
    print("!!!! missing python packages: " + " ".join(missing))
    print("!!!! Install on the LOGIN node, into this same venv:")
    print("!!!!     pip install --no-deps " + " ".join(missing))
    sys.exit(1)
print("[guard] tokenizer conversion deps present")
DEPCHECK

# TQA_LIMIT=4 runs the whole three-stage pipeline on 4 questions. Worth doing once:
# the two allenai judges have never been run in this repo, and a 10-minute smoke that
# proves they load and answer is cheaper than discovering it after the full pass.
LIMIT_FLAG=""
if [ -n "${TQA_LIMIT:-}" ]; then
  LIMIT_FLAG="--limit $TQA_LIMIT"
  echo "=== SMOKE RUN: only $TQA_LIMIT questions. The summary is NOT the Q1 answer. ==="
fi

echo "=== Q1 stage 1: unsteered generation, 64 holdout questions ==="
# --max-new-tokens 48 is explicit and load-bearing: Q2 must steer at the same budget or
# this baseline is not its control. The decoder here mirrors dct_steer_utils.generate,
# which is what reach_steer.py calls in Q2.
python3 src/tqa_baseline.py --generate --model "$MODEL" --device cuda \
  --max-new-tokens 48 $LIMIT_FLAG

echo "=== Q1 stage 2: the allenai truth and info judges, plus the gold check ==="
# The gold check re-scores TruthfulQA's own correct and incorrect answers, whose labels
# we already know. Without it the headline rate comes from a judge nobody validated on
# this holdout. This is what made the refusal control readable (validated at 0.969).
python3 src/tqa_baseline.py --judge --device cuda $JUDGE_FLAGS $LIMIT_FLAG

echo "=== Q1 stage 3: rates, Wilson intervals, and the gate verdict ==="
python3 src/tqa_baseline.py --summarize

echo "=== Q1 summary ==="
cat tqa_baseline_summary.csv
