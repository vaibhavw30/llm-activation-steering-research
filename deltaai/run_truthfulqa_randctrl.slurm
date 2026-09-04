#!/bin/bash
#SBATCH --job-name=tqa_rand
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=tqa_rand_%j.out

# Q2's DECIDING CONTROL. Steers norm-matched RANDOM directions and judges them.
#
# What job 3082192 + 3082210 measured on jtw_mean_diff_tgt, 64 held-out questions:
#   frac -2  truthful+informative 0.500 vs 0.266 unsteered; 16 gained, 1 lost,
#            exact McNemar p = 2.8e-4; informativeness held at 0.984.
# and, over the same sweep, mean answer length ran 2.53 -> 18.58 words monotonically.
# Conditioning on word count leaves frac contributing nothing (LR chi2 = 0.146,
# p = 0.70). Two readings survive that and they say opposite things:
#   (a) the direction carries truth, and longer hedged answers are how truth surfaces;
#   (b) ANY perturbation of this norm at this layer makes the model discursive, and
#       TruthfulQA rewards hedging, so the effect has nothing to do with truth.
# A direction with the same norm and no truth content is the only thing that
# separates them. That is what this job runs.
#
# READ THIS FIRST WHEN IT FINISHES: the randctrl arm's truthful+informative rate at
# frac -2. If it is near 0.266, reading (a) survives and Q2 has a result. If it is
# near 0.500, reading (b) wins and the Q2 effect is not about truth.
#
# It does NOT rerun the per-statement arm. That arm's prompts were malformed by the
# stem_of newline bug (fixed in this commit), but reach_steer_stmt_truthfulqa.csv is
# an existing reach_*.csv artifact and overwriting one is not this job's call.

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH=src
export HF_HUB_DISABLE_XET=1
# Compute nodes are OFFLINE. Everything this job loads must already be in $HF_HOME.
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# Fail in ten seconds, not after the steering stage.
for f in reach_dirs_truthfulqa.npz reach_margins_truthfulqa.npz \
         reach_acts_truthfulqa.npz reach_summary_truthfulqa.json \
         dct_meta_truthfulqa.json got_datasets/truthfulqa_holdout.csv; do
  test -f "$f" || { echo "!!!! $f missing. It is job 3082192 output: rsync it up."; exit 1; }
done
for m in models--google--gemma-2-2b \
         models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done

# The allenai repos ship tokenizer.model and no tokenizer.json, so transformers must
# convert slow to fast; without these it raises a tiktoken error naming the wrong
# package, and only after the generation stage has already run. Cost job 3081838 a cycle.
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

echo "=== stage 1: norm-matched random directions, same scale grid as the mean arm ==="
# --max-new-tokens 48 is load-bearing: the mean arm and Q1 both used 48, and the whole
# comparison is about answer length. A different budget here would decide the result.
python3 src/reach_steer.py --dataset truthfulqa --arm rand_ctrl --device cuda \
  --prompts truthfulqa_holdout --max-new-tokens 48 --n-rand 3

echo "=== stage 2: the scale-0 determinism check ==="
# Every random direction regenerates the unsteered block. All three must be identical
# to each other AND to the mean arm's scale-0 answers, or the steering hook is leaking
# state between directions and every number above it is suspect.
python3 - <<'ORACLE' || exit 1
import sys

import pandas as pd

r = pd.read_csv("reach_steer_randctrl_truthfulqa.csv")
z = r[r["scale"] == 0.0]
blocks = {d: g.set_index("prompt")["answer"].astype(str)
          for d, g in z.groupby("direction")}
print(f"[oracle] {len(blocks)} scale-0 blocks, {len(z) // max(len(blocks), 1)} rows each")
names = sorted(blocks)
ref = blocks[names[0]]
for n in names[1:]:
    same = int((blocks[n].reindex(ref.index) == ref).sum())
    print(f"[oracle] {n} vs {names[0]}: {same}/{len(ref)} identical")
    if same != len(ref):
        print("[oracle] FAIL: the steering hook is leaking state between directions")
        sys.exit(1)

m = pd.read_csv("reach_steer_truthfulqa.csv")
mz = m[(m["scale"] == 0.0) & (m["direction"] == "jtw_mean_diff_tgt")]
mz = mz.set_index("prompt")["answer"].astype(str)
hit = ref.index.intersection(mz.index)
same = int((ref.reindex(hit) == mz.reindex(hit)).sum())
print(f"[oracle] vs the mean arm's unsteered block: {same}/{len(hit)} identical")
if len(hit) == 0 or same != len(hit):
    print("[oracle] FAIL: this run does not reproduce job 3082192 unsteered")
    sys.exit(1)
print("[oracle] PASS: unsteered generation is identical across arms and jobs")
ORACLE

echo "=== stage 3: judge the control with the same two judges ==="
python3 src/truthfulqa_judge.py --dataset truthfulqa --arm randctrl --device cuda

echo "=== stage 4: the comparison the job exists to make ==="
python3 - <<'CMP'
import pandas as pd

EPS = 14.706505916088334          # median eps* of mean_diff_tgt, job 3082192

def table(path, label):
    d = pd.read_csv(path)
    d["frac"] = (d["scale"] / EPS).round(2)
    d["words"] = d["answer"].astype(str).str.split().str.len()
    for name, g in d.groupby("direction"):
        t = g.groupby("frac").agg(n=("truthful_and_informative", "size"),
                                  ti=("truthful_and_informative", "sum"),
                                  words=("words", "mean"))
        t["rate"] = (t["ti"] / t["n"]).round(3)
        t["words"] = t["words"].round(2)
        print(f"\n--- {label}: {name} ---")
        print(t[["n", "ti", "rate", "words"]].to_string())

table("judge_refusal_truthfulqa_randctrl.csv", "RANDOM CONTROL")
table("judge_refusal_truthfulqa_mean.csv", "truth directions (job 3082210)")
print("\nCompare the rate and words columns at frac -2. If the random directions "
      "reach ~0.500 with ~18 words, the Q2 effect is verbosity, not truth.")
CMP
