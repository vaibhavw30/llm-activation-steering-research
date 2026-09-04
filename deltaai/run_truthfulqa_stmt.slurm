#!/bin/bash
#SBATCH --job-name=tqa_stmt
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=tqa_stmt_%j.out

# Q2's per-statement arm, regenerated after the stem_of newline fix, and judged.
#
# WHY THIS RERUNS SOMETHING THAT ALREADY EXISTS. Job 3082192 generated this arm with
# the pre-fix stem_of, which flattened the "Q: ...\nA: ..." newline, so every one of
# its 960 prompts reads "Q: ... A: ...". The newline is the only marker of where the
# question ends, and truthfulqa_judge.question_of refuses to parse without it, which
# is what killed the stmt half of job 3082210. The CSV is not salvageable: the prompts
# the model actually saw were malformed, so the generations are wrong, not just their
# labels. stem_of is fixed and tested as of commit ecc834b.
#
# Overwriting an existing reach_*.csv is not something this project does, so stage 0
# MOVES the old one to reach_steer_stmt_truthfulqa.malformed-prompts.csv and refuses
# to clobber that archive if it is already there. Nothing is deleted by this job.
#
# WHAT THE ARM IS, because it reads backwards otherwise. arm_per_stmt samples 200
# label-1 statements, and prep_truthfulqa sets label 1 = UNTRUTHFUL. So the prompt is
# the opening of an UNTRUTHFUL answer minus its final word, and the generation is how
# the model continues it. Steering toward truthful is being asked to break the model
# out of an untruthful answer it has already started. That is why stage 3 judges
# head + continuation (truthfulqa_judge.ARMS_CONTINUING_THE_PROMPT): judging the
# continuation alone would score a sentence fragment and discard the untruthful
# lead-in that the whole arm is built around.
#
# EXPLORATORY. run_truthfulqa_reach.slurm already says so and it is still true: the
# mean arm is Q2's result (docs/Q2_TRUTHFULQA_STEERING.md), this is not. It is a
# different population from the one the direction was fit on, which is the D2 finding.

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
         dct_meta_truthfulqa.json; do
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

echo "=== stage 0: move the malformed arm aside, if that is what is there ==="
# Decided by reading the file, never by assuming. A CSV whose prompts carry the
# newline is a POST-fix run: leave it alone and let stage 1 skip, exactly as the
# randctrl job does. A CSV without it is the malformed one and gets archived.
python3 - <<'TRIAGE' || exit 1
import os
import sys

import pandas as pd

SRC = "reach_steer_stmt_truthfulqa.csv"
META = "reach_steer_stmt_meta_truthfulqa.csv"

if not os.path.exists(SRC):
    print("[stage 0] no existing arm, nothing to move")
    sys.exit(0)

d = pd.read_csv(SRC)
good = int(d["prompt"].astype(str).str.contains("\n").sum())
print(f"[stage 0] {SRC}: {len(d)} rows, {good} prompts carry the newline")
if good == len(d):
    print("[stage 0] this is a POST-fix run. Keeping it, stage 1 will skip.")
    sys.exit(0)
if good:
    print(f"!!!! {good} of {len(d)} prompts have the newline. A half-and-half file is "
          "neither the malformed run nor a clean one; refusing to guess.")
    sys.exit(1)

for src in (SRC, META):
    if not os.path.exists(src):
        continue
    dst = src.replace(".csv", ".malformed-prompts.csv")
    if os.path.exists(dst):
        print(f"!!!! {dst} already exists. Refusing to overwrite the archive; move or "
              "remove it by hand if you really mean to.")
        sys.exit(1)
    os.rename(src, dst)
    print(f"[stage 0] {src} -> {dst}")
TRIAGE

echo "=== stage 1: regenerate the per-statement arm with the fixed stem_of ==="
# --max-new-tokens 48 matches the mean arm and Q1. --prompt-mode stem is the arm.
if [ -s reach_steer_stmt_truthfulqa.csv ]; then
  echo "[stage 1] reach_steer_stmt_truthfulqa.csv exists, keeping it and skipping"
  wc -l reach_steer_stmt_truthfulqa.csv
else
  python3 src/reach_steer.py --dataset truthfulqa --device cuda --arm per_stmt \
    --prompt-mode stem --max-new-tokens 48
fi

echo "=== stage 2: the fix actually took ==="
# The whole reason this job exists. Every prompt must carry the newline AND parse with
# the judge's own parser, checked with the judge's own function rather than a lookalike
# regex, so this cannot pass while stage 3 fails the way job 3082210 did.
python3 - <<'ORACLE' || exit 1
import sys

import pandas as pd

from truthfulqa_judge import question_of

d = pd.read_csv("reach_steer_stmt_truthfulqa.csv")
bad = [p for p in d["prompt"].astype(str) if "\n" not in p]
print(f"[oracle] {len(d)} rows, {len(bad)} prompts without a newline")
if bad:
    print(f"[oracle] FAIL: stem_of is still flattening. First: {bad[0]!r}")
    sys.exit(1)
for p in d["prompt"].astype(str):
    question_of(p)
print(f"[oracle] PASS: all {len(d)} prompts parse with truthfulqa_judge.question_of")
ORACLE

echo "=== stage 3: judge head + continuation ==="
python3 src/truthfulqa_judge.py --dataset truthfulqa --arm stmt --device cuda

echo "=== stage 4: what moved, per dose ==="
python3 - <<'CMP'
import numpy as np
import pandas as pd

d = pd.read_csv("judge_refusal_truthfulqa_stmt.csv")
m = pd.read_csv("reach_steer_stmt_meta_truthfulqa.csv")
# Each statement has its OWN eps*, so scale is not comparable across statements the
# way the mean arm's is. frac is, and the meta file is where eps* lives.
d = d.join(m[["eps_star"]].reset_index(drop=True))
# np.nan, not pd.NA: Series.round raises TypeError on NAType, and a statement whose
# eps* is 0 (g_read already on the truthful side) contributes only its scale-0 row.
d["frac"] = (d["scale"] / d["eps_star"].replace(0.0, np.nan)).round(2)
d["words"] = d["judged_answer"].astype(str).str.split().str.len()
t = d.groupby("frac", dropna=False).agg(n=("truthful_and_informative", "size"),
                                        ti=("truthful_and_informative", "sum"),
                                        words=("words", "mean"))
t["rate"] = (t["ti"] / t["n"]).round(3)
t["words"] = t["words"].round(2)
print(t[["n", "ti", "rate", "words"]].to_string())
print("\nThese are 200 label-1 (UNTRUTHFUL) statements, so frac 0 should sit LOW and "
      "the question is whether the negative doses lift it. This arm is exploratory: "
      "the mean arm in docs/Q2_TRUTHFULQA_STEERING.md is Q2's result.")
CMP
