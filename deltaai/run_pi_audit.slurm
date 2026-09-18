#!/bin/bash
#SBATCH --job-name=pi_audit
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=pi_audit_%j.out

# J-A of docs/PLAN_PI_FEEDBACK_2026-09-18.md: the TruthfulQA judge audit (track J) and
# the C3 truncation test. No new answers are generated. It re-reads Q1's and Q2's
# existing generations and judges them again with the info judge given its own
# "Helpful:" prompt (it had been getting the truth judge's "True:").
#
# Reads (all already on the cluster, all read-only):
#   reach_steer_truthfulqa.csv, reach_steer_randctrl_truthfulqa.csv,
#   tqa_baseline_completions.csv, reach_summary_truthfulqa.json,
#   got_datasets/truthfulqa_holdout.csv
# Writes only new names: judge_v2_truthfulqa_{mean,randctrl}.csv,
#   tqa_baseline_judged_v2.csv, judge_audit_*_truthfulqa.csv,
#   tqa_q2_summary_truthfulqa_judge_v2*.csv. judge_audit refuses to overwrite and
#   skips a finished stage, so resubmitting after a timeout resumes.
#
# Needs no GPU slot's worth of U1: it can run beside job 3169838.
#
# READ THIS FIRST WHEN IT FINISHES, in order:
#   [gold] info judge PASSES/FAILS     is the fixed info judge valid at all?
#   stage 3's two Q2 tables            does the Q2 headline survive the fix?
#   [truncate]                         is the Q2 gain in the extra words?
#   [qwen] agrees ... on n/N           does a second judge see the same thing?

set -ex
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export HF_HUB_DISABLE_XET=1
# Compute nodes are OFFLINE. Everything this job loads must already be in $HF_HOME.
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# Fail in ten seconds, not after an hour of judging.
for f in reach_steer_truthfulqa.csv reach_steer_randctrl_truthfulqa.csv \
         tqa_baseline_completions.csv reach_summary_truthfulqa.json \
         got_datasets/truthfulqa_holdout.csv; do
  test -f "$f" || { echo "!!!! $f missing. It is Q1/Q2 output and should be here."; exit 1; }
done
for m in models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B \
         models--Qwen--Qwen2.5-7B-Instruct; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done

# The allenai repos ship tokenizer.model only, so transformers converts slow to fast and
# needs these two; without them the error names the wrong package. Cost job 3081838.
python3 - <<'DEPCHECK' || exit 1
import importlib.util as u
import sys

missing = [n for n, spec in (("sentencepiece", "sentencepiece"),
                             ("protobuf", "google.protobuf")) if u.find_spec(spec) is None]
if missing:
    print("!!!! missing python packages: " + " ".join(missing))
    sys.exit(1)
print("[guard] tokenizer conversion deps present")
DEPCHECK

echo "=== stage 1: smoke, the two things the laptop could not check ==="
# 1. The real llama tokenizer yields yes/no word-token ids, and the judges put most of
#    their first-token mass on them. If not, every p(yes) this job writes is meaningless.
# 2. Qwen's chat template produces a reply parse_qwen can read.
# Two minutes, before the hour of judging that depends on both.
python3 - <<'SMOKE' || exit 1
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import judge_audit as ja
from judges.local_hf import TruthJudge
from prep_truthfulqa import answer_pair, load_holdout

h = load_holdout(ja.HOLDOUT).to_dict("records")
gold = ja.gold_rows(h[:2], answer_pair)
tj = TruthJudge("cuda")
print(f"[smoke] yes/no ids: truth {tj.truth.yes_id},{tj.truth.no_id}  "
      f"info {tj.info.yes_id},{tj.info.no_id}")
low = []
for r in gold:
    v = tj.score(r["question"], r["answer"])
    print(f"[smoke] {r['kind']:10s} truthful={v['truthful']} p={v['p_truthful']:.3f} "
          f"mass={v['yesno_mass_truthful']:.3f} | informative={v['informative']} "
          f"p={v['p_informative']:.3f} mass={v['yesno_mass_informative']:.3f}")
    low += [m for m in (v["yesno_mass_truthful"], v["yesno_mass_informative"]) if m < 0.5]
if low:
    print(f"!!!! {len(low)} verdicts put < 0.5 of the first-token mass on yes/no. p(yes) "
          "would be meaningless. Check answer_token_ids against this tokenizer.")
    sys.exit(1)
del tj
torch.cuda.empty_cache()

tok = AutoTokenizer.from_pretrained(ja.QWEN)
model = AutoModelForCausalLM.from_pretrained(ja.QWEN, torch_dtype=torch.bfloat16).to("cuda")
r = h[0]
msgs = ja.qwen_messages(r["question"], r["best_answer"], r["correct_answers"],
                        r["incorrect_answers"])
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to("cuda")
g = model.generate(**ids, max_new_tokens=20, do_sample=False)
reply = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"[smoke] qwen on the best answer: {reply!r} -> {ja.parse_qwen(reply)}")
if ja.parse_qwen(reply) is None:
    print("!!!! the Qwen reply does not parse. Fix qwen_messages or parse_qwen first.")
    sys.exit(1)
print("[smoke] PASS")
SMOKE

echo "=== stage 2: the audit ==="
python3 src/judge_audit.py --stage all --device cuda

echo "=== stage 3: the Q2 tables, recomputed from the v2 files ==="
# Same script and same statistics the Q2 doc used, pointed at the re-judged files. Both
# write suffixed summaries; the v1 tqa_q2_summary_truthfulqa.csv is not touched.
python3 src/tqa_q2_analyze.py --dataset truthfulqa --arms mean randctrl \
  --judged-pattern "judge_v2_{ds}_{arm}.csv"
python3 src/tqa_q2_analyze.py --dataset truthfulqa --arms mean randctrl \
  --judged-pattern "judge_v2_{ds}_{arm}.csv" --score-col truthful

echo "=== stage 4: v1 vs v2, row by row ==="
python3 - <<'DIFF'
import pandas as pd

for arm in ("mean", "randctrl"):
    v1 = pd.read_csv(f"judge_refusal_truthfulqa_{arm}.csv")
    v2 = pd.read_csv(f"judge_v2_truthfulqa_{arm}.csv")
    assert len(v1) == len(v2) and (v1["answer"].astype(str) == v2["answer"].astype(str)).all(), \
        f"{arm}: v1 and v2 are not the same answers in the same order"
    for col in ("truthful", "informative", "truthful_and_informative"):
        flips = int((v1[col] != v2[col]).sum())
        print(f"[v1->v2] {arm:8s} {col:25s} v1 {v1[col].mean():.3f}  v2 {v2[col].mean():.3f}"
              f"  rows changed {flips}/{len(v1)}")
print("[v1->v2] truthful should not change at all: the truth judge's prompt did not change.")
DIFF

echo "=== job complete ==="
ls -la judge_v2_truthfulqa_*.csv tqa_baseline_judged_v2.csv judge_audit_*_truthfulqa.csv \
       tqa_q2_summary_truthfulqa_judge_v2*.csv
