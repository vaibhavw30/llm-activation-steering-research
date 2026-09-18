#!/bin/bash
#SBATCH --job-name=tqa_dct_s2
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=05:00:00
#SBATCH --output=tqa_dct_s2_%j.out

# J-B of docs/PLAN_PI_FEEDBACK_2026-09-18.md (round 1): DCT discovery on TruthfulQA
# (track D), MAG's verdict channel on TruthfulQA (G0), the readout half of the transfer
# matrix (X) and the probe/XGBoost row of the dataset card (C).
#
# NEEDS, and stage 0 checks: U1 (job 3169838) finished, so its fit dct_V/U_truthfulqa.pt
# is here, and J-A (job 3169899) finished, so judge_audit_gold_truthfulqa.csv is here.
# The screen is judged with J-A's fixed prompts and scores truthful-and-informative only
# if J-A's gold check passed; otherwise truthful alone. src/tqa_discovery.py decides.
#
# Writes only new names, never at a Q2 path:
#   dct_s2_truthfulqa/         the second fit (--seed=326), in its own directory, so the
#                              refit cannot touch U1's dct_V/U or any dct_meta file
#   dct_geometry_, dct_stability_, dct_screen_, dct_screen_judged_, dct_selection_ (json),
#   mag_g0_, readout_transfer_, direction_cosines_  *_truthfulqa.csv
#   jb_truthfulqa/             mag_acts / mag_dir _truthfulqa.npz, analyze.py's results
# A resubmit after a timeout resumes: the fit is skipped if complete, finished stages are
# skipped, the screen resumes per direction and the judge per 64-row block.
#
# READ THIS FIRST WHEN IT FINISHES, in order:
#   [select] the positive control's gain      can the screen see Q2's known effect at all?
#   [select] S-beh winner or "no stable winner"
#   [stability] factor rows vs random         is the factorization a property of the data?
#   [G0] ALIVE / dead, per phrasing
#   [readout] cities -> truthfulqa and back

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

DS=truthfulqa
S2=dct_s2_${DS}

echo "=== stage 0: preflight ==="
for f in activations/acts_${DS}.npz activations/acts_cities.npz \
         truth_dir_${DS}.npz truth_dir_tgt_${DS}.npz truth_dir_cities.npz \
         reach_summary_${DS}.json reach_margins_${DS}.npz reach_acts_${DS}.npz \
         reach_dirs_${DS}.npz dct_meta_${DS}.json got_datasets/${DS}.csv \
         got_datasets/${DS}_holdout.csv mag_dir_cities.npz; do
  test -f "$f" || { echo "!!!! $f missing. It is Q0/Q2 or cities output: rsync it up."; exit 1; }
done
for f in dct_V_${DS}.pt dct_U_${DS}.pt dct_meta_${DS}.dctfit.json; do
  test -f "$f" || { echo "!!!! $f missing: U1 (job 3169838) has not finished its fit."; exit 1; }
done
test -f judge_audit_gold_${DS}.csv \
  || { echo "!!!! judge_audit_gold_${DS}.csv missing: J-A (job 3169899) has not run."; exit 1; }
for m in models--google--gemma-2-2b models--allenai--truthfulqa-truth-judge-llama2-7B \
         models--allenai--truthfulqa-info-judge-llama2-7B; do
  test -d "$HF_HOME/hub/$m" \
    || { echo "!!!! $m not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }
done
python3 - <<'DEPCHECK' || exit 1
import importlib.util as u
import sys

need = {"sentencepiece": "sentencepiece", "protobuf": "google.protobuf",
        "scikit-learn": "sklearn", "xgboost": "xgboost"}
missing = [n for n, spec in need.items() if u.find_spec(spec) is None]
if missing:
    print("!!!! missing python packages: " + " ".join(missing))
    sys.exit(1)
print("[guard] all packages present")
DEPCHECK
# The Q2 meta must be the Q2 meta, not a DCT fit's copy (U1's stage-1 swap).
python3 -c "
import json, sys
m = json.load(open('dct_meta_${DS}.json'))
sys.exit(0 if m.get('num_factors') is None and m['source_layer'] == 11 else
         '!!!! dct_meta_${DS}.json is not the Q2 meta')"

echo "=== stage 1: D-R1, the second DCT fit, seed 326, in its own directory ==="
# Same arguments as U1's fit (deltaai/run_truthfulqa_dct.slurm) except --seed, which
# changes both the 64-statement draw and the initialisation. The equals form on
# --token-idxs and --seed is required: "-3:" begins with a dash (job 3085375).
# Non-fatal: without it D-R1 is not measured, and every other stage still runs.
mkdir -p ${S2}
ln -sfn "$SLURM_SUBMIT_DIR/got_datasets" "${S2}/got_datasets"
if [ -f ${S2}/dct_V_${DS}.pt ] && [ -f ${S2}/dct_U_${DS}.pt ] && [ -f ${S2}/dct_meta_${DS}.json ]; then
  echo "[stage 1] second fit already complete, skipped"
else
  SCALE=$(python3 -c "import json;print(json.load(open('dct_meta_${DS}.json'))['input_scale'])")
  (cd ${S2} && python3 "$SLURM_SUBMIT_DIR/src/run_dct_data.py" --dataset ${DS} --device cuda \
     --source-layer 11 --target-layer 20 --num-factors 512 --num-iters 30 --num-samples 64 \
     --balanced --max-length 96 --token-idxs=-3: --scale "$SCALE" --seed=326) \
    || echo "!!!! the second fit FAILED. D-R1 is not measured this run; continuing."
fi

echo "=== stage 2: smoke, every stage end to end on 16 items, smoke_ outputs ==="
python3 src/tqa_discovery.py --stage all --device cuda --limit 16 --prefix smoke_

echo "=== stage 3: the full run ==="
python3 src/tqa_discovery.py --stage all --device cuda

echo "=== stage 4: the card's probe/XGBoost row, in its own process ==="
# analyze.py imports xgboost. torch and xgboost in one process is the libomp clash
# recorded in tqa_baseline.py, so it runs alone, in jb_truthfulqa/ where its
# results_/directions_/plot_ files cannot touch anything at the repo root.
mkdir -p jb_truthfulqa
ln -sfn "$SLURM_SUBMIT_DIR/activations/acts_${DS}.npz" jb_truthfulqa/acts_${DS}.npz
if [ -f jb_truthfulqa/results_${DS}.csv ]; then
  echo "[stage 4] results_${DS}.csv exists, skipped"
else
  (cd jb_truthfulqa && python3 "$SLURM_SUBMIT_DIR/src/analyze.py" ${DS}) \
    || echo "!!!! analyze.py FAILED; the card row is missing, everything else stands"
fi

echo "=== job complete ==="
ls -la dct_*_${DS}.csv dct_selection_${DS}.json mag_g0_${DS}.csv readout_transfer_${DS}.csv \
       direction_cosines_${DS}.csv ${S2}/ jb_truthfulqa/
