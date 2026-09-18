#!/bin/bash
#SBATCH --job-name=tqa_dct
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=verbose,closest
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=02:30:00
#SBATCH --output=tqa_dct_%j.out
# Wall 2:30 (was 6:00, cut 2026-09-18 to fit backfill): the fit took 7 min in job 3085406
# and the Q2 reach job did stages 2-3 plus generation in under 59 min (04:11 -> 05:09).

# U1: the UNSUPERVISED arm, on the one dataset where truth demonstrably moves.
# docs/PLAN_ADVISOR_NOTES_2026-09.md section U1.
#
# WHY THIS EXISTS. Every unsupervised null this project has (DCT_VS_TRUTH_FINDINGS,
# DCT_VS_XGBOOST_FINDINGS, WARM_DCT_RESULTS, and DCT_VS_MAG_ON_TRUTH, which replicates
# the null with a second and mechanically unrelated miner) was measured on cities and
# common_claim. Those are the datasets with a 5.7% spontaneous-false rate and a readout
# at balanced accuracy 0.500 on the population we steer. Q2 showed that confound made
# the SUPERVISED direction look dead when it was not. The unsupervised nulls inherit it.
#
# The geometric half of that null stands and this job does not re-ask it: a cosine at the
# fit point does not depend on behavioural headroom. What this job re-asks is the
# BEHAVIOURAL half, which warm-DCT settled by steering where there was nothing to move.
#
# THE REGISTERED OUTCOME, written before the run. MIN_ACC_1D = 0.6 gates threshold
# validity and only 3 of 74 directions clear it on TruthfulQA. The four dct_u readouts
# will most likely FAIL that gate and get thresh02 = nan, so no valid eps* and nothing to
# steer. That is a result, not a wasted run: it says the unsupervised causal directions
# cannot read TruthfulQA truth one-dimensionally, which is the 3-of-74 finding again
# rather than a new mystery. Stage 4 prints exactly this and says which way it went.
#
# TWO THINGS THIS JOB REFUSES TO OVERWRITE, because both are finished Q2 output.
#   1. dct_meta_truthfulqa.json. run_dct_data.py rewrites it unconditionally, and it
#      carries source_layer 11 / target_layer 20 / input_scale 60.94 that Q2 ran against
#      and that reach_steer.load_meta reads. Stage 1 saves it, lets the DCT fit write its
#      own, then restores the Q2 file and archives the DCT one beside it.
#   2. reach_dirs / reach_margins / reach_curve / reach_summary _truthfulqa.*. The
#      margins rerun regenerates all four. It therefore runs in its OWN directory,
#      u1_truthfulqa/, with the read-only inputs symlinked in. Nothing at the repo root
#      is written by stages 2 or 3.

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
WORK=u1_truthfulqa

# Fail in ten seconds, not after the factorization.
for f in activations/acts_${DS}.npz truth_dir_${DS}.npz truth_dir_tgt_${DS}.npz reach_acts_${DS}.npz \
         dct_meta_${DS}.json got_datasets/${DS}.csv; do
  test -f "$f" || { echo "!!!! $f missing. It is Q0/Q2 output: rsync it up."; exit 1; }
done
test -d "$HF_HOME/hub/models--google--gemma-2-2b" \
  || { echo "!!!! gemma-2-2b not in $HF_HOME/hub. Stage it on the LOGIN node."; exit 1; }

# An existing factorization is finished output like any other: never refit over it. A
# COMPLETE one is resumed from rather than refused. Job 3085406 finished the fit and the
# meta restore in 7 minutes, then died in stage 2 or 3, and the fit it left is good.
# Complete means: both V and U present, the fit's archived meta present, and that meta
# says 512 factors at 11 -> 20. Anything less (one of V/U, no archived meta) is the
# half-present state reach_hop.optional_artifacts silently treats as absent, and is
# still refused.
FIT_DONE=0
if [ -e dct_V_${DS}.pt ] || [ -e dct_U_${DS}.pt ] || [ -e dct_meta_${DS}.dctfit.json ]; then
  python3 - <<'RESUME' || exit 1
import json
import os
import sys

need = ["dct_V_truthfulqa.pt", "dct_U_truthfulqa.pt", "dct_meta_truthfulqa.dctfit.json"]
missing = [f for f in need if not os.path.exists(f)]
if missing:
    print(f"!!!! a partial factorization is on disk, missing {missing}. This job does "
          "not refit over it or resume from it. Move the pieces aside by hand.")
    sys.exit(1)
m = json.load(open("dct_meta_truthfulqa.dctfit.json"))
got = (m.get("num_factors"), m.get("source_layer"), m.get("target_layer"))
if got != (512, 11, 20):
    print(f"!!!! the archived fit is (num_factors, source, target) = {got}, "
          "not (512, 11, 20). Not resuming from it.")
    sys.exit(1)
q2 = json.load(open("dct_meta_truthfulqa.json"))
if q2.get("num_factors") is not None:
    print("!!!! dct_meta_truthfulqa.json is a fit's copy, not the Q2 meta. Recover it "
          "from dct_meta_truthfulqa.q2.json.bak by hand first.")
    sys.exit(1)
print(f"[resume] complete factorization on disk {got}, Q2 meta intact: skipping stage 1")
RESUME
  FIT_DONE=1
fi

if [ "$FIT_DONE" = 1 ]; then
  echo "=== stage 1: SKIPPED, resuming from the existing factorization ==="
else
echo "=== stage 1: the DCT factorization, unsupervised, no labels ==="
# --source-layer 11 --target-layer 20 are NOT the script's defaults (6 and 18). They are
# the layers Q0's make_reach_meta chose and Q2 ran against, and the whole point of this
# job is a comparison at the same point in the network.
#
# --max-length 96 matches Q0. TruthfulQA rows are prompt PLUS answer and 4 of 1488 run
# past the 64-token default; a truncated row is read mid-answer instead of at its last
# token, which is the fit point the contrast depends on.
#
# --scale skips DCT's own SteeringCalibrator and reuses the input_scale already in
# dct_meta_truthfulqa.json. Same calibrator, same layers, same data, so it would land on
# this number anyway; passing it makes the perturbation norm identical to the one Q2's
# grid was built from, which is what makes the two arms comparable rather than adjacent.
# Back up only what is worth backing up. If a previous attempt died between the fit and
# the restore, dct_meta_<ds>.json is the FIT's copy, and cp would then overwrite a good
# backup with a bad one and lose the Q2 meta for good. num_factors is the tell: the Q2
# file has it null because no factorization ever ran.
python3 - <<'PREBACKUP' || exit 1
import json
import sys

m = json.load(open("dct_meta_truthfulqa.json"))
if m.get("num_factors") is not None:
    print(f"!!!! dct_meta_truthfulqa.json is a DCT fit's copy (num_factors="
          f"{m['num_factors']}), not the Q2 meta. A previous run died mid-restore. "
          "Recover the Q2 file from dct_meta_truthfulqa.q2.json.bak by hand before "
          "rerunning; this job will not overwrite the backup with it.")
    sys.exit(1)
print("[stage 1] dct_meta_truthfulqa.json is the Q2 meta, backing it up")
PREBACKUP
cp -p dct_meta_${DS}.json dct_meta_${DS}.q2.json.bak
SCALE=$(python3 -c "import json;print(json.load(open('dct_meta_${DS}.json'))['input_scale'])")
echo "[stage 1] reusing input_scale=$SCALE from the Q2 meta"

python3 src/run_dct_data.py --dataset ${DS} --device cuda \
  --source-layer 11 --target-layer 20 \
  --num-factors 512 --num-iters 30 --num-samples 64 --balanced \
  --max-length 96 --token-idxs=-3: --scale "$SCALE"
# The equals form on --token-idxs is REQUIRED, not style. "-3:" begins with a dash, so
# `--token-idxs "-3:"` makes argparse read it as a flag and die with "expected one
# argument". Job 3085375 lost a submission to exactly this.

# Put the Q2 meta back and keep the DCT one under its own name. Order matters: the
# factorization has already written its version over the Q2 file by this point.
mv dct_meta_${DS}.json dct_meta_${DS}.dctfit.json
mv dct_meta_${DS}.q2.json.bak dct_meta_${DS}.json
python3 - <<'RESTORE' || exit 1
import json
q2 = json.load(open("dct_meta_truthfulqa.json"))
assert q2["source_layer"] == 11 and q2["target_layer"] == 20, q2
assert q2["num_factors"] is None, "the Q2 meta was not restored, it is the DCT fit's"
print(f"[stage 1] Q2 meta restored intact: input_scale={q2['input_scale']}")
RESTORE
fi

echo "=== stage 2: a private directory, so the margins rerun cannot touch Q2 ==="
# reach_margins hardcodes reach_dirs_<ds>.npz and reach_margins_<ds>.npz in the cwd and
# has no output-suffix flag. Rather than patch a script that produced finished artifacts,
# give it a different cwd. Inputs are symlinked read-only; outputs land here.
mkdir -p ${WORK}
# truth_dir_<ds>.npz (the SOURCE-layer direction) is required by
# reach_hop.validate_inputs, not just truth_dir_tgt. Job 3085406 left it out and died
# in stage 3 with "missing truth_dir_truthfulqa.npz", after a good fit.
for f in truth_dir_${DS}.npz truth_dir_tgt_${DS}.npz reach_acts_${DS}.npz \
         dct_V_${DS}.pt dct_U_${DS}.pt dct_meta_${DS}.json got_datasets activations; do
  ln -sfn "$SLURM_SUBMIT_DIR/$f" "${WORK}/$(basename $f)"
done
# Optional: when present it adds the cos_vq landmark. Link it iff Q2 had it, so the
# rerun's bookkeeping matches Q2's rather than silently dropping a column.
if [ -e mag_dir_${DS}.npz ]; then ln -sfn "$SLURM_SUBMIT_DIR/mag_dir_${DS}.npz" "${WORK}/"; fi
cd ${WORK}

python3 - <<'GUARD' || exit 1
import sys

from reach_hop import optional_artifacts, validate_inputs

# The exact check reach_margins runs first, run here so a missing input fails in
# stage 2 with every problem listed, not one per resubmission.
validate_inputs("truthfulqa")
print("[guard] reach_hop.validate_inputs passed in the work directory")
ok = optional_artifacts("truthfulqa")["dct"]
print(f"[guard] optional_artifacts()['dct'] = {ok}")
if not ok:
    print("[guard] FAIL: the dct_u battery members will be SKIPPED and this job would "
          "silently reproduce the 74-direction run. Check the V/U symlinks.")
    sys.exit(1)
GUARD

echo "=== stage 3: rebuild the direction battery, now with dct_u_0..3 ==="
# K_DCT_U = 4, N_RAND = 64, so the battery goes 74 -> 78 directions. The four new ones
# are DCT's top-potency U vectors, ranked by ||U_i||, used as TARGET-LAYER READOUTS: the
# certificate machinery treats them exactly like mean_diff_tgt, same fit_threshold, same
# pullback, same eps*. No code change anywhere makes that happen.
python3 "$SLURM_SUBMIT_DIR/src/reach_margins.py" --dataset ${DS} --stage dirs --device cuda
python3 "$SLURM_SUBMIT_DIR/src/reach_margins.py" --dataset ${DS} --stage vjp --device cuda
python3 "$SLURM_SUBMIT_DIR/src/reach_analyze.py" --dataset ${DS}

echo "=== stage 4: the registered question, answered ==="
python3 - <<'REPORT'
import json

import numpy as np

d = np.load("reach_dirs_truthfulqa.npz", allow_pickle=True)
names = [str(x) for x in d["names"]]
acc = np.asarray(d["acc1d"], float)
th = np.asarray(d["thresh02"], float)
MIN_ACC_1D = 0.6

# reach_analyze writes `directions` as a dict keyed by direction name, each value
# carrying group / median_margin / rand_pct and, only when the threshold was valid,
# median_eps_star. A direction that failed MIN_ACC_1D has no such key at all, which is
# why .get is the right accessor here and a missing value prints as None.
summary = json.load(open("reach_summary_truthfulqa.json"))["directions"]
eps = {name: row.get("median_eps_star") for name, row in summary.items()}
pct = {name: row.get("rand_pct") for name, row in summary.items()}

print(f"\n[u1] battery is {len(names)} directions "
      f"(74 in the Q2 run, plus {sum(n.startswith('dct_u_') for n in names)} dct_u)")
valid = [n for n, a in zip(names, acc) if a >= MIN_ACC_1D]
print(f"[u1] clearing MIN_ACC_1D = {MIN_ACC_1D}: {len(valid)} of {len(names)} -> {valid}")

print("\n--- the four unsupervised readouts ---")
rows = [(n, a, t) for n, a, t in zip(names, acc, th) if n.startswith("dct_u_")]
for n, a, t in rows:
    print(f"  {n:12s} acc1d={a:.3f}  thresh02={t}  "
          f"median_eps_star={eps.get(n)}  rand_pct={pct.get(n)}")

cleared = [n for n, a, _ in rows if a >= MIN_ACC_1D]
print()
if not rows:
    print("[u1] !!!! no dct_u members in the battery. Stage 2's guard should have "
          "caught this; the run is void.")
elif cleared:
    print(f"[u1] AGAINST the registered expectation: {cleared} clear MIN_ACC_1D. The "
          "unsupervised causal directions CAN read TruthfulQA truth one-dimensionally. "
          "They have a valid eps*, so the next step is to steer them through "
          "reach_steer and judge, exactly as the mean arm was.")
else:
    print("[u1] AS REGISTERED: no dct_u readout clears MIN_ACC_1D, so none has a valid "
          "eps* and there is nothing to steer. This is the 3-of-74 readout finding "
          "again, now shown to cover the unsupervised directions too, and NOT a new "
          "mystery. The behavioural half of the DCT null stays open: it was not "
          "re-tested here, because the certificate has no boundary to aim at.")
REPORT

echo "=== job complete. Q2 artifacts at the repo root are untouched. ==="
ls -la "$SLURM_SUBMIT_DIR"/reach_dirs_truthfulqa.npz "$SLURM_SUBMIT_DIR"/dct_meta_truthfulqa.json
