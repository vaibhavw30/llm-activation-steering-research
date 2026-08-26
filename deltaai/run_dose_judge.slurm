#!/bin/bash
#SBATCH --job-name=dose-judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=dose_judge_%j.out
# D1 judging: score the dose-response completions TRUE / FALSE / INCOHERENT.
#
# USE THE OLMo JUDGE, NOT THE API. Not for convenience: judge_mag_steer_<ds>.csv and
# judge_reach_steer_<ds>.csv were both produced by this judge under this rubric, and D1 exists
# to be compared against them. Swapping in a different judge would change the instrument in the
# middle of the audit, and every "the naive arm did X, D1 did Y" statement would be confounded
# by the judge rather than by the dose. Same judge, same rubric, same prompt.
#
# The paths are given explicitly rather than through --mag, so this cannot read or write any
# committed judge artifact. Input dose_<ds>.csv -> output judge_dose_<ds>.csv.
#
# DEPENDS ON: dose_<ds>.csv from run_dose.slurm. Submit only after squeue is empty AND both
# dose CSVs are complete (see the row-count check in deltaai/D1_DOSE_RUN.md step 6).
#
# Cost: about 4,640 completions per dataset, 9,280 total, roughly 3.2x the MAG E4 judge run
# that took 40 minutes. The 4h wall is sized from that with headroom.
set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate     # the judge env (transformers 5.x), NOT .venv-dct-gpu
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# Short smoke first so a wrong path fails in seconds rather than after the full sweep.
srun python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset cities --limit 24 \
    --steer-input dose_cities.csv \
    --steer-output judge_dose_smoke_cities.csv \
    --steer-plot plot_judge_dose_smoke_cities.png

for ds in cities common_claim_true_false; do
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') judging D1 $ds ====="
  srun python3 src/judge_results.py --mode steer --backend olmo --device cuda \
      --dataset "$ds" \
      --steer-input "dose_${ds}.csv" \
      --steer-output "judge_dose_${ds}.csv" \
      --steer-plot "plot_judge_dose_${ds}.png" \
    || echo "!!!! $ds judge FAILED, continuing"
done
echo "=== dose judge done -> judge_dose_<ds>.csv $(date) ==="
echo "rsync judge_dose_*.csv + dose_yesno_*.csv + dose_meta_*.json back, then run"
echo "  ./.venv/bin/python src/dose_analyze.py --dataset <ds>"
