#!/bin/bash
# setup_judge_env.sh — one-time setup for the OLMo-3 LLM-as-judge on NCSA DeltaAI (GH200, ARM64).
#
# Run this ON A LOGIN NODE (it needs internet to pip-install + download the model).
#
#   bash deltaai/setup_judge_env.sh
#
# WHY A SEPARATE VENV FROM .venv-dct-gpu:
#   The DCT env pins transformers==4.51.3 (critical for SlicedModel). OLMo-3 chat templates need
#   transformers>=5. Upgrading the DCT env would break DCT, so the judge gets its own env
#   (.venv-judge-gpu) layered on the SAME GH200 torch module. Both envs coexist.
set -e
cd "$(dirname "$0")/.."   # repo root

echo "=== Loading DeltaAI PyTorch module (provides torch for GH200/aarch64) ==="
module load python/miniforge3_pytorch

echo "=== Creating judge venv that inherits the module's torch (--system-site-packages) ==="
python3 -m venv --system-site-packages .venv-judge-gpu
source .venv-judge-gpu/bin/activate

echo "=== Installing transformers 5.x + judge deps (NOT torch — it comes from the module) ==="
pip install --upgrade pip
# transformers==5.12.1 is the version the judge code was written and validated against locally.
# matplotlib is for the FALSE-vs-INCOHERENT steering plot; numpy comes with matplotlib.
pip install "transformers==5.12.1" accelerate matplotlib

echo "=== Verifying torch (from module) is compatible with transformers 5.x and sees the GPU ==="
python3 - <<'PY'
import torch, transformers
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
# OLMo-3 must be a known architecture in this transformers build:
from transformers import AutoConfig  # noqa: F401
print("transformers 5.x import OK — OLMo-3 chat template supported")
PY

echo
echo "=== Pre-download the OLMo-3 judge model on THIS login node (compute nodes have no internet) ==="
echo "OLMo-3 is fully open (allenai) — NO license gate, unlike gemma. Just:"
echo "  export HF_HOME=\$HOME/hf_cache          # same cache the DCT jobs use; keep off home quota"
echo "  export HF_HUB_DISABLE_XET=1"
echo "  hf download allenai/Olmo-3-7B-Instruct  # ~14 GB into \$HF_HOME"
echo
echo "Setup complete. Ensure the Task-3 INPUTS exist in the repo root on the cluster first:"
echo "  steer_supervised_cities.csv, steer_supervised_common_claim_true_false.csv  (from run_steer.slurm)"
echo "  interpret_top10_cities.md, interpret_top10_common_claim_true_false.md      (from run_interpret.slurm)"
echo "Then edit deltaai/run_judge.slurm (set --account), and: sbatch deltaai/run_judge.slurm"
