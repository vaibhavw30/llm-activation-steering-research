#!/bin/bash
# setup_env.sh — one-time environment setup for DCT on NCSA DeltaAI (GH200, ARM64).
#
# Run this ON A LOGIN NODE (it needs internet to pip-install + download the model).
# It layers our deps on top of DeltaAI's prebuilt PyTorch module, so we do NOT
# pip-install torch (the module already has an ARM+CUDA build tuned for GH200).
#
#   bash deltaai/setup_env.sh
#
set -e
cd "$(dirname "$0")/.."   # repo root

echo "=== Loading DeltaAI PyTorch module (provides torch for GH200/aarch64) ==="
module load python/miniforge3_pytorch

echo "=== Creating venv that inherits the module's torch (--system-site-packages) ==="
python3 -m venv --system-site-packages .venv-dct-gpu
source .venv-dct-gpu/bin/activate

echo "=== Installing the paper-pinned transformers + our deps (NOT torch) ==="
pip install --upgrade pip
# transformers==4.51.3 is the critical pin for SlicedModel compatibility.
pip install "transformers==4.51.3" scipy tqdm pandas

echo "=== Verifying torch sees the GPU ==="
python3 - <<'PY'
import torch, transformers
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
PY

echo
echo "=== Hugging Face auth (gemma-2-2b is gated) ==="
echo "If you have not already, run:   hf auth login     (paste a read token)"
echo "Then pre-download the model on THIS login node so compute nodes (no internet) can read the cache:"
echo "  export HF_HOME=\$HOME/hf_cache   # put cache on scratch, not home quota"
echo "  hf download google/gemma-2-2b"
echo
echo "Setup complete. Edit deltaai/run_dct.slurm (set --account), then: sbatch deltaai/run_dct.slurm"
