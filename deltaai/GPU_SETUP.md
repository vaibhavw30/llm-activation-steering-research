# Running DCT on NCSA DeltaAI (GH200)

DeltaAI nodes are **NVIDIA GH200 Grace-Hopper superchips**: the CPU is **ARM64
(aarch64)** and each node has **4 Hopper GPUs** (~96 GB HBM each). Partition: `ghx4`.

Because the CPU is ARM, we do **not** pip-install torch (the x86 wheels won't work).
Instead we use DeltaAI's prebuilt PyTorch module and layer our deps on top.

gemma-2-2b in fp32 is ~10 GB → fits on a single GH200 with huge headroom, so 1 GPU is
plenty (no multi-GPU / Ray needed).

---

## Step 1 — Get the code onto DeltaAI

**Recommended: git clone** (login node has internet).
```bash
# from your laptop: push this repo to GitHub first (it gitignores the big .npz/.pt/.venv)
ssh <you>@login.deltaai.ncsa.illinois.edu
cd $WORK            # or wherever you keep projects
git clone <your-repo-url> llm-activation-steering-research
cd llm-activation-steering-research
```

**Alternative: rsync from your laptop** (no git remote needed):
```bash
rsync -av --exclude '.venv*' --exclude 'activations' --exclude '*.npz' \
  ./ <you>@login.deltaai.ncsa.illinois.edu:/work/<...>/llm-activation-steering-research/
```
The `got_datasets/*.csv` are small and travel with the repo either way.

## Step 2 — One-time environment setup (on a login node)

```bash
bash deltaai/setup_env.sh
```
This loads `python/miniforge3_pytorch`, creates `.venv-dct-gpu` (inheriting the module's
torch via `--system-site-packages`), and pip-installs `transformers==4.51.3` + scipy/tqdm/pandas.

## Step 3 — Hugging Face auth + pre-download the model

gemma-2-2b is gated, and **compute nodes usually have no internet** — so cache the model
on the login node first:
```bash
source .venv-dct-gpu/bin/activate
hf auth login                       # paste a read token; accept license at hf.co/google/gemma-2-2b
export HF_HOME=$HOME/hf_cache    # keep the ~10 GB cache off your home quota
hf download google/gemma-2-2b
```
(Use the same `HF_HOME` in the job script — it already defaults to `$HOME/hf_cache`.)

## Step 4 — Edit and submit the job

```bash
# set your allocation in the script:
#   #SBATCH --account=ACCOUNT_NAME   ->   your account (see: sacctmgr show assoc user=$USER)
sbatch deltaai/run_dct.slurm
```

Monitor:
```bash
squeue -u $USER
tail -f dct_<jobid>.out
```

The default job runs **Stage 1 verification** (`run_dct_minimal.py`, num_factors=64,
max_iters=10) — a few minutes on GH200 vs ~90 min on CPU. Once that prints
`=== DONE ===` with V shape (2304, 64), uncomment the **Stage 3** lines in
`run_dct.slurm` for the real runs on the true/false datasets.

## Interactive debugging (optional)

```bash
srun -A ACCOUNT_NAME --partition=ghx4-interactive --nodes=1 --tasks=1 \
  --tasks-per-node=1 --cpus-per-task=8 --mem=32g --gpus-per-node=1 --time=01:00:00 --pty bash
# then, inside the node:
module load python/miniforge3_pytorch && source .venv-dct-gpu/bin/activate
export HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
python3 src/run_dct_minimal.py --device cuda --num-factors 64 --max-iters 10
```
(`ghx4-interactive` is capped at 2 h.)

---

## What changed from the CPU setup (and why)

| Issue | CPU setup | DeltaAI (GH200) |
|---|---|---|
| torch | pip CPU wheel | **module `python/miniforge3_pytorch`** (ARM+CUDA) |
| device | `--device cpu` | `--device cuda` (scripts auto-fallback if no GPU) |
| attention | `attn_implementation="eager"` | **same** — flash/SDPA still won't support the calibrator's forward-AD (jvp) |
| transformers | 4.51.3 (pinned) | **4.51.3 (same pin)** — required for SlicedModel |
| scale | num_factors=64, 1 prompt | scale up: num_factors 512–1024, many statements |

## Expected speedup

Stage 1 took ~90 min on CPU. On a GH200 expect **a few minutes**. Stage 3 at paper scale
(512–1024 factors, 30 iters, 64 statements) — infeasible on CPU — should be **tens of
minutes to ~1–2 h** on one GH200. Bump `--num-factors`, `--num-samples`, `--num-iters`
as the allocation allows.
