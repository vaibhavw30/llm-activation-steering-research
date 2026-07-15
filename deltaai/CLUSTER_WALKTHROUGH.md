# DeltaAI Cluster Walkthrough — operating SLURM and running every script

The single operator's guide: how the GH200 cluster works, how to drive it with SLURM, and the exact
order to run **all** the project's Python scripts (which on the cluster, which on the laptop) from a
cold start to the PI figures.

**Companion docs** (this one ties them together):
`GPU_SETUP.md` (why the env is built the way it is) · `MY_RUN_STEPS.md` (one-time NCSA/Duo + env
setup, filled-in) · `FUNNEL_RUN_STEPS.md` (the detailed interpret+steer walkthrough with "what you
should see" at each step) · `JUDGE_RUN_STEPS.md` (the OLMo judge, Task 3).

---

## 0. Mental model (read this once)

- **Two machines.** Your **💻 laptop** (Apple M-series, CPU/MPS) does the cheap, fast analysis and
  plotting. **🖥️ DeltaAI** (NCSA GH200: ARM64 CPU + ~96 GB Hopper GPU) does the heavy model forward
  passes (DCT, generation, the 7B judge). You move files between them with `rsync`.
- **You never run compute on the login node.** You `sbatch` a job script; SLURM finds a free GPU
  node, runs it, and writes a log. The login node is only for editing, `rsync`, `sbatch`, and
  pre-downloading models (it has internet; **compute nodes do not**).
- **Two Python envs on the cluster, both layered on the same torch module** (`module load
  python/miniforge3_pytorch` provides the ARM+CUDA torch — never `pip install torch`):
  | env | transformers | used by | why separate |
  |---|---|---|---|
  | `.venv-dct-gpu` | **4.51.3** (pinned) | DCT, interpret, steer | 4.51.3 is required for DCT's SlicedModel |
  | `.venv-judge-gpu` | **5.12.1** | the OLMo judge + validate | OLMo-3 needs transformers ≥5; upgrading the DCT env would break DCT |
- **Everything persists** across SSH sessions: your home dir, both venvs, and the `$HOME/hf_cache`
  model cache. You only redo setup if something's missing.

## 1. Your coordinates (DeltaAI)

| | value |
|---|---|
| Login host | `dtai-login.delta.ncsa.illinois.edu` (NCSA password + **Duo push**: type `1`, approve on phone) |
| User | `vwudaru` · Project `CIS260948` · **Account `bhhv-dtai-gh`** (the `--account` for every job) |
| Login prompt looks like | `vwudaru@gh-login01:~>` |
| Repo on cluster | `~/llm-activation-steering-research` |
| Repo on laptop | `/Users/vaibhav.wudaru/llm-activation-steering-research` |
| Batch partition | `ghx4` · Interactive partition | `ghx4-interactive` (2 h cap) |
| Model cache | `export HF_HOME=$HOME/hf_cache` (keep off the home quota) |
| Allocation | ~312 GPU-hr — **always keep a tight `--time`** (a hang can't drain it) |

---

## 2. SLURM cheat-sheet (how to operate the cluster)

### Submit / watch / stop

```bash
sbatch deltaai/run_dct.slurm          # queue a job -> prints "Submitted batch job 123456"
squeue -u vwudaru                     # your queue. ST: PD=pending, R=running. EMPTY = all done.
scancel 123456                        # cancel one job   |   scancel -u vwudaru = cancel all yours
sacct -j 123456 --format=JobID,State,Elapsed,MaxRSS,ReqTRES%40   # after it ends: did it OK? how long?
```

### Read the logs (IMPORTANT gotcha)

Job output goes to `*_%j.out` (e.g. `dct_123456.out`) in the dir you submitted from. The logs look
**blank** because tqdm progress bars overwrite the line with carriage returns. Always read them with
`grep -a` (treat as text), never `tail -f` (it freezes your terminal and eats keystrokes):

```bash
grep -a -E "Job|Saved|done|Error|Traceback" dct_123456.out    # the real content
```

### The #SBATCH header, line by line (top of every `deltaai/*.slurm`)

```bash
#SBATCH --job-name=dct           # label in squeue
#SBATCH --account=bhhv-dtai-gh   # WHICH ALLOCATION to bill  (edit ACCOUNT_NAME -> bhhv-dtai-gh)
#SBATCH --partition=ghx4         # GH200 batch partition
#SBATCH --nodes=1 --gpus-per-node=1   # 1 node, 1 GPU (a 7B/2B model fits with huge headroom)
#SBATCH --cpus-per-task=16 --mem=64g  # CPU/RAM for the node
#SBATCH --time=00:30:00          # HARD wall cap — job is killed at this; protects the budget
#SBATCH --output=dct_%j.out      # log file (%j = job id)
```
Inside, the body does: `module load python/miniforge3_pytorch` → `source .venv-*/bin/activate` →
export `HF_HOME`/`TRANSFORMERS_OFFLINE=1` → `srun python3 src/....py --device cuda`.

### Interactive node (for debugging, capped 2 h)

```bash
srun -A bhhv-dtai-gh --partition=ghx4-interactive --nodes=1 --tasks=1 --tasks-per-node=1 \
  --cpus-per-task=8 --mem=32g --gpus-per-node=1 --time=01:00:00 --pty bash
# then inside the node:
module load python/miniforge3_pytorch && source .venv-dct-gpu/bin/activate
export HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
python3 src/run_dct_minimal.py --device cuda --num-factors 64 --max-iters 10
```

---

## 3. The whole pipeline: which script runs where, in what order

Dependencies flow top→bottom. **🖥️ = GPU job via SLURM**, **💻 = laptop CPU** (the geometry `.venv`).

| # | Script | Where | Reads | Writes | Run it with |
|---|---|---|---|---|---|
| 1 | `extract.py` | 💻 or 🖥️ | `got_datasets/<ds>.csv` + gemma | `acts_<ds>.npz` | `python src/extract.py cities.csv` |
| 2 | `run_dct_data.py` | 🖥️ | `got_datasets/<ds>.csv` + gemma | `dct_V_<ds>.pt` | via `run_dct.slurm` (Stage 3) |
| 3 | `export_truth_dir.py` | 💻 | `acts_<ds>.npz` | `truth_dir_<ds>.npz` | `python src/export_truth_dir.py --dataset cities` |
| 4 | `interpret_top10.py` | 🖥️ | `dct_V_<ds>.pt` + gemma | `interpret_top10_<ds>.md` | via `run_interpret.slurm` |
| 5 | `steer_supervised.py` | 🖥️ | `truth_dir_<ds>.npz` + gemma | `steer_supervised_<ds>.csv/.md` | via `run_steer.slurm` |
| 6 | `judge_results.py` | 🖥️ (judge env) | `steer_supervised_<ds>.csv`, `interpret_top10_<ds>.md` + OLMo | `judge_steer_*`, `plot_judge_*`, `judge_interpret_*` | via `run_judge.slurm` |
| 7 | `validate_judge.py` | 🖥️ (judge env) | `got_datasets/<ds>.csv` + OLMo | `judge_validation_<ds>.csv` (+ PASS/FAIL) | via `run_judge.slurm` (gate) |
| 8 | `analyze.py` | 💻 | `acts_<ds>.npz` | `results_<ds>.csv`, `directions_<ds>.csv`, `plot_<ds>_*.png` | `python src/analyze.py cities` |
| 9 | `compare_directions.py` | 💻 | `acts_<ds>.npz`, `dct_V_<ds>.pt` | `compare_<ds>.csv` | `python src/compare_directions.py --dataset cities` |
| 10 | `subspace_top_k.py` / `subspace_xgb.py` | 💻 | `acts_`, `dct_V_` | `subspace_*_<ds>.csv` | `python src/subspace_xgb.py --dataset cities` |
| 11 | `cross_dataset.py` | 💻 | `acts_`, `dct_V_` (both ds) | transfer CSV | `python src/cross_dataset.py` |
| 12 | `viz_*.py` | 💻 | the CSVs above | `plot_findings_*.png` | `python src/viz_findings.py` etc. |

**The GPU dependency chain (what forces the SLURM job order):**
`run_dct` → produces `dct_V_*` → **`run_interpret`** needs it. `export_truth_dir` (laptop) → ship up
→ **`run_steer`** needs it. **`run_judge`** needs the outputs of *both* `run_interpret` and
`run_steer`. So: **DCT → (interpret ∥ steer) → judge**.

> `dct_train.py` is a lower-level trainer (defaults to a Qwen model) used by the DCT internals — not
> part of the funnel run; ignore it unless you're changing DCT itself.

---

## 4. Ship code up / pull results back (rsync)

Run these from a **💻 laptop** terminal (prompt says `…@Vaibhavs-MacBook…`, not `vwudaru@gh-login01`).
Each `rsync`/`ssh` prompts for password + Duo (`1`, approve).

**Push the repo up** (excludes venvs, activations, big `.npz`, git):
```bash
rsync -av --exclude '.venv*' --exclude 'activations' --exclude '*.npz' --exclude '.git' \
  /Users/vaibhav.wudaru/llm-activation-steering-research/ \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```
**Push a specific file the excludes skip** (e.g. the truth-dir `.npz` the steer job needs):
```bash
rsync -av /Users/vaibhav.wudaru/llm-activation-steering-research/truth_dir_cities.npz \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```
**Pull results back** (quote the glob so it expands on the cluster):
```bash
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/judge_steer_*.csv' \
  'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/plot_judge_*.png' \
  /Users/vaibhav.wudaru/llm-activation-steering-research/
```

---

## 5. Full run, cold start to figures (the checklist)

### One-time (skip anything already done — see `MY_RUN_STEPS.md`)
1. 💻 Enroll NCSA Duo; confirm `ssh vwudaru@dtai-login.delta.ncsa.illinois.edu` works.
2. 🖥️ `bash deltaai/setup_env.sh` → builds `.venv-dct-gpu` (transformers 4.51.3).
3. 🖥️ `bash deltaai/setup_judge_env.sh` → builds `.venv-judge-gpu` (transformers 5.12.1).
4. 🖥️ Pre-download models on the **login node** (compute nodes are offline):
   ```bash
   export HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1
   source .venv-dct-gpu/bin/activate && hf auth login   # accept gemma license at hf.co/google/gemma-2-2b
   hf download google/gemma-2-2b
   source .venv-judge-gpu/bin/activate
   hf download allenai/Olmo-3-7B-Instruct               # open model, no gate
   ```

### Every run
5. 💻 Make the laptop-side inputs, then ship everything up:
   ```bash
   cd /Users/vaibhav.wudaru/llm-activation-steering-research
   .venv/bin/python src/extract.py cities.csv            # -> acts_cities.npz (if not already made)
   .venv/bin/python src/extract.py common_claim_true_false.csv
   .venv/bin/python src/export_truth_dir.py --dataset cities
   .venv/bin/python src/export_truth_dir.py --dataset common_claim_true_false
   # push repo (Section 4), then push the truth_dir_*.npz explicitly (Section 4 second recipe)
   ```
6. 🖥️ SSH in, set the account in **every** job script, submit in dependency order:
   ```bash
   ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
   cd ~/llm-activation-steering-research
   sed -i 's/ACCOUNT_NAME/bhhv-dtai-gh/' deltaai/run_dct.slurm deltaai/run_interpret.slurm \
       deltaai/run_steer.slurm deltaai/run_judge.slurm
   grep -- '--account' deltaai/*.slurm        # all four should read bhhv-dtai-gh

   sbatch deltaai/run_dct.slurm               # 1) DCT -> dct_V_*.pt  (edit Stage-3 lines on first real run)
   #   wait for it to finish (squeue empty), confirm: grep -a -E "DONE|Saved|Error" dct_*.out
   sbatch deltaai/run_interpret.slurm         # 2) needs dct_V_*.pt
   sbatch deltaai/run_steer.slurm             # 2) needs truth_dir_*.npz (parallel with interpret)
   #   wait for BOTH; confirm: grep -a -E "Saved|done|Error" interpret_*.out steer_*.out
   sbatch deltaai/run_judge.slurm             # 3) needs interpret + steer outputs; runs the validate gate first
   squeue -u vwudaru                          # re-run until empty
   grep -a -E "PASS|FAIL|Saved|done|Error" judge_*.out
   ```
7. 💻 Pull the results back (Section 4) — you want `judge_steer_*.csv`, `plot_judge_steering_*.png`,
   `judge_interpret_*.csv`, plus `interpret_top10_*.md` and `steer_supervised_*` for reading.
8. 💻 Run the laptop analysis + plots (need `acts_*` and the pulled `dct_V_*`):
   ```bash
   .venv/bin/python src/analyze.py cities
   .venv/bin/python src/compare_directions.py --dataset cities
   .venv/bin/python src/subspace_xgb.py --dataset cities
   .venv/bin/python src/cross_dataset.py
   .venv/bin/python src/viz_findings.py        # + viz_steer.py, viz_subspace_xgb.py, viz_funnel.py
   ```

---

## 6. Gotchas we already hit

- **`tail -f` freezes the terminal** — it never returns and swallows keystrokes. Use `grep -a`, or
  Ctrl-C to escape.
- **Logs look blank** — tqdm carriage returns; `grep -a -E "Saved|done|Job|PASS|FAIL"` shows the real lines.
- **`sed`/`cp` print nothing on success** — silence = worked; verify with `grep`.
- **`Error configuring interconnect`** — transient bad node; just `sbatch` again once.
- **`… No such file` for a `.npz` / `dct_V_*`** — a dependency wasn't shipped or a prior job didn't
  finish; re-check Section 3's chain, ship the missing file, resubmit.
- **`ssh -v` keepalive spam** — log in without `-v`.
- **Wrong env** — the judge job MUST source `.venv-judge-gpu` (transformers 5.x); the DCT/interpret/
  steer jobs MUST source `.venv-dct-gpu` (4.51.3). The scripts already do this — don't cross them.

## 7. Budget discipline (your PI's rule)

Every `deltaai/*.slurm` caps `--time` (30 min for the funnel/judge jobs, 2 h for DCT). Keep it tight:
a hung job is killed at the cap instead of draining the ~312 GPU-hr allocation. Each funnel/judge job
is only a few minutes of real GPU time. Check spend with `sacct -j <id> --format=JobID,Elapsed`.
