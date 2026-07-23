# DeltaAI Cluster Operations — the master guide

**The one doc to read before touching the cluster.** It covers the reusable machinery once
(login, the two envs, SLURM, the two-rsync pattern, the gotchas, budget discipline) and then indexes
every experiment's own runbook with its exact submit commands. Read Parts 1–4 once; jump to Part 5
for whichever experiment you're running.

> **Why this doc and not `CLUSTER_WALKTHROUGH.md`?** That one is still accurate for the original
> *funnel* (DCT → interpret/steer → judge), but it predates the newer experiments (MAG extraction,
> MAG E4, length-steering, warm-DCT). This doc is the current superset and points back to it for the
> funnel detail.

---

## Part 1 — Mental model (read once)

- **Two machines.** Your **💻 laptop** (Apple M-series) does the cheap analysis, direction-building,
  and plotting. **🖥️ DeltaAI** (NCSA GH200: ARM64 CPU + ~96 GB Hopper GPU) does the heavy model
  forward passes — DCT fits, gemma generation/steering, the 7B OLMo judge. You move files between
  them with `rsync`.
- **You never run compute on the login node.** You `sbatch` a job script; SLURM finds a free GPU
  node, runs it, and writes a log. The **login node** is only for editing, `rsync`, `sbatch`, and
  pre-downloading models (it has internet; **compute nodes do not**).
- **Everything persists** across SSH sessions: your home dir, both venvs, and the `$HOME/hf_cache`
  model cache. You redo setup only if something's missing.
- **Delivery is by `rsync` from the laptop, not `git pull`.** Every runbook ships code up with
  `rsync ./`. You do **not** need the branch pushed to GitHub for a cluster run. (Some runbooks say
  "pushed to GitHub" in their preamble — that's aspirational; the actual mechanism is the rsync in
  their Step 1.)

### Your coordinates

| | value |
|---|---|
| Login host | `dtai-login.delta.ncsa.illinois.edu` |
| Auth | **NCSA password + Duo push** — type `1`, approve on phone — **on every `ssh`/`rsync`** (twice is expected) |
| User / Project | `vwudaru` / `CIS260948` |
| **Account** (`--account` on every job) | **`bhhv-dtai-gh`** |
| Repo (both sides) | `~/llm-activation-steering-research` |
| Batch partition | `ghx4` · Interactive | `ghx4-interactive` (2 h cap) |
| Model cache | `export HF_HOME=$HOME/hf_cache` |
| Allocation | ~312 GPU-hr — **always keep a tight `--time`**; a hang can't drain it |

---

## Part 2 — The two Python environments

Both are layered on the **same** GH200 torch module — never `pip install torch` (the ARM+CUDA build
comes from the module). Pick the env by *what the job runs*:

| env | transformers | used by | why separate |
|---|---|---|---|
| `.venv-dct-gpu` | **4.51.3** (pinned) | DCT fits, gemma **generation & steering** (warm-DCT, MAG E4, length, interpret, supervised steer, extraction) | 4.51.3 is required for DCT's `SlicedModel` |
| `.venv-judge-gpu` | **5.12.1** | the OLMo-3 judge + `validate_judge` | OLMo-3 needs transformers ≥5; upgrading the DCT env would break DCT |

Every `*.slurm` already sources the right one — **don't cross them**. A judge job in the DCT env (or
vice-versa) fails on the model load.

**One-time build** (login node, has internet — skip if the venvs already exist):
```bash
bash deltaai/setup_env.sh          # -> .venv-dct-gpu   (transformers 4.51.3)
bash deltaai/setup_judge_env.sh    # -> .venv-judge-gpu (transformers 5.12.1)
```

**One-time model pre-download** (login node — compute nodes are offline):
```bash
export HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1
source .venv-dct-gpu/bin/activate && hf auth login   # accept gemma license at hf.co/google/gemma-2-2b
hf download google/gemma-2-2b
source .venv-judge-gpu/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/Olmo-3-7B-Instruct')"
```
(gemma is gated → needs `hf auth login` + a one-click license accept; OLMo is open, no gate.)

---

## Part 3 — SLURM cheat-sheet

### Submit / watch / stop
```bash
sbatch deltaai/run_<job>.slurm       # queue -> "Submitted batch job 123456"
squeue -u vwudaru                    # ST: PD=pending, R=running. EMPTY = done.
squeue -u vwudaru --start            # estimated start time while PD
scancel 123456                       # cancel one   |   scancel -u vwudaru = cancel all yours
sacct -j 123456 --format=JobID,State,Elapsed,MaxRSS,ReqTRES%40   # after it ends
```

### Reading logs — the #1 gotcha
Output goes to `<name>_<jobid>.out` in the dir you submitted from. Logs look **blank** because tqdm
progress bars overwrite the line with carriage returns. Read them with `grep -a` (treat as text).
`tail -f` is fine for a live job you're babysitting, but it **freezes the terminal** and eats
keystrokes if the job hung — prefer:
```bash
grep -a -E "===|wrote|done|Saved|Error|Traceback|FAILED|PASS|FAIL" <name>_<jobid>.out
```

### The `#SBATCH` header (top of every `deltaai/*.slurm`)
```bash
#SBATCH --account=ACCOUNT_NAME     # <- sed-filled to bhhv-dtai-gh in each runbook's Step 2
#SBATCH --partition=ghx4           # GH200 batch partition
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --cpus-per-task=16 --mem=64g
#SBATCH --time=03:00:00            # HARD wall cap — killed at this; protects the budget
#SBATCH --output=<name>_%j.out
```
Body pattern: `module load python/miniforge3_pytorch` → `source .venv-*/bin/activate` → export
`HF_HOME` / `HF_HUB_DISABLE_XET=1` / `TRANSFORMERS_OFFLINE=1` → `PYTHONPATH=src python3 src/….py --device cuda`.

### Filling the account (every runbook's Step 2, copy-paste)
The account is lifted automatically from `run_dct.slurm` (which the runbooks' rsync deliberately does
*not* overwrite):
```bash
ACC=$(grep -o -- '--account=[^ ]*' deltaai/run_dct.slurm | head -1 | cut -d= -f2)
sed -i "s/ACCOUNT_NAME/$ACC/" deltaai/run_<jobA>.slurm deltaai/run_<jobB>.slurm
grep -- --account deltaai/run_<jobA>.slurm deltaai/run_<jobB>.slurm   # must print bhhv-dtai-gh, not ACCOUNT_NAME
```

### Interactive node (debugging only, 2 h cap)
```bash
srun -A bhhv-dtai-gh --partition=ghx4-interactive --nodes=1 --tasks=1 --tasks-per-node=1 \
  --cpus-per-task=8 --mem=32g --gpus-per-node=1 --time=01:00:00 --pty bash
# inside:
module load python/miniforge3_pytorch && source .venv-dct-gpu/bin/activate
export HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
```

---

## Part 4 — Moving files (the two-rsync pattern)

Run from a **💻 laptop** terminal. Each `rsync`/`ssh` prompts for password + Duo.

**Why two rsyncs.** rsync does **not** honor `.gitignore`. The repo root holds ~1.5 GB+ of
`*_acts_*.npz` activation caches and `dct_*.pt` factor files that the steering/judge jobs never read
— send them and the transfer appears to hang. So:

1. **First rsync — the code**, excluding all big binaries:
```bash
rsync -av \
  --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```
   (Excludes `run_dct.slurm` so the cluster's copy — the account source for Step 2 — stays intact.)

2. **Second rsync — only the artifacts this job actually needs**, named explicitly (varies per
   experiment; see Part 5). Example (warm-DCT):
```bash
rsync -av \
  dct_meta_*.json truth_dir_*.npz dct_V_*.pt dct_U_*.pt \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

**Pull results back** (quote the glob so it expands on the cluster):
```bash
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{judge_*_*.csv,plot_*_*.png}' ./
```

---

## Part 5 — The experiments (each has a self-contained runbook)

Every experiment is a **steer job → judge job** pair (the funnel is the exception, below). All follow
the identical shape: laptop builds/confirms inputs → two-rsync up → sed the account → `sbatch` the
steer job → wait for its CSVs → `sbatch` the judge job → pull results → visualize on the laptop.

| Experiment | Runbook | Steer/gen job → Judge job | Needs uploaded (2nd rsync) | Key outputs | Rough GPU time |
|---|---|---|---|---|---|
| **Warm-started DCT** | [DCT_WARM_RUN.md](DCT_WARM_RUN.md) | `run_dct_warm.slurm` → `run_dct_warm_judge.slurm` | `dct_meta_*.json`, `truth_dir_*.npz`, cold `dct_V_*.pt`/`dct_U_*.pt` | `dct_warm_dirs_*.npz`, `dct_warm_geometry_*.csv`, `dct_warm_steer_*.csv`, `judge_dct_warm_steer_*.csv` | train+assemble+steer ~2–4 h; judge ~25 min |
| **MAG E4 steering** | [MAG_E4_RUN.md](MAG_E4_RUN.md) | `run_mag_steer.slurm` → `run_mag_judge.slurm` | `mag_dir_*.npz` (rebuilt fresh first) | `mag_steer_*.csv`, `mag_verdict_flips_*.csv`, `judge_mag_steer_*.csv` | steer ~15–40 min; judge ~25 min |
| **Length-steering** | [LENGTH_STEER_RUN.md](LENGTH_STEER_RUN.md) | `run_length_steer.slurm` → `run_length_judge.slurm` | `mag_dir_*.npz`, `truth_dir_*.npz` | `length_steer_*.csv`, `length_prefixes_*.csv`, `judge_length_*.csv` | steer ~2 h; judge ~25 min |
| **MAG extraction** | [MAG_EXTRACT_RUN.md](MAG_EXTRACT_RUN.md) | `run_mag_extract.slurm` (single job) | `got_datasets/*.csv` (travel with the repo) | `mag_acts_*.npz` | ~tens of min |
| **The original funnel** | [CLUSTER_WALKTHROUGH.md](CLUSTER_WALKTHROUGH.md) §5, [FUNNEL_RUN_STEPS.md](FUNNEL_RUN_STEPS.md), [JUDGE_RUN_STEPS.md](JUDGE_RUN_STEPS.md) | `run_dct` → (`run_interpret` ∥ `run_steer`) → `run_judge` | `truth_dir_*.npz` (for steer) | `dct_V_*.pt`, `interpret_top10_*.md`, `steer_supervised_*.csv`, `judge_steer_*`, `judge_interpret_*` | DCT ~1–2 h; rest minutes |

**Ordering within a pair is a hard dependency:** the judge job reads the steer job's CSV, so only
`sbatch` the judge **after** `squeue` is empty and the steer CSVs exist. Each judge script routes its
output to an experiment-specific prefix (`--steer-output judge_<exp>_…` / `--mag`), so **runs never
overwrite each other's judge results.**

### Worked example — warm-DCT, end to end

```bash
# 💻 LAPTOP — confirm the 8 input artifacts exist (4 per dataset), then two-rsync up
cd ~/llm-activation-steering-research
python3 -c "
import json, os, numpy as np
for ds in ['cities','common_claim_true_false']:
    assert os.path.exists(f'dct_meta_{ds}.json') and os.path.exists(f'dct_V_{ds}.pt') and os.path.exists(f'dct_U_{ds}.pt')
    t=np.load(f'truth_dir_{ds}.npz'); assert 'mean_diff' in t and 'layer' in t
    print(ds,'ok, layer', json.load(open(f'dct_meta_{ds}.json'))['layer'], int(t['layer']))"
rsync -av --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
rsync -av dct_meta_*.json truth_dir_*.npz dct_V_*.pt dct_U_*.pt \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/

# 🖥️ CLUSTER — account, submit train+steer, wait, submit judge
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research
ACC=$(grep -o -- '--account=[^ ]*' deltaai/run_dct.slurm | head -1 | cut -d= -f2)
sed -i "s/ACCOUNT_NAME/$ACC/" deltaai/run_dct_warm.slurm deltaai/run_dct_warm_judge.slurm
sbatch deltaai/run_dct_warm.slurm
squeue -u vwudaru                                        # wait until empty
grep -a -E "===|wrote|FAILED" dct_warm_*.out            # sanity
ls -la dct_warm_dirs_*.npz dct_warm_steer_*.csv         # expect 2 of each
sbatch deltaai/run_dct_warm_judge.slurm
squeue -u vwudaru                                        # wait until empty
ls judge_dct_warm_steer_*.csv                           # expect 2

# 💻 LAPTOP — pull back, visualize
cd ~/llm-activation-steering-research
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{dct_warm_geometry_*.csv,dct_warm_steer_*.csv,judge_dct_warm_steer_*.csv,dct_warm_dirs_*.npz}' ./
PYTHONPATH=src .venv/bin/python src/viz_dct_warm.py --dataset cities
PYTHONPATH=src .venv/bin/python src/viz_dct_warm.py --dataset common_claim_true_false
```

For the other experiments the shape is identical — swap the script names, the 2nd-rsync artifacts,
and the viz command per the table above and the experiment's own runbook.

---

## Part 6 — Gotchas we've already hit

| Symptom | Cause / fix |
|---|---|
| Log file looks **blank** | tqdm carriage returns. `grep -a -E "===\|wrote\|done\|FAILED"` shows the real lines. |
| `tail -f` **freezes the terminal** | It never returns on a hung job. Ctrl-C to escape; use `grep -a`. |
| `sed` / `cp` print nothing | Silence = success. Verify with the following `grep`. |
| `Error configuring interconnect` | Transient bad node, only on `srun`-wrapped jobs (the judge). `sbatch` again; if it recurs, drop the `srun ` prefix from the `python3` line. |
| `.out` shows `TRANSFORMERS_OFFLINE` / cache error | Model not in `$HF_HOME` for that env. On the **login** node: `source .venv-<env>/bin/activate && export HF_HOME=$HOME/hf_cache && hf download <model>`, resubmit. |
| `FileNotFoundError` for a `.npz`/`.pt`/`.csv` on the cluster | A dependency wasn't in the 2nd rsync, or the prior job didn't finish. Confirm the file exists locally, resend it explicitly, resubmit. |
| Job stuck `PD` a long time | Normal queueing. `squeue -u vwudaru --start` for the estimate. |
| One `!!!! <ds>/<combo> FAILED — continuing` line | By design the loop continues; the other datasets/fits still run and assembly tolerates missing warm fits. Grab the `.out` and read the traceback for that one combo. |
| Wrong env error on model load | Judge job MUST be `.venv-judge-gpu` (transformers 5.x); everything else `.venv-dct-gpu` (4.51.3). The scripts set this — don't hand-edit. |

---

## Part 7 — Budget discipline (the PI's rule)

Every `*.slurm` caps `--time` (1–4 h depending on the job). Keep it tight: a hung job is killed at
the cap instead of draining the ~312 GPU-hr allocation. Real GPU time per job is minutes to a couple
hours. Check spend with `sacct -j <id> --format=JobID,Elapsed`. Never remove the `--time` cap.
