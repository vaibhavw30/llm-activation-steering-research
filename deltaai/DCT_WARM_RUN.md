# Warm-Started DCT — Cluster Run

**Two jobs, one purpose:** train the 16 warm-started DCT fits (2 seeds — `mean_diff`, `grad` —
× 4 anchor λ — `0.0, 0.3, 1.0, 3.0` — × 2 datasets) on a DeltaAI GH200, assemble each seed×λ
combo into a named direction, steer gemma-2-2b with every direction, then have the OLMo judge
score TRUE/FALSE/INCOHERENT on the steered completions. Both datasets (`cities`,
`common_claim_true_false`) run in each job.

This doc is self-contained. Every command is copy-pasteable; there are **no placeholders**.

**Already done (nothing to set up):**
- All warm-DCT code + the job scripts `deltaai/run_dct_warm.slurm` and
  `deltaai/run_dct_warm_judge.slurm` are pushed to GitHub.
- The two cluster envs (`.venv-dct-gpu` for gemma warm-train/steer, `.venv-judge-gpu` for the
  OLMo judge), the model caches, and your Duo/password login all exist from the extraction and
  MAG E4 / length-steering runs.

**Login reminder:** DeltaAI uses your **NCSA password + a Duo push** on every `ssh`/`rsync`.
Being prompted twice is expected, not an error.

---

## Step 1 — 💻 LAPTOP: confirm the seed/cold artifacts, then rsync up

The warm job needs, for both datasets: `dct_meta_<ds>.json` (source-layer metadata), and
`truth_dir_<ds>.npz` (the `mean_diff` seed direction, at the source layer). It also needs the
**cold** DCT factors `dct_V_<ds>.pt` / `dct_U_<ds>.pt` — the warm fit reuses the cold `V` (per
`funnel_utils.load_dct`) and only re-anchors factor 0. Confirm all six files are present before
pushing anything up:

```bash
cd ~/llm-activation-steering-research
python3 -c "
import json, os
import numpy as np
for ds in ['cities', 'common_claim_true_false']:
    assert os.path.exists(f'dct_meta_{ds}.json'), (ds, 'dct_meta missing')
    assert os.path.exists(f'dct_V_{ds}.pt'), (ds, 'cold dct_V missing')
    assert os.path.exists(f'dct_U_{ds}.pt'), (ds, 'cold dct_U missing')
    t = np.load(f'truth_dir_{ds}.npz')
    assert 'mean_diff' in t and 'layer' in t, (ds, 'truth_dir')
    meta = json.load(open(f'dct_meta_{ds}.json'))
    print(ds, 'ok — dct_meta layer:', meta.get('layer'), '| truth_dir layer:', int(t['layer']))
"
```

✅ **Check (laptop):** prints `ok` for both `cities` and `common_claim_true_false`, and the
`dct_meta layer` matches the `truth_dir layer` for each dataset (the warm fit asserts this at
load time — a mismatch means the seed was exported at the wrong layer).

Then push the code and the seed/cold artifacts (one password + Duo push each):

```bash
rsync -av \
  --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude 'run_dct.slurm' --exclude '*.npz' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/

rsync -av \
  dct_meta_cities.json dct_meta_common_claim_true_false.json \
  truth_dir_cities.npz truth_dir_common_claim_true_false.npz \
  dct_V_cities.pt dct_V_common_claim_true_false.pt \
  dct_U_cities.pt dct_U_common_claim_true_false.pt \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

> The first rsync excludes **all** `*.npz` and `*.pt` — rsync does NOT honor `.gitignore`, so
> without this it uploads the ~1.5 GB of `mag_acts_*.npz` / `dct_V_*.pt` activation and factor
> files (most of which the warm job never reads) and appears to hang. The `dct_meta_*.json`,
> `truth_dir_*.npz`, and cold `dct_V_*.pt` / `dct_U_*.pt` files — the only seed/cold artifacts
> the warm job needs — go up via the **second** rsync instead. Excludes `run_dct.slurm` too,
> since Step 2 reads the account from the cluster's copy of it.

Then log in:

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research
```

✅ **Check (cluster):** `ls dct_meta_*.json truth_dir_*.npz dct_V_*.pt dct_U_*.pt` shows all
eight files (four per dataset).

---

## Step 2 — 🖥️ CLUSTER: put your account in both job scripts

The account name is copied automatically from the DCT script (kept intact by Step 1's exclude):

```bash
ACC=$(grep -o -- '--account=[^ ]*' deltaai/run_dct.slurm | head -1 | cut -d= -f2)
sed -i "s/ACCOUNT_NAME/$ACC/" deltaai/run_dct_warm.slurm deltaai/run_dct_warm_judge.slurm
grep -- --account deltaai/run_dct_warm.slurm deltaai/run_dct_warm_judge.slurm
```

✅ **Check:** both lines print your real account (e.g. `bXXX-dtai-gh`), **not** `ACCOUNT_NAME`.
If either still says `ACCOUNT_NAME`, run `accounts`, copy the DeltaAI/ghx4 account string, and
edit the `#SBATCH --account=` line by hand.

---

## Step 3 — 🖥️ CLUSTER: submit the warm train + steer job

```bash
sbatch deltaai/run_dct_warm.slurm
squeue -u $USER              # PD = queued, R = running
tail -f dct_warm_*.out       # live; Ctrl-C stops watching, not the job
```

What you'll see: per dataset, 8 `=== warm train <ds> <seed> lam=<λ> ===` blocks (2 seeds × 4 λ),
each ending with a `[warm] wrote dct_warm_V_<ds>_<seed>_<lamtag>.pt` line, then
`=== assemble directions <ds> ===` (`[dirs] wrote dct_warm_dirs_<ds>.npz ... and
dct_warm_geometry_<ds>.csv`), then `=== warm steer <ds> ===` (`[warm/steer] wrote
dct_warm_steer_<ds>.csv`).

**Expected: ~2–4 h** (16 warm fits — 2 seeds × 4 λ × 2 datasets — at `num_factors=64`, plus
direction assembly and steering, all on one GH200).

✅ **Done when:** the `.out` ends with `=== dct warm done ...` and
`ls -la dct_warm_dirs_*.npz dct_warm_steer_*.csv` shows **2 of each**.

---

## Step 4 — 🖥️ CLUSTER: submit the judge job

Only after Step 3's CSVs exist (the judge reads `dct_warm_steer_<ds>.csv`):

```bash
sbatch deltaai/run_dct_warm_judge.slurm
tail -f dct_warm_judge_*.out
```

The `--steer-input`/`--steer-output`/`--steer-plot` flags route output to
`judge_dct_warm_steer_<ds>.csv` + `plot_judge_dct_warm_<ds>.png`, so nothing overwrites the MAG
E4 or length-steering judge results.

**Expected: ~10–25 min** (40 min wall).

✅ **Done when:** the `.out` ends with `=== dct warm judge done ...` and
`ls judge_dct_warm_steer_*.csv plot_judge_dct_warm_*.png` shows 2 of each.

---

## Step 5 — 💻 LAPTOP: bring the results back, then visualize

```bash
cd ~/llm-activation-steering-research
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{dct_warm_geometry_*.csv,dct_warm_steer_*.csv,judge_dct_warm_steer_*.csv,dct_warm_dirs_*.npz}' ./

PYTHONPATH=src .venv/bin/python src/viz_dct_warm.py --dataset cities
PYTHONPATH=src .venv/bin/python src/viz_dct_warm.py --dataset common_claim_true_false
```

Then tell Claude it's done. Claude builds the warm-DCT figures and writes the results into the
relevant docs file.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `AssertionError` on layer mismatch during warm fit | `truth_dir_<ds>.npz`'s `layer` and `dct_meta_<ds>.json`'s `layer` disagree — the seed was exported at a different source layer than the cold DCT run used. Re-export `truth_dir_<ds>.npz` at the correct source layer on the laptop, redo the second rsync, resubmit. |
| `FileNotFoundError: dct_V_<ds>.pt` (cold factors missing) on cluster | The first rsync's `--exclude '*.pt'` dropped it and the second rsync didn't include it. Confirm `dct_V_cities.pt dct_V_common_claim_true_false.pt dct_U_cities.pt dct_U_common_claim_true_false.pt` are present locally, resend via Step 1's second rsync. |
| Job stuck `PD` for a long time | Normal queueing; `squeue -u $USER --start` for the estimate. |
| `.out` shows `TRANSFORMERS_OFFLINE` / cache error | Model cache missing on the env: `source .venv-dct-gpu/bin/activate && export HF_HOME=$HOME/hf_cache && hf download google/gemma-2-2b` (or the OLMo model for the judge env) on the **login** node, resubmit. |
| Judge job: `FileNotFoundError: dct_warm_steer_...csv` | Step 3 didn't finish, or only partially wrote its CSVs. Confirm `ls dct_warm_steer_*.csv` shows 2 files before submitting the judge. |
| One `<ds> <seed> lam=<λ>` block prints `!!!! ... FAILED — continuing` | The other 15 warm fits still run; `dct_warm_directions.py` will simply have fewer directions to assemble for that dataset. Send Claude the `.out` contents. |
| `!!!! steer <ds> FAILED — continuing` | Direction assembly for that dataset produced no usable directions, or the steer script errored. Check `dct_warm_geometry_<ds>.csv` exists and is non-empty before resubmitting just the steer step by hand. |
| One dataset prints `!!!! <ds> FAILED — continuing` (judge job) | The other still runs. Send Claude the `.out` contents. |
