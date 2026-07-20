# Length-Steering — Cluster Run

**Two jobs, one purpose:** run the long-generation length-steering behavioral eval on a DeltaAI
GH200 — inject the calibrated `mean_diff` and `resid_pc1` directions into gemma-2-2b at three
taus (0, 0.3, 1) over 96-token generations, log the intrinsic signals (entropy/max_prob/rep3) at
each cutoff, then have the OLMo judge score TRUE/FALSE/INCOHERENT on the judged prefixes. Both
datasets (`cities`, `common_claim_true_false`) run in each job.

This doc is self-contained. Every command is copy-pasteable; there are **no placeholders**.

**Already done (nothing to set up):**
- All length-steering code + the job scripts `deltaai/run_length_steer.slurm` and
  `deltaai/run_length_judge.slurm` are pushed to GitHub.
- The two cluster envs (`.venv-dct-gpu` for gemma gen, `.venv-judge-gpu` for the OLMo judge),
  the model caches, and your Duo/password login all exist from the extraction and MAG E4 runs.

**Login reminder:** DeltaAI uses your **NCSA password + a Duo push** on every `ssh`/`rsync`.
Being prompted twice is expected, not an error.

---

## Step 1 — 💻 LAPTOP: confirm the direction files, then rsync up

The gen job needs `mag_dir_<ds>.npz` (carries `resid_pc1_unit`, `A_prefix_norm`, `layer`) and
`truth_dir_<ds>.npz` (carries `mean_diff`, `layer`) for both datasets already sitting in the repo
root. Confirm they're present and hold the right keys before pushing anything up:

```bash
cd ~/llm-activation-steering-research
python3 -c "
import numpy as np
for ds in ['cities', 'common_claim_true_false']:
    m = np.load(f'mag_dir_{ds}.npz')
    t = np.load(f'truth_dir_{ds}.npz')
    assert 'resid_pc1_unit' in m and 'A_prefix_norm' in m and 'layer' in m, (ds, 'mag_dir')
    assert 'mean_diff' in t and 'layer' in t, (ds, 'truth_dir')
    print(ds, 'ok — mag_dir keys:', list(m.keys()), '| truth_dir keys:', list(t.keys()))
"
```

✅ **Check (laptop):** prints `ok` for both `cities` and `common_claim_true_false` with no
`AssertionError`.

Then push the code and the four direction files (one password + Duo push each):

```bash
rsync -av \
  --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude 'run_dct.slurm' --exclude '*.npz' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/

rsync -av mag_dir_cities.npz mag_dir_common_claim_true_false.npz \
  truth_dir_cities.npz truth_dir_common_claim_true_false.npz \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

> The first rsync excludes **all** `*.npz` — rsync does NOT honor `.gitignore`, so without this it
> uploads the ~1.5 GB of `mag_acts_*.npz` activation files (which the steer job never reads) and
> appears to hang. The `mag_dir_*.npz` / `truth_dir_*.npz` direction files — the only npz the job
> needs — go up via the **second** rsync instead. Excludes `run_dct.slurm` too, since Step 2 reads
> the account from the cluster's copy of it.

Then log in:

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research
```

✅ **Check (cluster):** `python3 -c "import numpy as np; d=np.load('mag_dir_cities.npz'); print('resid_pc1_unit' in d, 'A_prefix_norm' in d)"`
prints `True True`.

---

## Step 2 — 🖥️ CLUSTER: put your account in both job scripts

The account name is copied automatically from the DCT script (kept intact by Step 1's exclude):

```bash
ACC=$(grep -o -- '--account=[^ ]*' deltaai/run_dct.slurm | head -1 | cut -d= -f2)
sed -i "s/ACCOUNT_NAME/$ACC/" deltaai/run_length_steer.slurm deltaai/run_length_judge.slurm
grep -- --account deltaai/run_length_steer.slurm deltaai/run_length_judge.slurm
```

✅ **Check:** both lines print your real account (e.g. `bXXX-dtai-gh`), **not** `ACCOUNT_NAME`.
If either still says `ACCOUNT_NAME`, run `accounts`, copy the DeltaAI/ghx4 account string, and edit
the `#SBATCH --account=` line by hand.

---

## Step 3 — 🖥️ CLUSTER: submit the steering job

```bash
sbatch deltaai/run_length_steer.slurm
squeue -u $USER              # PD = queued, R = running
tail -f length_steer_*.out   # live; Ctrl-C stops watching, not the job
```

**Expected: up to ~2 h wall** (two datasets × two directions × three taus × 96-token generations).
Log out freely.

✅ **Done when:** the `.out` ends with `=== length steer done ...` and
`ls -la length_steer_*.csv length_prefixes_*.csv` shows **2 of each**.

---

## Step 4 — 🖥️ CLUSTER: submit the judge job

Only after Step 3's CSVs exist (the judge reads `length_prefixes_<ds>.csv`):

```bash
sbatch deltaai/run_length_judge.slurm
tail -f length_judge_*.out
```

The `--steer-input`/`--steer-output`/`--steer-plot` flags route output to
`judge_length_<ds>.csv` + `plot_judge_length_<ds>.png`, so nothing overwrites the MAG E4 or
supervised judge results.

**Expected: ~10–25 min** (40 min wall).

✅ **Done when:** the `.out` ends with `=== length judge done ...` and
`ls judge_length_*.csv plot_judge_length_*.png` shows 2 of each.

---

## Step 5 — 💻 LAPTOP: bring the results back, then visualize

```bash
cd ~/llm-activation-steering-research
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{length_steer_*.csv,length_prefixes_*.csv,judge_length_*.csv}' ./

PYTHONPATH=src .venv/bin/python src/viz_length.py --dataset cities
PYTHONPATH=src .venv/bin/python src/viz_length.py --dataset common_claim_true_false
```

Then tell Claude it's done. Claude builds the length-steering figures and writes the results
into the relevant docs file.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `KeyError: 'resid_pc1_unit'` in `length_steer_*.out` | The stale `mag_dir` went up (or didn't go up at all). Redo Step 1's check + the second rsync, resubmit. |
| Job stuck `PD` for a long time | Normal queueing; `squeue -u $USER --start` for the estimate. |
| `.out` shows `TRANSFORMERS_OFFLINE` / cache error | Model cache missing on the env: `source .venv-dct-gpu/bin/activate && export HF_HOME=$HOME/hf_cache && hf download google/gemma-2-2b` (or the OLMo model for the judge env) on the **login** node, resubmit. |
| Judge job: `FileNotFoundError: length_prefixes_...csv` | Step 3 didn't finish, or only partially wrote its CSVs. Confirm `ls length_prefixes_*.csv` shows 2 files before submitting the judge. |
| `srun: error: ... Error configuring interconnect` | Neither script uses `srun` (both call `python3` directly), so this shouldn't recur; if it does, check for a stray `srun` prefix before resubmitting. |
| One dataset prints `!!!! <ds> FAILED — continuing` | The other still runs. Send Claude the `.out` contents. |
