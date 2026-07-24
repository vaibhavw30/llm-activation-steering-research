# Backward-Reachability Audit — DeltaAI Run (this instance)

Concrete, copy-paste runbook for **this** repo and **this** cluster account. Every placeholder is
already filled — no `ACCOUNT_NAME`, no `USER`, no `PROJECT_DIR`. Read
[CLUSTER_OPERATIONS.md](CLUSTER_OPERATIONS.md) Parts 1–4 once for the machinery (login, the two envs,
SLURM, the two-rsync pattern); the phase-**decision** logic (which phase becomes the headline) lives
in [../docs/REACHABILITY_RUNBOOK.md](../docs/REACHABILITY_RUNBOOK.md). This doc is the operational
recipe: exact commands, exact order, exact checks.

Symbols: **💻 = laptop terminal** · **🖥️ = cluster (after `ssh`)**. Every `ssh`/`rsync` prompts for
**NCSA password + Duo push** (type `1`, approve on phone) — twice per batch is expected.

---

## Your coordinates (this instance)

| | value |
|---|---|
| Login host | `dtai-login.delta.ncsa.illinois.edu` |
| User / Project | `vwudaru` / `CIS260948` |
| **`--account`** (every job) | **`bhhv-dtai-gh`** |
| Repo (both sides) | `~/llm-activation-steering-research` |
| Partition | `ghx4` (batch) |
| Compute env | `.venv-dct-gpu` (transformers 4.51.3) — margins/svd/steer/jlens/linerr |
| Judge env | `.venv-judge-gpu` (transformers 5.12.1) — the OLMo-3 judge |
| Model cache | `export HF_HOME=$HOME/hf_cache` (gemma pre-downloaded, gated) |
| Datasets / hops | `cities` (11 → 20) · `common_claim_true_false` (13 → 22) |

**The five jobs** (all in `deltaai/`, all account-`sed`'d in Phase 2):

| Job | `--time` cap | env | Reads | Writes |
|---|---|---|---|---|
| `run_reach_margins.slurm` | 06:00 | dct | inputs below | `reach_acts_*.npz`, `reach_dirs_*.npz`, `reach_margins_*.npz` |
| `run_reach_svd.slurm` | 04:00 | dct | acts + dirs + dct_V/U + truth_dir | `reach_svd_*/`, `reach_svd_energy_*.csv`, `reach_svd_summary_*.csv` |
| `run_reach_steer.slurm` | 05:00 | dct | **+ `reach_summary_*.json`** (analyze first) | `reach_steer_*.csv`, `reach_steer_readout_*.csv`, `reach_steer_stmt_*.csv`, `reach_jlens_*.csv` |
| `run_reach_linerr.slurm` | 02:00 | dct | acts + dirs + landmarks | `reach_linerr_*.csv`, `reach_linerr_summary_*.csv` |
| `run_reach_judge.slurm` | 02:00 | **judge** | `reach_steer_*.csv`, `reach_steer_stmt_*.csv` | `judge_reach_steer_*.csv`, `judge_reach_steer_stmt_*.csv`, `plot_judge_reach_*.png` |

---

## Phase 0 — one-time cluster prerequisites (🖥️ login node, has internet)

**(a) scikit-learn into the DCT env.** `.venv-dct-gpu` ships without sklearn (`setup_env.sh` installs
only transformers/scipy/tqdm/pandas), but `reach_margins --stage dirs` and `reach_jlens` both call
`fit_probe_dir` / `fit_threshold`, which need it. Install once:

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
python -c "import sklearn" 2>/dev/null && echo "sklearn already present" || pip install scikit-learn
python -c "import sklearn, sys; print('sklearn', sklearn.__version__)"    # confirm
```

**(b) gemma cached?** The margins/svd/steer/linerr jobs run offline (`TRANSFORMERS_OFFLINE=1`). If
`google/gemma-2-2b` isn't already in `$HF_HOME` for this env, pre-download it now (see
CLUSTER_OPERATIONS Part 2 — needs `hf auth login` + the one-click license accept). The DCT/MAG runs
already did this; skip unless a job later dies with a cache/offline error.

---

## Phase 0b — laptop smoke + input check (💻, before any rsync)

Confirm the pipeline is green locally and every input artifact `validate_inputs` demands exists
(6 per dataset). **`validate_inputs` fails fast** on a missing file or a layer mismatch, so this check
is exactly what the cluster jobs enforce at startup.

```bash
cd ~/llm-activation-steering-research

# 1) unit suite for the reach modules (pure-python; ~5 s)
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_reach_hop.py tests/test_reach_margins.py tests/test_reach_analyze.py \
  tests/test_reach_linerr.py tests/test_viz_reach.py -q

# 2) the six required inputs per dataset (+ the src/tgt layer-match guard)
PYTHONPATH=src .venv/bin/python - <<'PY'
import reach_hop as rh
for ds in ("cities", "common_claim_true_false"):
    rh.validate_inputs(ds)          # raises SystemExit on any missing file / layer mismatch
    src, tgt, scale = rh.load_meta(ds)
    print(f"{ds}: OK  src={src} tgt={tgt} input_scale={scale:.4g}")
PY

# 3) OPTIONAL end-to-end CPU dry-run on 8 statements (throwaway — delete before rsync)
PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage acts --limit 8 --device cpu
PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage dirs
PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage vjp  --limit 8 --device cpu
PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset cities
PYTHONPATH=src .venv/bin/python src/viz_reach.py    --dataset cities
# then delete the throwaway --limit 8 artifacts so the cluster regenerates them at full size:
rm -f reach_acts_cities.npz reach_dirs_cities.npz reach_margins_cities.npz \
      reach_curve_cities.csv reach_summary_cities.json plot_reach_*_cities.png
rm -rf reach_chunks_cities
```

Expected: step 1 all pass; step 2 prints `cities: OK src=11 tgt=20 …` and
`common_claim_true_false: OK src=13 tgt=22 …`.

---

## Phase 1 — rsync up (💻, two batches)

**rsync #1 — code** (excludes big binaries; `got_datasets/*.csv` travels here):

```bash
cd ~/llm-activation-steering-research
rsync -av \
  --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

**rsync #2 — the reach inputs** (exactly what `validate_inputs` + the battery + landmarks read;
2 files per dataset for each name):

```bash
rsync -av \
  dct_meta_cities.json dct_meta_common_claim_true_false.json \
  truth_dir_cities.npz truth_dir_common_claim_true_false.npz \
  truth_dir_tgt_cities.npz truth_dir_tgt_common_claim_true_false.npz \
  mag_dir_cities.npz mag_dir_common_claim_true_false.npz \
  dct_V_cities.pt dct_V_common_claim_true_false.pt \
  dct_U_cities.pt dct_U_common_claim_true_false.pt \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

> `dct_V_*.pt` / `dct_U_*.pt` are likely already on the cluster from the DCT runs. If
> `ssh … 'ls ~/llm-activation-steering-research/dct_V_*.pt'` lists them, you may drop the two `.pt`
> lines from rsync #2. `truth_dir_tgt_*.npz` come from Project C's `export_target_dir.py`
> (gitignored) — they must be sent; `validate_inputs` requires them and checks their `layer` == tgt.

---

## Phase 2 — cluster jobs (🖥️, GH200)

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research

# (i) fill the account into all five reach jobs (lifted from the un-rsync'd run_dct.slurm)
ACC=$(grep -o -- '--account=[^ ]*' deltaai/run_dct.slurm | head -1 | cut -d= -f2)
sed -i "s/ACCOUNT_NAME/$ACC/" \
  deltaai/run_reach_margins.slurm deltaai/run_reach_svd.slurm \
  deltaai/run_reach_steer.slurm deltaai/run_reach_linerr.slurm deltaai/run_reach_judge.slurm
grep -H -- --account deltaai/run_reach_*.slurm       # must all print bhhv-dtai-gh, not ACCOUNT_NAME
```

### Step A — margins (gates everything)

```bash
sbatch deltaai/run_reach_margins.slurm
squeue -u vwudaru                                    # wait until EMPTY (acts+dirs+vjp, both datasets)
grep -a -E "===|wrote|FAILED|Traceback" reach_margins_*.out
ls -la reach_acts_*.npz reach_dirs_*.npz reach_margins_*.npz    # expect 2 of each
```

### Step B — svd + linerr (independent of analyze; can run concurrently)

```bash
sbatch deltaai/run_reach_svd.slurm
sbatch deltaai/run_reach_linerr.slurm
squeue -u vwudaru                                    # wait until EMPTY
grep -a -E "===|wrote|WARNING|FAILED|Traceback" reach_svd_*.out reach_linerr_*.out
ls reach_svd_summary_*.csv reach_linerr_summary_*.csv           # expect 2 of each
```

### Step C — analyze on the login node (seconds; produces the summary steer needs)

`reach_steer` reads `reach_summary_<ds>.json`, produced by `reach_analyze`. sklearn is installed
(Phase 0a) and the margins npz are already on the cluster, so run analyze right here:

```bash
module load python/miniforge3_pytorch && source .venv-dct-gpu/bin/activate
export HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH=src python3 src/reach_analyze.py --dataset cities
PYTHONPATH=src python3 src/reach_analyze.py --dataset common_claim_true_false
ls reach_summary_*.json reach_curve_*.csv                       # expect 2 of each
```

### Step D — steer + jlens, then judge

```bash
sbatch deltaai/run_reach_steer.slurm
squeue -u vwudaru                                    # wait until EMPTY
grep -a -E "===|wrote|FAILED|Traceback" reach_steer_*.out
ls reach_steer_*.csv reach_steer_stmt_*.csv reach_jlens_*.csv   # expect the steer/readout/stmt/meta + jlens CSVs

sbatch deltaai/run_reach_judge.slurm                 # ONLY after the steer CSVs exist
squeue -u vwudaru                                    # wait until EMPTY
grep -a -E "===|wrote|FAILED|Traceback" reach_judge_*.out
ls judge_reach_steer_*.csv judge_reach_steer_stmt_*.csv plot_judge_reach_*.png
```

**Hard ordering:** judge reads the steer CSVs — only `sbatch` it after `squeue` is empty **and** the
`reach_steer_*.csv` exist. Each `!!!! <ds> … FAILED — continuing` line means one dataset failed but
the loop went on; grab that `.out` and read the traceback for that dataset.

---

## Phase 3 — rsync results back (💻, one batch; excludes weights)

```bash
cd ~/llm-activation-steering-research
rsync -av \
  'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{reach_acts_*.npz,reach_dirs_*.npz,reach_margins_*.npz,reach_curve_*.csv,reach_summary_*.json,reach_svd_summary_*.csv,reach_svd_energy_*.csv,reach_steer_*.csv,reach_steer_readout_*.csv,reach_steer_stmt_*.csv,reach_jlens_*.csv,reach_linerr_*.csv,judge_reach_*.csv,plot_judge_reach_*.png}' \
  ./
```

> `reach_acts_*.npz` and `reach_margins_*.npz` are ~200 MB/dataset (jtw rows are float16 to keep this
> small) — needed locally because `reach_analyze` and `viz_reach` read them. If bandwidth hurts, drop
> `reach_acts_*.npz` from the glob and keep `reach_summary`/`reach_curve` (already produced in Step C)
> — but `viz_reach.fig_margins`/`fig_geometry` will then need them re-sent.

---

## Phase 4 — local analysis + figures (💻)

Analyze already ran on the cluster (Step C), so locally just (re-)summarize linerr and render:

```bash
cd ~/llm-activation-steering-research
# (re)generate the Phase-5 validity-radius summary locally (or trust the cluster's copy)
PYTHONPATH=src .venv/bin/python src/reach_linerr.py --dataset cities --summarize
PYTHONPATH=src .venv/bin/python src/reach_linerr.py --dataset common_claim_true_false --summarize

# figures: margins, reachability curves (+ Phase-5 trust-region overlay), source geometry, svd, jlens
PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset cities
PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset common_claim_true_false
ls plot_reach_*_cities.png plot_reach_*_common_claim_true_false.png
```

`viz_reach.fig_curves` overlays `reach_linerr_summary_<ds>.csv`'s `jtw_mean_diff_tgt` `eps20` as the
trust region **only if that file is present** — so run the `--summarize` lines before the viz lines.

---

## Decision gates between phases

The *interpretation* logic (which phase is the headline) is in
[../docs/REACHABILITY_RUNBOOK.md](../docs/REACHABILITY_RUNBOOK.md) "Decision gates". In brief, read
these after the run:

- **Gate P1** — `reach_summary_<ds>.json` `verdict`: `unreachable-tail` ⇒ Phase 2 (mechanism) is the
  story; `reachable-candidate` ⇒ Phase 3 (steering) is the headline experiment.
- **Gate P2** — `reach_svd_summary_<ds>.csv` mean `dctV_overlap_top16` **< 0.3** is a
  STOP-AND-DIAGNOSE (the `run_reach_svd.out` prints a `WARNING` line): the linearized SVD picture
  disagrees with DCT's V; don't trust P1 margins until resolved.
- **Gate P3** — the 2×2 from `judge_reach_steer_*.csv` × `reach_steer_readout_*.csv`: readout moves &
  behavior moves ⇒ lever found; readout moves & behavior doesn't ⇒ LiSeCo-style dissociation.
- **Gate P5** — restate every P1–P3 claim inside vs outside the `reach_linerr_summary` validity
  radius; if the radius ≪ `input_scale`, phrase the (un)reachability claim at the radius.

---

## GPU-budget note

Allocation is ~312 GPU-hr; every job caps `--time` (2–6 h) — **never remove the cap**, a hang is
killed at the wall instead of draining the budget. Real spend is well under each cap (margins is the
long one: full acts extraction + vjp over the ~74-direction battery for both datasets). Check with
`sacct -j <id> --format=JobID,State,Elapsed`.

## Reach-specific gotchas

| Symptom | Cause / fix |
|---|---|
| Job dies in **seconds** with `[reach] input validation FAILED` | A file missing from rsync #2 (esp. `truth_dir_tgt_*.npz`). Send it, resubmit. |
| `[reach] layer mismatch` at startup | `truth_dir_*` / `truth_dir_tgt_*` `layer` ≠ `dct_meta` src/tgt. You uploaded a stale direction file; rebuild/re-send the matching one. |
| `ModuleNotFoundError: sklearn` in `reach_margins`/`reach_jlens` | Phase 0a not done in `.venv-dct-gpu`. `pip install scikit-learn` on the login node, resubmit. |
| `run_reach_steer` crashes reading `reach_summary_*.json` | Step C (analyze) not run before submitting steer. Run analyze, resubmit. |
| `.out` looks blank | tqdm carriage returns — read with `grep -a -E "===\|wrote\|FAILED"`. |
| `reach_svd` prints a `WARNING … disagrees with DCT's V` | Not a crash — it's **Gate P2** firing. Diagnose before trusting P1. |
