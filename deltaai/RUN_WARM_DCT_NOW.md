# Run Warm-DCT NOW — this exact instance

A copy-paste execution sheet for launching **the warm-started DCT run that is committed and ready on
branch `feat/mag-e4-steering` right now.** Everything generic has already been resolved for this
specific moment — no decisions, no placeholders. If you're reading this weeks later, prefer the
general [DCT_WARM_RUN.md](DCT_WARM_RUN.md); this one is a snapshot.

## State as of this session (verified, not assumed)

- **Code:** 9 warm-DCT commits on `feat/mag-e4-steering`, HEAD `1607381`. Tests green (13/13 warm +
  38/38 dct/funnel/steer).
- **Branch is NOT pushed** to `origin`. **This does not matter** — Step 1 ships code up by `rsync`,
  not `git pull`. Do **not** push just to run this.
- **All 8 input artifacts exist locally** (checked this session): `dct_meta_{cities,common_claim_true_false}.json`,
  `truth_dir_{…}.npz`, cold `dct_V_{…}.pt`, `dct_U_{…}.pt`. Step 1's guard will re-confirm.
- **Account:** `bhhv-dtai-gh`. **Login:** `vwudaru@dtai-login.delta.ncsa.illinois.edu`, NCSA
  password + Duo push (type `1`, approve) on every `ssh`/`rsync`.
- **⚠️ One thing to eyeball before you spend 2–4 GPU-hours:** the anchor is *scale-relative*
  (`V0 += λ·‖G_V[:,0]‖·seed`, commit `8d20fb0`) — my engineering fix after the plan's absolute anchor
  turned out inert. It was validated on real gemma (cos ladder λ{0.3,1,3} → {0.33,0.70,0.95}). This
  is the experiment's core knob; if you want to sign off on it first, say so before running.

---

## Step 1 — 💻 LAPTOP: confirm inputs, two-rsync up

```bash
cd ~/llm-activation-steering-research

# guard: all 8 inputs present + layer agreement (warm fit asserts this)
python3 -c "
import json, os, numpy as np
for ds in ['cities','common_claim_true_false']:
    assert os.path.exists(f'dct_meta_{ds}.json'), (ds,'dct_meta')
    assert os.path.exists(f'dct_V_{ds}.pt') and os.path.exists(f'dct_U_{ds}.pt'), (ds,'cold factors')
    t = np.load(f'truth_dir_{ds}.npz'); assert 'mean_diff' in t and 'layer' in t, (ds,'truth_dir')
    meta = json.load(open(f'dct_meta_{ds}.json'))
    assert meta['source_layer'] == int(t['layer']), (ds,'LAYER MISMATCH', meta['source_layer'], int(t['layer']))
    print(ds, 'ok — source_layer', meta['source_layer'])
"

# rsync #1 — code only (excludes the big caches rsync would otherwise upload)
rsync -av --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/

# rsync #2 — only the artifacts the warm job reads
rsync -av dct_meta_cities.json dct_meta_common_claim_true_false.json \
  truth_dir_cities.npz truth_dir_common_claim_true_false.npz \
  dct_V_cities.pt dct_V_common_claim_true_false.pt \
  dct_U_cities.pt dct_U_common_claim_true_false.pt \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```
✅ guard prints `ok — layer N` for both datasets, no `LAYER MISMATCH`.

---

## Step 2 — 🖥️ CLUSTER: log in, set account, submit train+steer

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research

# account is lifted from run_dct.slurm (kept intact by Step 1's exclude)
ACC=$(grep -o -- '--account=[^ ]*' deltaai/run_dct.slurm | head -1 | cut -d= -f2)
sed -i "s/ACCOUNT_NAME/$ACC/" deltaai/run_dct_warm.slurm deltaai/run_dct_warm_judge.slurm
grep -- --account deltaai/run_dct_warm.slurm deltaai/run_dct_warm_judge.slurm   # -> bhhv-dtai-gh, not ACCOUNT_NAME

sbatch deltaai/run_dct_warm.slurm
squeue -u vwudaru        # PD=queued, R=running; poll until EMPTY
```
This runs **16 warm fits** (2 datasets × 2 seeds `{mean_diff,grad}` × 4 λ `{0.0,0.3,1.0,3.0}`), then
per dataset assembles directions and steers. **Expected ~2–4 h** (`--time` cap 3 h).

Watch progress (don't `tail -f` a possibly-hung job):
```bash
grep -a -E "=== warm|wrote|=== assemble|=== warm steer|FAILED|done" dct_warm_*.out
```
✅ **Done when** `squeue` is empty and:
```bash
ls -la dct_warm_dirs_*.npz dct_warm_geometry_*.csv dct_warm_steer_*.csv   # 2 of each
```
> If one line reads `!!!! <ds> <seed> lam=<λ> FAILED — continuing`, that's tolerated: the other 15
> fits run and assembly skips the missing one. `raw_mean_diff`/`raw_grad`/`cold_top` never depend on
> warm fits. Save the `.out` and send it to me.

---

## Step 3 — 🖥️ CLUSTER: submit the judge (only after Step 2's CSVs exist)

```bash
sbatch deltaai/run_dct_warm_judge.slurm
squeue -u vwudaru        # poll until EMPTY
grep -a -E "=== warm judge|FAILED|done" dct_warm_judge_*.out
```
Judge env is `.venv-judge-gpu` (OLMo-3); output routes to `judge_dct_warm_steer_<ds>.csv` +
`plot_judge_dct_warm_<ds>.png` — won't touch MAG/length judge results. **Expected ~10–25 min**.
✅ **Done when:**
```bash
ls judge_dct_warm_steer_*.csv plot_judge_dct_warm_*.png   # 2 of each
```

---

## Step 4 — 💻 LAPTOP: pull results back, visualize

```bash
cd ~/llm-activation-steering-research
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{dct_warm_geometry_*.csv,dct_warm_steer_*.csv,judge_dct_warm_steer_*.csv,dct_warm_dirs_*.npz}' ./

PYTHONPATH=src .venv/bin/python src/viz_dct_warm.py --dataset cities
PYTHONPATH=src .venv/bin/python src/viz_dct_warm.py --dataset common_claim_true_false
```
Then tell me it's done — I'll build the warm-DCT figures and write up the section.

---

## What you're looking for in the results

Per dataset there are two artifacts that answer the experiment:

- **`dct_warm_geometry_<ds>.csv`** — is the anchor working as a knob? `cos(warm factor 0, seed)`
  should climb with λ (≈0 → 0.33 → 0.70 → 0.95). It also carries `cold_top` — whether cold DCT's own
  most-potent factor is truth-aligned (expected: no).
- **`judge_dct_warm_steer_<ds>.csv`** — the behavioral payoff, over the two-sided τ sweep
  `{-1,-0.6,-0.3,0,+0.3,+0.6,+1}`. **−τ pushes toward FALSE (lying), +τ toward TRUE.** Read the
  verdict fractions vs τ:
  - **−τ → more FALSE, sign-dependent** = the warm/anchored direction became a *real causal truth
    lever* — the headline positive result (the thing the raw supervised axis failed to be in §5 of the
    since-last-meeting doc).
  - **both ends → more INCOHERENT, symmetric** = still just a *degradation* lever, same null as before.
  - **flat** = inert.

The interesting comparison is **across λ**: does anchoring harder toward the supervised seed turn an
inert/degrading direction into a truth lever, or not? That contrast is the whole point of the run.
