# Task 3 — Run the OLMo-3 LLM-as-judge on DeltaAI (GH200)

Makes funnel **Test 1** (interpret top-10 DCT vectors) and **Test 2** (supervised steering) quantitative:
- **steer mode** → `judge_steer_<ds>.csv` + `plot_judge_steering_<ds>.png` — the FALSE-vs-INCOHERENT
  curve that separates *"steering made the model lie"* (causal) from *"it just degraded to gibberish"*.
- **interpret mode** → `judge_interpret_<ds>.csv` — the 0/10 "manipulates truthfulness" count.

Judge model: **`allenai/Olmo-3-7B-Instruct`** — fully open, no license gate, fp16 ≈14 GB (trivial on a
96 GB GH200). We run it here rather than on the 24 GB laptop, which swap-thrashes on a 7B model.

> **Why not the DCT env?** `.venv-dct-gpu` pins `transformers==4.51.3` (critical for DCT's SlicedModel).
> OLMo-3 needs `transformers>=5`. So the judge gets its **own** venv, `.venv-judge-gpu`, on the same
> torch module. The two envs coexist; nothing about the DCT setup changes.

---

## Prerequisite — the input files must already be on the cluster

The judge **consumes** the funnel outputs (they are gitignored, so they don't travel with `git clone` —
they're produced on the cluster by the earlier jobs):

| Judge mode | Needs in repo root | Produced by |
|---|---|---|
| steer | `steer_supervised_cities.csv`, `steer_supervised_common_claim_true_false.csv` | `run_steer.slurm` |
| interpret | `interpret_top10_cities.md`, `interpret_top10_common_claim_true_false.md` | `run_interpret.slurm` |

If you already ran those funnel jobs, the files are there — check with `ls steer_supervised_*.csv
interpret_top10_*.md`. If not, `sbatch deltaai/run_steer.slurm` and `deltaai/run_interpret.slurm` first
(or `rsync` the four files up from the laptop — they're small).

## Step 1 — One-time judge env (on a login node)

```bash
bash deltaai/setup_judge_env.sh
```
Creates `.venv-judge-gpu` (inherits the module torch via `--system-site-packages`) and installs
`transformers==5.12.1` + `accelerate` + `matplotlib`. It prints a torch/transformers compatibility
check — confirm `CUDA available: True` and `transformers: 5.12.1` before continuing.

## Step 2 — Pre-download the judge model (login node has internet; compute nodes don't)

```bash
source .venv-judge-gpu/bin/activate
export HF_HOME=$HOME/hf_cache          # same cache the DCT jobs use
export HF_HUB_DISABLE_XET=1
hf download allenai/Olmo-3-7B-Instruct # ~14 GB; NO gate/token needed (open model)
```

## Step 3 — Edit and submit the judge job

```bash
# set your allocation:  #SBATCH --account=ACCOUNT_NAME  ->  (sacctmgr show assoc user=$USER)
sbatch deltaai/run_judge.slurm
```
The job runs, in order: a 24-row **plumbing smoke** → the **validation gate**
(`src/validate_judge.py`: judges 100 gold cities statements and aborts if accuracy < 0.85, so you
never build curves from a judge that can't tell true from false) → the full cities + common_claim
steer sweeps → both interpret runs. Watch it:
```bash
squeue --me
tail -f judge_<jobid>.out
```

## Step 4 — Pull the results back to the laptop

```bash
# from your laptop:
rsync -av <you>@login.deltaai.ncsa.illinois.edu:/work/<...>/llm-activation-steering-research/'judge_steer_*.csv plot_judge_steering_*.png judge_interpret_*.csv' ./
```
These are the PI deliverables: the FALSE-vs-INCOHERENT plots and the interpret count. They're gitignored,
so they won't clobber anything and won't get committed.

## Sanity checks on the output (before trusting the numbers)

- **Gate passed?** In `judge_<jobid>.out`, the validation gate prints an `accuracy … -> PASS/FAIL`
  line and a confusion table. PASS (>= 0.85 on gold cities labels) is your license to trust the
  curves; on FAIL the job aborts and you fall back to another backend or note the caveat. Run it
  standalone anytime: `python3 src/validate_judge.py --backend olmo --dataset cities --device cuda`.
- **Expected shape:** for a *real* causal truth effect, the `−` (away-from-truth) side of the steer plot
  should raise **FALSE**, not just **INCOHERENT**. Mostly INCOHERENT at the extremes = degradation, not lying.
- **interpret:** `n/10 manipulate truthfulness` — the thesis expects **0/10** (top DCT vectors are
  geography/format/tone, not a truth knob).
