# DO THIS NOW — run the OLMo-3 judge on DeltaAI (Task 3)

Your exact steps for **this** run. Goal: produce the judged **FALSE-vs-INCOHERENT** steering curves
and the **0/10 interpret count** on the GH200 (the 24 GB laptop swap-thrashes on a 7B model).

**State going in (why these steps):**
- The judge code is on `main`/GitHub but **not on the cluster yet** → Step 1 ships it up.
- `.venv-judge-gpu` and the OLMo model **don't exist on the cluster yet** → Steps 3–4 are one-time.
- The judge's inputs (`steer_supervised_*.csv`, `interpret_top10_*.md`) already exist on your laptop
  from the earlier funnel run → the Step-1 rsync carries them up too (they're not `.npz`, so not excluded).

**Coordinates:** User `vwudaru` · Account `bhhv-dtai-gh` · Login
`dtai-login.delta.ncsa.illinois.edu` (NCSA password → Duo, type `1`, approve on phone).
**Legend:** 💻 = a terminal on your Mac · 🖥️ = a terminal SSH'd into DeltaAI. Every command is one
line so it pastes cleanly. If a step's output differs from "**You should see**", stop and fix it
before continuing.

---

## STEP 1 — 💻 LAPTOP — ship the new code + the judge inputs up

```bash
rsync -av --exclude '.venv*' --exclude 'activations' --exclude '*.npz' --exclude '.git' /Users/vaibhav.wudaru/llm-activation-steering-research/ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```
- Prompts for **password → Duo** (`1`, approve).
- **You should see:** a file list that includes `deltaai/run_judge.slurm`, `deltaai/setup_judge_env.sh`,
  `src/validate_judge.py`, `src/judges/olmo_judge.py`, `steer_supervised_cities.csv`,
  `interpret_top10_cities.md`, then a `sent … bytes` summary. ✅
  If it printed rsync's *usage/help* text, the line broke on paste — redo as one line.

## STEP 2 — 💻 LAPTOP — log in

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
```
- password → Duo `1` → approve.
- **You should see:** the DeltaAI banner and your prompt become `vwudaru@gh-login01:~>`. ✅ (now 🖥️)

## STEP 3 — 🖥️ CLUSTER — build the judge env (one-time, ~2–3 min)

```bash
cd ~/llm-activation-steering-research && bash deltaai/setup_judge_env.sh
```
- **You should see:** it creates `.venv-judge-gpu`, installs `transformers==5.12.1` + accelerate +
  matplotlib, then prints a verify block. Confirm these lines: **`CUDA available: True`** and
  **`transformers: 5.12.1`** and **`transformers 5.x import OK`**. ✅
- ❗ If it says CUDA not available, that's fine here (login node has no GPU) — what matters is
  `transformers: 5.12.1` imported cleanly. The GPU check re-runs inside the job on a compute node.
- ❗ If pip errors that transformers 5.12.1 needs a newer torch than the module provides, tell me —
  we pick a newer `module load` or pin a compatible transformers.

## STEP 4 — 🖥️ CLUSTER — download the judge model on the login node (~14 GB, one-time)

Compute nodes have no internet, so cache it here first. OLMo-3 is **open — no license/token needed**.
Use the library directly (the `hf` CLI isn't on PATH in a plain venv shell — it lives in the module):
```bash
source .venv-judge-gpu/bin/activate && export HF_HOME=$HOME/hf_cache && export HF_HUB_DISABLE_XET=1 && python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/Olmo-3-7B-Instruct')"
```
- **You should see:** download progress bars, ending back at the prompt with the files under
  `$HOME/hf_cache/hub/models--allenai--Olmo-3-7B-Instruct`. ✅ (~14 GB, a few minutes)
- ❗ If you'd rather use the `hf` CLI: run `module load python/miniforge3_pytorch` first, then
  `hf download allenai/Olmo-3-7B-Instruct`.

## STEP 5 — 🖥️ CLUSTER — confirm the judge inputs are present

```bash
ls -la steer_supervised_cities.csv steer_supervised_common_claim_true_false.csv interpret_top10_cities.md interpret_top10_common_claim_true_false.md
```
- **You should see:** all four files listed (they came up in Step 1). ✅
- ❗ If any is **missing**: the judge can't run without it. Fastest fix — re-ship just those from the
  laptop (💻 new tab): `rsync -av /Users/vaibhav.wudaru/llm-activation-steering-research/steer_supervised_* /Users/vaibhav.wudaru/llm-activation-steering-research/interpret_top10_* vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/`

## STEP 6 — 🖥️ CLUSTER — set your account in the job script

```bash
sed -i 's/ACCOUNT_NAME/bhhv-dtai-gh/' deltaai/run_judge.slurm && grep -- '--account' deltaai/run_judge.slurm
```
- **You should see:** `#SBATCH --account=bhhv-dtai-gh` (not `ACCOUNT_NAME`). ✅

## STEP 7 — 🖥️ CLUSTER — submit the job

```bash
sbatch deltaai/run_judge.slurm
```
- **You should see:** `Submitted batch job <NNNNNNN>`. ✅ **Write down that job number.**
- The job runs, in order: a 24-row plumbing smoke → the **validation gate** (aborts if the judge
  can't hit 0.85 on gold cities labels) → the full cities + common_claim steer sweeps → both
  interpret runs. Capped at `--time=00:30:00`.

## STEP 8 — 🖥️ CLUSTER — wait, then confirm it passed

Re-run until the list is **empty** (a few minutes; `ST`: `PD`=pending, `R`=running):
```bash
squeue -u vwudaru
```
Then read the real log lines (use `grep -a` — the log looks blank due to progress bars; **never**
`tail -f`, it freezes the terminal):
```bash
grep -a -E "PASS|FAIL|accuracy|saved plot_judge|wrote judge|Error|Traceback" judge_*.out
```
- **You should see:** an **`accuracy … -> PASS`** line from the gate, then `wrote judge_steer_*.csv`,
  `saved plot_judge_steering_*.png`, and `wrote judge_interpret_*.csv`. ✅
- ❗ If you see **`-> FAIL`**: the gate stopped the run on purpose — the OLMo judge disagreed with
  gold labels too often. Don't trust any curves; send me the confusion table from the log and we
  decide the fallback. (Itself a finding worth telling your PI.)
- ❗ `Error configuring interconnect` → transient bad node; just `sbatch deltaai/run_judge.slurm` again.

## STEP 9 — 💻 LAPTOP — pull the results back

New 💻 tab (⌘T), or `exit` the cluster first. Prompt must say `…@Vaibhavs-MacBook…`:
```bash
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/judge_steer_*.csv' 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/plot_judge_steering_*.png' 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/judge_interpret_*.csv' /Users/vaibhav.wudaru/llm-activation-steering-research/
```
- password → Duo `1`. **You should see:** the two `judge_steer_*.csv`, two
  `plot_judge_steering_*.png`, and two `judge_interpret_*.csv` transfer. ✅

## STEP 10 — 💻 LAPTOP — read the result (what it means)

```bash
open plot_judge_steering_cities.png plot_judge_steering_common_claim_true_false.png
```
Look at the **negative-scale (away-from-truth)** side of each plot:
- **FALSE line rises** as scale goes negative → steering the truth direction made the model **lie** →
  a *real causal* truth effect. This is the strong result.
- **Only INCOHERENT rises**, FALSE stays flat → steering just **degraded** the model, didn't flip
  truth → the effect is coherence, not a clean truth axis (the caveat the keyword heuristic couldn't resolve).

And the interpret count:
```bash
grep -a "manipulate" judge_*.out    # or open judge_interpret_cities.csv
```
- **0/10** top DCT vectors manipulate truthfulness → the null is legitimate (top causal levers are
  geography/format/tone, not truth). **≥1/10** → look at which vector, tell me.

Then tell me the numbers and I'll fold them into the funnel writeup and update the "Test 1/2 were
hand-read/keyword-scored" caveats to the quantitative judged versions.

---

## RERUN (corrected design) — one chained submission

After the first run we found the judge was scoring gemma's rambling completions whole (a correct
"four…" got marked FALSE on its tail). The fix is committed: 32 factual prompts (was 8), 8-token
completions, and an answer-only judge rubric (see `docs/INVESTIGATION_steering_validity.md`). To rerun
end-to-end with **one** command that regenerates the steer sweep and auto-judges it when it succeeds:

```bash
# 🖥️ CLUSTER, in the repo, after `git pull` and setting the account in both scripts:
bash deltaai/submit_rerun.sh
```
- Chains `run_steer.slurm` → `--dependency=afterok` → `run_judge.slurm` (judge starts only if steer
  succeeds; if steer fails, SLURM auto-cancels the judge).
- It re-judges the existing `interpret_top10_*.md` too (DCT vectors unchanged — only the rubric
  changed), so the interpret count is recomputed without a new DCT job.
- Prints both job IDs, the `squeue`/`grep -a` watch commands, and the pull-back rsync. Then Steps 9–10
  above are unchanged.
