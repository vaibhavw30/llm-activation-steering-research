# DO THIS NOW — the corrected steering rerun (chained steer → judge)

Step-by-step for the **corrected** rerun after the validity investigation
(`docs/INVESTIGATION_steering_validity.md`). One chained submission regenerates the steer sweep with
the fixed design and auto-judges it.

**What changed since the first run (already committed — `0f9dc41`):**
- `steer_supervised.py`: **8 → 32 factual prompts** (statistical power) and **`--max-new-tokens 24 → 8`**
  (the completion is now the *answer*, not a rambling paragraph the judge misjudged).
- `judge_results.py` `STEER_SYS`: **answer-only** rubric (judge the direct answer, ignore the tail).
- `judge_results.py` `INTERPRET_SYS`: **strict flip-only** rubric (manipulates_truth=true ONLY if it
  negates an established fact, e.g. Paris→Germany).
- `submit_rerun.sh`: chains `run_steer.slurm` → `--dependency=afterok` → `run_judge.slurm`.

**What's already on the cluster from the first run (persists — do NOT redo):** `.venv-dct-gpu`,
`.venv-judge-gpu`, gemma + OLMo cached in `$HF_HOME`, `truth_dir_*.npz`, `interpret_top10_*.md`.
So this rerun skips env build + model download — it's just: ship code → submit → pull.

**Coordinates:** `vwudaru` · account `bhhv-dtai-gh` · login `dtai-login.delta.ncsa.illinois.edu`
(password → Duo `1` → approve). **Legend:** 💻 laptop · 🖥️ cluster. Commands are single-line.

---

## STEP 1 — 💻 LAPTOP — ship the corrected code up

```bash
rsync -av --exclude '.venv*' --exclude 'activations' --exclude '*.npz' --exclude '.git' /Users/vaibhav.wudaru/llm-activation-steering-research/ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```
- password → Duo `1`.
- **You should see:** a file list including `deltaai/submit_rerun.sh`, `deltaai/run_steer.slurm`,
  `src/steer_supervised.py`, `src/judge_results.py`, then a `sent … bytes` line. ✅
  (If it printed rsync's usage text, the line broke on paste — redo as one line.)

## STEP 2 — 💻 LAPTOP — log in

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
```
- **You should see:** the banner and prompt `vwudaru@gh-login01:~>`. ✅ (now 🖥️)

## STEP 3 — 🖥️ CLUSTER — set your account in BOTH job scripts

`run_judge.slurm` may already be set from the first run; set both to be safe (the submit script
refuses to run if either still says `ACCOUNT_NAME`):
```bash
cd ~/llm-activation-steering-research && sed -i 's/ACCOUNT_NAME/bhhv-dtai-gh/' deltaai/run_steer.slurm deltaai/run_judge.slurm && grep -- '--account' deltaai/run_steer.slurm deltaai/run_judge.slurm
```
- **You should see:** `#SBATCH --account=bhhv-dtai-gh` for **both** files. ✅

## STEP 4 — 🖥️ CLUSTER — submit the chained rerun

```bash
bash deltaai/submit_rerun.sh
```
- The script guards the account + inputs, then submits both jobs.
- **You should see:** `submitted STEER job:  <A>` and `submitted JUDGE job:  <B>   (afterok:<A> …)`,
  then the watch/pull-back commands. ✅ **Write down A and B.**
- ❗ If it errors `missing truth_dir_*.npz` or `missing interpret_top10_*.md`: those inputs aren't on
  the cluster. Re-ship from the laptop (💻 new tab):
  `rsync -av /Users/vaibhav.wudaru/llm-activation-steering-research/truth_dir_*.npz /Users/vaibhav.wudaru/llm-activation-steering-research/interpret_top10_*.md vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/`
  then re-run Step 4.

## STEP 5 — 🖥️ CLUSTER — watch the chain run

```bash
squeue --me
```
- **You should see:** the STEER job `R` (running), the JUDGE job with REASON `(Dependency)` — it waits
  for steer. When steer finishes, the judge flips `(Dependency)` → `PD` → `R` on its own. ✅
- Re-run until the list is **empty** (both done; a few minutes each on the GH200).
- (Don't `tail -f` — it freezes the terminal. Use the grep in Step 6.)

## STEP 6 — 🖥️ CLUSTER — confirm steer saved and the judge passed the gate

```bash
grep -a -E "Saved|Error|Traceback" steer_<A>.out ; grep -a -E "PASS|FAIL|accuracy|saved plot_judge|wrote judge|Error" judge_<B>.out
```
- **You should see:** two `Saved steer_supervised_*.md` lines, then an **`accuracy … -> PASS`** line
  from the gate, then `wrote judge_steer_*.csv`, `saved plot_judge_steering_*.png`,
  `wrote judge_interpret_*.csv`. ✅
- ❗ If steer FAILED, SLURM auto-cancelled the judge (afterok). Diagnose: `sacct -j <A>` and
  `grep -a -E "Error|truth_dir|Traceback" steer_<A>.out`.
- ❗ If the gate printed `-> FAIL`: the judge disagreed with gold cities labels — send me the
  confusion table; we pick a fallback before trusting anything.

## STEP 7 — 💻 LAPTOP — pull the corrected results back

New 💻 tab (⌘T) or `exit` first; prompt must say `…@Vaibhavs-MacBook…`:
```bash
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/judge_steer_*.csv' 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/plot_judge_steering_*.png' 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/judge_interpret_*.csv' /Users/vaibhav.wudaru/llm-activation-steering-research/
```
- **You should see:** the two `judge_steer_*.csv`, two `plot_judge_steering_*.png`, two
  `judge_interpret_*.csv` transfer. ✅
- ⚠️ This **overwrites** the first-run files locally — intended (v2 supersedes v1; v1's findings are
  already captured in `docs/INVESTIGATION_steering_validity.md`).

## STEP 8 — 💻 LAPTOP — what the corrected results should show

```bash
cd /Users/vaibhav.wudaru/llm-activation-steering-research && open plot_judge_steering_*.png
```
Then tell me the numbers — I'll re-run the same statistical investigation on the corrected data. What
we're checking (predictions from the investigation):
- **Baseline (scale 0) should be almost all TRUE now** (short answers, answer-only rubric) — vs the
  ~50% TRUE artifact last time. If scale 0 isn't ~all TRUE, the instrument still isn't clean; stop and
  tell me.
- **The null on lie-asymmetry should *tighten*** — with 32 prompts we can now *bound* it (rule out a
  FALSE-asymmetry Δ ≳ 0.13), not just fail to find one.
- **The degradation trend (INCOHERENT ↑ with |scale|) may now reach significance** — that's the real,
  weak effect.
- **interpret should read 0/10** manipulate truthfulness (strict flip-only rubric), matching the
  hand drill-in: `grep -a "manipulate" judge_<B>.out`.

Then I fold the corrected numbers into `docs/INVESTIGATION_steering_validity.md` and the funnel writeup.
