# Refusal Positive Control — DeltaAI Run (Track A)

Copy-paste runbook for the refusal positive control: A1 model screen, A2 dataset prep
(already done locally — see Phase 0), A3/A4 extraction + meta + calibration, P1/P3 reach
margins + steering + judging, and A8's verdict. Plan:
[../docs/superpowers/plans/2026-07-29-horizon1-refusal-sae.md](../docs/superpowers/plans/2026-07-29-horizon1-refusal-sae.md)
(Horizon-1 §1.1). Coordinates, envs,
and the `SlicedModel` gotcha are identical to
[REACH_RUN.md](REACH_RUN.md)/[REACH_H0_RUN.md](REACH_H0_RUN.md) — account `bhhv-dtai-gh`,
user `vwudaru@dtai-login.delta.ncsa.illinois.edu`, partition `ghx4`, `.venv-dct-gpu`
(transformers 4.51.3) for everything except the OLMo spot-check, which needs
`.venv-judge-gpu`.

**Why this run exists.** The truth run (cities, common_claim) produced a dissociation:
the certificate was locally exact yet steering past eps\* moved the readout and left
completions alone (`readout-only`). That is consistent with either "the instrument is
broken" or "truth is not a behaviorally actuatable variable in gemma-2-2b" — only a
concept with a *known* behavioral handle can tell those apart. Refusal (Arditi et al.,
arXiv:2406.11717) is that concept, run through the identical code path, hook, and eps\*
arithmetic. **Polarity:** `refusal.csv` uses label 1 = harmless, label 0 = harmful, so the
certificate describes *inducing* refusal on harmless instructions (full headroom; the
opposite direction needs a model that refuses unprompted, which a base model may not do —
`reach_steer` sweeps ± scales so we get that direction for free without betting on it).
**Linearization point:** `--prompt-mode full` puts the certificate's linearization point at
the prompt's own final token — a refusal instruction *is* the generation prompt, so the
per-statement arm here does not need the "chop the last word" trick that created the
Horizon-0 context-shift confound on truth.

**The four cluster jobs:**

| Job | `--time` | Runs | Writes |
|---|---|---|---|
| `run_refusal_screen.slurm` | 00:40 | A1: `refusal_screen.py` on both `google/gemma-2-2b` and `google/gemma-2-2b-it`, unsteered | `refusal_screen_google_gemma_2_2b.csv`, `refusal_screen_google_gemma_2_2b_it.csv`, the `DECISION` lines |
| `run_refusal_prep.slurm` | 03:00 | A3 extract, A4 layer-sweep meta, calibration, direction exports | `activations/acts_refusal.npz`, `dct_meta_refusal.json`, `truth_dir_refusal.npz`, `truth_dir_tgt_refusal.npz` |
| `run_refusal_reach.slurm` | 08:00 | smoke, P1 margins + analyze, P3 mean + per-statement steering, substring judging, A8 verdict | `reach_acts/dirs/margins_refusal.npz`, `reach_curve_refusal.csv`, `reach_summary_refusal.json`, `reach_steer*_refusal*.csv`, `judge_refusal_refusal_{mean,stmt}.csv`, `reach_control_refusal_{mean,stmt}.{csv,json}` |
| `run_refusal_spotcheck.slurm` | 01:30 | OLMo spot-check re-scoring (`.venv-judge-gpu`) | `judge_refusal_spotcheck_refusal_{mean,stmt}.csv` |

---

## Phase 0 — laptop, before anything else

```bash
cd ~/llm-activation-steering-research
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q      # 399 passed, 1 skipped (as of 2026-08-27)
.venv/bin/python src/prep_refusal.py                     # needs internet (AdvBench + Alpaca)
ls -l got_datasets/refusal.csv got_datasets/refusal_holdout.csv
```

`refusal.csv` is the balanced (label 1 = harmless / label 0 = harmful) fit set;
`refusal_holdout.csv` never enters direction fitting — it supplies both the A1 screening
prompts and the mean-arm generation prompts, so behavioral evaluation stays leakage-free.
If these two files already exist from a prior local run, re-running `prep_refusal.py` is
harmless (deterministic, `seed=42`) but not required.

## Phase 1 — rsync up (laptop)

Same exclude list as `REACH_H0_RUN.md` Phase 1. The two new files are CSVs under
`got_datasets/`, so the existing command carries them without modification.

```bash
cd ~/llm-activation-steering-research
rsync -av \
  --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

## Phase 2 — the model gate (cluster) — **STOP HERE**

`run_refusal_screen.slurm` screens *both* candidates in one job, so the
`google/gemma-2-2b-it` license must be accepted at
huggingface.co/google/gemma-2-2b-it before you submit it, even if you expect to keep the
base model.

### Phase 2a — stage the `-it` weights on the LOGIN node (one time, required)

Compute nodes are offline and all four jobs now run with `TRANSFORMERS_OFFLINE=1`, so a
checkpoint that is not already in `$HF_HOME` cannot be fetched from inside a job. The base
model is staged from the truth runs; `google/gemma-2-2b-it` almost certainly is not. Skip
this and the screen job burns a queue cycle and exits 1 on a missing `-it` CSV.

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research
source .venv-dct-gpu/bin/activate
export HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1
hf auth login          # only if the -it license was just accepted; gated repo needs a token
hf download google/gemma-2-2b-it
```

Verify both checkpoints resolve offline before spending a queue slot. This is the exact
check the job does, run on the login node where a failure costs nothing:

```bash
HF_HOME=$HOME/hf_cache TRANSFORMERS_OFFLINE=1 python3 - <<'EOF'
from transformers import AutoTokenizer, AutoConfig
for m in ("google/gemma-2-2b", "google/gemma-2-2b-it"):
    try:
        AutoTokenizer.from_pretrained(m); AutoConfig.from_pretrained(m)
        print(f"OK      {m}")
    except Exception as e:
        print(f"MISSING {m}: {type(e).__name__}")
EOF
```

Two `OK` lines means Phase 2b can proceed. A `MISSING` line means `hf download` did not
complete for that model, and the screen job would fail the same way.

Disk: the `-it` checkpoint is another ~5 GB in `$HF_HOME`. Check `du -sh $HOME/hf_cache`
and your quota before downloading.

### Phase 2b — submit the screen

```bash
sed -i "s/ACCOUNT_NAME/bhhv-dtai-gh/" deltaai/run_refusal_*.slurm
grep -h -- --account deltaai/run_refusal_*.slurm     # verify before submitting — must print bhhv-dtai-gh, not ACCOUNT_NAME
sbatch deltaai/run_refusal_screen.slurm
squeue -u vwudaru        # wait until EMPTY
```

Read both `DECISION INPUT` lines and the final `DECISION:` line in
`refusal_screen_*.out`, and apply the pre-registered rule:

**Also note the harmless-prompt refusal rate**, printed on both `DECISION INPUT` lines and
again in the job's `[decision] harmless-prompt refusal rate (REPORTING ONLY, gates
nothing)` line. It gates nothing, but it is the number that decides how to read the
verdict eight hours later: `REFUSAL_MARKERS` counts `"as an ai"` / `"as a language
model"`, which are benign openers on harmless Alpaca instructions (especially on `-it`).
If the chosen model's harmless rate is well above 0, expect a nonzero `frac=0` baseline in
`reach_control` and a real possibility that a large perturbation *lowers* the rate — which
still earns `actuatable`, since `moved` thresholds `|delta|`. Write the number down now,
and see "The sign of the deciding frac" below.

- **harmful-prompt refusal rate for `google/gemma-2-2b` >= 0.10** -> keep the base model.
  This is the preferred outcome: identical model to the truth run, so the contrast has no
  model confound.
  ```bash
  sbatch deltaai/run_refusal_prep.slurm
  ```
- **< 0.10** -> fall back to the instruct model. `google/gemma-2-2b` is a BASE model that
  may never refuse anything at all, so this is an expected, not a failure, outcome.
  Rebuild the dataset **on the laptop** with the chat template applied, then re-rsync and
  submit with an environment override:
  ```bash
  # laptop
  .venv/bin/python src/prep_refusal.py --chat-template google/gemma-2-2b-it
  rsync -av got_datasets/refusal.csv got_datasets/refusal_holdout.csv \
    vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/got_datasets/
  ```
  ```bash
  # cluster
  REFUSAL_MODEL=google/gemma-2-2b-it sbatch deltaai/run_refusal_prep.slurm
  ```
  **Do not re-submit `run_refusal_screen.slurm` after the `-it` re-prep.** That job calls
  `refusal_screen.py --chat-template`, and `refusal_screen.chat_wrap` would apply the
  template *again* to the now-already-templated `got_datasets/refusal_holdout.csv` —
  double-templating every prompt (a nested `<start_of_turn>user` inside a user turn) and
  producing a refusal rate that measures nothing. The screen has already served its
  purpose at this point; the gate is decided. If you must re-screen after the re-prep,
  drop `--chat-template`, because the holdout file already carries the template.

  Templating happens **once, in `prep_refusal.py`'s `apply_template`, at dataset-build
  time** — never at generation time. `extract.py`, `reach_margins.py`, and
  `reach_steer.py` must all tokenize the *identical* string, or the certificate's
  linearization point stops being the prompt's own last token, reintroducing the exact
  context-shift confound this design exists to avoid. Do not "simplify" this by templating
  later in the pipeline. **Record the model change as a stated caveat in the writeup.**

## Phase 2 result — decided 2026-08-27, job 3034032

```
[decision] base google/gemma-2-2b     harmful refusal rate = 0.000
[decision] it   google/gemma-2-2b-it  harmful refusal rate = 0.969   (31/32)
[decision] harmless-prompt rate (gates nothing): base=0.000  it=0.031
[decision] DECISION: use google/gemma-2-2b-it --chat-template (base_rate=0.000 < 0.10)
```

The base checkpoint never refuses anything, so the fallback branch is taken and
**`google/gemma-2-2b-it` is the model for the whole refusal control**. Two consequences
that must reach the writeup:

- **Stated caveat.** The positive control runs on a different checkpoint than the truth
  run (`gemma-2-2b`). A `readout-only` verdict here therefore cannot be blamed on the
  model being incapable of refusal: at 0.969 this checkpoint plainly refuses. But an
  `actuatable` verdict does not automatically transfer back to the truth run's model.
- **The harmless baseline is low.** At 0.031 (1 of 32) the nonzero-`frac=0` hazard the
  screen warns about is small, but the sign check at the deciding frac is still required.

`got_datasets/refusal.csv` and `refusal_holdout.csv` were rebuilt on the laptop with
`--chat-template google/gemma-2-2b-it` and re-rsynced. Verified after the rebuild: 976 fit
rows (488/488), 64 holdout (32/32), exactly two `<start_of_turn>` markers per row (no
double-templating), every row ending at the open model turn, and fit/holdout overlap 0.

`prep_refusal.apply_template` and `refusal_screen.chat_wrap` issue byte-identical
`apply_chat_template(..., tokenize=False, add_generation_prompt=True)` calls, so the string
the 0.969 gate measured is exactly the string stored in the CSVs. Every downstream stage
(`extract.py:125`, `reach_steer.py:92`, `dct_steer_utils.generate:64`) tokenizes with the
default `add_special_tokens=True`, adding a BOS on top of the template's own `<bos>` in all
cases. That is consistent across stages, which is what the linearization point requires.

**Do not re-submit `run_refusal_screen.slurm` now.** It would apply the template a second
time to the already-templated holdout file. See the warning in Phase 2b.

---

## Phase 2.5 — prep result and a commitment made BEFORE the reach job ran

Job 3034813, completed 2026-08-27 in ~14 min, no errors.

```
model google/gemma-2-2b-it   source_layer 5   target_layer 14
input_scale 24.5649          token_idxs "-3:"   num_samples 64
```

**The layer sweep is uninformative, and this must be stated up front.**

```
layer 0: 0.500   layer 1: 0.995   layer 2: 1.000   ... layers 2-15: 1.000   layers 16-26: 0.990-0.995
```

The probe is at ceiling by layer 2. Harmful-vs-harmless is lexically separable, exactly as
`make_reach_meta`'s docstring predicts, so every layer from 2 up is within 0.01 of every
other and the sweep cannot distinguish them. `source_layer = 5` is therefore the
`MIN_SOURCE_LAYER` floor firing, not a layer the data selected. The rule ran as
pre-registered; the point here is that its input carried no signal.

(The 0.500 at layer 0 is the pipeline behaving correctly, not a bug: with `token_idxs
"-3:"` the read positions are `<start_of_turn>model\n`, identical across every templated
prompt, so before any attention has run they contain nothing about the instruction.)

**Why the run proceeds anyway.** The lexical objection bites the probe, not the
experiment. The outcome measure is behavioral refusal on held-out prompts, which is how
Arditi et al. answer the same objection about the same harmful/harmless contrast. If
steering induces refusal on harmless instructions, the direction is functionally a refusal
direction however easy the fit set was. `target_layer` is 14, so the readout the
certificate targets sits at 54% depth, where the refusal literature places it.

**The commitment, made before any steering result exists.** The two verdicts are not
symmetric, and deciding that after seeing a null would be post-hoc:

- **`actuatable`** -> the instrument is validated. No follow-up needed. The truth
  dissociation is a fact about truth, not about the method.
- **`readout-only`** -> **the control is NOT decisive and must not be reported as such.**
  A null at layer 5 is ambiguous between "the instrument cannot actuate" (the finding this
  control exists to establish) and "layer 5 was the wrong injection site, chosen by a floor
  rather than by evidence". Resolving it requires a second arm at a literature-standard
  injection layer before any claim about the instrument is made.

---

## Phase 3 — prep, then reach (cluster)

Wait for `run_refusal_prep.slurm` (submitted above), then check the meta before spending
the 8-hour reach budget on it:

```bash
squeue -u vwudaru        # wait until EMPTY
cat dct_meta_refusal.json
```

Check `source_layer >= 5` (the earliest-within-tolerance rule is floored there because
harmful-vs-harmless is lexically separable and accuracy saturates near layer 0) and that
`input_scale` is a number, not `null` (`null` means `calibrate_scale.py` didn't run or
died — do not proceed). `num_factors` and `num_iters` are `null` by design (see Gotchas).

```bash
sbatch deltaai/run_refusal_reach.slurm
squeue -u vwudaru        # poll until EMPTY (up to 8h cap; real usage likely well under)
sacct -j <jobid> --format=JobID,ExitCode   # non-zero = a P3 stage FAILED, named below; 0 = all OK
grep -a -E "===|VERDICT|slice fidelity|optional artifact|wrote|FAILED|Traceback" refusal_reach_*.out
```

**The smoke step.** Before the full run, this job runs `reach_margins.py --limit 8` and
hard-stops (no "continuing") if it fails: `load_model_and_slice`, `stage_vjp`, and
`reach_linerr.compute` are not covered by any test, so this is the first time that code
runs against a real model. Check the smoke section of the log, and specifically look for
the `[reach] slice fidelity cos=…` line. If a job instead dies with `SlicedModel
unfaithful`, **stop** — its Jacobians would be of the wrong map (see Gotchas).

## Phase 4 — spot-check (cluster)

```bash
sbatch deltaai/run_refusal_spotcheck.slurm
squeue -u vwudaru        # wait until EMPTY
grep -a "spot-check" refusal_spotcheck_*.out
```

This re-runs the substring pass (identical output) and adds the OLMo agreement check —
see "The kappa cross-check" below before reading the numbers.

## Phase 5 — rsync back and plot (laptop)

Pull the sidecar JSONs along with the CSVs — `reach_control_<ds>_<arm>.json`, not the
`.out` log, is the artifact of record (see "The sidecar artifact" below):

```bash
cd ~/llm-activation-steering-research
rsync -av \
  'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{dct_meta_refusal.json,reach_summary_refusal.json,reach_curve_refusal.csv,reach_steer_refusal.csv,reach_steer_readout_refusal.csv,reach_steer_stmt_refusal.csv,reach_steer_stmt_meta_refusal.csv,judge_refusal_refusal_mean.csv,judge_refusal_refusal_stmt.csv,judge_refusal_spotcheck_refusal_mean.csv,judge_refusal_spotcheck_refusal_stmt.csv,reach_control_refusal_mean.csv,reach_control_refusal_mean.json,reach_control_refusal_stmt.csv,reach_control_refusal_stmt.json,refusal_screen_google_gemma_2_2b.csv,refusal_screen_google_gemma_2_2b_it.csv,reach_margins_refusal.npz,reach_dirs_refusal.npz,reach_acts_refusal.npz}' \
  ./
PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset refusal
```

---

## The sidecar artifact — read this, not the log

Each `reach_control.py` invocation writes **two** files: `reach_control_<ds>_<arm>.csv`
(the per-frac table, for plotting) and `reach_control_<ds>_<arm>.json` (the artifact of
record). The JSON carries everything needed to audit the verdict without re-deriving it
from logs: the verdict string, `min_delta`, the `direction` used, the per-frac `table`
(same numbers as the CSV, keyed by `frac_eps_star`), `column_semantics` (a self-describing
dict explaining every CSV column), and three row-accounting counts —
`dropped_unmatched_rows` (mean arm: judged rows whose scale had no matching readout entry
— can silently delete crossing fracs), `dropped_never_steered_rows` (per-statement arm:
statements whose own `eps_i <= 0`, excluded — see "Excluded statements" below), and
`empty_completion_rows` (completions that were blank, scored as not-refused, not an
error). Each `table` bucket additionally carries `n_refused` (the exact refusal count
behind `frac_refused`), `refused_ci95_lo` / `refused_ci95_hi` (a 95% Wilson interval for
that rate), and `too_small_to_adjudicate`; the sidecar's top level carries
`min_bucket_n` and `undersized_fracs`, plus `deciding_frac`,
`deciding_delta_vs_baseline` (the **signed** delta at the frac that earned an
`actuatable`) and `refusal_fell_at_deciding_frac`. See "Reading the results" below for how
to weigh them. For dataset `refusal` there are two sidecars per run: `reach_control_refusal_
mean.json` and `reach_control_refusal_stmt.json`.

## Reading the results — decision gates

| VERDICT | Meaning | Action |
|---|---|---|
| `actuatable` | Crossing the boundary changes behavior | **The instrument is valid.** The truth dissociation is a fact about truth, not the method. Proceed to the main-venue dissociation-with-instrument paper — but only after the primary-arm and interval checks below. **First check the SIGN**: read the sidecar's `deciding_delta_vs_baseline`, and if it is negative (or `refusal_fell_at_deciding_frac` is `true`, or the log prints the `refusal FELL` WARNING) this verdict is *not* evidence of actuation — see "The sign of the deciding frac" below. |
| `readout-only` | Crossing moves the readout, not behavior — refusal fails the same way truth did | **STOP.** The instrument cannot demonstrate actuation on a concept where the literature says actuation exists. Recalibrate with Julian before spending more GPU time; the likely suspects are the all-position steering-hook convention and the choice of `Jᵀw` as the input direction. |
| `inert` | Behavior moves without a readout crossing | Off-target steering. Check whether `g_read` moves at all in `reach_control_refusal_*.json`'s `table`. |
| `no-crossing` | Neither | Underpowered sweep: `1.5 × input_scale` is capping the scales below eps\*. Report the capped fraction before concluding anything. |
| *(no verdict — `SystemExit`)* | Every non-baseline frac bucket has `n < 5`, so no cell may be named at all | **Not a result — a failed measurement.** The job exits non-zero, no `reach_control_*.{csv,json}` is written, and the message names the undersized fracs. The sweep was fully clamped (`scale_grid`'s `1.5 × input_scale` cap firing per statement, so each clamped statement got its own `frac = cap/eps_i` bucket). Fix `input_scale` / inspect the per-statement `eps_i` and re-run P3. Do **not** hand-read this as `no-crossing`: that row is a measurement, this is an artefact. |

Both arms (`mean`, `stmt`) get their own verdict — read both sidecars. When they
disagree, **the per-statement (`stmt`) arm is the primary arm**: its frac buckets hold
~200 rows each, against 32 for the mean arm (the harmless holdout), and each of its
subjects is steered at *its own* `eps_i` rather than at the dataset-wide
`median_eps_star`. So it is both better powered and a tighter test of the certificate.

The mean arm is the supporting arm, and its `actuatable` must never be read on its own.
`MIN_DELTA = 0.10` is a hard threshold on a point estimate, so at n=32 **4 of 32 prompts**
changing refusal status is enough to cross it. Before treating a `mean`-arm `actuatable`
as the headline result, check:

1. the `stmt` arm's verdict, and
2. the sidecar's `n_refused` / `n` and `refused_ci95_lo` / `refused_ci95_hi` (95% Wilson
   interval) for the crossing bucket **and** for the frac=0 baseline. At 4 vs 0 of 32 the
   two intervals overlap heavily; that is a suggestive result, not a demonstrated one.

The interval is reporting only — it does **not** enter the verdict. The verdict logic is
deliberately unchanged so it stays the pre-registered rule; the interval is how you
decide how much weight to put on it.

`min_bucket_n` / `undersized_fracs` in the sidecar: a frac bucket with `n < 5`
(`reach_control.MIN_BUCKET_N`) is barred from deciding **any** verdict cell, because
`scale_grid`'s `1.5 × input_scale` clamp can give a clamped subset of statements
idiosyncratic `frac = cap/eps_i` values that land in their own `n=1` bucket. Those
buckets are still shown in the table with their `n` and a `too_small_to_adjudicate` flag
— never dropped — so you can see how much of the sweep was clamped. If **every**
non-baseline bucket is flagged, no verdict is emitted at all (last row of the table
above). The floor is a **clamp-artefact filter, not a statistical power gate**: at n=5
a 5/5 rate still has a 95% Wilson CI of (0.57, 1.00), so clearing the floor says only
"this bucket is not a clamp remnant" — how much weight its rate carries is the
`n_refused` / `n` and interval question above.

### The sign of the deciding frac

`reach_control` decides "behavior moved" on `|delta| ≥ 0.10` — a symmetric
did-anything-change screen, pre-registered and deliberately unchanged. But the
*prediction* on this polarity is **directional**: `refusal.csv` has label 1 = harmless, so
crossing pushes a harmless prompt into the **harmful** halfspace, which should make the
model **refuse more** (`src/prep_refusal.py`'s docstring). A refusal rate that *falls* by
≥10 points at the crossing frac still returns `actuatable`, and that is evidence *against*
actuation, not for it.

So on any `actuatable`, read three sidecar keys before anything else:

- `deciding_frac` — the frac that earned the verdict (the one nearest the boundary, if
  several did),
- `deciding_delta_vs_baseline` — its **signed** delta, and
- `refusal_fell_at_deciding_frac` — `true` iff any deciding frac's delta is negative.

All three are `null`/`false` when the verdict is not `actuatable`. When the flag is
`true` the log also prints a `WARNING: refusal FELL where it was predicted to RISE`. The
likely mechanism is mundane: `REFUSAL_MARKERS` includes `"as an ai"` and `"as a language
model"`, reliable refusal signals on harmful prompts (Arditi's setting) but common benign
openers on harmless Alpaca instructions — so a large perturbation that degrades fluency
*removes* them and lowers the rate. Check the Phase-2 harmless-prompt baseline (below);
if it was well above 0, this is the first explanation to rule out.

### Two warnings that do not change the verdict but change how to read it

Both are printed to the job log **and** recorded in the sidecar; check for both before
trusting an `actuatable`, because "any sufficiently large perturbation moves behavior" is
the primary confound a positive control has to rule out.

- **Near-boundary baseline** (`near_boundary: true` in the sidecar, with
  `baseline_g_read` and `g_read_swing_at_pm1`). If the smallest non-baseline `|frac|`
  already crosses, the certificate's **magnitude** claim (~eps\* of perturbation needed)
  is untested for this prompt population — the verdict may say `actuatable` on a boundary
  that was already underfoot, not one the sweep actually had to reach.
- **Off-target movement**: the log prints a `WARNING` when behavior moved at a frac that
  did **not** cross (this is the `inert` case) *or* when it moved at a non-crossing frac
  **in addition to** a genuine crossing (verdict still `actuatable`, but read it knowing
  behavior also moved somewhere the certificate says nothing about). This is not a
  separate sidecar boolean — recompute it from the sidecar's `table` if reading after the
  fact: a frac with `crossed: 0` but `|delta_vs_baseline| >= min_delta` (0.10) is
  off-target.

A third log-only warning is worth a glance: **crossing at a positive frac**. Per the
certificate's own sign convention, crossings should only occur at negative fracs (eps\* is
positive only where `g > 0`); a positive-frac crossing suggests the linearization sign is
wrong or the readout is non-monotone.

### Three hard-stop errors you may hit

`reach_control.py` refuses to emit a verdict rather than guess in three situations —
all `SystemExit`, all mean the run needs to be re-examined, not silently re-run:

- **Vacuous baseline** — the unsteered readout is already inside the target halfspace
  (`baseline g_read <= 0`) before any steering happens. This means the prompt population
  (`refusal_holdout.csv` for the mean arm, the per-statement prompts for `stmt`) does not
  sit where `t02` (the target-layer threshold) was fit — the crossing test is vacuous by
  construction and cannot support any verdict, including `no-crossing`.
- **Baseline-only table** — no non-baseline frac bucket exists at all. Either the scale
  grid was fully clamped (eps\* far exceeds `1.5 * input_scale`, so every steered frac
  rounds into the frac=0 bucket), or, in the per-statement arm, every sampled statement
  already had its own `g_i <= 0` (`eps_i = 0`), so `scale_grid` returned only the
  baseline. Check `input_scale` **and** the per-statement `eps_i` values before assuming
  it's a clamp.
- **Nothing adjudicable** — non-baseline buckets exist, but *every one* of them has
  `n < 5` (`MIN_BUCKET_N`), i.e. they are all `frac = cap/eps_i` clamp remnants. The
  message names them. This is the fully-clamped, underpowered case; no cell may be named
  off artefact buckets, least of all `readout-only` (which would both halt the programme
  and coincide with the pre-existing truth result). Fix the sweep and re-run P3.

### Excluded statements (per-statement arm only)

Statements whose own `eps_i == 0` were never steered at all (`scale_grid` returns `[0.0]`
only for them) and are excluded from the control table — `align_stmt_rows` drops them and
reports the count as `dropped_never_steered_rows` in the sidecar and in the job log
(`"dropped N never-steered statement row(s)"`). A large count means most of the 200 sampled
statements already sat inside the target halfspace before steering, and the per-statement
arm is weak evidence either way — worth checking before leaning on the `stmt` verdict.

### The kappa cross-check

`refusal_judge.py --spot-check` reports **raw agreement** and **Cohen's kappa** between
the substring judge and the OLMo judge on a random 60-row subsample per arm (printed at
the end of `refusal_spotcheck_*.out`: `spot-check n=… agreement=… kappa=… unparseable=…`,
and in `judge_refusal_spotcheck_refusal_{mean,stmt}.csv`). **If kappa < 0.6**, the two
judges disagree enough that the headline substring-based refusal rate needs a stated
caveat in the writeup — report both numbers, don't just take the substring rate at face
value. OLMo replies that can't be confidently parsed as REFUSED/COMPLIED (`unparseable`)
are counted separately and **excluded** from both agreement and kappa, not defaulted to
either label.

---

## Gotchas

Copied verbatim from `REACH_H0_RUN.md`:

> `dct.SlicedModel` divides gemma-2 inputs by √d expecting the HF model to re-multiply
> `inputs_embeds` by its normalizer. **transformers ≥ 5 no longer does** (the local
> `.venv` is 5.12.1), which silently made the hop map unrelated to the real forward
> (cos ≈ 0.04–0.23). `reach_hop.load_model_and_slice` now runs a startup fidelity probe
> and auto-applies a √d compensation — every hop job prints `[reach] slice fidelity
> cos=1.000000 (input compensation x…)` as its first model line. **If a job instead dies
> with `SlicedModel unfaithful`, stop:** the environment can't reproduce the forward and
> any margins it produced would be Jacobians of the wrong map. The cluster's
> `.venv-dct-gpu` (4.51.3) is faithful with factor 1 — the audit's margins were computed
> there and stand.

Three more, specific to the refusal run:

- `dct_meta_refusal.json` has **`"num_factors": null`** by design — the minimal positive
  control (decided 2026-07-29) trains no DCT factors at all. Do not read this as a failed
  fit; only `input_scale` needs to be non-null.
- `reach_margins.py` prints one line per absent optional artifact —
  `[reach] optional artifact absent for refusal: dct — dct_u battery members +
  cos_dctv landmark disabled` and the same for `mag` — dct is expected, not an error.
  It means the battery has no `dct_u_*` control-group members and
  `reach_margins_refusal.npz` has no `cos_vq`/`cos_dctv` keys; `viz_reach.py` will render
  "no mag_dir landmark" in the second geometry panel instead of that cosine curve.
- The per-statement control table (`reach_steer_stmt_meta_refusal.csv`,
  `reach_control_refusal_stmt.{csv,json}`) is indexed by `frac_eps_star`, **not** by raw
  `scale` — every statement has its own eps\*, so raw scales are all distinct and grouping
  by scale would give `n=1` buckets.

---

## Commit

```bash
git add deltaai/REFUSAL_RUN.md
git commit -m "docs(refusal): cluster runbook with model gate and verdict decision table"
```
