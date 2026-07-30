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
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q      # 253 passed, 1 skipped with Track A alone; 266 once Track B lands
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

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research
sed -i "s/ACCOUNT_NAME/bhhv-dtai-gh/" deltaai/run_refusal_*.slurm
grep -h -- --account deltaai/run_refusal_*.slurm     # verify before submitting — must print bhhv-dtai-gh, not ACCOUNT_NAME
sbatch deltaai/run_refusal_screen.slurm
squeue -u vwudaru        # wait until EMPTY
```

Read both `DECISION INPUT` lines and the final `DECISION:` line in
`refusal_screen_*.out`, and apply the pre-registered rule:

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
  Templating happens **once, in `prep_refusal.py`'s `apply_template`, at dataset-build
  time** — never at generation time. `extract.py`, `reach_margins.py`, and
  `reach_steer.py` must all tokenize the *identical* string, or the certificate's
  linearization point stops being the prompt's own last token, reintroducing the exact
  context-shift confound this design exists to avoid. Do not "simplify" this by templating
  later in the pipeline. **Record the model change as a stated caveat in the writeup.**

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
error). For dataset `refusal` there are two sidecars per run: `reach_control_refusal_
mean.json` and `reach_control_refusal_stmt.json`.

## Reading the results — decision gates

| VERDICT | Meaning | Action |
|---|---|---|
| `actuatable` | Crossing the boundary changes behavior | **The instrument is valid.** The truth dissociation is a fact about truth, not the method. Proceed to the main-venue dissociation-with-instrument paper. |
| `readout-only` | Crossing moves the readout, not behavior — refusal fails the same way truth did | **STOP.** The instrument cannot demonstrate actuation on a concept where the literature says actuation exists. Recalibrate with Julian before spending more GPU time; the likely suspects are the all-position steering-hook convention and the choice of `Jᵀw` as the input direction. |
| `inert` | Behavior moves without a readout crossing | Off-target steering. Check whether `g_read` moves at all in `reach_control_refusal_*.json`'s `table`. |
| `no-crossing` | Neither | Underpowered sweep: `1.5 × input_scale` is capping the scales below eps\*. Report the capped fraction before concluding anything. |

Both arms (`mean`, `stmt`) get their own verdict — read both sidecars, they can disagree.

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

### Two hard-stop errors you may hit

`reach_control.py` refuses to emit a verdict rather than guess in two situations —
both `SystemExit`, both mean the run needs to be re-examined, not silently re-run:

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
