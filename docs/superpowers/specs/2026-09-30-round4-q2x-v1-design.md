# Round 4: Q2x, crossing the certificate on TruthfulQA, and V1, the certificate's closed check

Approved in chat 2026-09-30, approach A (thin new modules over the existing harness). Two
independent jobs, submitted together as round 4 of `PLAN_PI_FEEDBACK_2026-09-18.md`:
`deltaai/run_q2x.slurm` and `deltaai/run_reach_validate.slurm`. They depend on no other
round's output and on each other not at all.

## Why

Two claims in the project rest on a certificate that has never been tested where it matters.

1. **Q2 never crossed the boundary.** On TruthfulQA the readout g fell 37.88 -> 15.05 from
   frac 0 to -2 and never reached 0. Linear extrapolation (slope 10.84 per eps\* unit) puts the
   crossing at **3.49 eps\***. Meanwhile J-A's C3 showed the -2 gain is form: cut to the
   unsteered length, truthful-and-informative is 0.297 against 0.266 at baseline (Q2 doc
   section 12.5). Refusal flipped only once the boundary was crossed. So the open question is
   whether TruthfulQA *content* moves once the push crosses, or whether only form keeps
   growing.
2. **The certificate's arithmetic was never closed.** Every eps\* is a prediction. V1 compares
   prediction and realization at the output layer, where the readout is the decision and
   cannot dissociate from it, across all 26 source layers and three contexts.
   `src/reach_validate.py` (commit `cf7320d`, 15 tests) is written; it has no job.

## Constraint: do not touch what the queued jobs import

J-B and J-C (jobs 3282837/3282838) are pending, and rounds 2-3 follow. They import, at run
time on the cluster, `tqa_discovery`, `tqa_confirm`, `judge_audit`, `reach_steer`,
`reach_hop`, `tqa_baseline`, `tqa_q2_analyze`, `prep_truthfulqa`, `dct_steer_utils`,
`xfer_*`, `tqa_mc` and `tqa_learned`. **This work only imports them; it edits none of them.**
The only existing module edited is `reach_validate.py`, which no queued job imports. The
round-4 rsync lists files explicitly (section "Submission") instead of syncing `src/`.

---

## Q2x: does crossing move content? (`src/tqa_q2x.py`, `deltaai/run_q2x.slurm`)

### Registered question

Once the push along Q2's direction crosses the certificate's boundary (readout g < 0), is the
answer's **content** more truthful, beyond what norm-matched random directions do? Content is
scored by C3: each answer is cut to the same question's unsteered word count and re-judged.

### Directions

All unit vectors, signed toward truthful.

| name | what | source |
|---|---|---|
| `q2_mean_diff` | Q2's steering direction | `tqa_discovery.q2_vector()` (J^T w mean, from root `reach_margins_truthfulqa.npz`), times `Q2_TRUTHFUL_SIGN` |
| `rand_0..7` | norm-matched random directions | `np.random.default_rng(Q2X_RAND_SEED = 31)` Gaussian rows, unit-normalised. The seed differs from J-C's 13, so the null is a fresh draw |

### Doses

`DOSES = (0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)` x Q2's eps\* (`tqa_discovery.q2_eps_star()`,
14.7065), all toward truthful. Dose 2 is Q2's registered frac -2 and is this job's **anchor**.
The largest push, 73.5, is under `reach_steer`'s cap of 1.5 x input_scale = 91.4. Dose 0 is
generated once, as `baseline`.

### Data and generation

- **Questions:** the 64 holdout questions (`tqa_discovery.HOLDOUT`), the same set as Q1, Q2 and
  D3. No selection happens in this job.
- **Generation:** with exactly the primitives `tqa_discovery.steer_block` uses:
  `prep_truthfulqa.prompt_of`, `dct_steer_utils.generate_raw` (greedy, repetition penalty
  1.3), `tqa_baseline.first_answer` and `MAX_NEW_TOKENS = 48`. That is Q2's path and budget,
  one answer at a time. The steering layer is 11 (`reach_hop.load_meta`).
- **Why not `steer_block` itself:** it flattens the raw completion's newlines before writing,
  and the newline is the only marker of whether the answer ended on its own. So Q2x runs its
  own loop over the same primitives. It writes the same columns as `steer_block`
  (`direction, scale, frac, question, prompt, completion, answer`) plus the measurements below,
  computed from the raw text in the same pass.
- **Size:** 9 directions x 6 doses x 64 + 64 baseline = **3,520 generations**.

### Measurements per row

Columns written in the same pass, with the hook set to that block's vector:

- `g`: layer-20 readout of `mean_diff_tgt` at the prompt's last token, computed with
  `reach_steer.read_g`. It uses `w` and `thresh02` from root `reach_dirs_truthfulqa.npz`,
  located by name, and the target layer from `load_meta`. `crossed = g < 0`.
- `budget_hit`: true when the raw completion, stripped, contains **no newline**. In that case
  the answer never ended and the 48-token budget cut it. A raw-length test would be useless:
  Q1 measured that every generation runs to the budget by inventing a next turn (job 3081925,
  64 of 64), so only the newline separates "answer finished" from "answer cut".
- `words` and `distinct_ratio`: distinct words over words in `answer`, a repetition and
  degeneration signal.

Judging (stage `judge`) runs `judge_audit.judge_resumable` with
`judges.local_hf.TruthJudge(device).score` (the v2 prompts) on every row, giving `truthful`,
`informative`, `truthful_and_informative` and the `p_*` columns.

Truncation (stage `truncate`) cuts each steered row's `answer` to its own question's
**baseline** word count, using `judge_audit.truncate_words`, and re-judges it. The baseline
comes from this job's dose-0 answer, not from Q1/Q2's files. A row already at or under that
length is copied with `cut = 0` and is not re-judged. This is C3 per dose and per direction.

### Statistics (stage `summary`, also runnable on the LAPTOP)

`tqa_confirm.summarize(rows, col)` is applied twice: to the full and the truncated judged
rows, both with `col = truthful`, and again with `truthful_and_informative` for the table. It
gives, per (direction, dose):
- rate with a Wilson interval;
- exact McNemar against `baseline`, paired by question;
- a one-sided permutation p against the 8 random directions' paired gains at the same dose.

From the readout and form columns, per (direction, dose): `median_g`, `crossed_share`,
`mean_words`, `budget_hit_share`, `median_distinct_ratio`.

The score column is `truthful`. J-A's gold check failed the info judge's 0.9 bar by one row
(Q2 doc section 12.2), so the plan's fallback applies. Truthful-and-informative is reported
alongside, never selected on.

### Registered readings (fixed before the run)

- **R0, the gate.** At dose 2, `q2_mean_diff`'s full-answer truthful rate falls inside Q2's v2
  Wilson interval at frac -2, **[0.3958, 0.6337]** (`tqa_q2_summary_truthfulqa_judge_v2_truthful.csv`).
  If it falls outside, the summary prints `GATE FAILED` and no R1/R2 reading is made.
- **R1, arithmetic.** The **realized crossing dose** `d_x` is the smallest dose at which
  `q2_mean_diff`'s `median_g < 0`. It is reported against the predicted 3.49. If no dose up to 5
  crosses, R1 reads "no crossing by 5 eps\*". R2 is then read at dose 5 and labelled
  *uncrossed*.
- **R2, primary.** Read at every dose >= `d_x`. Across those doses, the McNemar p's are
  Holm-corrected.
  - **(a) content moves:** the *truncated* truthful gain has Holm-adjusted McNemar p < 0.05
    against baseline **and** beats all 8 random directions' truncated gains at that dose
    (perm p = 1/9).
  - **(b) form only:** (a) fails, while the *full* truthful score meets the same bar or its
    gain exceeds the truncated gain by at least 0.10.
  - **(c) degeneration:** the full truthful rate is below the dose-2 anchor's, **and** either
    `budget_hit_share >= 0.5` or `median_distinct_ratio < 0.5`.
  - Outcomes are classified per dose. The headline is the classification at `d_x`.
- **Prediction, registered now: (b).** C1-C3 put the -2 gain in length, and the readout is
  close to linear in the push. A crossing that moves content would be the first actuatable
  truth result in the project, and the largest revision to its claim.

### Stages, resumption, files

`STAGES = ("steer", "judge", "truncate", "summary")`, all run by `--stage all`.

| stage | resumes | output (`PREFIX` + name) |
|---|---|---|
| steer | per (direction, dose) block of 64; generation and readout in one pass, one row each | `q2x_steer_truthfulqa.csv` |
| judge | per 64 rows (`judge_resumable`) | `q2x_judged_truthfulqa.csv` |
| truncate | per 64 rows | `q2x_trunc_judged_truthfulqa.csv` |
| summary | refuses to overwrite | `q2x_summary_truthfulqa.csv`, `q2x_outcome_truthfulqa.json` |

`--limit N` caps questions and keeps `q2_mean_diff` plus one random direction. It requires
`--prefix`, as `tqa_discovery.main` does.

---

## V1: the certificate's closed check (`deltaai/run_reach_validate.slurm`, `reach_validate.py`)

### What runs

`reach_validate.py --compute` then `--analyze` on **cities**, then on **truthfulqa**: 64
statements each (seed 42), every source layer (26), three contexts.

| context | what it is |
|---|---|
| decl | the bare statement, where J is linearized |
| stem | the statement minus its last word, the generation context |
| quest | `Q_TRUTH` + statement + `Q_SUFFIX`, where the next token is the verdict |

The target is the output-layer contrastive readout `a = unit(mean W_U[yes] - mean W_U[no])`.
Inputs on the cluster: `reach_acts_{ds}.npz` and `dct_meta_{ds}.json` for both datasets, and
gemma-2-2b in the HF cache.

### Code change in `reach_validate.py`

- `--prefix` (default `""`), applied to `reach_validate_{ds}.npz` and
  `reach_validate_summary_{ds}.csv`, in both compute (write) and analyze (read and write).
  `--limit` without `--prefix` is refused.
- Neither stage overwrites. Compute refuses if its npz exists; analyze refuses if its summary
  exists. The refusal names the file to move aside.
- The ratio_A gate is restated (below). It now prints which reading applies instead of a
  single all-layers verdict.

### Registered readings

| reading | expectation |
|---|---|
| `ratio_A`, short hops (source layers >= 20) | within 0.10 of 1. Failing this is a **bug** in the pullback/injection convention, and nothing else in the file is read |
| `ratio_A`, deep sources | drift that grows with depth is reported as **nonlinearity of the multi-layer map**, not a bug |
| `ratio_B` (decl certificate at stem) | cities ≈ 0.07 (D1's 8-35x collapse). TruthfulQA: no registered value |
| `ratio_B_own` | ≈ 1. A low `ratio_B` with `ratio_B_own` ≈ 1 puts the failure in transport, not linearization |
| `ratio_Q_own` + flip counts | crossing zero flips the yes/no argmax in ≥ 90% of crossed statements, the no-dissociation check |
| **actionable** | any source layer with median `ratio_B >= 0.5` holds its gain across the context shift and is a candidate steering layer |

The current gate (`ARITHMETIC_TOL` checked at every layer) becomes: worst `ratio_A` deviation
over layers >= 20 decides bug vs not-bug. The worst over all layers is printed as the
nonlinearity figure.

---

## Jobs

Both jobs carry the repo's guards:
- preflight on inputs and HF cache (compute nodes are offline);
- a prefixed smoke run of every stage, then deletion of the smoke files;
- refusal to overwrite outputs;
- resumable stages;
- `--flag=value` for values with a leading dash;
- `ACCOUNT_NAME` placeholder, filled by the sed;
- `.venv-dct-gpu`, `HF_HOME=$HOME/hf_cache`, offline.

| | `run_q2x.slurm` | `run_reach_validate.slurm` |
|---|---|---|
| wall | 5:00 | 3:00 |
| expected | ~1-2 h generation and readout, minutes of judging | ~1-2 h |
| models | gemma-2-2b, both allenai judges | gemma-2-2b |
| preflight files | `reach_summary/dirs/margins/acts_truthfulqa` (all four read by `q2_vector` via `reach_steer._load_common`), `dct_meta_truthfulqa.json`, `got_datasets/truthfulqa_holdout.csv`, `tqa_q2_summary_truthfulqa_judge_v2_truthful.csv` | `reach_acts_{cities,truthfulqa}.npz`, `dct_meta_{cities,truthfulqa}.json` |
| smoke | `--stage all --limit 2 --prefix smoke_` | cities `--compute --analyze --limit 4 --layers 0,25 --prefix smoke_` |

`deltaai/submit_pi_feedback.sh` gets a `round4` case that submits both jobs with no
dependency between them.

## Submission

Round 4 is independent of rounds 1-3. Default order: after round 3. It can take any slot that
frees earlier, at the operator's call.

The rsync for round 4 names its files, so the modules imported by queued jobs are not touched
mid-queue:
`src/tqa_q2x.py`, `src/reach_validate.py`, `deltaai/run_q2x.slurm`,
`deltaai/run_reach_validate.slurm`, `deltaai/submit_pi_feedback.sh`.

## Testing (laptop, before any submission)

- `tests/test_tqa_q2x.py`, all on fakes:
  - the dose grid and its cap;
  - the random directions: unit norm, fixed seed, distinct from J-C's;
  - the sign of `q2_mean_diff`;
  - the truncation reference (cut to the baseline length, `cut = 0` when already short);
  - `budget_hit` from raw text, with and without a newline;
  - the steer loop on a fake model and tokenizer, writing `steer_block`'s columns plus `g`;
  - `crossing_dose`, including the no-crossing case;
  - the outcome classifier on planted tables for (a), (b), (c) and the R0 gate failure;
  - Holm across doses;
  - stage resumption;
  - `--limit` refused without `--prefix`;
  - the summary end to end on fake judged CSVs.
- `tests/test_reach_validate.py`: `--prefix` reaches both outputs; no overwrite in either
  stage; the restated gate separates the short-hop bug case from deep-layer drift.
- A CPU smoke of `tqa_q2x.py --stage steer --limit 1 --prefix smoke_` on the laptop with
  gemma-2-2b, if it is in the local HF cache, so the readout pass runs for real once before
  the cluster sees it.

## Files

| file | change |
|---|---|
| `src/tqa_q2x.py` | new |
| `tests/test_tqa_q2x.py` | new |
| `deltaai/run_q2x.slurm` | new |
| `src/reach_validate.py` | `--prefix`, no-overwrite, restated gate |
| `tests/test_reach_validate.py` | new tests |
| `deltaai/run_reach_validate.slurm` | new |
| `deltaai/submit_pi_feedback.sh` | `round4` case |

Results docs (`docs/Q2X_CROSSING.md`, `docs/V1_PIPELINE_VALIDATION.md`) are written when the
outputs come back, not now.
