# TruthfulQA (track Q): the cluster run

**What this covers:** getting `got_datasets/truthfulqa.csv` from the laptop onto DeltaAI and
turning it into the activation and direction artifacts the rest of track Q needs. One job,
`run_truthfulqa_prep.slurm`, roughly 1 GPU-hour with a 3-hour cap.

Read [`CLUSTER_OPERATIONS.md`](CLUSTER_OPERATIONS.md) Parts 1 to 4 once before your first run.
This runbook assumes the two venvs and the model cache already exist, which they do from the
refusal and token-space runs.

**Every block below is labeled 💻 LAPTOP or 🖥️ CLUSTER. Check the label before you run it.**

| | |
|---|---|
| Login | `ssh vwudaru@dtai-login.delta.ncsa.illinois.edu` |
| Auth | NCSA password, then Duo: type `1`, approve on your phone. Every `ssh` and every `rsync`. |
| Repo, both sides | `~/llm-activation-steering-research` |
| Account | `bhhv-dtai-gh` |
| Env used | `.venv-dct-gpu` (the script sources it; do not hand-edit) |

---

## Phase 0. Confirm the inputs. 💻 LAPTOP

The dataset is built on the laptop, not the cluster: `prep_truthfulqa.py` downloads
`truthfulqa/truthful_qa` from Hugging Face and **compute nodes have no internet**. It is already
built and committed, so this is a check, not a build.

```bash
cd ~/llm-activation-steering-research && ./.venv/bin/python -c "
import pandas as pd
d = pd.read_csv('got_datasets/truthfulqa.csv'); h = pd.read_csv('got_datasets/truthfulqa_holdout.csv')
print('fit', d.shape, dict(d.label.value_counts()))
print('holdout', h.shape)
assert len(d) == 1488 and d.label.sum() == 744 and len(h) == 64
print('OK')"
```

Expect `fit (1488, 6) {0: 744, 1: 744}` / `holdout (64, 7)` / `OK`.

---

## Phase 1. Ship the code up. 💻 LAPTOP

`got_datasets/*.csv` are plain CSVs, so they travel with the code rsync. **This job needs no
second rsync**, which is the usual step for artifacts. Everything it reads either goes up here or
is produced by the job itself.

```bash
cd ~/llm-activation-steering-research && rsync -av --exclude '.git' --exclude '.venv*' --exclude 'activations/' --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

The `*.npz` and `*.pt` excludes matter: the repo root holds well over a gigabyte of activation
caches this job never reads, and sending them makes the transfer look hung. `run_dct.slurm` is
excluded so the cluster's copy stays as it is.

Watch the file list go by and confirm you see `got_datasets/truthfulqa.csv`,
`got_datasets/truthfulqa_holdout.csv`, `src/prep_truthfulqa.py`, `src/extract.py` and
`deltaai/run_truthfulqa_prep.slurm`.

---

## Phase 2. Log in, fill the account, dry run. 🖥️ CLUSTER

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
```

Then, on the login node:

```bash
cd ~/llm-activation-steering-research && sed -i 's/--account=ACCOUNT_NAME/--account=bhhv-dtai-gh/' deltaai/run_truthfulqa_prep.slurm && grep -- --account deltaai/run_truthfulqa_prep.slurm
```

That must print `#SBATCH --account=bhhv-dtai-gh`. If it still says `ACCOUNT_NAME` the sed did not
match, and submitting will be rejected for an invalid account. **Do not** try to lift the account
out of `run_dct.slurm` with the old `ACC=$(grep ...)` idiom that some older runbooks show: the
cluster's copy of that file still holds the unfilled placeholder, so it substitutes the
placeholder with itself and looks like a no-op failure. Type the account literally.

Confirm gemma is already in the offline cache, because the compute node cannot fetch it:

```bash
ls $HOME/hf_cache/hub | grep -i gemma
```

You want a `models--google--gemma-2-2b` entry. If it is missing, get it **on the login node**,
which does have internet, then come back:

```bash
export HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 && source .venv-dct-gpu/bin/activate && hf download google/gemma-2-2b
```

Dry-run the submission before spending any allocation:

```bash
sbatch --test-only deltaai/run_truthfulqa_prep.slurm
```

It should report an estimated start time and submit nothing.

---

## Phase 3. Submit and watch. 🖥️ CLUSTER

```bash
sbatch deltaai/run_truthfulqa_prep.slurm
```

Note the job id it prints. Then:

```bash
squeue -u vwudaru
```

`PD` is pending, `R` is running, and an empty result means it finished. `squeue -u vwudaru --start`
gives an estimated start while it is pending.

**Read the log with `grep -a`, not `cat`.** The file looks blank in a pager because tqdm rewrites
its line with carriage returns:

```bash
grep -a -E "===|wrote|Saved|WARNING|Error|Traceback|FAILED" truthfulqa_prep_<jobid>.out
```

What a healthy run prints, in order:

| line | meaning |
|---|---|
| `=== Q0 activations, every layer ===` | extraction started |
| `Saved acts_truthfulqa.npz` | 1488 statements through 27 hidden states |
| `=== layer sweep + meta ===` | picking the source layer |
| `[meta] layer sweep: rows COMPUTED from ...` | swept from the activations, which is what you want |
| `[meta] wrote dct_meta_truthfulqa.json: src=... -> tgt=...` | source and target layers chosen |
| `=== calibration ...` then `=== direction exports ...` | the last two stages |
| the `cat` of `dct_meta_truthfulqa.json` | the final artifact, with a non-null `input_scale` |

**One line to check for specifically:**

```bash
grep -a "WARNING" truthfulqa_prep_<jobid>.out
```

Silence is correct. If it reports statements hitting `--max-length`, those rows had their
activation read mid-answer rather than at the answer's last token, and the run is not usable as
the fit set. Raise `--max-length` in the script and resubmit.

Verify the outputs exist:

```bash
ls -la activations/acts_truthfulqa.npz dct_meta_truthfulqa.json truth_dir_truthfulqa.npz truth_dir_tgt_truthfulqa.npz
```

---

## Phase 4. Pull the small artifacts back. 💻 LAPTOP

The activation npz stays on the cluster; it is large and every consumer of it runs there. Bring
back the meta and the directions, which the laptop analysis reads.

```bash
cd ~/llm-activation-steering-research && rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{dct_meta_truthfulqa.json,truth_dir_truthfulqa.npz,truth_dir_tgt_truthfulqa.npz}' ./
```

Quote the brace glob so it expands on the cluster, not in your local shell.

Then read the layer choice, which is the first real result of the run:

```bash
cat dct_meta_truthfulqa.json
```

`source_layer` and `target_layer` are worth comparing against cities (11 to 20) and common_claim
(13 to 22). A very different source layer means the probe separates truthfulness at a different
depth in the Q/A format than it does on declarative statements, which is itself something to
report.

---

## If it goes wrong

| Symptom | Cause and fix |
|---|---|
| `.out` is blank in a pager | tqdm carriage returns. Use `grep -a`. |
| `TRANSFORMERS_OFFLINE` or cache error | gemma not in `$HF_HOME` for `.venv-dct-gpu`. Download it on the login node as in Phase 2, resubmit. |
| `MISSING got_datasets/truthfulqa.csv` | The rsync in Phase 1 did not land. Rerun it and confirm the file appears in the transfer list. |
| `[meta] refusing to overwrite existing dct_meta_truthfulqa.json` | A previous run already wrote it. That guard exists because a calibrated `input_scale` costs a GPU run and cannot be recomputed from the script. Delete it deliberately, or pass `--force`, only if you mean to redo the whole prep. |
| `[meta] model mismatch` | `extract.py` and `make_reach_meta.py` were given different `--model` values. Both come from `$TQA_MODEL` in the script, so this means the npz is left over from an earlier run with the other checkpoint. Delete `activations/acts_truthfulqa.npz` and resubmit. |
| `[meta] layer sweep: rows READ FROM EXISTING results_truthfulqa.csv` | A leftover results CSV is deciding the source layer instead of these activations. Delete it and resubmit. |
| Job stuck `PD` for a long time | Normal queueing. `squeue -u vwudaru --start`. |
| Wrong env on model load | The script sources `.venv-dct-gpu`. Do not hand-edit it to the judge env. |

Cancel with `scancel <jobid>`. Check what a finished job actually cost with
`sacct -j <jobid> --format=JobID,State,Elapsed`.

---

## What this does and does not unlock

This produces the fit-side artifacts for track Q. It does **not** answer Q1, the unsteered
baseline, which is the gate the plan puts before everything else: measure gemma-2-2b's truthful
and informative rate on the 64 held-out questions with no steering at all, and **stop if it is
near ceiling** the way it is on cities.

Q1 needs its own script and its own job, plus the two allenai judges
(`truthfulqa-truth-judge-llama2-7B` and `truthfulqa-info-judge-llama2-7B`, both wrapped already
by `TruthJudge` in `src/judges/local_hf.py`) pre-downloaded on the login node. That is the next
thing to write.
