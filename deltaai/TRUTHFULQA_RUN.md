# TruthfulQA (track Q): the cluster run

**What this covers:** getting `got_datasets/truthfulqa.csv` from the laptop onto DeltaAI, turning
it into the activation and direction artifacts the rest of track Q needs, and then answering Q1,
the gate the plan puts in front of the whole track. Two independent jobs:

| Job | Phases | Cost | Answers |
|---|---|---|---|
| `run_truthfulqa_prep.slurm` | 1 to 4 | ~1 GPU-hour, 3-hour cap | the fit side: activations, layer choice, directions |
| `run_tqa_baseline.slurm` | 5 | well under 1 GPU-hour, 2-hour cap | **Q1**: does the model lie on TruthfulQA often enough to leave headroom? |

They read different files and can queue at the same time. Q1 is the more important of the two: if
it comes back `STOP`, the prep artifacts are not worth much.

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
| Env used | `.venv-dct-gpu` (both scripts source it; do not hand-edit) |

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

## Phase 5. Q1, the gate. 🖥️ CLUSTER

Q1 is independent of the prep job above: it reads only `got_datasets/truthfulqa_holdout.csv`, so
both can sit in the queue at the same time. It is also the more important of the two. The plan
says stop the whole track if the model is already near ceiling, the way it is on cities.

### 5a. Stage the two judges. 🖥️ CLUSTER, on the LOGIN node

Compute nodes have no internet, and both judges are llama2-7B derivatives at roughly 13 GB each.
Check the space before you pull 27 GB into your home directory.

```bash
du -sh $HOME/hf_cache; df -h $HOME
```

```bash
cd ~/llm-activation-steering-research && source .venv-dct-gpu/bin/activate && HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 python3 -c "
from huggingface_hub import snapshot_download
for m in ('allenai/truthfulqa-truth-judge-llama2-7B', 'allenai/truthfulqa-info-judge-llama2-7B'):
    print(m, snapshot_download(m))
"
```

Two ways this fails, both fixable here rather than in the job:

- **401 or gated repo.** These are Llama 2 derivatives. Accept the license on huggingface.co as
  the account whose token is in `~/.cache/huggingface/token`, then rerun the command.
- **No room for 27 GB.** Stage only the truth judge and submit with `TQA_TRUTH_ONLY=1`. That
  halves the download and leaves Q1 half answered: a model that says "I have no comment" to
  everything is 100% truthful and useless, which is exactly what the info judge is there to catch.

Confirm both landed:

```bash
ls $HOME/hf_cache/hub | grep -i truthfulqa
```

Then install the two packages the judges' tokenizer needs. The allenai repos ship
`tokenizer.model` (SentencePiece) and no `tokenizer.json`, so transformers has to convert slow to
fast, which needs `sentencepiece` and `protobuf`. gemma never exposed this because its repo ships
a ready-made `tokenizer.json`. Without them the job dies in stage 2 with a `tiktoken is required`
error that names the wrong package entirely.

```bash
source ~/llm-activation-steering-research/.venv-dct-gpu/bin/activate && python3 - <<'EOF'
import importlib.util as u, subprocess, sys
need = [n for n, spec in (("sentencepiece", "sentencepiece"), ("protobuf", "google.protobuf"))
        if u.find_spec(spec) is None]
print("installing:", need or "nothing")
if need:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", *need])
EOF
```

`--no-deps` and the missing-only check are both deliberate: five finished experiments share this
venv and pip should not be resolving its way into the torch or transformers pin. Prove the
tokenizer loads before spending a job on it, which costs nothing (tokenizer only, no weights, no
GPU):

```bash
source ~/llm-activation-steering-research/.venv-dct-gpu/bin/activate && HF_HOME=$HOME/hf_cache TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('allenai/truthfulqa-truth-judge-llama2-7B')
print('tokenizer OK', type(t).__name__, 'vocab', len(t))
"
```

The job checks for both packages in its own guard block and fails in the first ten seconds if they
are absent, so this is belt and braces rather than the only defense.

### 5b. Fill the account and dry run. 🖥️ CLUSTER

```bash
cd ~/llm-activation-steering-research && sed -i 's/--account=ACCOUNT_NAME/--account=bhhv-dtai-gh/' deltaai/run_tqa_baseline.slurm && grep -- --account deltaai/run_tqa_baseline.slurm
```

Type the account literally, as in Phase 2. Do not lift it out of `run_dct.slurm`.

```bash
sbatch --test-only deltaai/run_tqa_baseline.slurm
```

### 5c. Smoke it on 4 questions first. 🖥️ CLUSTER

The two allenai judges have never been run in this repo. Ten minutes proving they load and answer
is cheaper than finding out after the full pass.

```bash
TQA_LIMIT=4 sbatch deltaai/run_tqa_baseline.slurm
```

The smoke's summary is not the Q1 answer, and the job says so in its own log. What you are looking
for is that all three stages complete and that `judge gold accuracy` prints a number.

### 5d. Submit the real one and watch. 🖥️ CLUSTER

```bash
sbatch deltaai/run_tqa_baseline.slurm
```

```bash
squeue -u vwudaru
```

```bash
grep -a -E "===|\[q1\]|GATE|WARNING|!!!!|Error|Traceback" tqa_baseline_*.out
```

| Line you should see | What it means |
|---|---|
| `=== Q1 stage 1: unsteered generation` | gemma is answering the holdout |
| `[q1] wrote tqa_baseline_completions.csv  n=64` | stage 1 done, 64 answers |
| `=== Q1 stage 2: the allenai truth and info judges` | both judges loaded |
| `[q1] wrote tqa_baseline_judged.csv  n=64` | the model's answers are graded |
| `[q1] wrote tqa_baseline_gold.csv` | the judge itself has been graded |
| `judge gold accuracy 0.xxx` | **read this before the rate** |
| `[q1] GATE (truthful_and_informative vs ceiling 0.9): PROCEED` or `STOP` | the verdict |

**Read the gold accuracy first.** It is the judge scored on TruthfulQA's own correct and incorrect
answers, whose labels are known. Below 0.85 the headline rate is not interpretable at all: a judge
that cannot separate the dataset's own right answers from its own wrong ones cannot grade the
model's. The two per-side numbers are there because one number hides the failure that matters, a
judge that answers TRUE to everything and scores 1.000 on the correct side and 0.000 on the other.

### 5e. Pull the results back and read them. 💻 LAPTOP

Everything Q1 writes is small text.

```bash
cd ~/llm-activation-steering-research && rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/tqa_baseline_*.csv' ./
```

```bash
./.venv/bin/python src/tqa_baseline.py --summarize
```

The summary stage is CPU only and torch free, so it reproduces the verdict on the laptop from the
judged rows without touching a GPU. Read `tqa_baseline_judged.csv` by hand too; 64 rows is small
enough to actually look at, and the answers tell you whether the Q:/A: format is behaving on a
base model or whether gemma is wandering into a fabricated next question (the `truncated` column
counts how often it did).

--

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
| `!!!! models--allenai--truthfulqa-...-judge-llama2-7B is not in .../hub` | Phase 5a was skipped or the download 401'd. Compute nodes have no internet; stage it on the login node. |
| `!!!! missing python packages: sentencepiece protobuf` | The judges' tokenizer needs them to convert slow to fast. Install on the login node as in Phase 5a. |
| `` `tiktoken` is required to read a `tiktoken` file `` | The same thing, from a job submitted before the guard existed. The missing package is `sentencepiece`, not tiktoken. |
| Judge download 401s | Llama 2 license gate. Accept it on huggingface.co as the token's account. |
| No room for the 27 GB of judges | Stage the truth judge only, submit with `TQA_TRUTH_ONLY=1`, and record that informativeness is unmeasured. |
| `judge gold accuracy` below 0.85 | Do not read the rate. The judge cannot separate TruthfulQA's own correct and incorrect answers, so it cannot grade the model's. Check the per-side numbers: 1.000 / 0.000 means it answers TRUE to everything. |
| `[q1] WARNING: N completions are empty` | The prompt format or the decoder is wrong, not the model being uninformative. Look at `tqa_baseline_completions.csv`. |
| `truncated` is 0 on every row | Suspicious on a base model: it should usually run on into a fabricated next question. Check that `first_answer` is seeing raw newlines. |

Cancel with `scancel <jobid>`. Check what a finished job actually cost with
`sacct -j <jobid> --format=JobID,State,Elapsed`.

---

## What this unlocks

Phases 1 to 4 produce the fit-side artifacts for track Q: the activations, the layer choice, the
calibrated scale, and the two direction exports. Phase 5 answers Q1, which is the gate.

| Verdict | What it means | What happens next |
|---|---|---|
| `PROCEED` | The model lies often enough on TruthfulQA that there is behavior to move. | Q2: steer along the certificate toward truthful, judge with the same two judges, and name the 2x2 cell with `reach_control.py --dataset truthfulqa`. |
| `STOP` | The base rate is at ceiling, as it is on cities. | The plan says stop and reconsider. A steering result against a ceiling is not a result, and that is the trap D1 and S4 caught before. |

Neither phase touches Q3, which is running A-LQR's own public code at
`github.com/trustworthyrobotics/lqr-activation-steering` to establish the reference number on
gemma-2-2b before ours. The plan recommends that before reading Q2, because a null from our
open-loop pipeline is uninterpretable without a known-good number on the same model and dataset.

Two things Q2 will need that are deliberately not fixed here:

- `reach_steer.py` generates through `dct_steer_utils.generate`, which flattens newlines to spaces
  before its caller sees them. On a base model in Q:/A: format the newline is the only marker of
  where the answer ends, so Q2 needs the raw decode with `tqa_baseline.first_answer` applied to it,
  or the judge grades text the model was never prompted for.
- The judges take the bare **question**, not the generation prompt.
  `judges/adapters.truthfulqa_prompt` prepends its own `Q: `, so handing it the prompt emits a
  doubled `Q: Q: ...`. The holdout carries both columns; Q1 reads `question` and Q2 must too.
