# Round 4: Q2x and V1

Spec: `docs/superpowers/specs/2026-09-30-round4-q2x-v1-design.md`. Both jobs are independent
of rounds 1-3 and of each other. They submit together with `round4` when two slots are free.

## 0. Before the first submit: the Q2x CPU smoke (LAPTOP)

Q2x reads Q2's root `reach_{acts,margins,dirs}_truthfulqa.npz`. Those are not on the laptop.
The copies in `u1_truthfulqa/` are U1's rerun, not Q2's: do not use them. Pull Q2's, one Duo
push:

```bash
cd ~/llm-activation-steering-research && rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{reach_acts_truthfulqa.npz,reach_margins_truthfulqa.npz,reach_dirs_truthfulqa.npz}' .
```

Then one question through all 13 blocks on CPU (`.venv-dct` pins transformers 4.51.3, as on
the cluster):

```bash
PYTHONPATH=src .venv-dct/bin/python src/tqa_q2x.py --stage steer --device cpu --limit 1 --prefix smoke_
```

```bash
PYTHONPATH=src .venv/bin/python -c "import pandas as pd; d=pd.read_csv('smoke_q2x_steer_truthfulqa.csv'); print(d[['direction','frac','g','crossed','budget_hit','words','answer']].to_string())"
```

Pass: 13 rows, the baseline `g` positive (Q2's median was 37.9), and `q2_mean_diff`'s `g`
falling as the dose rises. A negative baseline `g`, or a `g` that rises with dose, means the
readout or the sign is wrong: do not submit. Clean up after:

```bash
rm -f smoke_q2x_*
```

## 1. Push only round 4's files (LAPTOP)

Queued jobs import other `src/` modules at run time, so do not sync all of `src/` mid-queue.

```bash
cd ~/llm-activation-steering-research && rsync -av src/tqa_q2x.py src/reach_validate.py vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/src/ && rsync -av deltaai/run_q2x.slurm deltaai/run_reach_validate.slurm deltaai/submit_pi_feedback.sh vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/deltaai/
```

## 2. Fill the account and submit (LAPTOP, runs on the CLUSTER over ssh)

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu 'cd ~/llm-activation-steering-research && sed -i "s/--account=ACCOUNT_NAME/--account=bhhv-dtai-gh/" deltaai/run_*.slurm && grep -h -- --account deltaai/run_q2x.slurm deltaai/run_reach_validate.slurm && bash deltaai/submit_pi_feedback.sh round4'
```

The submitter refuses if this would put you over ghx4's 2 jobs. In that case wait for a slot.
Both jobs resume if resubmitted after a timeout: Q2x per (direction, dose) block and per 64
judged rows, V1 per dataset.

## 3. Pull the results (LAPTOP)

```bash
cd ~/llm-activation-steering-research && rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{q2x_*_truthfulqa.*,tqa_q2x_*.out,reach_validate_*}' .
```

## 4. Read

- `tqa_q2x_<id>.out`:
  - the R0 line, which must PASS;
  - R1, the realized crossing against the predicted 3.49;
  - the HEADLINE.
- `reach_validate_<id>.out`: for each dataset, the gate line, then `ratio_B` by layer.
- To re-run the Q2x summary on the laptop at any time:
  `PYTHONPATH=src .venv/bin/python src/tqa_q2x.py --stage summary`. Move
  `q2x_summary_truthfulqa.csv` and `q2x_outcome_truthfulqa.json` aside first.
