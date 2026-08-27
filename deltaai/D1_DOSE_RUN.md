# D1 dose-response: cluster runbook

Gate 2 of the steering validity audit. What the experiment is and why, plus the pre-registered
decision rule, is in `docs/D1_DOSE_RESPONSE.md`. This file is only the mechanics.

Read `deltaai/CLUSTER_OPERATIONS.md` Parts 1 to 4 once before your first cluster run. Coordinates
used below: user `vwudaru`, host `dtai-login.delta.ncsa.illinois.edu`, account `bhhv-dtai-gh`,
partition `ghx4`. Every `ssh` and `rsync` prompts for the NCSA password then a Duo push.

**Shape:** generation job, then judge job, then analysis on the laptop.

```
run_dose.slurm  ->  dose_<ds>.csv          ->  run_dose_judge.slurm  ->  judge_dose_<ds>.csv
   (gen, GPU)       dose_yesno_<ds>.csv         (OLMo judge, GPU)              |
                    dose_meta_<ds>.json                                        v
                          |______________________________________________  dose_analyze.py
                                                                            (laptop, CPU)
```

**Cost, measured 2026-08-26:** generation runs at 14.2 s per (direction, dose) cell, so about
34 min per dataset and **1.2 h for both**, model loads included. The judge is still an estimate at
about 2 h. Walls are capped at `06:00:00` and `04:00:00` so a hang cannot drain the allocation;
the generation wall could safely come down to `03:00:00` and would schedule better under backfill.

To watch progress and project the finish:

```bash
n=$(grep -a -c "rel=" dose_*.out); t=$(squeue -u $USER -h -o "%M" | head -1); echo "$n $t" | awk '{split($2,a,"-"); if(length(a)>1){d=a[1];hms=a[2]}else{d=0;hms=a[1]}; split(hms,b,":"); if(length(b)==3){s=b[1]*3600+b[2]*60+b[3]}else if(length(b)==2){s=b[1]*60+b[2]}else{s=b[1]}; s+=d*86400; r=s/$1; printf "cells %d/288  elapsed %s  %.1f s/cell  projected total %.1f h\n", $1, $2, r, 288*r/3600}'
```

288 cells total, 144 per dataset. The rate is stable across datasets because `FACTUAL_PROMPTS`
and `YESNO_STATEMENTS` are fixed lists, not dataset-derived.

---

## Step 0. Confirm the inputs exist on the laptop

D1 reads six committed artifacts per dataset and writes nothing over them.

```bash
cd ~/llm-activation-steering-research
PYTHONPATH=src ./.venv/bin/python -c "
import json, numpy as np
for ds in ['cities','common_claim_true_false']:
    md = np.load(f'mag_dir_{ds}.npz'); td = np.load(f'truth_dir_{ds}.npz')
    meta = json.load(open(f'dct_meta_{ds}.json'))
    summ = json.load(open(f'reach_summary_{ds}.json'))
    mz = np.load(f'reach_margins_{ds}.npz', allow_pickle=True)
    acts = np.load(f'reach_acts_{ds}.npz', allow_pickle=True)
    L = {int(md['layer']), int(td['layer']), int(meta['source_layer'])}
    assert len(L) == 1, f'{ds}: layers disagree {L}'
    h = float(np.median(np.linalg.norm(np.asarray(acts['h_src'], np.float64), axis=1)))
    e = summ['directions']['mean_diff_tgt']['median_eps_star']
    print(f'{ds}: layer {L.pop()}  median||h_src|| {h:.2f}  eps* {e:.3f}  = rel {e/h:.4f}')
"
```

Expect exactly:

```
cities: layer 11  median||h_src|| 118.06  eps* 2.802  = rel 0.0237
common_claim_true_false: layer 13  median||h_src|| 151.28  eps* 10.689  = rel 0.0707
```

If the layers disagree the script refuses to run anyway, on the grounds that a shared magnitude
axis across differently-fit directions is meaningless.

## Step 0b. Run the harness tests before spending a GPU hour

Five seconds, no model, no GPU. This is the check that the sweep does what it claims.

```bash
./.venv/bin/python -m pytest tests/test_dose_response.py tests/test_dose_analyze.py -q
```

Expect `16 passed`. They stub generation with a function whose output encodes the injected norm,
then assert the injected magnitude equals `rel * median ||h_src||`, that the baseline block is
genuinely unsteered, that `--resume` refills exactly the missing cells including one written to
only a single output file, that a finished run cannot be clobbered, that a dead hook aborts, and
that the clean-window rule fires on a planted effect and stays silent on a norm-matched one.

## Step 1. Two-rsync up

rsync does not honour `.gitignore`, so the code goes first with the big binaries excluded, then
only the artifacts this job needs, named explicitly.

```bash
rsync -av --exclude '.git' --exclude '.venv*' --exclude 'activations/' \
  --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' \
  ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/

rsync -av mag_dir_cities.npz mag_dir_common_claim_true_false.npz \
          truth_dir_cities.npz truth_dir_common_claim_true_false.npz \
          reach_acts_cities.npz reach_acts_common_claim_true_false.npz \
          reach_margins_cities.npz reach_margins_common_claim_true_false.npz \
  vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

`dct_meta_*.json` and `reach_summary_*.json` are small and travel with the first rsync.

The second rsync names about 312 MB, but `reach_acts_*` and `reach_margins_*` were produced on
the cluster in the first place, so rsync will compare and almost certainly skip them. Let it
decide rather than trying to guess which are already there.

## Step 2. Fill the account, and verify it

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research
sed -i 's/--account=ACCOUNT_NAME/--account=bhhv-dtai-gh/' \
  deltaai/run_dose.slurm deltaai/run_dose_judge.slurm
grep -- --account deltaai/run_dose.slurm deltaai/run_dose_judge.slurm
```

Both lines must print `bhhv-dtai-gh`. If either still says `ACCOUNT_NAME`, SLURM will reject the
job for an invalid account.

Do **not** use the `ACC=$(grep ... run_dct.slurm)` idiom that appears in the older runbooks. The
cluster's copy of `run_dct.slurm` still holds the unfilled placeholder, so that substitutes the
placeholder with itself and silently does nothing.

## Step 3. Submit the generation job

```bash
sbatch deltaai/run_dose.slurm
squeue -u vwudaru
```

It loops both datasets in one job. If cities finishes and common_claim is killed by the wall,
cities is complete and intact; see step 5.

## Step 4. Read the liveness line first

Logs look blank because of carriage returns; read them with `grep -a`.

```bash
grep -a -E "assert|warn|===|wrote|FAILED|Traceback" dose_*.out
```

**The first thing to look for, before anything else:**

```
[D1][assert] hook live: rel=5.0 changes the text ('...' -> '...')
```

At five times the median activation norm the completion cannot stay byte-identical. If instead
the job died with `HOOK IS DEAD`, the `forward_pre_hook` never fired and every null the run would
have produced is an artifact of the harness, not a fact about the model. Fix that before reading
another line. Same role as the oracle line in `run_token_steer.slurm`.

**The second thing:**

```
[D1] baseline done; N yes, M no, K unparsed, mean margin +0.0xxx
```

and possibly

```
[D1][warn] only XX% of unsteered yes/no prompts produced a parseable yes or no.
```

If that warning appears, the flip rate is partly measuring whether steering made the model answer
at all rather than whether it made it lie, because `flipped` requires the steered answer to parse
and differ from a baseline that did not parse. Lead the writeup with the verdict margin in that
case. The caveat is inherited from `src/mag/steer.py:88`, so it applies to the committed
`mag_verdict_flips_*.csv` too. See `docs/D1_DOSE_RESPONSE.md` section 3.7.

Progress lines look like:

```
  sup_grad rel=+0.3000 alpha=+35.42 flips=3/24
```

144 of them per dataset, six directions by twenty-four non-zero doses.

## Step 5. Check completeness, and resume if the wall killed it

```bash
wc -l dose_*.csv
```

Expect per dataset: `dose_<ds>.csv` = 4,641 rows (1 header + 32 baseline + 6 x 24 x 32) and
`dose_yesno_<ds>.csv` = 3,481 rows (1 header + 24 baseline + 6 x 24 x 24).

Short of that, resubmit with resume set. It skips every cell already complete in **both** files
and regenerates any cell a mid-cell kill left in only one of them:

```bash
RESUME=1 sbatch deltaai/run_dose.slurm
```

The job refuses to start on an existing output without this, so a plain resubmit cannot silently
clobber a partial run.

## Step 6. Submit the judge job

Only after `squeue -u vwudaru` is empty and the row counts above are right.

```bash
sbatch deltaai/run_dose_judge.slurm
squeue -u vwudaru
grep -a -E "=====|wrote|FAILED|Traceback" dose_judge_*.out
ls -la judge_dose_cities.csv judge_dose_common_claim_true_false.csv
```

This uses OLMo-3-7B-Instruct in `.venv-judge-gpu`, the same judge and rubric that produced
`judge_mag_steer_*.csv` and `judge_reach_steer_*.csv`. That is deliberate: D1 exists to be read
against those, and changing the scorer mid-audit would confound every comparison with the judge
rather than the dose. It also means D1 needs no `ANTHROPIC_API_KEY`.

The job runs a 24-row smoke first, writing `judge_dose_smoke_cities.csv`, so a wrong path fails in
seconds instead of after 9,280 completions. That smoke file is disposable.

## Step 7. Pull back and analyse on the laptop

```bash
cd ~/llm-activation-steering-research
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{dose_*.csv,dose_meta_*.json,judge_dose_*.csv}' ./
./.venv/bin/python src/dose_analyze.py --dataset cities
./.venv/bin/python src/dose_analyze.py --dataset common_claim_true_false
```

Writes `dose_summary_<ds>.csv`, `dose_window_<ds>.csv`, `plot_d1_dose_<ds>.png`,
`plot_d1_margin_<ds>.png`, and prints the verdict:

- `N clean windows out of 120 cells` with `N > 0` is the **PI is right** branch. Behaviour moved
  above a norm-matched random vector at a dose where coherence held.
- `0 clean windows` is the **null is real** branch on the discrete outcomes.
- The margin block printed after it is scored separately and is never merged into the window
  count. A null window count with a monotone signed margin is branch C, the most informative
  outcome available, and it is invisible to every measurement this project has made so far.

Then fill sections 4 and 5 of `docs/D1_DOSE_RESPONSE.md`. Do not edit sections 1 to 3: they are
the registration.

---

## Gotchas specific to this job

| Symptom | Cause and fix |
|---|---|
| `HOOK IS DEAD` at startup | The steering hook is not firing. Nothing else in the run would mean anything. Check `dct_steer_utils.Steerer` against the model's layer signature. |
| `refusing to run: injection layer disagrees` | `mag_dir`, `truth_dir` and `dct_meta.source_layer` do not agree. One of the three artifacts in the second rsync is stale. Resend all of them. |
| `dose_<ds>.csv already exists` | A previous run left output. `RESUME=1 sbatch` to continue it, or delete it deliberately. This guard is why a restart cannot destroy hours of generation. |
| `FileNotFoundError: reach_margins_<ds>.npz` | Missing from the second rsync. It is excluded from the first by `--exclude '*.npz'`. |
| Judge job fails on model load | Wrong env. The judge is `.venv-judge-gpu` (transformers 5.x); generation is `.venv-dct-gpu` (4.51.3). The scripts set this, do not hand-edit. |
| `Error configuring interconnect` | Transient bad node, only on the `srun`-wrapped judge job. Resubmit; if it recurs, drop the `srun ` prefix from the `python3` lines. |
| Log looks blank | Carriage returns from progress output. Use `grep -a`, not `cat`. |
