# Token-space program: the actual cluster run, step by step

This is the operational sheet for **one specific run**: E0, E1, E2, E3, E4, E6, E7, E8 on
DeltaAI, both datasets. Copy-paste top to bottom.

- The **why** for each experiment is in [`docs/TOKEN_SPACE_PROGRAM.md`](../docs/TOKEN_SPACE_PROGRAM.md).
- The **algebra** is in [`docs/math_map.tex`](../docs/math_map.tex).
- The **reusable cluster machinery** (envs, SLURM, the two-rsync pattern) is in
  [`CLUSTER_OPERATIONS.md`](CLUSTER_OPERATIONS.md). Read that once; this sheet assumes it.
- [`TOKEN_SPACE_RUN.md`](TOKEN_SPACE_RUN.md) is the interpretation guide: what each number
  means once it lands. This sheet is only the mechanics.

**Coordinates:** `vwudaru@dtai-login.delta.ncsa.illinois.edu`, account `bhhv-dtai-gh`,
partition `ghx4`, repo `~/llm-activation-steering-research`, env `.venv-dct-gpu`.
Password plus a Duo push on **every** `ssh` and `rsync`.

**Budget:** about 8 GPU-hr of wall-clock caps against roughly 475 remaining. Four jobs.

**No judge job.** The outcome variable is `hit_target`, a comparison of the next-token
argmax against a target token id. That is mechanical. `.venv-judge-gpu` is not used
anywhere in this run. The only thing a judge could add is a fluency read on the
`completion` column, and that is optional and last.

---

## Step 0 (laptop): run E8 before you touch the cluster

```bash
.venv/bin/python -m pytest tests/test_token_geom.py tests/test_token_steer.py tests/test_token_sens.py -q
```

Expect `37 passed`. These are the PI's claim 3 assertions: the RMSNorm adjoint pair, the
cone certificate against a full synthetic vocabulary, which layer each hook actually
fires on, whether `positions=last` touches only the last row, and the `sqrt(T)` energy
the broadcast convention spends without reporting it.

If any of these fail, **stop**. Everything downstream is an actuator claim, and a run
whose actuator is wrong produces output that looks fine and means nothing.

---

## Step 1 (laptop): two rsyncs up

**First rsync, the code.** rsync does not honor `.gitignore`, and the repo root holds
well over a gigabyte of activation caches the cluster jobs never read.

```bash
rsync -av --exclude '.git' --exclude '.venv*' --exclude 'activations/' --exclude '*.pt' --exclude '*.npz' --exclude 'run_dct.slurm' ./ vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

`got_datasets/*.csv` rides along in this one (they are csv, not npz), which is what E0
reads. `run_dct.slurm` is excluded so the cluster's copy stays intact: Step 2 lifts the
account out of it.

**Second rsync, the three artifacts these jobs actually need.** Named explicitly.

```bash
rsync -av reach_dirs_cities.npz reach_dirs_common_claim_true_false.npz reach_margins_cities.npz reach_margins_common_claim_true_false.npz dct_meta_cities.json dct_meta_common_claim_true_false.json vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/
```

| file | size | who reads it | if you skip it |
|---|---|---|---|
| `reach_dirs_<ds>.npz` | 708K each | `token_geom` geom stage, `token_sens` | the `mean_diff_tgt_asis` and `probe_grad_tgt_asis` alpha columns vanish, which are the two directions every previous steering run used. Do not skip |
| `reach_margins_<ds>.npz` | 94M and 125M | `token_steer --dirs jtw_legacy` only | the `jtw_legacy` continuity arm dies with a clear error. The rest of the run is unaffected |
| `dct_meta_<ds>.json` | 4K each | `viz_token` layer figure | the src/tgt layer markers are missing from one plot |

That 219 MB of `reach_margins` buys exactly one arm. It is the arm that answers "what
would the direction we actually steered with have done here", so it is worth the
transfer, but if the link is slow you can drop those two files, run everything else, and
send them later.

**Do not upload `token_acts_*.npz` or `token_geom_*.npz`.** Step 3 regenerates them on
the cluster in minutes, and that doubles as a GPU-versus-CPU consistency check on the
geometry we already have locally.

---

## Step 2 (login node): fill in the account

```bash
ssh vwudaru@dtai-login.delta.ncsa.illinois.edu
cd ~/llm-activation-steering-research

sed -i 's/--account=ACCOUNT_NAME/--account=bhhv-dtai-gh/' deltaai/run_token_geom.slurm deltaai/run_token_steer.slurm deltaai/run_token_jac.slurm deltaai/run_token_sens.slurm
grep -- --account deltaai/run_token_*.slurm
sacctmgr -n show assoc user=$USER format=account%20,partition%20
```

The grep must print `bhhv-dtai-gh` four times, and `sacctmgr` must list that account. If
it does not, the account name here is stale and the real one is whatever `sacctmgr`
reports.

**Do not lift the account out of `run_dct.slurm`.** Other runbooks tell you to, and on
this cluster it silently fails: the cluster's copy of that file still holds the
unfilled `ACCOUNT_NAME` placeholder, so `sed "s/ACCOUNT_NAME/$ACC/"` substitutes the
placeholder with itself and the grep afterwards still shows `ACCOUNT_NAME`. It looks
like the sed did not run. It ran and did nothing, which is worse, because the next thing
you do is submit a job that SLURM rejects for an invalid account.

**Also run E8 here**, in the environment the jobs will use:

```bash
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
python3 -m pytest tests/test_token_geom.py tests/test_token_steer.py tests/test_token_sens.py -q
```

If `pytest` is not installed in `.venv-dct-gpu`, `pip install pytest` (the login node has
internet). The `run_token_sens.slurm` job also runs these, but it degrades to a loud
warning if pytest is absent rather than killing a two-hour job over a missing tool, so
confirming it here is the reliable check.

---

## Step 3: E0, the geometry (35 to 45 minutes, against a 1 h cap)

```bash
sbatch deltaai/run_token_geom.slurm
squeue --me
```

Three invocations: `cities` with false-country targets, `common_claim` with runner-up
targets, and a runner-up lower bound for `cities` under `--tag runnerup` so it does not
clobber the first.

**This is the one job where the GPU barely matters.** Only the extraction is a forward
pass. The cost is the cone solve, which is pure numpy on the CPU: 20,000 dual iterations
per active-set round, per statement, at two sites. Measured locally at 16 minutes for
`cities` (2 to 3 binding faces per statement) and 8 for `common_claim` (usually 1), and
this job runs three geometry stages. So 35 to 45 minutes is normal and the `--time=01:00:00`
cap is tighter than it looks.

Watch progress with the per-statement line rather than guessing:

```bash
grep -a "\[geom\]" token_geom_<jobid>.out | tail -3
```

If it does hit the cap, nothing is lost: each invocation writes its own outputs when it
finishes. `ls -la token_*` shows what survived, then raise the cap and resubmit only what
is missing (`--stage geom` skips the extraction if `token_acts_<ds>.npz` is already
there).

**Gate before continuing.** Read these off the log:

```bash
grep -a -E "===|cone proved|delta_rel_z|alpha|rmsnorm_penalty|wrote|FAILED|Traceback" token_geom_<jobid>.out
```

| line | expected | what a miss means |
|---|---|---|
| `post-norm cone proved N/N` | 200/200, both datasets | anything less and those rows hit `MAX_FACES` and are not certificates |
| `delta_rel_z median` | about 0.034 (`cities`), 0.011 (`common_claim`) | a large disagreement with the local run means the GPU path differs from the CPU path, which is a bug, not a result |
| the `alpha` block | 0.004 to 0.016 | if alpha comes back order 1, the whole budget-units diagnosis is wrong and the run should stop until that is understood |

This step is a reproduction. We already have these numbers locally. If they do not match,
the disagreement is the finding and nothing after it should be believed.

---

## Step 4: E1 + E4 + E7, the naive steer (cap 4 h, expect 2 to 3)

```bash
sbatch deltaai/run_token_steer.slurm
```

Four arms per dataset, eight runs total: post-norm, pre-norm, post-norm at the last
position only, and post-norm with `repetition_penalty=1.3`.

**Read the oracle line before anything else, in every arm:**

```bash
grep -a -E "assert|===|hit_rate|FAILED|Traceback" token_steer_<jobid>.out
```

```
[assert] oracle @ +1.00 hit rate 1.000 (HARNESS OK)
```

At the post-norm site that flip is a theorem: Step 3 verified in closed form that the
displacement makes the target token the full-vocabulary argmax. If the running model
does not emit it, the injection code is wrong and every other row in the file is
meaningless. A rate below about 0.9 means stop and debug, not "interesting, the oracle
underperformed".

Outputs: `token_steer_<ds>_<site>_<positions>_rp<x>.csv`, eight files.

---

## Step 5: E2, the layer sweep (cap 3 h)

```bash
sbatch deltaai/run_token_jac.slurm
```

One backward per batch gives `||J_l^T a||` for all 26 layers under both injection
conventions, plus `--save-dirs` writes the pullback vectors Step 6 needs (about 50 MB per
dataset).

```bash
grep -a -E "cheapest layer|broadcast|===|wrote|FAILED" token_jac_<jobid>.out
```

**Write down the layer.** The line reads `cheapest layer (broadcast injection)`. Steps 6
and 7 both take it as a parameter. The prediction on record is that it is not 11 or 13,
which are the layers DCT and the probe picked and which were never chosen by a
steerability criterion.

Steps 4 and 5 are independent and can be in the queue at the same time.

---

## Step 6: E3, the certified steer at that layer (cap 3 h)

```bash
sbatch --export=ALL,TOKEN_STEER_LAYER=<L> deltaai/run_token_jac.slurm
```

Same script. With `TOKEN_STEER_LAYER` set it re-runs the sweep, which is cheap and
revalidates, and then injects `unit(J_L^T a)` at the certified budget.

This is the arm that decides the shape of the writeup. If the flip lands at the certified
budget, the result becomes "certified reachability works, but only at layers with
adequate control authority, and decodability peaks are systematically not those layers",
which is a considerably stronger claim than the null we currently have.

---

## Step 7: E6, the sensitivity spectrum (cap 3 h)

```bash
sbatch deltaai/run_token_sens.slurm                                  # defaults to layer 13
sbatch --export=ALL,TOKEN_SENS_LAYER=<L> deltaai/run_token_sens.slurm   # and at the cheapest layer
```

The PI's claim 4. Independent of Steps 4 through 6, so it can go in the queue alongside
them; the second submission needs `<L>` from Step 5.

**Read the post-norm assertion first:**

```bash
grep -a -E "assert|spread|scale-dependence|===|wrote|FAILED" token_sens_<jobid>.out
```

```
[assert] postnorm identity: max |gain-1| 1.3e-04, max |slope - a.u| 4.6e-05 (HARNESS OK)
```

At the post-norm site the map to `z` is the identity, so gain is 1 and slope is `a . u`
as arithmetic. That arm is a harness check, not a result. This assertion already passed
locally, so a failure on the cluster means the GPU path differs from the CPU path.

Then the two numbers, which answer different questions and must not be merged:

| line | reading |
|---|---|
| `spread p90/p10` | near 1: the layer is close to isotropic and a norm budget is a fair currency. Orders of magnitude: it is not, and the budget must be gain-normalized |
| `median scale-dependence` | 1.0 is exactly linear across a 400x budget range. Far from 1 is the chaos claim proper, and it would mean no fixed budget certifies anything at any scale |

"Anisotropic but linear" is a units problem and `eps = eps*/alpha` already fixes it.
Only the second reading would obstruct the certificate.

---

## Step 8 (laptop): pull results and plot

```bash
rsync -av 'vwudaru@dtai-login.delta.ncsa.illinois.edu:~/llm-activation-steering-research/{token_geom_*.csv,token_jac_*.csv,token_steer_*.csv,token_sens_*.csv,token_*_*.out}' ./
```

The npz files stay on the cluster unless you need them; `token_jac_<ds>.npz` alone is
about 50 MB per dataset.

```bash
PYTHONPATH=src .venv/bin/python src/viz_token.py --dataset cities
PYTHONPATH=src .venv/bin/python src/viz_token.py --dataset common_claim_true_false
```

Writes `plot_token_budget_*`, `plot_token_alpha_*`, `plot_token_layers_*`,
`plot_token_sens_*` (one per site), and `plot_signed_steer_*`.

---

## Submission order, condensed

```bash
# after Steps 0 to 2
sbatch deltaai/run_token_geom.slurm     # gate on "cone proved 200/200"
sbatch deltaai/run_token_steer.slurm    # gate on the oracle line
sbatch deltaai/run_token_jac.slurm      # read the cheapest layer L
sbatch deltaai/run_token_sens.slurm     # gate on the postnorm identity
# then, with L in hand
sbatch --export=ALL,TOKEN_STEER_LAYER=<L> deltaai/run_token_jac.slurm
sbatch --export=ALL,TOKEN_SENS_LAYER=<L>  deltaai/run_token_sens.slurm
```

Only Step 3 blocks the others: `token_steer`, `token_jac`, and `token_sens` all read
`token_geom_<ds>.npz`. After that everything is parallel except the two `<L>` jobs.

---

## Resyncing code mid-run

You will patch a script between jobs. Use **`-avR`**, not `-av`, and **drop the tilde**
from the destination:

```bash
rsync -avR src/token_steer.py deltaai/run_token_jac.slurm vwudaru@dtai-login.delta.ncsa.illinois.edu:llm-activation-steering-research/
```

Remote paths are already relative to your home, so `llm-activation-steering-research/`
is what you want. Writing `:~/llm-activation-steering-research/` works under plain `-av`
but fails under `-R`, which issues an explicit `mkdir` and gets
`mkdir "/u/vwudaru/~/llm-activation-steering-research" failed`. The absolute
`:/u/vwudaru/llm-activation-steering-research/` also works if you prefer it.

Without `-R`, rsync copies named files *into* the destination directory and discards
their paths, so `src/token_steer.py` lands at the repo root as `token_steer.py`. The
transfer reports success, the byte count looks right, and the cluster keeps running the
old code. Verified on 2026-08-04: an hour of GPU time went into jobs running a stale
script, and the tell was that the rsync output listed bare filenames with no directory
prefixes.

Always confirm on the cluster by grepping for a string that only exists in the new
version, never by trusting the rsync summary:

```bash
grep -c jtw_last src/token_steer.py     # must be non-zero
```

**Type it at a Mac prompt.** Run from the cluster shell it syncs the cluster's own files
onto themselves: the file list comes back empty and the byte count is small. Check the
`total size` against your local `wc -c` before believing a transfer.

**Re-sed the account after every `.slurm` rsync.** The laptop's copy always carries
`ACCOUNT_NAME`, so a resync silently reverts Step 2 and the next `sbatch` fails with
`Invalid account or account/partition combination specified`:

```bash
sed -i 's/--account=ACCOUNT_NAME/--account=bhhv-dtai-gh/' deltaai/run_token_<job>.slurm
```

Step 1's first rsync does not have this problem because it syncs `./` and preserves the
whole tree. Step 1's second rsync does not either, because those artifacts genuinely
belong at the repo root.

---

## Traps specific to this run

**Three assertions, three different jobs.** E0's `cone proved N/N`, E1's oracle hit rate,
E6's postnorm identity. Each one is a closed-form theorem about its own job. They exist
so a broken harness is distinguishable from a real negative result, which is precisely
what the previous negative result could not do. Check all three. Do not read past a
failed one.

**Never overwrite a `reach_*` artifact.** Those are inputs to results already written up.
Every script in this run only reads them. If a job writes one, that is a bug.

**gemma's tokenizer left-pads.** `attention_mask.sum(1) - 1` silently reads a pad row.
Use the flip-argmax form in `reach_hop.last_nonpad_index`. This bit `token_geom.py`
during development and cost a full extraction pass.

**Two conventions for the final norm gain.** `Gemma2RMSNorm` returns
`normalized * (1 + weight)`, so `hidden_states[-1]` already carries the gain and pairs
with the plain tied `E`. The other convention folds the gain into the unembedding and
pairs with the pre-gain vector. Identical logits, identical argmax, **different
distances**, and every budget in this program is a distance.

**Logs look blank.** tqdm overwrites lines with carriage returns. Always
`grep -a`, never plain `cat`. `tail -f` freezes the terminal if the job hung.

**`--time` is a hard kill, not a hint.** The caps here are generous against measured
runtimes. If a job hits its cap it was hung, and rerunning it with a longer cap without
finding out why is how an allocation disappears.
