# Token-space program: cluster runbook

Companion to `docs/TOKEN_SPACE_PROGRAM.md` (why) and `docs/math_map.tex` (the algebra).
Environment as usual: DeltaAI GH200, account `bhhv-dtai-gh`, partition `ghx4`,
`.venv-dct-gpu` for everything here. No judge environment is needed for the core arms,
which is the big saving over the previous design.

Before submitting, replace `ACCOUNT_NAME` in each `.slurm` with the real account.

---

## Why there is barely any judging this time

The outcome variable is `hit_target`: does the model's next-token argmax equal the
target token id. That is a mechanical comparison, not an LLM verdict. The OLMo judge
is needed only for the free-text `completion` column, and only if we want a fluency or
truthfulness read on top of the flip. Run the judge last, on the arms that generated
text, or skip it for the first pass.

---

## Order

### Step 0: E8, the actuator assertions (4 seconds, laptop or login node)

```bash
pytest tests/test_token_geom.py tests/test_token_steer.py -q
```

This is the PI's claim 3 ("check that the steering vector is what you want it to be")
turned into something that stays checked. 31 tests, no model, no GPU: the RMSNorm site
adjoint and its radial annihilation, the cone certificate verified against a full
synthetic vocabulary at three sizes, which layer each hook actually fires on, whether
`positions=last` touches only the last row, and the `sqrt(T)` energy the broadcast
convention spends without reporting it.

Run it before every submission. It costs nothing and it is the only step that can tell
you the harness is wrong *before* you spend GPU-hours producing output you would have
to discard.

### Step 1: E0, the geometry (minutes)

```bash
sbatch deltaai/run_token_geom.slurm
```

Writes, per dataset: `token_acts_<ds>.npz` (z post-norm and h pre-norm at the stem's
last token), `token_geom_<ds>.csv`, `token_geom_<ds>.npz`.

**Read from stdout before going further:**

- `post-norm cone proved  N/N`. This should be N/N. Each proved row means an explicit
  full-vocabulary argmax check confirmed the target wins at the computed displacement.
  Anything less means the active set hit `MAX_FACES` and those rows are not certificates.
- `delta_rel_z` median. This is the fraction of the activation norm a flip costs.
- The `alpha` / `eps_req` block. If `alpha` is order 0.01 for our directions, the
  behavioral null is explained by the actuator and not by truth. If `alpha` is order 1,
  that explanation dies and D1 through D4 are back on the table.
- `rmsnorm_penalty`. How much more a pre-norm injection costs than a post-norm one.

### Step 2: E1 + E4 + E7, the naive steer and its ablations (about 2 GPU-hr)

```bash
sbatch deltaai/run_token_steer.slurm
```

**Read the oracle line first, in every arm:**

```
[assert] oracle @ +1.00 hit rate 1.000 (HARNESS OK)
```

At the post-norm site this flip is a theorem, verified in closed form by Step 1. A hit
rate below about 0.9 means the injection code is wrong, and no other row in the output
means anything until that is fixed. Stop and debug.

With the harness confirmed, the trichotomy the PI asked for reads off the table:

| observation | conclusion |
|---|---|
| target token appears, text stays fluent | the last layer is fine; the hop broke it |
| text degenerates | the linear-feature assumption fails at the last layer |
| nothing changes at the old budget, flips at `eps/alpha` | the budget never bought a flip |

Compare across arms: `postnorm` vs `prenorm` isolates RMSNorm; `positions all` vs
`positions last` gives the broadcast gain; `rep-penalty 1.0` vs `1.3` says how much
the existing runs were confounded by a non-argmax decoder.

Outputs: `token_steer_<ds>_<site>_<positions>_rp<x>.csv`.

### Step 3: E2, the layer sweep (about 2 GPU-hr)

```bash
sbatch deltaai/run_token_jac.slurm
```

One backward per batch gives `||J_l^T a||` for all 26 layers under both injection
conventions. This is the experiment the literature calls decisive: layer choice
dominates steering success, and the decodability peak is systematically not the causal
peak (LEACE section 5.3). We chose layer 11 for `cities` because DCT and the probe both
peaked there, and neither is a steerability criterion.

Read `cheapest layer (broadcast injection)` off stdout.

### Step 4: E3, the certified steer at the cheapest layer (about 2 GPU-hr)

```bash
sbatch --export=ALL,TOKEN_STEER_LAYER=<L> deltaai/run_token_jac.slurm
```

Re-runs the sweep (cheap, and it revalidates) then injects `unit(J_L^T a)` at the
certified budget. If the flip lands here, the result becomes "certified reachability
works, but only at layers with adequate control authority, and decodability peaks are
systematically not those layers," which is a much stronger paper than the null.

### Step 5: E6, the directional sensitivity spectrum (about 2 GPU-hr)

```bash
sbatch deltaai/run_token_sens.slurm
sbatch --export=ALL,TOKEN_SENS_LAYER=<L> deltaai/run_token_sens.slurm   # at the cheapest layer
```

The PI's claim 4: if the same budget buys wildly different amounts of movement in
different directions, then a scalar `eps` is the wrong currency and the certificate
needs a gain-normalized budget instead. Independent of Steps 2 through 4, so it can be
submitted alongside them.

**Read the postnorm assertion first:**

```
[assert] postnorm identity: max |gain-1| 2.4e-07, max |slope - a.u| 1.1e-06 (HARNESS OK)
```

At the post-norm site the map to `z` is the identity, so `gain` is 1 and `slope` is
`a . u` as a matter of arithmetic. That arm is a check on the harness, not a result.
The layer arms are the experiment.

Two numbers come out, and they answer different questions:

| number | reading |
|---|---|
| `spread p90/p10` of gain across directions at fixed budget | near 1: the layer is close to isotropic and `eps` is a fair currency. Orders of magnitude: it is not, and the budget must be gain-normalized |
| `scale-dependence`, gain at the largest budget over gain at the smallest | 1.0 is exactly linear over the whole range. Far from 1 is the chaos claim, and it means no fixed budget works at any scale |

The distinction matters because "anisotropic but linear" is a fixable units problem,
while "chaotic" would be a genuine obstruction to the certificate. Claim 4 asserts the
second; nothing in our data has ever tested which one it is.

### Step 6: figures and the free audit (laptop, no GPU)

```bash
PYTHONPATH=src python src/signed_steer_audit.py --dataset cities
PYTHONPATH=src python src/viz_token.py --dataset cities
```

---

## Traps, all of which have already bitten something

**gemma's tokenizer left-pads.** `attention_mask.sum(1) - 1` silently reads a pad row.
Use the flip-argmax form in `reach_hop.last_nonpad_index`. This bit `token_geom.py`
during development and the guard is now the only thing standing between us and a whole
run measured at the wrong position.

**Two conventions for the final norm gain.** HF's `Gemma2RMSNorm` returns
`normalized * (1 + weight)`, so `hidden_states[-1]` already carries the gain and pairs
with the plain tied `E`. Folding the gain into the unembedding is the other valid
convention and pairs with the pre-gain vector. Identical logits, identical argmax,
**different distances**, and every budget here is a distance. `token_geom.load_unembed`
documents which one we use.

**Never re-run or overwrite an existing truth artifact.** The `reach_*` files for
`cities` and `common_claim_true_false` are inputs to results already written up. The
token-space scripts only ever read them.

**Do not import xgboost in the same process as torch** on macOS ARM (two libomp copies,
RC=139). Not an issue on the cluster, but `viz_token.py` is a laptop script.

**Do not index into a compressed npz inside a loop.** `np.load(...)["jtw"]` re-inflates
the whole array on every access. `token_jac.py` writes a `(n, 26, 2304)` float16 array
under `--save-dirs`; bind it once outside the loop.

---

## Artifact inventory

| file | written by | used by |
|---|---|---|
| `token_acts_<ds>.npz` | `token_geom.py --stage extract` | geom, `token_jac`, `token_steer` |
| `token_geom_<ds>.{csv,npz}` | `token_geom.py --stage geom` | `token_jac`, `token_steer`, `viz_token` |
| `token_jac_<ds>.{csv,npz}` | `token_jac.py` | `token_steer --dirs jtw_token`, `viz_token` |
| `token_steer_<ds>_<tag>.csv` | `token_steer.py` | analysis, optional judge |
| `token_sens_<ds>_<site>_<positions>.csv` | `token_sens.py` | `viz_token`, the claim-4 verdict |
| `signed_steer_<ds>.csv` | `signed_steer_audit.py` | `viz_token` |
| `stolen_probability_cities.csv` | `stolen_probability.py` | done, 182/182 achievable |

`stolen_probability.py` has no `common_claim` counterpart and does not need one. It was
written to ask whether a target token is argmax-achievable *at all*, before we had
certificates. `token_geom` now exhibits, for each of 400 statements across both
datasets, an explicit activation at which the target wins a full-vocabulary argmax,
which is a constructive achievability proof for exactly the targets we steer toward.
The general check survives as prior-art defense (Grivas et al., arXiv:2203.06462,
report the pathology is rare above d around 200, and we are at d = 2304), not as a
gate.
