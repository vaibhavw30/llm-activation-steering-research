# Truth in gemma-2-2b: certified reachable, behaviorally inert

**The finding in one sentence.** A linear probe reads "is this statement true?" off the model's
activations at 99%, and you can push the activations until that probe flips from TRUE to FALSE
with R-squared 0.999 against push size, and the model goes right on telling the truth. We then
proved the pushing machinery works by using the identical pipeline to induce refusal in a chat
model, where behavior moves decisively. So the null is a fact about **truth**, not about the
instrument.

> The analogy that lands: we found the thermometer, not the thermostat. You can hold a lighter
> under the thermometer and watch the number climb. The room stays cold.

A literature search found no published work reporting a certified-reachable-but-behaviorally-inert
dissociation with a mechanism attached. That dissociation is the contribution.

---

## Start here

| If you want | Read | Length |
|---|---|---|
| **The whole project, cold, in plain English** | [`docs/PLAIN_ENGLISH_WALKTHROUGH.md`](docs/PLAIN_ENGLISH_WALKTHROUGH.md) | 45 min |
| The 20-minute spoken version | [`docs/MEETING_20MIN.md`](docs/MEETING_20MIN.md) | 20 min |
| Which doc covers what | [`docs/README.md`](docs/README.md) | 5 min |
| How to run anything on the cluster | [`deltaai/README.md`](deltaai/README.md) | 5 min |
| Where the open questions are | [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) | 30 min |

**New to the project: read `docs/PLAIN_ENGLISH_WALKTHROUGH.md` first and nothing else.** It is
written to be read by someone who has not seen the code, it walks the eight experiments in the
order they happened, and every number in it is read off a named artifact in this repo.

---

## Where it lands

The publishable object is not either result alone. It is the contrast, produced by one pipeline
holding everything constant except the concept being steered.

| Cell | Meaning | Observed in |
|---|---|---|
| **actuatable** | readout crosses the certified boundary and behavior follows | refusal, per-statement arm |
| **readout-only** | readout crosses and behavior does not | truth, cities |
| **no-crossing** | the push never reaches the boundary | refusal, held-out mean arm |
| **inert** | neither moves | none so far |

---

## The chain, one line per step

Each experiment exists because the one before it produced a result that could not be interpreted
without it. Full detail for every row is in the walkthrough.

| # | Question | Answer | Write-up |
|---|---|---|---|
| 01 | Is truth encoded linearly? | Yes on clean data (cities 0.990 linear vs 0.993 XGBoost), with real non-linear headroom on messy data (+0.082) | [`docs/EXPLAINER.md`](docs/EXPLAINER.md) |
| 02 | Is the truth direction causally special? | No. Unsupervised DCT never surfaces it, and the non-linear headroom is not in DCT's subspace either | [`docs/DCT_VS_TRUTH_FINDINGS.md`](docs/DCT_VS_TRUTH_FINDINGS.md) |
| 03 | Reframe as backward reachability | Certificate `eps*`: the smallest nudge at layer 11 that lands in the probe's FALSE half at layer 20 | [`docs/REACHABILITY_RUNBOOK.md`](docs/REACHABILITY_RUNBOOK.md) |
| 04 | Does hitting the certified target change behavior? | **No.** Readout crosses inside the trust radius on cities; lie rate stays at baseline. Jacobian overlap with concept space 0.062 | [`docs/REACH_AUDIT_FINDINGS.md`](docs/REACH_AUDIT_FINDINGS.md) |
| 05 | Was the target set the problem? | Yes. Retargeting to token space turns 0.000 flips into 200/200 at the output layer and 34% pulled back to layer 16 | [`docs/TOKEN_SPACE_FINDINGS.md`](docs/TOKEN_SPACE_FINDINGS.md) |
| 06 | Do our own negative results survive an audit? | All four registered assumptions were refuted, most sharply S4: 200/200 token flips succeed, at most 5.5% made the claim false | [`docs/AUDIT_SUMMARY.md`](docs/AUDIT_SUMMARY.md) |
| 07 | Did we only ever push too hard? | No. Pre-registered dose sweep, 0 clean windows out of 120 cells, on both datasets | [`docs/D1_DOSE_RESPONSE.md`](docs/D1_DOSE_RESPONSE.md) |
| 08 | Can the pipeline move *any* behavior? | **Yes.** Refusal, `gemma-2-2b-it`, layers 5 to 14: 14 flips against 0, Mantel-Haenszel OR 24.2 for crossing | [`docs/REFUSAL_POSITIVE_CONTROL.md`](docs/REFUSAL_POSITIVE_CONTROL.md) |

The audit experiments S1 to S4 each have their own doc: [`S1_ASYMMETRY.md`](docs/S1_ASYMMETRY.md),
[`S2_LAYER_SWEEP.md`](docs/S2_LAYER_SWEEP.md), [`S3_COMMON_AXIS.md`](docs/S3_COMMON_AXIS.md),
[`S4_TARGET_CENSUS.md`](docs/S4_TARGET_CENSUS.md).

---

## Repository map

```
README.md                  you are here
CLAUDE.md                  the original Project-1 build guide, kept for provenance
docs/                      every write-up; see docs/README.md for the index
deltaai/                   NCSA DeltaAI (GH200) SLURM scripts + runbooks; see deltaai/README.md
got_datasets/              input CSVs: cities, sp_en_trans, companies, common_claim
src/                       all code; run as `.venv/bin/python src/<name>.py` from the repo root
tests/                     pytest suite, 430 passing
results/                   Project-1 probe CSVs and plots
activations/               extracted activations (gitignored, large)
(repo root)                experiment artifacts land here: *.csv, *.json, *.npz, plot_*.png
```

`src/` by program:

| Program | Files |
|---|---|
| P1 geometry of truth | `extract.py` `analyze.py` `summary.py` |
| DCT (third-party code + our drivers) | `dct.py` `dct_train.py` `run_dct_*.py` `dct_warm*.py` `apply_dct_vector.py` |
| MAG | `mag/` `run_mag.py` `viz_mag*.py` |
| Backward reachability | `reach_*.py` `make_reach_meta.py` `export_target_dir.py` |
| Token space | `token_geom.py` `token_jac.py` `token_sens.py` `token_steer.py` `token_conclusions.py` |
| Refusal positive control | `prep_refusal.py` `refusal_screen.py` `refusal_judge.py` `refusal_analyze.py` |
| Steering-validity audit (S1 to S4) | `audit_asymmetry.py` `audit_layer_sweep.py` `audit_common_axis.py` `audit_target_census.py` |
| D1 dose-response | `dose_response.py` `dose_analyze.py` `calibrate_scale.py` |
| SAE decomposition | `sae_load.py` `sae_decompose.py` `viz_sae.py` |
| Judging | `judge_results.py` `judges/` `validate_judge.py` |

### Reading an artifact

Artifacts are named `<program>_<what>_<dataset>.<ext>` at the repo root, where dataset is
`cities` or `common_claim_true_false`. **Never pool the two datasets**; they behave differently
and every finding is reported per dataset.

One rule that bites: read every `token_steer_*.csv` through `token_conclusions.load_arm`, never a
bare `pd.read_csv`. The arms have different column conventions and `load_arm` normalizes them.

```python
from token_conclusions import load_arm
df = load_arm("cities", "postnorm_all", rp=1.0)
```

---

## Running things

Everything runs from the repo root. There is no bare `python` on the dev machine; use the venv
interpreter explicitly.

**Project 1, geometry of truth** (CPU, `.venv`):

```bash
.venv/bin/python src/extract.py cities.csv
.venv/bin/python src/analyze.py cities
.venv/bin/python src/summary.py
```

**The audit experiments** (CPU, seconds each, read-only over committed artifacts):

```bash
PYTHONPATH=src .venv/bin/python src/audit_target_census.py --dataset cities
```

**Anything involving the model** runs on DeltaAI. Start at [`deltaai/README.md`](deltaai/README.md),
which maps each experiment to its runbook and SLURM script.

**Tests:**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

### Two hazards worth knowing before you run anything

- **Never import `xgboost` in the same process as `torch`.** Two copies of libomp segfault on
  macOS ARM (exit 139). `analyze.py` keeps them in separate processes for this reason.
- **Never index into a loaded `.npz` inside a loop.** Each access re-inflates the whole array.
  Pull the array out once, then loop.

---

## Environments

| venv | Where | Contents |
|---|---|---|
| `.venv` | laptop | CPU: torch, transformers, scikit-learn, anthropic |
| `.venv-dct` | laptop | DCT's pinned stack (transformers 4.51.3, required by the paper's code) |
| `.venv-dct-gpu` | DeltaAI | built on the cluster torch module, see `deltaai/setup_env.sh` |
| `.venv-judge-gpu` | DeltaAI | local OLMo judge backend, see `deltaai/setup_judge_env.sh` |

Commands in the runbooks are labeled **LAPTOP** or **CLUSTER**. Check the label before running
one; the two-machine workflow has already caused a misdirected rsync.

---

## Status, as of 28 August 2026

**Closed.** The truth null is established and audited. The refusal positive control passed and
opened the publication gate. Seven of the PI's ten items from the last meeting are answered
(the mapping is section 3 of the walkthrough).

**Open.**

- **T1**, the token-space behavioral test with a semantically meaningful target set. S4 calibrated
  the bar: a stoplist removes 178 of the 200 old targets, and T1 must beat 5.5%.
- **Horizon-0 item 0.2**, refitting the truth probe on generation-stem activations. The refusal
  control applies its chat template once at dataset-build time, so its linearization point is the
  generation prompt's last token, while the truth run fits on full statements and reads on
  prefixes. The 2x2 above therefore changes two variables, not one, and this is the experiment
  that separates them.
- **System-level synthesis**, deferred on the PI's own "might be overkill" pending D1, which is
  now done.

Related work: Julian's A-LQR paper is arXiv:2604.19018, *Local Linearity of LLMs Enables
Activation Steering via Model-Based Linear Optimal Control*. Our per-statement fits corroborate
its local-linearity assumption strongly (R-squared 0.999 across a nine-layer hop, 0% wrong-sign),
and our two failure modes are exactly what a closed loop would fix.
