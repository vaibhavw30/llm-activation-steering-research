# Can you steer an LLM's "sense of truth"? Probing, control theory, and causal tests on gemma-2-2b

An interpretability research project on `google/gemma-2-2b` and `gemma-2-2b-it`. It asks whether
the linear "truth direction" that probes find in an LLM's activations is a **lever** that controls
what the model says, or only a **readout** of it. The project combines ML research with the
engineering needed to answer that rigorously: a tested Python codebase, GPU jobs on a national
supercomputer, pre-registered statistics, and LLM judges that are themselves validated.

> **In one sentence:** a linear probe reads "is this statement true?" off the model's activations
> at 99% accuracy. You can push the activations until that probe flips from TRUE to FALSE, and the
> model keeps telling the truth. The same pipeline *does* move behavior when pointed at refusal,
> so the null is a fact about how truth is represented, not a broken instrument.
>
> *We found the thermometer, not the thermostat: hold a lighter under it and the number climbs,
> but the room stays cold.*

**At a glance:** ~21k lines of Python across 93 modules · 690 passing tests · 45 SLURM jobs on
NCSA DeltaAI (NVIDIA GH200) · 250+ commits since June 2026 · 70 write-ups. Every number below
traces to a committed artifact.

---

## Highlights

| | |
|---|---|
| **Research question** | Is "truth" in an LLM a controllable direction, or a readout the model doesn't act on? |
| **Methods** | Linear vs. non-linear probing (logistic regression vs. XGBoost) across all 26 layers. Unsupervised steering-vector discovery (DCT, MAG). Jacobian-based backward reachability, borrowed from control theory. Sparse-autoencoder decomposition (GemmaScope). Activation steering with forward hooks. Gradient-trained steering vectors. |
| **Key result** | Truth is **linearly readable** (99% probe accuracy on clean data) but **not linearly actuatable**. The probe flips while behavior stays at baseline, on both fact datasets. |
| **Validation** | A positive control on refusal. The same pipeline flips 14 prompts into refusal against 0 the other way (exact McNemar p = 1.2e-4), and crossing the certified boundary predicts refusal with odds ratio 24.2. |
| **Engineering** | Resumable, smoke-tested GPU jobs. A 690-test pytest suite with fake models, so model code is tested on a laptop. Pre-registered decision rules. Four silent bugs that would each have produced a wrong scientific conclusion were caught and fixed (see below). |
| **Stack** | Python 3.13, PyTorch, Hugging Face Transformers, scikit-learn, XGBoost, NumPy/SciPy/pandas, Matplotlib, pytest, SLURM, Llama-2-7B / OLMo-3-7B judges |

---

## Results

### 1. Truth is linearly encoded on clean data, with non-linear headroom on messy data

For every statement in four true/false datasets, the project extracts the residual-stream
activation at every layer. It then trains a linear probe and a gradient-boosted tree probe on each
layer. On clean facts ("The city of X is in Y") the linear probe is as good as XGBoost: 0.990 vs
0.993. On messy general claims, XGBoost opens an 8.2-point gap, so some of the truth signal there
is not linear.

<p align="center"><img src="results/plots/plot_summary_maxgap.png" width="560" alt="Non-linear gap (XGBoost minus linear probe accuracy) at the best layer, per dataset"></p>

### 2. The probe can be flipped, and the model still tells the truth

Following a control-theory framing, a certificate `eps*` is computed: the smallest nudge at layer
11 that moves the layer-20 activation into the probe's FALSE half-space. The nudge is built from
the Jacobian between the two layers. On `cities` the certificate is inside the region where the
linearization holds, and the readout actuates almost perfectly linearly (R² = 0.999). Yet the
model's lie rate stays at its 3% baseline at every magnitude, and its completions stay coherent
and true.

<p align="center"><img src="plot_reach_audit_dissociation.png" width="760" alt="Behavior stays truthful while the target-layer readout moves under steering"></p>

The project then tried to explain the null away and failed:

- A pre-registered dose sweep found **0 clean windows out of 120 cells**, so it is not a matter of
  pushing too hard or too softly.
- Retargeting to the model's next-token decision flips 200/200 outputs. But a follow-up audit of
  our own methodology showed that at most 5.5% of those flips made the statement false. Most of
  them changed the token " the" rather than the country.
- That audit tested four working assumptions behind the steering experiments and refuted all
  four. So instead of trusting the harness, the project built a positive control for it (below).

### 3. The same pipeline moves refusal (the positive control)

A null is only meaningful if the instrument can detect an effect. Run unchanged on refusal in
`gemma-2-2b-it`, the pipeline:

- flips 14 compliant prompts into refusal against 0 the other way (exact McNemar p = 1.2e-4);
- produces the opposite effect when the push is reversed (odds ratio 26.6);
- shows that **crossing the certified boundary**, not the size of the push, predicts refusal
  (Mantel-Haenszel odds ratio 24.2 with push magnitude held fixed).

| Outcome | Meaning | Observed on |
|---|---|---|
| **actuatable** | readout crosses the boundary and behavior follows | refusal |
| **readout-only** | readout crosses, behavior does not | truth (cities) |
| **inert** | behavior moves without a readout crossing | truth (TruthfulQA) |

### 4. TruthfulQA: behavior moves, but through form

On TruthfulQA the base model is truthful only 28.1% of the time, so there is room to move. On 64
held-out questions, steering along the Jacobian-projected truth direction raises the truthful
rate from **0.281 to 0.516**. The paired test gives 16 questions gained and 1 lost, and three
norm-matched random directions stay at about 0.286. The gain runs through answer length: the
steered model hedges more, and TruthfulQA rewards hedging. The current round of experiments is
designed to separate *content* from *form* (see "Status").

A literature search found no published report of a certified-reachable but behaviorally inert
dissociation with a mechanism attached. That dissociation is the project's main contribution. It
connects to A-LQR (arXiv:2604.19018): our per-statement fits corroborate that paper's
local-linearity assumption (R² = 0.999 across a nine-layer hop), and our failure modes are the
kind a closed-loop controller would address.

---

## Engineering highlights

**Running on a supercomputer.** Anything that touches the model runs on NCSA DeltaAI, on NVIDIA
GH200 nodes, through 45 SLURM scripts. Each job:

- runs a **preflight** that fails fast on missing inputs;
- runs a **smoke test** on a few examples with prefixed output files, then deletes them;
- is **resumable at block granularity**, so a timeout never loses completed work and nothing is
  overwritten.

The partition allows two jobs per user and queues run 12+ hours, so a wasted slot costs a day.
That is why most of the rigor goes into catching bugs *before* submission.

**Testing ML code without a GPU.** The 690-test suite runs in about 8 seconds on a laptop. Model
code is tested against small fake tokenizers and fake residual networks whose behavior is known by
construction. That makes it possible to test log-probability masking, left-padding and position
ids, the gradient path of steering hooks, and optimizer convergence to a known optimum. Tests are
written to fail on the specific bug they guard against: for example, a training test that fails
if the learning rate is zero.

**Silent bugs caught before they became wrong conclusions:**

- **A library upgrade silently corrupted a model slice.** transformers ≥ 5 stopped re-scaling
  gemma-2's input embeddings. A third-party model slice then produced outputs with cosine ≈ 0.1 to
  the real hidden states, with no error. A startup fidelity probe now detects and compensates for
  it.
- **A judge was fed the wrong prompt.** The TruthfulQA "informativeness" judge had been getting
  the truth judge's prompt template (`True:` instead of `Helpful:`). This was found while
  auditing the judge, and a re-judging job was queued before any claim rested on that column.
- **A script run polluted the full run.** Run as a script, a module was imported twice, so a
  smoke run would have left unprefixed training outputs that the full run would silently reuse.
  This was found in review and is now pinned by a test that runs the file as `__main__`.
- **An optimizer converged to the wrong answer.** Projected Adam at a per-coordinate learning rate
  settles about 37° off the optimum in 2,304 dimensions (cosine 0.79). Adam on the tangent-projected
  gradient reaches 0.997+. This was verified on a problem with a known optimum before any GPU time
  was spent.

**Statistics that hold up.**

- **Paired tests** on the same held-out items: exact McNemar for 0/1 outcomes, Wilcoxon for
  margins.
- **Permutation nulls** from norm-matched random directions at the same dose.
- **Pre-registered bars**, written down before each job runs.
- **Wilson intervals** throughout.
- **Multiplicity-aware reporting:** primary claims are read at one registered dose, and everything
  else is labelled secondary.

**Validating the evaluators.** LLM-as-judge labels are checked against gold labels before being
trusted. The refusal judge scored 0.969 against a labelled gold set, the ceiling that set allows.
A 7B chat judge with apparently low agreement turned out to answer "refused" 63 to 77% of the time
regardless of input. The lesson, now encoded in a test: *a low agreement score is a trigger to run
a labelled check, not a verdict on either judge*.

---

## Research arc

Each step exists because the previous one produced a result that could not be interpreted without
it. The full story is in [`docs/PLAIN_ENGLISH_WALKTHROUGH.md`](docs/PLAIN_ENGLISH_WALKTHROUGH.md).

| # | Question | Answer | Write-up |
|---|---|---|---|
| 01 | Is truth encoded linearly? | Yes on clean data (0.990 linear vs 0.993 XGBoost); +0.082 non-linear headroom on messy claims | [`EXPLAINER.md`](docs/EXPLAINER.md) |
| 02 | Is the truth direction causally special? | No. Unsupervised DCT never surfaces it | [`DCT_VS_TRUTH_FINDINGS.md`](docs/DCT_VS_TRUTH_FINDINGS.md) |
| 03 | Reframe as backward reachability | Certificate `eps*`, the smallest layer-11 nudge that lands in the probe's FALSE set at layer 20 | [`REACHABILITY_RUNBOOK.md`](docs/REACHABILITY_RUNBOOK.md) |
| 04 | Does hitting the target change behavior? | **No.** The readout moves, the lie rate stays at baseline | [`REACH_AUDIT_FINDINGS.md`](docs/REACH_AUDIT_FINDINGS.md) |
| 05 | Was the target set the problem? | Token-space targets flip 200/200 outputs, but the audit shows they were mostly flips of " the" | [`TOKEN_SPACE_FINDINGS.md`](docs/TOKEN_SPACE_FINDINGS.md) |
| 06 | Do our own steering experiments survive an audit? | All four registered assumptions were refuted. Most sharply S4: at most 5.5% of the 200/200 certified flips made the claim false | [`AUDIT_SUMMARY.md`](docs/AUDIT_SUMMARY.md) |
| 07 | Did we push too hard, or too softly? | No. 0 clean windows out of 120 dose cells | [`D1_DOSE_RESPONSE.md`](docs/D1_DOSE_RESPONSE.md) |
| 08 | Can the pipeline move *any* behavior? | **Yes**, refusal: 14 vs 0, odds ratio 24.2 for crossing | [`REFUSAL_POSITIVE_CONTROL.md`](docs/REFUSAL_POSITIVE_CONTROL.md) |
| 09 | Does truth steer where there is headroom? | TruthfulQA truthful 0.281 → 0.516, beats random directions, runs through answer length | [`Q2_TRUTHFULQA_STEERING.md`](docs/Q2_TRUTHFULQA_STEERING.md) |
| 10 | Content or form, and what is the ceiling? | Queued on the cluster, see Status | [`PLAN_PI_FEEDBACK_2026-09-18.md`](docs/PLAN_PI_FEEDBACK_2026-09-18.md) |

---

## Status (19 September 2026)

**Done and audited:**
- the truth null on both fact datasets;
- the refusal positive control;
- TruthfulQA baseline and steering.

**Queued on the cluster:** five jobs in three rounds, answering the PI's latest feedback.

- **Re-judging.** Re-judge the TruthfulQA informativeness column with the correct prompt, and
  hand-label 64 answers blind as an independent check on the judge.
- **Unsupervised discovery on TruthfulQA.** Run DCT and MAG on TruthfulQA, then a **transfer
  matrix**: every direction, from both datasets, steered on both datasets.
- **J-E, a judge-free test.** Score TruthfulQA by the model's own log-probability of reference
  answers, which removes the judge and answer length from the measurement. Plus a **learned
  ceiling**: the best single steering vector at each norm, trained by gradient descent. The
  ceiling says whether a null means "truth isn't steerable this way" or "nothing is".

---

## Repository map

```
README.md          you are here
docs/              every write-up; docs/README.md is the index
deltaai/           SLURM scripts and runbooks for NCSA DeltaAI (GH200); deltaai/README.md maps them
got_datasets/      input CSVs: cities, sp_en_trans, companies, common_claim, truthfulqa, refusal
src/               all code, run from the repo root as `.venv/bin/python src/<name>.py`
tests/             pytest suite, 690 passing
results/           probe results and plots from the first experiment
(repo root)        experiment artifacts: <program>_<what>_<dataset>.{csv,json,npz,png}
```

| Program | Files in `src/` |
|---|---|
| Linear vs. non-linear probing | `extract.py` `analyze.py` `summary.py` |
| Unsupervised steering discovery | `dct.py` `dct_train.py` `run_dct_*.py` `mag/` `run_mag.py` |
| Backward reachability | `reach_*.py` `make_reach_meta.py` |
| Token-space steering | `token_geom.py` `token_jac.py` `token_sens.py` `token_steer.py` |
| Refusal positive control | `prep_refusal.py` `refusal_screen.py` `refusal_judge.py` `refusal_analyze.py` |
| Audits and dose response | `audit_*.py` `dose_response.py` `dose_analyze.py` |
| SAE decomposition | `sae_load.py` `sae_decompose.py` |
| TruthfulQA and transfer | `prep_truthfulqa.py` `tqa_*.py` `xfer_*.py` |
| Judges | `judges/` `judge_audit.py` `truthfulqa_judge.py` |

## Running things

The laptop runs the analysis and the tests. Anything that loads the model runs on DeltaAI: start
at [`deltaai/README.md`](deltaai/README.md). Commands in the runbooks are labelled **LAPTOP** or
**CLUSTER**.

```bash
.venv/bin/python src/extract.py cities.csv
```

```bash
.venv/bin/python src/analyze.py cities
```

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Two hazards:
- **Never import `xgboost` and `torch` in the same process.** Two copies of libomp segfault on
  macOS ARM.
- **Keep the analysis and DCT environments separate.** DCT pins transformers 4.51.3, and the
  analysis environment runs 5.x.

| venv | Where | For |
|---|---|---|
| `.venv` | laptop | analysis, tests (torch, transformers, scikit-learn) |
| `.venv-dct` | laptop | DCT's pinned stack (transformers 4.51.3) |
| `.venv-dct-gpu` | DeltaAI | model runs, built on the cluster's torch module |
| `.venv-judge-gpu` | DeltaAI | 7B judge models |

## Further reading

| If you want | Read |
|---|---|
| The whole project in plain English (45 min) | [`docs/PLAIN_ENGLISH_WALKTHROUGH.md`](docs/PLAIN_ENGLISH_WALKTHROUGH.md) |
| The 20-minute spoken version | [`docs/MEETING_20MIN.md`](docs/MEETING_20MIN.md) |
| The open questions | [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) |
| Related work | [`docs/LITERATURE.md`](docs/LITERATURE.md) |
